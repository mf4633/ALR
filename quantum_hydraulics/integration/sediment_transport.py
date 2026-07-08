"""
Quasi-Unsteady Sediment Transport Engine.

Steps through a hydrograph computing fractional bedload transport,
active-layer grain-size evolution (Hirano 1971), Egiazaroff hiding/exposure
correction, Exner equation bed evolution, and morphodynamic feedback.

References:
  Hirano, M. (1971). River bed degradation with armoring.
  Meyer-Peter, E. & Muller, R. (1948). Formulas for bed-load transport.
  Egiazaroff, I. (1965). Calculation of non-uniform sediment concentrations.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from quantum_hydraulics.integration.swmm_2d import RHO, NU, G, _vectorized_colebrook_white


# ── Grain-size classes ────────────────────────────────────────────────────

@dataclass
class SedimentFraction:
    """Single grain-size class within a mixture."""
    name: str
    d_mm: float
    tau_c_psf: float
    rho_s: float = 5.14  # slugs/ft3 (quartz)

    @property
    def d_ft(self) -> float:
        return self.d_mm / 304.8


@dataclass
class GrainSizeDistribution:
    """
    Multi-fraction grain-size distribution.

    ``percentages`` tracks what fraction of the active layer each size
    class occupies (sums to 1.0).
    """
    fractions: List[SedimentFraction]
    percentages: np.ndarray

    def __post_init__(self):
        self.percentages = np.asarray(self.percentages, dtype=np.float64)
        total = self.percentages.sum()
        if total > 0:
            self.percentages /= total

    @classmethod
    def default_sand_gravel(cls) -> "GrainSizeDistribution":
        """Standard 6-fraction sand-gravel mix."""
        fracs = [
            SedimentFraction("fine_sand",       0.25, 0.04),
            SedimentFraction("medium_sand",     0.50, 0.10),
            SedimentFraction("coarse_sand",     1.00, 0.15),
            SedimentFraction("very_coarse_sand", 2.00, 0.22),
            SedimentFraction("fine_gravel",     5.00, 0.35),
            SedimentFraction("medium_gravel",  10.00, 0.55),
        ]
        pct = np.array([0.10, 0.25, 0.25, 0.20, 0.12, 0.08])
        return cls(fracs, pct)

    @property
    def n_fractions(self) -> int:
        return len(self.fractions)

    def _cumulative(self):
        """Return sorted (d_mm, cumulative_pct) arrays."""
        d = np.array([f.d_mm for f in self.fractions])
        order = np.argsort(d)
        d_sorted = d[order]
        cum = np.cumsum(self.percentages[order])
        return d_sorted, cum

    @property
    def d50_mm(self) -> float:
        d, cum = self._cumulative()
        return float(np.interp(0.50, cum, d))

    @property
    def d90_mm(self) -> float:
        d, cum = self._cumulative()
        return float(np.interp(0.90, cum, d))

    def copy(self) -> "GrainSizeDistribution":
        return GrainSizeDistribution(
            list(self.fractions), self.percentages.copy()
        )


# ── Active layer (Hirano 1971) ───────────────────────────────────────────

class ActiveLayerModel:
    """
    Mass-conserving Hirano (1971) active-layer model.

    State is tracked as ABSOLUTE solids volume per unit bed area, per fraction:
    ``s`` (active layer) and ``b`` (substrate reservoir), plus cumulative
    ``exported``/``imported`` so total sediment mass is conserved to machine
    precision (invariant sum(s + b + exported - imported) = const).

    Armoring is EMERGENT, not hard-coded: transport removes fractions
    size-selectively (supply-limited), the substrate resupplies parent
    composition to maintain active-layer thickness, and fractions below their
    critical shear accumulate as a coarse lag that throttles transport.

    (The previous version tracked only renormalized percentages -- no absolute
    mass existed to conserve -- and re-injected fine-rich substrate every scour
    step, which is why it could not correctly armor.)
    """

    def __init__(self, initial_gradation: GrainSizeDistribution,
                 porosity: float = 0.40, substrate_depth_ft: float = 8.0):
        self.surface = initial_gradation.copy()      # view; .percentages = F_i
        self.porosity = porosity
        self.n = initial_gradation.n_fractions
        self._d_mm = np.array([f.d_mm for f in initial_gradation.fractions])
        self._d_median = initial_gradation.d50_mm    # fines = d < initial d50

        f0 = np.asarray(initial_gradation.percentages, dtype=np.float64)
        f0 = f0 / max(f0.sum(), 1e-30)
        self._f0 = f0.copy()

        d90_0 = self._d_pct(f0, 0.90)
        self.thickness_ft = max(2.0 * d90_0 / 304.8, 0.05)
        self.s = f0 * (1.0 - porosity) * self.thickness_ft   # active solids/area
        self.b = f0 * (1.0 - porosity) * substrate_depth_ft  # substrate reservoir
        self.exported = np.zeros(self.n)
        self.imported = np.zeros(self.n)
        self._total0 = self._total()
        self._sync_surface()

    def _total(self) -> float:
        return float(np.sum(self.s + self.b + self.exported - self.imported))

    @property
    def mass_residual(self) -> float:
        """Sediment mass balance residual (ft); should stay ~0."""
        return self._total() - self._total0

    def _F(self) -> np.ndarray:
        """Surface volume fractions from absolute active-layer volumes."""
        T = self.s.sum()
        return self.s / T if T > 0 else self._f0.copy()

    def _d_pct(self, F: np.ndarray, p: float) -> float:
        order = np.argsort(self._d_mm)
        cum = np.cumsum(np.asarray(F)[order])
        cum = cum / max(cum[-1], 1e-30)
        return float(np.interp(p, cum, self._d_mm[order]))

    def _sync_surface(self):
        self.surface.percentages = self._F()

    @property
    def is_armored(self) -> bool:
        """True if fine fractions (< initial d50) are mostly depleted."""
        F = self._F()
        fine_pct = float(np.sum(F[self._d_mm < self._d_median]))
        return fine_pct < 0.05

    def update(self, qs_out: np.ndarray, qs_in: np.ndarray,
               dt: float, length: float, depth: float) -> float:
        """
        Advance the active layer one sub-step and return the bed change dz (ft).

        Parameters
        ----------
        qs_out, qs_in : np.ndarray
            Per-fraction volumetric transport out of / into the reach
            (ft^3/ft/s), shape (n,).
        dt : float
            Sub-step duration (s).
        length : float
            Reach length (ft) -- the Exner divisor (NOT the width).
        depth : float
            Flow depth (ft), for the stability limiter.
        """
        # Length-based Exner, per fraction: solids/area eroded (>0 = scour)
        erod = (qs_out - qs_in) * dt / max(length, 1e-6)

        # Stability limiter (Courant on bulk bed change)
        dz = -erod.sum() / (1.0 - self.porosity)
        max_dz = 0.02 * max(depth, 0.1)
        if abs(dz) > max_dz > 0:
            scale = max_dz / abs(dz)
            erod = erod * scale
            qs_out = qs_out * scale
            qs_in = qs_in * scale

        # Supply limit: cannot erode more of fraction i than exists (active+sub)
        avail = self.s + self.b
        over = erod > avail
        if np.any(over):
            erod = np.where(over, avail, erod)
            qs_out = np.where(over, qs_in + erod * length / max(dt, 1e-30), qs_out)

        self.s = self.s - erod
        self.exported = self.exported + qs_out * dt / max(length, 1e-6)
        self.imported = self.imported + qs_in * dt / max(length, 1e-6)
        dz = -erod.sum() / (1.0 - self.porosity)

        # Any tiny residual negative from float error: borrow from same-fraction
        # substrate (mass-conserving, NOT a clamp-to-zero which would create mass)
        neg = self.s < 0
        if np.any(neg):
            fix = -self.s[neg]
            self.s[neg] = self.s[neg] + fix
            self.b[neg] = self.b[neg] - fix

        # Active-layer thickness maintenance via substrate exchange
        self.thickness_ft = max(2.0 * self._d_pct(self._F(), 0.90) / 304.8, 0.05)
        deficit = (1.0 - self.porosity) * self.thickness_ft - self.s.sum()
        if deficit > 0:                      # scour: pull parent material up
            Tb = self.b.sum()
            if Tb > 1e-15:
                take = min(deficit, Tb) * (self.b / Tb)
                self.s = self.s + take
                self.b = self.b - take
        elif deficit < 0:                    # aggradation: bury surface material
            Ts = self.s.sum()
            if Ts > 1e-15:
                give = (-deficit) * (self.s / Ts)
                self.s = self.s - give
                self.b = self.b + give

        self._sync_surface()
        return dz


# ── Channel reach ─────────────────────────────────────────────────────────

@dataclass
class ChannelReach:
    """Rectangular or trapezoidal channel for quasi-unsteady computation."""
    length_ft: float
    width_ft: float
    slope: float
    roughness_ks: float = 0.1
    side_slope: float = 0.0
    bed_elevation: float = 0.0

    def _geometry(self, depth):
        A = self.width_ft * depth + self.side_slope * depth ** 2
        P = self.width_ft + 2.0 * depth * np.sqrt(1.0 + self.side_slope ** 2)
        R = A / max(P, 0.01)
        return A, P, R

    def compute_normal_depth(self, Q: float, tol: float = 0.001,
                              max_iter: int = 50) -> float:
        """Bisection solver for normal depth using Colebrook-White."""
        if Q <= 0:
            return 0.01

        lo, hi = 0.01, 30.0
        for _ in range(max_iter):
            mid = (lo + hi) / 2.0
            A, P, R = self._geometry(mid)
            if R <= 0 or A <= 0:
                lo = mid
                continue
            V = Q / A
            Re = V * R / NU
            eps_D = self.roughness_ks / (4.0 * R)
            f_arr = _vectorized_colebrook_white(
                np.array([max(Re, 100.0)]), np.array([eps_D])
            )
            f = float(f_arr[0])
            # Q = A * sqrt(8gRS / f)
            Q_calc = A * np.sqrt(8.0 * G * R * self.slope / max(f, 1e-6))
            if Q_calc < Q:
                lo = mid
            else:
                hi = mid
            if abs(hi - lo) < tol:
                break
        return (lo + hi) / 2.0

    def compute_hydraulics(self, Q: float, depth: float) -> dict:
        """Compute V, Re, f, u*, tau for given Q and depth."""
        A, P, R = self._geometry(depth)
        V = Q / max(A, 0.01)
        Re = V * R / NU
        eps_D = self.roughness_ks / (4.0 * max(R, 0.01))
        f_arr = _vectorized_colebrook_white(
            np.array([max(Re, 100.0)]), np.array([eps_D])
        )
        f = float(f_arr[0])
        u_star = V * np.sqrt(f / 8.0)
        tau = RHO * u_star ** 2
        return {
            "depth": depth, "velocity": V, "area": A,
            "wetted_perimeter": P, "hydraulic_radius": R,
            "reynolds": Re, "friction_factor": f,
            "u_star": u_star, "bed_shear": tau,
        }


# ── Result containers ─────────────────────────────────────────────────────

@dataclass
class StepResult:
    """Results from a single quasi-unsteady time step."""
    time_hours: float
    discharge_cfs: float
    depth_ft: float
    velocity_fps: float
    bed_shear_psf: float
    total_transport_rate: float
    fractional_transport: np.ndarray
    delta_z_ft: float
    cumulative_z_ft: float
    surface_d50_mm: float
    surface_d90_mm: float
    is_armored: bool
    bed_elevation_ft: float


@dataclass
class SedimentTransportResults:
    """Accumulated results from a full quasi-unsteady run."""
    steps: List[StepResult]
    initial_gradation: GrainSizeDistribution
    final_gradation: GrainSizeDistribution
    channel_length: float
    channel_width: float

    @property
    def times(self): return np.array([s.time_hours for s in self.steps])
    @property
    def discharges(self): return np.array([s.discharge_cfs for s in self.steps])
    @property
    def bed_elevations(self): return np.array([s.bed_elevation_ft for s in self.steps])
    @property
    def cumulative_bed_change(self): return np.array([s.cumulative_z_ft for s in self.steps])
    @property
    def surface_d50(self): return np.array([s.surface_d50_mm for s in self.steps])
    @property
    def total_scour_ft(self): return self.steps[-1].cumulative_z_ft if self.steps else 0
    @property
    def max_scour_ft(self): return min(s.cumulative_z_ft for s in self.steps) if self.steps else 0
    @property
    def final_d50_mm(self): return self.final_gradation.d50_mm
    @property
    def armored(self): return self.steps[-1].is_armored if self.steps else False

    def get_assessment(self) -> str:
        scour = abs(self.max_scour_ft)
        if scour > 5.0:
            return "SEVERE"
        elif scour > 2.0:
            return "HIGH"
        elif scour > 0.5:
            return "MODERATE"
        else:
            return "LOW"


# ── Quasi-Unsteady Engine ────────────────────────────────────────────────

class QuasiUnsteadyEngine:
    """
    Steps through a hydrograph computing fractional transport,
    active-layer evolution, and morphodynamic bed feedback.

    Parameters
    ----------
    channel : ChannelReach
    sediment_mix : GrainSizeDistribution
    porosity : float
    upstream_feed_fraction : float
        0.0 = clear-water scour (dam), 1.0 = equilibrium feed
    computational_increment_hours : float
    bed_mixing_steps : int
    """

    def __init__(
        self,
        channel: ChannelReach,
        sediment_mix: GrainSizeDistribution,
        porosity: float = 0.40,
        upstream_feed_fraction: float = 0.0,
        computational_increment_hours: float = 1.0,
        bed_mixing_steps: int = 3,
        substrate_depth_ft: float = 8.0,
    ):
        self.channel = channel
        self.initial_gradation = sediment_mix.copy()
        self.active_layer = ActiveLayerModel(
            sediment_mix, porosity=porosity, substrate_depth_ft=substrate_depth_ft
        )
        self.porosity = porosity
        self.feed_fraction = upstream_feed_fraction
        self.increment_hours = computational_increment_hours
        self.mixing_steps = bed_mixing_steps
        self._hydrograph: List[Tuple[float, float]] = []  # (Q, duration_hours)

    def set_hydrograph_durations(self, records: List[Tuple[float, float]]):
        """Set hydrograph as (Q_cfs, duration_hours) records."""
        self._hydrograph = list(records)

    def set_hydrograph_timeseries(self, times_hours, discharges_cfs):
        """Convert time-series to duration records."""
        times = np.asarray(times_hours)
        Q = np.asarray(discharges_cfs)
        records = []
        for i in range(len(times) - 1):
            dt = times[i + 1] - times[i]
            q_avg = (Q[i] + Q[i + 1]) / 2.0
            records.append((q_avg, dt))
        self._hydrograph = records

    def _compute_fractional_transport(self, bed_shear, gradation):
        """
        Meyer-Peter Muller per fraction with Egiazaroff hiding/exposure.

        Returns transport rate per fraction (ft3/ft/s).
        """
        n = gradation.n_fractions
        q_frac = np.zeros(n)

        # Mean diameter of mixture
        d_m = sum(
            gradation.percentages[i] * gradation.fractions[i].d_ft
            for i in range(n)
        )
        d_m = max(d_m, 1e-8)

        for i in range(n):
            p_i = gradation.percentages[i]
            if p_i < 1e-6:
                continue

            frac = gradation.fractions[i]
            d_i = frac.d_ft
            rho_s = frac.rho_s
            s = rho_s / RHO

            # Critical shear for this fraction: the calibrated per-fraction table
            # value, with an Egiazaroff exposure easing applied ONLY to coarse
            # grains (xi^2 < 1).  Fines keep their (larger) table threshold, which
            # preserves the threshold spread that drives selective transport and
            # static armoring.  NOTE: the full Egiazaroff hiding form
            # (tau_c ~ xi^2) implements the equal-mobility hypothesis, which
            # collapses the per-fraction thresholds and eliminates clear-water
            # armoring entirely -- so it is deliberately not used here (verified).
            ratio = d_i / d_m
            if ratio > 0.01:
                xi = np.log(19.0) / np.log(max(19.0 * ratio, 1.01))
            else:
                xi = 1.0
            tau_c_corrected = frac.tau_c_psf * min(xi * xi, 1.0)

            # Shields parameter
            denom = (rho_s - RHO) * G * d_i
            if denom <= 0:
                continue
            tau_star = bed_shear / denom
            tau_star_c = tau_c_corrected / denom

            excess = tau_star - tau_star_c
            if excess <= 0:
                continue

            # MPM per fraction
            phi = 8.0 * excess ** 1.5
            q_i = p_i * phi * np.sqrt(max((s - 1.0) * G * d_i ** 3, 0.0))
            q_frac[i] = q_i

        return q_frac

    @property
    def mass_residual(self) -> float:
        """Sediment mass-balance residual (ft); should stay ~0."""
        return self.active_layer.mass_residual

    def run(self) -> SedimentTransportResults:
        """Execute the quasi-unsteady simulation."""
        steps = []
        cumulative_z = 0.0
        elapsed_hours = 0.0

        for Q, duration_hours in self._hydrograph:
            n_increments = max(1, int(np.ceil(duration_hours / self.increment_hours)))
            inc_hours = duration_hours / n_increments

            for _ in range(n_increments):
                dt_sub = (inc_hours * 3600.0) / self.mixing_steps

                for _ in range(self.mixing_steps):
                    # 1. Hydraulics at current bed elevation
                    depth = self.channel.compute_normal_depth(Q)
                    hyd = self.channel.compute_hydraulics(Q, depth)
                    tau = hyd["bed_shear"]

                    # 2. Fractional transport (per fraction, ft^3/ft/s)
                    q_frac = self._compute_fractional_transport(
                        tau, self.active_layer.surface
                    )
                    total_qs_out = q_frac.sum()
                    qs_in = q_frac * self.feed_fraction
                    total_qs_in = total_qs_out * self.feed_fraction

                    # 3. Mass-conserving active-layer + length-based Exner update;
                    #    returns the realized bed change (supply-limited).
                    delta_z = self.active_layer.update(
                        q_frac, qs_in, dt_sub, self.channel.length_ft, depth
                    )
                    cumulative_z += delta_z
                    self.channel.bed_elevation += delta_z

                elapsed_hours += inc_hours

                steps.append(StepResult(
                    time_hours=elapsed_hours,
                    discharge_cfs=Q,
                    depth_ft=depth,
                    velocity_fps=hyd["velocity"],
                    bed_shear_psf=tau,
                    total_transport_rate=total_qs_out,
                    fractional_transport=q_frac.copy(),
                    delta_z_ft=delta_z,
                    cumulative_z_ft=cumulative_z,
                    surface_d50_mm=self.active_layer.surface.d50_mm,
                    surface_d90_mm=self.active_layer.surface.d90_mm,
                    is_armored=self.active_layer.is_armored,
                    bed_elevation_ft=self.channel.bed_elevation,
                ))

        return SedimentTransportResults(
            steps=steps,
            initial_gradation=self.initial_gradation,
            final_gradation=self.active_layer.surface.copy(),
            channel_length=self.channel.length_ft,
            channel_width=self.channel.width_ft,
        )
