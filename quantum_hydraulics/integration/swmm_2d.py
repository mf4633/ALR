"""
SWMM 2D Mesh Post-Processor — Quantum turbulence analysis for 2D models.

Two-tier architecture:
  Tier 1: Vectorized numpy scan across all cells (V, Fr, Re, f, u*, tau, scour).
  Tier 2: Full vortex particle analysis at hotspot cells only.

Input: CSV with columns (time, cell_id, x, y, depth, vx, vy) exported from PCSWMM,
or numpy arrays passed directly.
"""

import csv
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

try:
    from scipy.spatial import cKDTree
    HAS_SCIPY = True
except ImportError:
    cKDTree = None
    HAS_SCIPY = False

try:
    from numba import njit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

from quantum_hydraulics.integration.swmm_node import (
    SedimentProperties,
    _compute_velocity_induction_fast,
)

# ── Physical constants (must match swmm_node.py and hydraulics.py) ──────────
RHO = 1.94       # slug/ft3
NU = 1.1e-5      # ft2/s
G = 32.2         # ft/s2
KAPPA = 0.41     # von Karman constant


# ── Vectorized Colebrook-White ──────────────────────────────────────────────

def _vectorized_colebrook_white(Re, epsilon_D, max_iter=20, tol=1e-8):
    """
    Solve Colebrook-White for arrays of Re and relative roughness.

    Parameters
    ----------
    Re : np.ndarray
        Reynolds numbers, shape (N,)
    epsilon_D : np.ndarray
        Relative roughness ks/(4R), shape (N,)

    Returns
    -------
    np.ndarray
        Darcy-Weisbach friction factor, shape (N,)
    """
    f = np.full_like(Re, 0.02, dtype=np.float64)

    # Mask: only solve for turbulent flow with valid Re
    mask = Re > 2300

    if not np.any(mask):
        # All laminar or zero
        result = np.where(Re > 1e-6, 64.0 / np.maximum(Re, 1e-6), 0.02)
        return result

    # Laminar
    f[~mask & (Re > 1e-6)] = 64.0 / Re[~mask & (Re > 1e-6)]

    # Turbulent: iterative Colebrook-White
    Re_m = Re[mask]
    eps_m = epsilon_D[mask]
    f_m = np.full(Re_m.shape, 0.02)

    for _ in range(max_iter):
        term1 = eps_m / 3.7
        term2 = 2.51 / (Re_m * np.sqrt(f_m))
        f_new = (-2.0 * np.log10(term1 + term2)) ** (-2)
        if np.all(np.abs(f_new - f_m) < tol):
            f_m = f_new
            break
        f_m = f_new

    f[mask] = f_m
    return f


# ── Vectorized scour / transport ────────────────────────────────────────────

def _vectorized_scour_risk(bed_shear, tau_c, steepness=2.5, midpoint=1.0):
    """Calibrated logistic scour risk and excess shear ratio for arrays."""
    excess = bed_shear / tau_c
    risk = 1.0 / (1.0 + np.exp(-steepness * (excess - midpoint)))
    return np.clip(risk, 0.0, 1.0), excess


def _vectorized_shields(bed_shear, rho_s, d_ft):
    """Shields parameter for arrays."""
    denom = (rho_s - RHO) * G * d_ft
    if denom <= 0:
        return np.zeros_like(bed_shear)
    return bed_shear / denom


def _vectorized_meyer_peter_muller(bed_shear, tau_c, d_ft, rho_s):
    """
    Meyer-Peter Muller bedload transport for arrays.

    Returns (transport_rate_lb_ft_s, scour_depth_ft_year)
    """
    s = rho_s / RHO
    denom = (rho_s - RHO) * G * d_ft

    tau_star = np.where(denom > 0, bed_shear / denom, 0.0)
    tau_star_c = tau_c / denom if denom > 0 else 0.047

    active = tau_star > tau_star_c
    excess = np.maximum(tau_star - tau_star_c, 0.0)
    phi = np.where(active, 8.0 * excess ** 1.5, 0.0)

    q_v = phi * np.sqrt(np.maximum((s - 1) * G * d_ft ** 3, 0.0))
    q_s = q_v * rho_s * G

    storm_hours = 100.0
    porosity = 0.4
    scour_depth = np.minimum(q_v * storm_hours * 3600 / (1.0 - porosity), 10.0)

    return q_s, scour_depth


# ── Data classes ────────────────────────────────────────────────────────────

@dataclass
class CellMetrics:
    """Tier 1 results for all cells at one timestep."""

    cell_id: np.ndarray
    x: np.ndarray
    y: np.ndarray
    depth: np.ndarray
    vx: np.ndarray
    vy: np.ndarray
    v_mag: np.ndarray
    froude: np.ndarray
    reynolds: np.ndarray
    friction_factor: np.ndarray
    u_star: np.ndarray
    bed_shear: np.ndarray
    scour_risk: np.ndarray
    shields: np.ndarray
    excess_shear_ratio: np.ndarray
    transport_rate: np.ndarray
    scour_depth: np.ndarray

    # Velocity gradients (computed separately, may be None)
    dvdx: Optional[np.ndarray] = None
    dvdy: Optional[np.ndarray] = None
    grad_mag: Optional[np.ndarray] = None

    @property
    def n_cells(self):
        return len(self.cell_id)

    def get_hotspot_indices(self, top_n, metric="v_mag"):
        """Return indices of top_n cells sorted descending by metric."""
        arr = getattr(self, metric)
        if arr is None:
            raise ValueError(f"Metric '{metric}' not computed")
        idx = np.argsort(arr)[::-1]
        return idx[:top_n]


@dataclass
class QuantumCellResult:
    """Tier 2 vortex particle results for a single cell."""

    cell_id: int
    x: float
    y: float
    tier1_bed_shear: float
    quantum_u_star: float
    quantum_bed_shear: float
    quantum_scour_risk: float
    quantum_shields: float
    quantum_excess_shear: float
    quantum_transport_rate: float
    quantum_scour_depth: float
    tke: float
    n_particles: int
    amplification_factor: float

    def to_dict(self):
        return {
            "cell_id": int(self.cell_id),
            "x": float(self.x),
            "y": float(self.y),
            "tier1_bed_shear": self.tier1_bed_shear,
            "quantum_bed_shear": self.quantum_bed_shear,
            "quantum_scour_risk": self.quantum_scour_risk,
            "quantum_shields": self.quantum_shields,
            "tke": self.tke,
            "n_particles": self.n_particles,
            "amplification_factor": self.amplification_factor,
        }


# ── Mesh2DResults ───────────────────────────────────────────────────────────

class Mesh2DResults:
    """
    2D mesh results for a single timestep.

    Structure-of-Arrays: all data as numpy arrays for vectorized computation.

    Parameters
    ----------
    cell_ids : np.ndarray
        Cell identifiers, shape (N,)
    x, y : np.ndarray
        Cell centroids (ft), shape (N,)
    depth : np.ndarray
        Water depth (ft), shape (N,)
    vx, vy : np.ndarray
        Velocity components (ft/s), shape (N,)
    roughness_ks : float
        Equivalent sand roughness (ft)
    sediment : SedimentProperties
        Sediment for scour analysis
    """

    def __init__(self, cell_ids, x, y, depth, vx, vy,
                 roughness_ks=0.1, sediment=None):
        self.cell_ids = np.asarray(cell_ids, dtype=np.int64)
        self.x = np.asarray(x, dtype=np.float64)
        self.y = np.asarray(y, dtype=np.float64)
        self.depth = np.asarray(depth, dtype=np.float64)
        self.vx = np.asarray(vx, dtype=np.float64)
        self.vy = np.asarray(vy, dtype=np.float64)
        self.ks = roughness_ks
        self.sediment = sediment or SedimentProperties.sand()

        n = len(self.cell_ids)
        assert all(len(a) == n for a in [self.x, self.y, self.depth, self.vx, self.vy])

        self._metrics: Optional[CellMetrics] = None

    @property
    def n_cells(self):
        return len(self.cell_ids)

    def compute_tier1(self) -> CellMetrics:
        """
        Tier 1: fast vectorized hydraulics at every cell.

        Computes velocity magnitude, Froude, Reynolds, Colebrook-White friction,
        friction velocity, bed shear, scour risk, Shields, and sediment transport.
        """
        depth_safe = np.maximum(self.depth, 0.01)
        v_mag = np.sqrt(self.vx ** 2 + self.vy ** 2)

        # Flow regime
        froude = v_mag / np.sqrt(G * depth_safe)
        reynolds = v_mag * depth_safe / NU

        # Friction (Colebrook-White, R ~ depth for 2D shallow flow)
        epsilon_D = self.ks / (4.0 * depth_safe)
        friction_factor = _vectorized_colebrook_white(reynolds, epsilon_D)

        # Friction velocity and bed shear
        u_star = v_mag * np.sqrt(friction_factor / 8.0)
        bed_shear = RHO * u_star ** 2

        # Scour metrics
        tau_c = self.sediment.critical_shear_psf
        d_ft = self.sediment.d50_mm / 304.8
        rho_s = self.sediment.density_slugs_ft3

        scour_risk, excess = _vectorized_scour_risk(
            bed_shear, tau_c,
            steepness=self.sediment.scour_steepness,
            midpoint=self.sediment.scour_midpoint,
        )
        shields = _vectorized_shields(bed_shear, rho_s, d_ft)
        transport_rate, scour_depth = _vectorized_meyer_peter_muller(
            bed_shear, tau_c, d_ft, rho_s
        )

        self._metrics = CellMetrics(
            cell_id=self.cell_ids,
            x=self.x, y=self.y,
            depth=self.depth,
            vx=self.vx, vy=self.vy,
            v_mag=v_mag,
            froude=froude,
            reynolds=reynolds,
            friction_factor=friction_factor,
            u_star=u_star,
            bed_shear=bed_shear,
            scour_risk=scour_risk,
            shields=shields,
            excess_shear_ratio=excess,
            transport_rate=transport_rate,
            scour_depth=scour_depth,
        )
        return self._metrics

    def compute_velocity_gradients(self, k_neighbors=6) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute dV/dx and dV/dy from neighboring cells via least-squares.

        Uses cKDTree for neighbor search. Returns (dvdx, dvdy, grad_mag).
        """
        if self._metrics is None:
            self.compute_tier1()

        n = self.n_cells
        v_mag = self._metrics.v_mag
        dvdx = np.zeros(n, dtype=np.float64)
        dvdy = np.zeros(n, dtype=np.float64)

        if not HAS_SCIPY or n < k_neighbors + 1:
            grad_mag = np.zeros(n)
            self._metrics.dvdx = dvdx
            self._metrics.dvdy = dvdy
            self._metrics.grad_mag = grad_mag
            return dvdx, dvdy, grad_mag

        pts = np.column_stack([self.x, self.y])
        tree = cKDTree(pts)

        # For each cell, fit V = a*x + b*y + c over k nearest neighbors
        _, idx = tree.query(pts, k=k_neighbors + 1)  # includes self

        for i in range(n):
            neighbors = idx[i, 1:]  # exclude self
            dx = self.x[neighbors] - self.x[i]
            dy = self.y[neighbors] - self.y[i]
            dv = v_mag[neighbors] - v_mag[i]

            # Least-squares: [dx, dy] @ [a, b] = dv
            A = np.column_stack([dx, dy])
            try:
                result, _, _, _ = np.linalg.lstsq(A, dv, rcond=None)
                dvdx[i] = result[0]
                dvdy[i] = result[1]
            except np.linalg.LinAlgError:
                pass

        grad_mag = np.sqrt(dvdx ** 2 + dvdy ** 2)
        self._metrics.dvdx = dvdx
        self._metrics.dvdy = dvdy
        self._metrics.grad_mag = grad_mag
        return dvdx, dvdy, grad_mag

    def get_hotspots(self, top_n=20, metric="v_mag"):
        """Return cell indices ranked by metric, descending."""
        if self._metrics is None:
            self.compute_tier1()
        return self._metrics.get_hotspot_indices(top_n, metric)

    def compute_tier2(self, cell_indices, n_particles=300,
                      cell_size=5.0) -> List[QuantumCellResult]:
        """
        Tier 2: full vortex particle analysis at selected cells.

        Injects particles in a local control volume, runs Biot-Savart
        velocity induction, and computes turbulence-augmented scour metrics.

        Parameters
        ----------
        cell_indices : array-like
            Indices of cells to analyze
        n_particles : int
            Particles per cell analysis
        cell_size : float
            Nominal cell size (ft) for control volume width

        Returns
        -------
        list of QuantumCellResult
        """
        if self._metrics is None:
            self.compute_tier1()

        rng = np.random.default_rng(42)
        results = []

        tau_c = self.sediment.critical_shear_psf
        d_ft = self.sediment.d50_mm / 304.8
        rho_s = self.sediment.density_slugs_ft3

        for idx in cell_indices:
            depth = float(self.depth[idx])
            v_mag = float(self._metrics.v_mag[idx])
            u_star_base = float(self._metrics.u_star[idx])
            tau_base = float(self._metrics.bed_shear[idx])

            if depth <= 0.01 or v_mag <= 0.01:
                continue

            # Local control volume
            width = cell_size
            length = max(2.0 * depth, cell_size)

            # Inject particles throughout control volume
            z_rand = rng.uniform(0.05 * depth, 0.95 * depth, n_particles)
            x_rand = rng.uniform(0, length, n_particles)
            y_rand = rng.uniform(-width / 2, width / 2, n_particles)

            positions = np.column_stack([x_rand, y_rand, z_rand])

            # Vorticity from log-law velocity profile
            z0 = max(self.ks / 30.0, 1e-6)
            z_safe = np.maximum(z_rand, z0)
            u_at_z = (u_star_base / KAPPA) * np.log(z_safe / z0)

            # Adaptive core size (smaller near bed)
            sigmas = 0.05 + (z_rand / depth) * 0.4

            vorticities = np.column_stack([
                u_at_z,
                np.zeros(n_particles),
                np.zeros(n_particles),
            ])

            # Biot-Savart velocity induction
            induced = _compute_velocity_induction_fast(
                positions, vorticities, sigmas
            )

            # Reynolds stress from induced velocity fluctuations
            u_fluct = induced[:, 0]
            w_fluct = induced[:, 2]
            reynolds_stress = np.mean(u_fluct * w_fluct)
            u_star_turb = np.sqrt(np.abs(reynolds_stress))

            # Effective friction velocity (augmented by turbulence)
            u_star_eff = max(u_star_turb, u_star_base * 1.2)
            tau_quantum = RHO * u_star_eff ** 2

            # TKE from induced velocities
            tke = 0.5 * np.mean(np.sum(induced ** 2, axis=1))

            # Recompute scour metrics with augmented shear (calibrated)
            excess_q = tau_quantum / tau_c
            _k = self.sediment.scour_steepness
            _m = self.sediment.scour_midpoint
            risk_q = 1.0 / (1.0 + np.exp(-_k * (excess_q - _m)))
            risk_q = np.clip(risk_q, 0.0, 1.0)

            shields_q = tau_quantum / ((rho_s - RHO) * G * d_ft) if d_ft > 0 else 0.0

            # Meyer-Peter Muller with augmented shear
            tau_star_q = tau_quantum / ((rho_s - RHO) * G * d_ft) if d_ft > 0 else 0.0
            tau_star_c = tau_c / ((rho_s - RHO) * G * d_ft) if d_ft > 0 else 0.047
            if tau_star_q > tau_star_c:
                s = rho_s / RHO
                phi = 8.0 * (tau_star_q - tau_star_c) ** 1.5
                q_v = phi * np.sqrt(max((s - 1) * G * d_ft ** 3, 0.0))
                transport_q = q_v * rho_s * G
                scour_q = min(q_v * 100.0 * 3600 / 0.6, 10.0)
            else:
                transport_q = 0.0
                scour_q = 0.0

            amplification = tau_quantum / tau_base if tau_base > 0 else 1.0

            results.append(QuantumCellResult(
                cell_id=int(self.cell_ids[idx]),
                x=float(self.x[idx]),
                y=float(self.y[idx]),
                tier1_bed_shear=tau_base,
                quantum_u_star=u_star_eff,
                quantum_bed_shear=tau_quantum,
                quantum_scour_risk=float(risk_q),
                quantum_shields=float(shields_q),
                quantum_excess_shear=float(excess_q),
                quantum_transport_rate=transport_q,
                quantum_scour_depth=scour_q,
                tke=float(tke),
                n_particles=n_particles,
                amplification_factor=amplification,
            ))

        return results


# ── SWMM2DPostProcessor ────────────────────────────────────────────────────

class SWMM2DPostProcessor:
    """
    Post-processor for 2D SWMM (PCSWMM) mesh results.

    Parameters
    ----------
    roughness_ks : float
        Equivalent sand roughness (ft), default 0.1
    sediment : SedimentProperties
        Sediment type, default sand
    cell_size : float
        Nominal mesh cell size (ft), default 5.0
    """

    def __init__(self, roughness_ks=0.1, sediment=None, cell_size=5.0):
        self.ks = roughness_ks
        self.sediment = sediment or SedimentProperties.sand()
        self.cell_size = cell_size

    def load_csv(self, csv_path) -> Dict[str, "Mesh2DResults"]:
        """
        Load 2D mesh results from CSV.

        Expected columns: time, cell_id, x, y, depth, vx, vy

        Returns dict mapping time label to Mesh2DResults.
        """
        # Group rows by timestep
        timesteps = {}  # {time_str: [rows]}

        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                t = row["time"].strip()
                if t not in timesteps:
                    timesteps[t] = []
                timesteps[t].append(row)

        results = {}
        for t, rows in timesteps.items():
            n = len(rows)
            cell_ids = np.array([int(r["cell_id"]) for r in rows], dtype=np.int64)
            x = np.array([float(r["x"]) for r in rows])
            y = np.array([float(r["y"]) for r in rows])
            depth = np.array([float(r["depth"]) for r in rows])
            vx = np.array([float(r["vx"]) for r in rows])
            vy = np.array([float(r["vy"]) for r in rows])

            results[t] = Mesh2DResults(
                cell_ids, x, y, depth, vx, vy,
                roughness_ks=self.ks,
                sediment=self.sediment,
            )

        return results

    def load_arrays(self, time_label, cell_ids, x, y, depth, vx, vy):
        """Create Mesh2DResults from numpy arrays."""
        return Mesh2DResults(
            cell_ids, x, y, depth, vx, vy,
            roughness_ks=self.ks,
            sediment=self.sediment,
        )

    def analyze(self, timestep_data, top_n_hotspots=20,
                tier2_particles=300, compute_gradients=True):
        """
        Full analysis across all timesteps.

        Parameters
        ----------
        timestep_data : dict
            {time_label: Mesh2DResults}
        top_n_hotspots : int
            Number of cells for Tier 2 analysis
        tier2_particles : int
            Particles per Tier 2 cell
        compute_gradients : bool
            Whether to compute velocity gradients

        Returns
        -------
        dict with keys: tier1, tier2, peak_envelope
        """
        all_tier1 = {}
        all_tier2 = {}

        # Track peak conditions across timesteps (element-wise max)
        peak_v_mag = None
        peak_bed_shear = None
        peak_scour_risk = None
        peak_time = None

        for t_label, mesh in timestep_data.items():
            # Tier 1
            metrics = mesh.compute_tier1()
            all_tier1[t_label] = metrics

            if compute_gradients:
                mesh.compute_velocity_gradients()

            # Track peaks
            if peak_v_mag is None:
                peak_v_mag = metrics.v_mag.copy()
                peak_bed_shear = metrics.bed_shear.copy()
                peak_scour_risk = metrics.scour_risk.copy()
                peak_time = t_label
            else:
                update_mask = metrics.v_mag > peak_v_mag
                peak_v_mag = np.maximum(peak_v_mag, metrics.v_mag)
                peak_bed_shear = np.maximum(peak_bed_shear, metrics.bed_shear)
                peak_scour_risk = np.maximum(peak_scour_risk, metrics.scour_risk)
                if np.sum(update_mask) > np.sum(~update_mask):
                    peak_time = t_label

        # Run Tier 2 on the peak timestep
        peak_mesh = timestep_data[peak_time]
        if peak_mesh._metrics is None:
            peak_mesh.compute_tier1()

        hotspot_idx = peak_mesh.get_hotspots(top_n_hotspots, metric="v_mag")
        tier2_results = peak_mesh.compute_tier2(
            hotspot_idx, n_particles=tier2_particles, cell_size=self.cell_size
        )
        all_tier2[peak_time] = tier2_results

        return {
            "tier1": all_tier1,
            "tier2": all_tier2,
            "peak_time": peak_time,
            "peak_v_mag": peak_v_mag,
            "peak_bed_shear": peak_bed_shear,
            "peak_scour_risk": peak_scour_risk,
            "hotspot_indices": hotspot_idx,
        }
