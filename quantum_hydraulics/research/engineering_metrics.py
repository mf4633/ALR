"""
Engineering Post-Processing Metrics.

Scenario-specific metrics computed on top of existing CellMetrics from
the Tier 1/Tier 2 pipeline.  Does NOT modify swmm_2d.py.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional

from quantum_hydraulics.integration.swmm_2d import (
    RHO, NU, G, _vectorized_colebrook_white, _vectorized_scour_risk,
)
from quantum_hydraulics.integration.swmm_node import SedimentProperties


# ── Permissible shear (USACE EM 1110-2-1601) ─────────────────────────────

PERMISSIBLE_SHEAR_PSF = {
    "bare_soil":   0.02,
    "grass_poor":  0.15,
    "grass_good":  0.35,
    "gravel_mulch": 0.60,
    "riprap_6in":  1.50,
    "riprap_12in": 3.50,
    "gabions":     5.00,
    "concrete":   25.00,
}


# ── Result containers ─────────────────────────────────────────────────────

@dataclass
class BankErosionAssessment:
    bank_shear: np.ndarray          # tau_bank per cell (psf)
    max_bank_shear: float           # psf
    mean_bank_shear: float          # psf
    k_bank: float                   # ratio used
    permissible_shear: float        # psf for selected bank material
    bank_material: str
    factor_of_safety: float         # permissible / max_bank_shear
    assessment: str                 # STABLE / EROSION LIKELY / FAILURE IMMINENT


@dataclass
class DegradationAssessment:
    upstream_mean_v: float          # ft/s
    downstream_mean_v: float        # ft/s
    upstream_transport: float       # lb/ft/s (reach-averaged)
    downstream_transport: float     # lb/ft/s
    transport_deficit: float        # downstream - upstream (positive = degrading)
    annual_degradation_ft: float    # ft/yr
    assessment: str                 # DEGRADING / STABLE / AGGRADING


@dataclass
class CulvertOutletAssessment:
    max_plunge_shear: float         # psf
    max_plunge_velocity: float      # ft/s
    jet_exit_velocity: float        # ft/s
    required_riprap_d50_in: float   # inches (Lane's formula)
    required_apron_length_ft: float # ft
    scour_risk: float               # 0-1
    assessment: str


@dataclass
class BendAssessment:
    outer_mean_shear: float         # psf
    inner_mean_shear: float         # psf
    approach_mean_shear: float      # psf
    amplification_factor: float     # outer / approach
    r_over_w: float                 # bend radius / channel width
    bend_scour_depth_ft: float      # Lacey-type estimate
    assessment: str


# ── Free surface correction (Bernoulli) ──────────────────────────────────

@dataclass
class FreeSurfaceResult:
    eta: np.ndarray                  # water surface elevation (ft)
    corrected_depth: np.ndarray      # depth after Bernoulli correction (ft)
    drawdown: np.ndarray             # approach_depth - corrected_depth (ft)
    max_drawdown: float
    corrected_bed_shear: np.ndarray  # psf (re-computed with corrected depth)
    corrected_scour_risk: np.ndarray
    approach_velocity: float
    approach_depth: float


def compute_free_surface_correction(
    tier1_metrics,
    approach_v: float,
    approach_depth: float,
    roughness_ks: float = 0.1,
    sediment: Optional[SedimentProperties] = None,
) -> FreeSurfaceResult:
    """
    Apply Bernoulli equation to correct the rigid-lid depth field.

    eta = approach_depth + (V_approach^2 - V(x,y)^2) / (2g)

    Then re-compute Colebrook-White friction and bed shear with corrected depths.
    """
    if sediment is None:
        sediment = SedimentProperties.sand()

    v_mag = tier1_metrics.v_mag

    # Bernoulli: surface elevation
    eta = approach_depth + (approach_v ** 2 - v_mag ** 2) / (2.0 * G)

    # Corrected depth (clamp to avoid negatives at supercritical points)
    corrected_depth = np.maximum(eta, 0.05)
    drawdown = approach_depth - corrected_depth

    # Re-compute hydraulics with corrected depth
    Re_corr = v_mag * corrected_depth / NU
    epsilon_D_corr = roughness_ks / (4.0 * corrected_depth)
    f_corr = _vectorized_colebrook_white(Re_corr, epsilon_D_corr)
    u_star_corr = v_mag * np.sqrt(f_corr / 8.0)
    bed_shear_corr = RHO * u_star_corr ** 2

    # Re-compute scour risk with corrected shear
    tau_c = sediment.critical_shear_psf
    scour_risk_corr, _ = _vectorized_scour_risk(
        bed_shear_corr, tau_c,
        steepness=sediment.scour_steepness,
        midpoint=sediment.scour_midpoint,
    )

    return FreeSurfaceResult(
        eta=eta,
        corrected_depth=corrected_depth,
        drawdown=drawdown,
        max_drawdown=float(drawdown.max()),
        corrected_bed_shear=bed_shear_corr,
        corrected_scour_risk=scour_risk_corr,
        approach_velocity=approach_v,
        approach_depth=approach_depth,
    )


# ── Bank erosion ──────────────────────────────────────────────────────────

def compute_bank_shear(tier1_metrics, bank_mask,
                       k_bank: float = 0.75,
                       bank_material: str = "bare_soil") -> BankErosionAssessment:
    """
    Compute bank shear from bed shear using USACE bank-to-bed ratio.

    Parameters
    ----------
    tier1_metrics : CellMetrics
        From Mesh2DResults.compute_tier1()
    bank_mask : np.ndarray of bool
        Which cells are bank cells
    k_bank : float
        Bank-to-bed shear ratio (default 0.75 per EM 1110-2-1601)
    bank_material : str
        Key into PERMISSIBLE_SHEAR_PSF
    """
    bank_shear = tier1_metrics.bed_shear * k_bank

    bank_cells = bank_shear[bank_mask]
    if len(bank_cells) == 0:
        return BankErosionAssessment(
            bank_shear=bank_shear, max_bank_shear=0, mean_bank_shear=0,
            k_bank=k_bank, permissible_shear=0, bank_material=bank_material,
            factor_of_safety=99.0, assessment="NO BANK CELLS",
        )

    max_bs = float(bank_cells.max())
    mean_bs = float(bank_cells.mean())
    perm = PERMISSIBLE_SHEAR_PSF.get(bank_material, 0.02)
    fos = perm / max_bs if max_bs > 0 else 99.0

    if fos < 0.8:
        assessment = "BANK FAILURE IMMINENT"
    elif fos < 1.0:
        assessment = "BANK EROSION LIKELY"
    elif fos < 1.5:
        assessment = "MARGINALLY STABLE"
    else:
        assessment = "BANK STABLE"

    return BankErosionAssessment(
        bank_shear=bank_shear,
        max_bank_shear=max_bs,
        mean_bank_shear=mean_bs,
        k_bank=k_bank,
        permissible_shear=perm,
        bank_material=bank_material,
        factor_of_safety=fos,
        assessment=assessment,
    )


# ── Bed degradation ──────────────────────────────────────────────────────

def compute_degradation(tier1_metrics, upstream_mask, downstream_mask,
                        channel_width: float,
                        porosity: float = 0.40,
                        storm_hours_per_year: float = 100.0) -> DegradationAssessment:
    """
    Compare transport capacity between upstream and downstream reaches.

    Positive deficit = downstream can carry more than upstream supplies → degradation.
    """
    v_up = float(tier1_metrics.v_mag[upstream_mask].mean())
    v_dn = float(tier1_metrics.v_mag[downstream_mask].mean())
    tr_up = float(tier1_metrics.transport_rate[upstream_mask].mean())
    tr_dn = float(tier1_metrics.transport_rate[downstream_mask].mean())

    deficit = tr_dn - tr_up  # lb/ft/s

    # Convert to annual degradation depth
    # transport_rate is lb/ft/s per unit width
    # Volume rate = transport_rate / (rho_s * g) in ft^3/ft/s... but MPM already
    # returns mass flux.  Use the same approach as swmm_2d.py scour_depth:
    #   scour_depth = q_v * hours * 3600 / (1 - porosity)
    # where q_v = transport_rate / (rho_s * g) approximately
    rho_s_g = 5.14 * G  # slug/ft3 * ft/s2 = lb/ft3/s2... need careful units
    # Actually: transport_rate from MPM is already in lb/ft/s
    # Volume flux q_v = transport_rate / (rho_s * g) in ft3/ft/s (approx)
    # But simpler: use deficit as indicator, degradation ~ deficit * time / (rho_bulk * width)
    # rho_bulk of sand ~ 100 lb/ft3 (compacted)
    rho_bulk = 100.0  # lb/ft3
    if deficit > 0:
        # ft3/ft/s of eroded volume per unit width
        vol_rate = deficit / rho_bulk  # ft3/ft/s per unit width... rough
        annual_depth = vol_rate * storm_hours_per_year * 3600 / (1.0 - porosity)
        annual_depth = min(annual_depth, 10.0)  # cap
    else:
        annual_depth = 0.0

    if deficit > 0.001:
        assessment = "DEGRADING"
    elif deficit < -0.001:
        assessment = "AGGRADING"
    else:
        assessment = "STABLE"

    return DegradationAssessment(
        upstream_mean_v=v_up,
        downstream_mean_v=v_dn,
        upstream_transport=tr_up,
        downstream_transport=tr_dn,
        transport_deficit=deficit,
        annual_degradation_ft=annual_depth,
        assessment=assessment,
    )


# ── Culvert outlet ────────────────────────────────────────────────────────

def compute_culvert_outlet(tier1_metrics, jet_mask, plunge_mask,
                           tailwater_depth: float = 2.0,
                           scour_steepness: float = 2.5,
                           scour_midpoint: float = 1.0) -> CulvertOutletAssessment:
    """
    Compute scour metrics at a culvert outlet plunge zone.
    """
    jet_v = float(tier1_metrics.v_mag[jet_mask].max()) if jet_mask.any() else 0
    plunge_shear = float(tier1_metrics.bed_shear[plunge_mask].max()) if plunge_mask.any() else 0
    plunge_v = float(tier1_metrics.v_mag[plunge_mask].max()) if plunge_mask.any() else 0

    # Riprap sizing per HEC-14: use JET EXIT velocity (not decayed plunge velocity)
    # because the stone must survive the initial impact.
    # Lane's formula: D50 = 0.020 * V^2 (inches) for V < 10 fps
    #                 D50 = 0.025 * V^2 for V >= 10 fps
    design_v = max(jet_v, plunge_v)  # conservative: use whichever is higher
    if design_v >= 10:
        riprap_d50 = 0.025 * design_v ** 2
    else:
        riprap_d50 = 0.020 * design_v ** 2

    # Apron length (HEC-14 guidance: function of tailwater depth and jet velocity)
    apron_length = 3.0 * tailwater_depth + 0.25 * design_v

    # Scour risk (reuse logistic formula)
    tau_c = 0.10  # sand
    excess = plunge_shear / tau_c if tau_c > 0 else 0
    scour_risk = 1.0 / (1.0 + np.exp(-scour_steepness * (excess - scour_midpoint)))
    scour_risk = float(np.clip(scour_risk, 0, 1))

    if scour_risk > 0.8:
        assessment = "CRITICAL — Energy dissipation REQUIRED"
    elif scour_risk > 0.5:
        assessment = "HIGH — Riprap apron recommended"
    elif scour_risk > 0.2:
        assessment = "MODERATE — Monitor outlet conditions"
    else:
        assessment = "LOW — Acceptable"

    return CulvertOutletAssessment(
        max_plunge_shear=plunge_shear,
        max_plunge_velocity=plunge_v,
        jet_exit_velocity=jet_v,
        required_riprap_d50_in=riprap_d50,
        required_apron_length_ft=apron_length,
        scour_risk=scour_risk,
        assessment=assessment,
    )


# ── Channel bend ──────────────────────────────────────────────────────────

def compute_bend_metrics(tier1_metrics, approach_mask, outer_mask, inner_mask,
                         R_centerline: float, channel_width: float,
                         depth: float = 4.0) -> BendAssessment:
    """
    Compute bend shear amplification and scour estimate.
    """
    approach_shear = float(tier1_metrics.bed_shear[approach_mask].mean()) if approach_mask.any() else 0
    outer_shear = float(tier1_metrics.bed_shear[outer_mask].mean()) if outer_mask.any() else 0
    inner_shear = float(tier1_metrics.bed_shear[inner_mask].mean()) if inner_mask.any() else 0

    amplification = outer_shear / approach_shear if approach_shear > 0 else 1.0
    r_over_w = R_centerline / channel_width

    # Lacey-type bend scour estimate (simplified)
    # Bend scour depth ≈ depth * (amplification - 1) * 2
    bend_scour = depth * max(0, amplification - 1.0) * 2.0

    if amplification > 1.5:
        assessment = "SEVERE — Outer bank protection REQUIRED"
    elif amplification > 1.2:
        assessment = "MODERATE — Outer bank protection recommended"
    elif amplification > 1.05:
        assessment = "MILD — Monitor outer bank"
    else:
        assessment = "NEGLIGIBLE"

    return BendAssessment(
        outer_mean_shear=outer_shear,
        inner_mean_shear=inner_shear,
        approach_mean_shear=approach_shear,
        amplification_factor=amplification,
        r_over_w=r_over_w,
        bend_scour_depth_ft=bend_scour,
        assessment=assessment,
    )
