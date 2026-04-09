"""
HEC-18 / HEC-RAS Empirical Scour Equations.

Exact implementations of the standard scour equations used by FHWA HEC-18
(5th Edition) and USACE HEC-RAS for bridge scour estimation.  These are
INDEPENDENT of Quantum Hydraulics' physics-based approach and serve as
benchmarks for comparison.

Equations implemented:
  - CSU pier scour (HEC-18 Eq. 7.3) with K1-K4 correction factors
  - Froehlich pier scour (Froehlich 1988)
  - Live-bed contraction scour (Modified Laursen 1960)
  - Clear-water contraction scour (Laursen 1963)
  - HIRE abutment scour (HEC-18 Eq. 7.2)
  - Froehlich abutment scour (HEC-18 Eq. 7.1)
  - Critical velocity for live-bed / clear-water determination

Units:  English (ft, ft/s, cfs) unless noted.

References
----------
- FHWA HEC-18, 5th Ed. (2012): Evaluating Scour at Bridges
- USACE HEC-RAS 1D Technical Reference Manual, Ch. 12
- Froehlich, D.C. (1988): Analysis of Onsite Measurements of Scour at Piers
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional

G = 32.174  # ft/s^2


# ═══════════════════════════════════════════════════════════════════════════
# Result containers
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class PierScourResult:
    """Result from CSU or Froehlich pier scour equation."""
    method: str             # "CSU" or "Froehlich"
    scour_depth_ft: float   # ys (ft)
    K1: float               # pier nose shape factor
    K2: float               # angle of attack factor
    K3: float               # bed condition factor
    K4: float               # armoring factor (CSU only, 1.0 for Froehlich)
    froude_number: float    # approach Fr
    a_effective_ft: float   # effective pier width (ft)
    y1_ft: float            # approach depth (ft)
    V1_fps: float           # approach velocity (ft/s)

    def to_dict(self):
        return {k: v for k, v in self.__dict__.items()}


@dataclass
class ContractionScourResult:
    """Result from Laursen live-bed or clear-water contraction scour."""
    method: str             # "live-bed" or "clear-water"
    scour_depth_ft: float   # ys = y2 - y0 (ft)
    y2_ft: float            # equilibrium depth in contracted section (ft)
    y0_ft: float            # existing depth in contracted section (ft)
    V_mean_fps: float       # mean velocity in approach (ft/s)
    Vc_fps: float           # critical velocity (ft/s)
    is_live_bed: bool       # True if V > Vc

    def to_dict(self):
        return {k: v for k, v in self.__dict__.items()}


@dataclass
class AbutmentScourResult:
    """Result from HIRE or Froehlich abutment scour equation."""
    method: str             # "HIRE" or "Froehlich"
    scour_depth_ft: float   # ys (ft)
    K1: float               # abutment shape factor
    K2: float               # angle correction factor
    ya_ft: float            # approach depth (ft)
    froude_number: float    # approach Fr

    def to_dict(self):
        return {k: v for k, v in self.__dict__.items()}


# ═══════════════════════════════════════════════════════════════════════════
# Critical velocity — determines live-bed vs clear-water
# ═══════════════════════════════════════════════════════════════════════════

def critical_velocity(y_ft: float, D50_ft: float) -> float:
    """
    Critical velocity for sediment transport initiation (HEC-18 Eq. 6.1).

    Vc = Ku * y^(1/6) * D50^(1/3)

    Parameters
    ----------
    y_ft : float
        Flow depth (ft)
    D50_ft : float
        Median grain size (ft)

    Returns
    -------
    float
        Critical velocity (ft/s)
    """
    Ku = 11.17  # English units
    return Ku * y_ft ** (1.0 / 6.0) * D50_ft ** (1.0 / 3.0)


def fall_velocity(D50_ft: float, temperature_F: float = 60.0) -> float:
    """
    Fall velocity of sediment particle (Rubey 1933, simplified).

    Parameters
    ----------
    D50_ft : float
        Median grain size (ft)
    temperature_F : float
        Water temperature (°F)

    Returns
    -------
    float
        Fall velocity omega (ft/s)
    """
    # Kinematic viscosity at temperature (simplified)
    T_C = (temperature_F - 32.0) * 5.0 / 9.0
    nu = 1.792e-5 / (1.0 + 0.0337 * T_C + 0.000221 * T_C ** 2)  # m^2/s
    nu_ft2 = nu * 10.764  # convert to ft^2/s

    D_m = D50_ft * 0.3048  # convert to meters for Rubey
    sg = 2.65  # specific gravity of quartz

    # Rubey's formula
    d = D50_ft
    if d > 0.007:  # > 2mm, gravel regime
        omega = 3.32 * np.sqrt((sg - 1.0) * G * d)
    elif d > 0.0003:  # 0.1mm - 2mm, sand regime
        omega = np.sqrt((sg - 1.0) * G * d * 2.0 / 3.0 +
                        36.0 * nu_ft2 ** 2 / d ** 2) - 6.0 * nu_ft2 / d
    else:  # fine silt
        omega = (sg - 1.0) * G * d ** 2 / (18.0 * nu_ft2)

    return max(omega, 1e-6)


# ═══════════════════════════════════════════════════════════════════════════
# CSU PIER SCOUR (HEC-18 Eq. 7.3)
# ═══════════════════════════════════════════════════════════════════════════

# K1 — Pier nose shape
PIER_SHAPE_K1 = {
    "square":   1.1,
    "round":    1.0,
    "circular": 1.0,
    "cylinder": 1.0,
    "group":    1.0,
    "sharp":    0.9,
}

# K3 — Bed condition
BED_CONDITION_K3 = {
    "clear-water":  1.1,
    "plane_bed":    1.1,
    "antidune":     1.1,
    "small_dunes":  1.1,
    "medium_dunes": 1.2,   # dune height 10-30 ft
    "large_dunes":  1.3,   # dune height >= 30 ft
}


def _compute_K2(theta_deg: float, L_over_a: float) -> float:
    """
    K2 — Angle of attack correction factor.

    K2 = (cos(theta) + L/a * sin(theta))^0.65

    Parameters
    ----------
    theta_deg : float
        Angle of attack (degrees), 0 = aligned with flow
    L_over_a : float
        Pier length-to-width ratio (max 12 used)
    """
    if theta_deg <= 5.0:
        return 1.0
    theta = np.radians(theta_deg)
    L_a = min(L_over_a, 12.0)
    return (np.cos(theta) + L_a * np.sin(theta)) ** 0.65


def _compute_K4(V1: float, y1: float, a: float,
                D50_ft: float, D95_ft: float) -> float:
    """
    K4 — Armoring correction factor for coarse bed materials.

    Applies when D50 >= 0.007 ft (2 mm) AND D95 >= 0.066 ft (20 mm).

    K4 = 0.4 * VR^0.15  (minimum 0.4)

    Parameters
    ----------
    V1, y1, a : float
        Approach velocity (ft/s), depth (ft), pier width (ft)
    D50_ft, D95_ft : float
        Median and 95th percentile grain size (ft)
    """
    if D50_ft < 0.007 or D95_ft < 0.066:
        return 1.0

    Ku = 11.17
    Vc50 = Ku * y1 ** (1.0 / 6.0) * D50_ft ** (1.0 / 3.0)
    Vc95 = Ku * y1 ** (1.0 / 6.0) * D95_ft ** (1.0 / 3.0)

    Vi50 = 0.645 * (D50_ft / a) ** 0.053 * Vc50
    Vi95 = 0.645 * (D95_ft / a) ** 0.053 * Vc95

    denom = Vc50 - Vi95
    if denom <= 0:
        return 1.0

    VR = (V1 - Vi50) / denom
    VR = max(VR, 0.0)

    K4 = 0.4 * VR ** 0.15
    return max(K4, 0.4)


def csu_pier_scour(
    V1: float,
    y1: float,
    a: float,
    pier_shape: str = "round",
    theta_deg: float = 0.0,
    L_over_a: float = 4.0,
    bed_condition: str = "clear-water",
    D50_ft: float = 0.002,
    D95_ft: float = 0.005,
) -> PierScourResult:
    """
    CSU pier scour equation (HEC-18 Eq. 7.3).

    ys = 2.0 * K1 * K2 * K3 * K4 * a * (y1/a)^0.35 * Fr1^0.43

    Scour depth limits:
      Fr1 <= 0.8:  ys <= 2.4 * a
      Fr1 >  0.8:  ys <= 3.0 * a

    Parameters
    ----------
    V1 : float
        Approach velocity directly upstream of pier (ft/s)
    y1 : float
        Approach flow depth directly upstream of pier (ft)
    a : float
        Pier width (ft)
    pier_shape : str
        One of: "square", "round", "circular", "cylinder", "group", "sharp"
    theta_deg : float
        Angle of attack (degrees)
    L_over_a : float
        Pier length / pier width ratio
    bed_condition : str
        One of: "clear-water", "plane_bed", "antidune", "small_dunes",
                "medium_dunes", "large_dunes"
    D50_ft : float
        Median grain size (ft)
    D95_ft : float
        95th percentile grain size (ft)

    Returns
    -------
    PierScourResult
    """
    Fr = V1 / np.sqrt(G * y1) if y1 > 0 else 0.0

    K1 = PIER_SHAPE_K1.get(pier_shape, 1.0)
    if theta_deg > 5.0:
        K1 = 1.0  # HEC-18: shape factor ignored at high angle
    K2 = _compute_K2(theta_deg, L_over_a)
    K3 = BED_CONDITION_K3.get(bed_condition, 1.1)
    K4 = _compute_K4(V1, y1, a, D50_ft, D95_ft)

    ys = 2.0 * K1 * K2 * K3 * K4 * a * (y1 / a) ** 0.35 * Fr ** 0.43

    # Depth limits
    if Fr <= 0.8:
        ys = min(ys, 2.4 * a)
    else:
        ys = min(ys, 3.0 * a)

    return PierScourResult(
        method="CSU",
        scour_depth_ft=ys,
        K1=K1, K2=K2, K3=K3, K4=K4,
        froude_number=Fr,
        a_effective_ft=a,
        y1_ft=y1,
        V1_fps=V1,
    )


# ═══════════════════════════════════════════════════════════════════════════
# FROEHLICH PIER SCOUR (Froehlich 1988)
# ═══════════════════════════════════════════════════════════════════════════

FROEHLICH_SHAPE_PHI = {
    "square": 1.3,
    "round":  1.0,
    "circular": 1.0,
    "cylinder": 1.0,
    "sharp":  0.7,
}


def froehlich_pier_scour(
    V1: float,
    y1: float,
    a: float,
    pier_shape: str = "round",
    theta_deg: float = 0.0,
    L_over_a: float = 4.0,
    D50_ft: float = 0.002,
) -> PierScourResult:
    """
    Froehlich pier scour equation (Froehlich 1988).

    ys = 0.32 * Phi * a'^0.62 * y1^0.47 * Fr1^0.22 * D50^-0.09 + a

    Parameters
    ----------
    V1, y1, a : float
        Approach velocity (ft/s), depth (ft), pier width (ft)
    pier_shape : str
        Pier nose shape
    theta_deg : float
        Angle of attack (degrees)
    L_over_a : float
        Pier length / pier width ratio
    D50_ft : float
        Median grain size (ft)
    """
    Fr = V1 / np.sqrt(G * y1) if y1 > 0 else 0.0
    Phi = FROEHLICH_SHAPE_PHI.get(pier_shape, 1.0)

    # Projected pier width
    theta = np.radians(theta_deg)
    L = a * min(L_over_a, 12.0)
    a_prime = a * np.cos(theta) + L * np.sin(theta)

    # Protect against zero/negative D50
    D50_safe = max(D50_ft, 1e-6)

    ys = 0.32 * Phi * a_prime ** 0.62 * y1 ** 0.47 * Fr ** 0.22 * D50_safe ** (-0.09) + a

    # Same depth limits as CSU
    if Fr <= 0.8:
        ys = min(ys, 2.4 * a)
    else:
        ys = min(ys, 3.0 * a)

    K2 = _compute_K2(theta_deg, L_over_a)

    return PierScourResult(
        method="Froehlich",
        scour_depth_ft=ys,
        K1=Phi, K2=K2, K3=1.0, K4=1.0,
        froude_number=Fr,
        a_effective_ft=a_prime,
        y1_ft=y1,
        V1_fps=V1,
    )


# ═══════════════════════════════════════════════════════════════════════════
# CONTRACTION SCOUR — Live-Bed (Modified Laursen 1960)
# ═══════════════════════════════════════════════════════════════════════════

def _laursen_K1_exponent(V_star: float, omega: float) -> float:
    """
    K1 exponent for live-bed contraction scour based on transport mode.

    V*/omega ratio determines K1:
      < 0.50:  K1 = 0.59  (mostly contact bed material)
      0.50-2.0: K1 = 0.64 (some suspended bed material)
      > 2.0:   K1 = 0.69  (mostly suspended bed material)
    """
    ratio = V_star / omega if omega > 0 else 0.0
    if ratio < 0.50:
        return 0.59
    elif ratio <= 2.0:
        return 0.64
    else:
        return 0.69


def live_bed_contraction_scour(
    y1: float,
    Q1: float,
    Q2: float,
    W1: float,
    W2: float,
    y0: float,
    slope: float = 0.002,
    D50_ft: float = 0.002,
    temperature_F: float = 60.0,
) -> ContractionScourResult:
    """
    Live-bed contraction scour (Modified Laursen 1960, HEC-18 Eq. 6.2).

    y2 = y1 * (Q2/Q1)^(6/7) * (W1/W2)^K1
    ys = y2 - y0

    Parameters
    ----------
    y1 : float
        Average depth in upstream main channel (ft)
    Q1 : float
        Flow in upstream main channel transporting sediment (cfs)
    Q2 : float
        Flow in contracted section (cfs)
    W1 : float
        Bottom width of upstream main channel (ft)
    W2 : float
        Bottom width of contracted section less pier widths (ft)
    y0 : float
        Existing depth in contracted section before scour (ft)
    slope : float
        Energy slope of upstream channel
    D50_ft : float
        Median grain size (ft)
    temperature_F : float
        Water temperature (°F)
    """
    # Shear velocity
    V_star = np.sqrt(G * y1 * slope)

    # Fall velocity
    omega = fall_velocity(D50_ft, temperature_F)

    # Transport mode exponent
    K1 = _laursen_K1_exponent(V_star, omega)

    # Equilibrium depth in contracted section
    y2 = y1 * (Q2 / Q1) ** (6.0 / 7.0) * (W1 / W2) ** K1

    ys = max(y2 - y0, 0.0)

    V_mean = Q1 / (W1 * y1) if (W1 * y1) > 0 else 0.0
    Vc = critical_velocity(y1, D50_ft)

    return ContractionScourResult(
        method="live-bed",
        scour_depth_ft=ys,
        y2_ft=y2,
        y0_ft=y0,
        V_mean_fps=V_mean,
        Vc_fps=Vc,
        is_live_bed=True,
    )


# ═══════════════════════════════════════════════════════════════════════════
# CONTRACTION SCOUR — Clear-Water (Laursen 1963)
# ═══════════════════════════════════════════════════════════════════════════

def clear_water_contraction_scour(
    Q2: float,
    W2: float,
    y0: float,
    D50_ft: float = 0.002,
    Dm_factor: float = 1.25,
) -> ContractionScourResult:
    """
    Clear-water contraction scour (Laursen 1963, HEC-18 Eq. 6.4).

    y2 = [Q2^2 / (C * Dm^(2/3) * W2^2)]^(3/7)
    ys = y2 - y0

    Parameters
    ----------
    Q2 : float
        Flow in contracted section (cfs)
    W2 : float
        Bottom width of contracted section less pier widths (ft)
    y0 : float
        Existing depth in contracted section before scour (ft)
    D50_ft : float
        Median grain size (ft)
    Dm_factor : float
        Dm = Dm_factor * D50 (default 1.25 per HEC-18)
    """
    C = 130.0  # English units
    Dm = Dm_factor * D50_ft

    if W2 <= 0 or Dm <= 0:
        return ContractionScourResult(
            method="clear-water", scour_depth_ft=0.0, y2_ft=y0,
            y0_ft=y0, V_mean_fps=0.0, Vc_fps=0.0, is_live_bed=False,
        )

    y2 = (Q2 ** 2 / (C * Dm ** (2.0 / 3.0) * W2 ** 2)) ** (3.0 / 7.0)
    ys = max(y2 - y0, 0.0)

    V_mean = Q2 / (W2 * y0) if (W2 * y0) > 0 else 0.0
    Vc = critical_velocity(y0, D50_ft)

    return ContractionScourResult(
        method="clear-water",
        scour_depth_ft=ys,
        y2_ft=y2,
        y0_ft=y0,
        V_mean_fps=V_mean,
        Vc_fps=Vc,
        is_live_bed=False,
    )


# ═══════════════════════════════════════════════════════════════════════════
# HIRE ABUTMENT SCOUR (HEC-18 Eq. 7.2)
#   Used when L/y1 > 25 (long abutments)
# ═══════════════════════════════════════════════════════════════════════════

ABUTMENT_SHAPE_K1 = {
    "vertical_wall":      1.00,
    "vertical_wingwall":  0.82,
    "spill_through":      0.55,
}


def _abutment_K2(theta_deg: float) -> float:
    """K2 angle correction for abutment scour."""
    return (theta_deg / 90.0) ** 0.13


def hire_abutment_scour(
    V1: float,
    y1: float,
    abutment_shape: str = "vertical_wall",
    theta_deg: float = 90.0,
) -> AbutmentScourResult:
    """
    HIRE abutment scour equation (HEC-18 Eq. 7.2).

    ys = 4 * y1 * (K1/0.55) * K2 * Fr1^0.33

    Used when abutment length / approach depth > 25.

    Parameters
    ----------
    V1 : float
        Approach velocity near abutment (ft/s)
    y1 : float
        Approach depth at abutment (ft)
    abutment_shape : str
        One of: "vertical_wall", "vertical_wingwall", "spill_through"
    theta_deg : float
        Angle between embankment and flow (degrees, 90 = perpendicular)
    """
    Fr = V1 / np.sqrt(G * y1) if y1 > 0 else 0.0
    K1 = ABUTMENT_SHAPE_K1.get(abutment_shape, 1.0)
    K2 = _abutment_K2(theta_deg)

    ys = 4.0 * y1 * (K1 / 0.55) * K2 * Fr ** 0.33

    return AbutmentScourResult(
        method="HIRE",
        scour_depth_ft=ys,
        K1=K1, K2=K2,
        ya_ft=y1,
        froude_number=Fr,
    )


# ═══════════════════════════════════════════════════════════════════════════
# FROEHLICH ABUTMENT SCOUR (HEC-18 Eq. 7.1)
# ═══════════════════════════════════════════════════════════════════════════

def froehlich_abutment_scour(
    V1: float,
    ya: float,
    L_prime: float,
    abutment_shape: str = "vertical_wall",
    theta_deg: float = 90.0,
) -> AbutmentScourResult:
    """
    Froehlich abutment scour equation (HEC-18 Eq. 7.1).

    ys = 2.27 * K1 * K2 * L'^0.43 * ya^0.57 * Fr^0.61 + ya

    Parameters
    ----------
    V1 : float
        Approach velocity (ft/s)
    ya : float
        Average depth of flow on the floodplain at the abutment (ft)
    L_prime : float
        Length of active flow obstructed by the abutment (ft)
    abutment_shape : str
        One of: "vertical_wall", "vertical_wingwall", "spill_through"
    theta_deg : float
        Angle between embankment and flow (degrees)
    """
    Fr = V1 / np.sqrt(G * ya) if ya > 0 else 0.0
    K1 = ABUTMENT_SHAPE_K1.get(abutment_shape, 1.0)
    K2 = _abutment_K2(theta_deg)

    ys = 2.27 * K1 * K2 * L_prime ** 0.43 * ya ** 0.57 * Fr ** 0.61 + ya

    return AbutmentScourResult(
        method="Froehlich",
        scour_depth_ft=ys,
        K1=K1, K2=K2,
        ya_ft=ya,
        froude_number=Fr,
    )


# ═══════════════════════════════════════════════════════════════════════════
# TOTAL SCOUR — Superposition per HEC-18 practice
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class TotalScourResult:
    """Combined scour from all components at a bridge cross-section."""
    pier_scour: Optional[PierScourResult]
    contraction_scour: Optional[ContractionScourResult]
    abutment_scour: Optional[AbutmentScourResult]
    total_scour_ft: float
    components: dict  # {component_name: depth_ft}

    def to_dict(self):
        return {
            "total_scour_ft": self.total_scour_ft,
            "components": self.components,
            "pier": self.pier_scour.to_dict() if self.pier_scour else None,
            "contraction": self.contraction_scour.to_dict() if self.contraction_scour else None,
            "abutment": self.abutment_scour.to_dict() if self.abutment_scour else None,
        }


def total_scour(
    pier: Optional[PierScourResult] = None,
    contraction: Optional[ContractionScourResult] = None,
    abutment: Optional[AbutmentScourResult] = None,
) -> TotalScourResult:
    """
    Compute total scour as superposition of components (HEC-18 practice).

    Total = pier scour + contraction scour  (at piers)
    Total = abutment scour + contraction scour  (at abutments)
    """
    components = {}
    total = 0.0

    if pier:
        components["pier"] = pier.scour_depth_ft
        total += pier.scour_depth_ft
    if contraction:
        components["contraction"] = contraction.scour_depth_ft
        total += contraction.scour_depth_ft
    if abutment:
        components["abutment"] = abutment.scour_depth_ft
        total += abutment.scour_depth_ft

    return TotalScourResult(
        pier_scour=pier,
        contraction_scour=contraction,
        abutment_scour=abutment,
        total_scour_ft=total,
        components=components,
    )
