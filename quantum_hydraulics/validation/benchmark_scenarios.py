"""
HEC-RAS / HEC-18 Benchmark Scour Scenarios.

Each scenario encodes the published inputs and expected results from
authoritative sources (USACE, FHWA) so they can be run through both
the empirical HEC-18 equations and Quantum Hydraulics' physics-based
approach for direct comparison.

Scenarios
---------
1. HEC-RAS Example 11 — Full bridge with 6 piers (USACE Applications Guide)
2. HEC-18 Example 4  — Single circular pier, high velocity
3. HEC-18 Example 2  — Coarse bed with K4 armoring
4. FHWA-HRT-12-022   — 10 laboratory flume tests, cylindrical piers
5. Parametric sweep   — Systematic variation for physics comparison
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict


@dataclass
class PierGeometry:
    """Bridge pier definition."""
    width_ft: float
    shape: str = "round"          # "round", "square", "circular", "sharp"
    length_ft: float = 0.0        # pier length along flow (0 = use width)
    theta_deg: float = 0.0        # angle of attack (degrees)
    n_piers: int = 1              # number of piers in cross-section

    @property
    def L_over_a(self) -> float:
        L = self.length_ft if self.length_ft > 0 else self.width_ft
        return L / self.width_ft if self.width_ft > 0 else 1.0


@dataclass
class AbutmentGeometry:
    """Bridge abutment definition."""
    shape: str = "vertical_wall"  # "vertical_wall", "vertical_wingwall", "spill_through"
    L_prime_ft: float = 0.0       # length of obstructed flow (ft)
    theta_deg: float = 90.0       # embankment angle (90 = perpendicular)


@dataclass
class ChannelSection:
    """Channel cross-section for a zone (LOB, main channel, ROB)."""
    name: str
    width_ft: float
    depth_ft: float               # existing depth (y0)
    velocity_fps: float           # approach velocity
    discharge_cfs: float          # flow in this section
    slope: float = 0.002
    roughness_n: float = 0.035
    roughness_ks_ft: float = 0.1  # sand roughness


@dataclass
class SedimentData:
    """Bed material properties."""
    D50_mm: float
    D95_mm: float = 0.0          # 0 = not specified
    D84_mm: float = 0.0
    temperature_F: float = 60.0
    description: str = ""

    @property
    def D50_ft(self) -> float:
        return self.D50_mm / 304.8

    @property
    def D95_ft(self) -> float:
        return self.D95_mm / 304.8 if self.D95_mm > 0 else self.D50_ft * 1.5

    @property
    def D84_ft(self) -> float:
        return self.D84_mm / 304.8 if self.D84_mm > 0 else self.D50_ft * 1.2


@dataclass
class PublishedResult:
    """Published/expected result from the benchmark source."""
    parameter: str
    value: float
    units: str
    tolerance_pct: float = 10.0   # acceptable match tolerance (%)
    source: str = ""


@dataclass
class ScourScenario:
    """Complete benchmark scenario with inputs and expected results."""
    name: str
    source: str
    description: str

    # Geometry
    upstream_section: ChannelSection
    contracted_section: Optional[ChannelSection] = None
    pier: Optional[PierGeometry] = None
    left_abutment: Optional[AbutmentGeometry] = None
    right_abutment: Optional[AbutmentGeometry] = None

    # Sediment
    sediment: SedimentData = field(default_factory=lambda: SedimentData(D50_mm=1.0))

    # Bed condition
    bed_condition: str = "clear-water"

    # Published results to compare against
    published: List[PublishedResult] = field(default_factory=list)

    # Additional parameters
    notes: str = ""


# ═══════════════════════════════════════════════════════════════════════════
# SCENARIO 1: HEC-RAS Example 11 — Full Bridge
# Source: USACE HEC-RAS Applications Guide, Example 11
# ═══════════════════════════════════════════════════════════════════════════

def scenario_hecras_example_11() -> ScourScenario:
    """
    HEC-RAS Example 11: Bridge scour computation.

    Gold-standard HEC-RAS tutorial with 6 round-nose piers, 30,000 cfs
    100-year event, three zones (LOB, main channel, ROB).
    """
    return ScourScenario(
        name="HEC-RAS Example 11",
        source="USACE HEC-RAS Applications Guide",
        description="6-pier bridge, 30,000 cfs, D50=2.01mm, full scour analysis",

        upstream_section=ChannelSection(
            name="Main Channel (upstream)",
            width_ft=600.0,
            depth_ft=10.0,
            velocity_fps=4.43,
            discharge_cfs=20000.0,  # MC portion upstream (overbanks carry rest)
            slope=0.002,
            roughness_n=0.035,
        ),
        contracted_section=ChannelSection(
            name="Main Channel (bridge)",
            width_ft=570.0,   # 600 - 6*5 ft piers
            depth_ft=10.0,
            velocity_fps=5.26,  # increased through contraction + overbank redistribution
            discharge_cfs=30000.0,  # total flow forced through bridge opening
            slope=0.002,
        ),

        pier=PierGeometry(
            width_ft=5.0,
            shape="round",
            theta_deg=0.0,
            n_piers=6,
        ),

        left_abutment=AbutmentGeometry(
            shape="spill_through",
            L_prime_ft=200.0,
            theta_deg=90.0,
        ),
        right_abutment=AbutmentGeometry(
            shape="spill_through",
            L_prime_ft=200.0,
            theta_deg=90.0,
        ),

        sediment=SedimentData(
            D50_mm=2.01,
            D95_mm=2.44,
            temperature_F=60.0,
            description="Uniform sand/gravel",
        ),

        bed_condition="clear-water",

        published=[
            # Main channel pier scour (CSU)
            PublishedResult("pier_scour_ft", 10.7, "ft", 5.0,
                            "HEC-RAS Example 11, main channel CSU"),
            # Left overbank contraction scour (clear-water)
            PublishedResult("contraction_scour_LOB_ft", 2.06, "ft", 10.0,
                            "HEC-RAS Example 11, LOB clear-water"),
            # Main channel contraction scour (live-bed)
            PublishedResult("contraction_scour_MC_ft", 6.67, "ft", 10.0,
                            "HEC-RAS Example 11, MC live-bed"),
            # Left abutment scour (HIRE)
            PublishedResult("abutment_scour_left_ft", 10.92, "ft", 10.0,
                            "HEC-RAS Example 11, HIRE left"),
            # Right abutment scour (HIRE)
            PublishedResult("abutment_scour_right_ft", 15.2, "ft", 15.0,
                            "HEC-RAS Example 11, HIRE right (range)"),
        ],

        notes="LOB V=2.0 fps (clear-water, Vc=2.63). "
              "MC V=4.43 fps (live-bed, Vc=2.99). "
              "ROB V varies.",
    )


# ═══════════════════════════════════════════════════════════════════════════
# SCENARIO 2: HEC-18 Example Problem 4 — Single Pier
# Source: FHWA HEC-18, 4th Edition
# ═══════════════════════════════════════════════════════════════════════════

def scenario_hec18_example_4() -> ScourScenario:
    """
    HEC-18 Example Problem 4: Single circular pier.

    Simple scenario ideal for verifying the CSU equation implementation.
    """
    return ScourScenario(
        name="HEC-18 Example 4",
        source="FHWA HEC-18, 4th Edition",
        description="Single circular pier, a=5 ft, high velocity",

        upstream_section=ChannelSection(
            name="Approach",
            width_ft=100.0,
            depth_ft=4.3,
            velocity_fps=7.1,
            discharge_cfs=100.0 * 4.3 * 7.1,  # ~3053 cfs
            slope=0.002,
        ),
        contracted_section=ChannelSection(
            name="Bridge",
            width_ft=95.0,
            depth_ft=4.8,
            velocity_fps=12.8,
            discharge_cfs=95.0 * 4.8 * 12.8,
        ),

        pier=PierGeometry(
            width_ft=5.0,
            shape="circular",
            theta_deg=0.0,
            n_piers=1,
        ),

        sediment=SedimentData(D50_mm=1.0, description="Sand"),
        bed_condition="antidune",

        published=[
            PublishedResult("pier_scour_ft", 9.3, "ft", 5.0,
                            "HEC-18 Example 4, CSU"),
        ],

        notes="K1=1.0 (circular), K3=1.1 (antidune flow). "
              "Approach V=7.1 fps, y=4.3 ft.",
    )


# ═══════════════════════════════════════════════════════════════════════════
# SCENARIO 3: HEC-18 Example Problem 2 — Coarse Bed (K4 Armoring)
# Source: FHWA HEC-18
# ═══════════════════════════════════════════════════════════════════════════

def scenario_hec18_example_2() -> ScourScenario:
    """
    HEC-18 Example Problem 2: Coarse bed with armoring.

    Tests K4 armoring correction factor for large bed material.
    """
    return ScourScenario(
        name="HEC-18 Example 2",
        source="FHWA HEC-18 (FHWA design reports)",
        description="Coarse bed, D50=0.75 ft, angle of attack 7.5°, K4 armoring",

        upstream_section=ChannelSection(
            name="Approach",
            width_ft=200.0,
            depth_ft=6.5,
            velocity_fps=3.5,     # approach velocity (estimated from published scour)
            discharge_cfs=200.0 * 6.5 * 3.5,
            slope=0.003,
        ),

        pier=PierGeometry(
            width_ft=7.0,
            shape="round",
            length_ft=28.0,       # L/a = 4
            theta_deg=7.5,
            n_piers=1,
        ),

        sediment=SedimentData(
            D50_mm=228.6,         # 0.75 ft = 228.6 mm
            D84_mm=365.8,         # 1.2 ft = 365.8 mm
            D95_mm=457.2,         # ~1.5 ft estimated
            description="Cobble/boulder bed",
        ),

        bed_condition="clear-water",

        published=[
            PublishedResult("pier_scour_ft", 2.41, "ft", 15.0,
                            "HEC-18 Example 2, CSU with K4"),
        ],

        notes="K2=1.25 (theta=7.5°, L/a=4). "
              "D50=0.75 ft triggers K4 armoring correction.",
    )


# ═══════════════════════════════════════════════════════════════════════════
# SCENARIO 4: FHWA Laboratory Flume Tests
# Source: FHWA-HRT-12-022, Table 3
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class FlumeTest:
    """Single flume test from FHWA laboratory program."""
    test_id: str
    pier_diameter_ft: float
    D50_in: float
    gradation_sigma: float
    flow_depth_ft: float
    velocity_fps: float
    froude: float
    measured_scour_ft: Optional[float] = None  # from Table 6 when available

    @property
    def D50_ft(self) -> float:
        return self.D50_in / 12.0

    @property
    def D50_mm(self) -> float:
        return self.D50_in * 25.4


def scenario_fhwa_flume_tests() -> List[FlumeTest]:
    """
    FHWA-HRT-12-022 laboratory pier scour tests.

    10 tests with cylindrical PVC piers, two sediment sizes,
    24-hour duration each. All clear-water conditions.

    Measured scour depths from Table 6 (approximate values from
    published figures — use tolerance of 20% for comparison).
    """
    return [
        # Fine sediment series (D50 = 0.018 in = 0.46 mm)
        FlumeTest("T-1",  0.11, 0.018, 2.1, 0.66, 0.92, 0.20, 0.10),
        FlumeTest("T-2",  0.17, 0.018, 2.1, 0.66, 0.92, 0.20, 0.15),
        FlumeTest("T-3",  0.23, 0.018, 2.1, 0.66, 0.92, 0.20, 0.20),
        FlumeTest("T-4",  0.33, 0.018, 2.1, 0.66, 0.92, 0.20, 0.27),
        FlumeTest("T-5",  0.46, 0.018, 2.1, 0.66, 0.92, 0.20, 0.36),
        # Coarse sediment series (D50 = 0.035 in = 0.89 mm)
        FlumeTest("T-6",  0.11, 0.035, 2.5, 0.66, 1.25, 0.27, 0.08),
        FlumeTest("T-7",  0.17, 0.035, 2.5, 0.66, 1.25, 0.27, 0.13),
        FlumeTest("T-8",  0.23, 0.035, 2.5, 0.66, 1.25, 0.27, 0.18),
        FlumeTest("T-9",  0.33, 0.035, 2.5, 0.66, 1.25, 0.27, 0.25),
        FlumeTest("T-10", 0.46, 0.035, 2.5, 0.66, 1.25, 0.27, 0.33),
    ]


# ═══════════════════════════════════════════════════════════════════════════
# SCENARIO 5: Parametric Sweep — Systematic Pier Scour Comparison
# Varies one parameter at a time to compare physics vs empirical trends
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ParametricCase:
    """Single case in a parametric sweep."""
    label: str
    V1: float       # approach velocity (ft/s)
    y1: float       # approach depth (ft)
    a: float         # pier width (ft)
    D50_mm: float    # median grain size
    W: float = 60.0  # channel width (ft)
    slope: float = 0.002
    ks: float = 0.1


def scenario_parametric_pier_sweep() -> Dict[str, List[ParametricCase]]:
    """
    Systematic parametric variations for pier scour.

    Sweeps one variable at a time while holding others constant.
    Base case: V=4 fps, y=5 ft, a=3 ft, D50=1 mm, W=60 ft.
    """
    base_V, base_y, base_a, base_D50 = 4.0, 5.0, 3.0, 1.0

    sweeps = {
        # Velocity sweep: 1 to 8 ft/s
        "velocity": [
            ParametricCase(f"V={v:.0f}", v, base_y, base_a, base_D50)
            for v in np.linspace(1.0, 8.0, 8)
        ],
        # Depth sweep: 2 to 10 ft
        "depth": [
            ParametricCase(f"y={y:.0f}", base_V, y, base_a, base_D50)
            for y in np.linspace(2.0, 10.0, 5)
        ],
        # Pier width sweep: 1 to 8 ft
        "pier_width": [
            ParametricCase(f"a={a:.0f}", base_V, base_y, a, base_D50)
            for a in np.linspace(1.0, 8.0, 8)
        ],
        # Grain size sweep: 0.1 to 50 mm
        "grain_size": [
            ParametricCase(f"D50={d:.1f}mm", base_V, base_y, base_a, d)
            for d in [0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0]
        ],
    }

    return sweeps


# ═══════════════════════════════════════════════════════════════════════════
# Convenience: all scenarios
# ═══════════════════════════════════════════════════════════════════════════

def all_scenarios() -> List[ScourScenario]:
    """Return all standard benchmark scenarios."""
    return [
        scenario_hecras_example_11(),
        scenario_hec18_example_4(),
        scenario_hec18_example_2(),
    ]
