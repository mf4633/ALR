"""
Validation modules for quantum hydraulics.

Contains:
- analytical: Analytical solutions for validation (Lamb-Oseen, Poiseuille, Kolmogorov)
- benchmarks: Pytest test suite for validation
- hec18_scour: HEC-18/HEC-RAS empirical scour equations (CSU, Froehlich, Laursen, HIRE)
- benchmark_scenarios: Published scour scenarios for cross-validation
"""

from quantum_hydraulics.validation.analytical import (
    lamb_oseen_vortex,
    poiseuille_velocity,
    kolmogorov_spectrum,
    wall_vorticity,
)

from quantum_hydraulics.validation.hec18_scour import (
    csu_pier_scour,
    froehlich_pier_scour,
    live_bed_contraction_scour,
    clear_water_contraction_scour,
    hire_abutment_scour,
    froehlich_abutment_scour,
    critical_velocity,
    total_scour,
)

__all__ = [
    "lamb_oseen_vortex",
    "poiseuille_velocity",
    "kolmogorov_spectrum",
    "wall_vorticity",
    "csu_pier_scour",
    "froehlich_pier_scour",
    "live_bed_contraction_scour",
    "clear_water_contraction_scour",
    "hire_abutment_scour",
    "froehlich_abutment_scour",
    "critical_velocity",
    "total_scour",
]
