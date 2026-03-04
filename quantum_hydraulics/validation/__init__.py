"""
Validation modules for quantum hydraulics.

Contains:
- analytical: Analytical solutions for validation (Lamb-Oseen, Poiseuille, Kolmogorov)
- benchmarks: Pytest test suite for validation
"""

from quantum_hydraulics.validation.analytical import (
    lamb_oseen_vortex,
    poiseuille_velocity,
    kolmogorov_spectrum,
    wall_vorticity,
)

__all__ = [
    "lamb_oseen_vortex",
    "poiseuille_velocity",
    "kolmogorov_spectrum",
    "wall_vorticity",
]
