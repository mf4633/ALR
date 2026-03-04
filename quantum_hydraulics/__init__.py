"""
Quantum Hydraulics
==================
A physics-based hydraulic simulation package using vortex particle methods
with observation-dependent (quantum-inspired) adaptive resolution.

Combines:
- First-principles physics (Colebrook-White, Kolmogorov cascade)
- True vortex particle method (Biot-Savart law, 3D particles)
- Observation-dependent resolution (adaptive core size sigma)
- Professional visualization with multiple themes

For engineers and researchers working on turbulent open-channel flow.
"""

__version__ = "1.0.0"
__author__ = "Michael Flynn"

# Core physics
from quantum_hydraulics.core.particle import VortexParticle
from quantum_hydraulics.core.hydraulics import HydraulicsEngine
from quantum_hydraulics.core.vortex_field import VortexParticleField, FieldState

# Visualization
from quantum_hydraulics.visualization.theme import Theme, THEMES, get_theme
from quantum_hydraulics.visualization.renderers import (
    plot_plan_view,
    plot_profile_view,
    plot_velocity_profile,
    plot_energy_spectrum,
    plot_detail_map,
)
from quantum_hydraulics.visualization.interactive import InteractiveSimulator
from quantum_hydraulics.visualization.export import export_animation, export_frames

# Integration (optional - requires pyswmm)
try:
    from quantum_hydraulics.integration.swmm_node import QuantumNode
    _HAS_PYSWMM = True
except ImportError:
    _HAS_PYSWMM = False

# Validation
from quantum_hydraulics.validation.analytical import (
    lamb_oseen_vortex,
    poiseuille_velocity,
    kolmogorov_spectrum,
)

# Quick analysis mode
from quantum_hydraulics.analysis import analyze, DesignResults, print_design_table

__all__ = [
    # Version
    "__version__",
    # Core
    "VortexParticle",
    "HydraulicsEngine",
    "VortexParticleField",
    "FieldState",
    # Quick Analysis
    "analyze",
    "DesignResults",
    "print_design_table",
    # Visualization
    "Theme",
    "THEMES",
    "get_theme",
    "plot_plan_view",
    "plot_profile_view",
    "plot_velocity_profile",
    "plot_energy_spectrum",
    "plot_detail_map",
    "InteractiveSimulator",
    "export_animation",
    "export_frames",
    # Integration
    "QuantumNode",
    # Validation
    "lamb_oseen_vortex",
    "poiseuille_velocity",
    "kolmogorov_spectrum",
]
