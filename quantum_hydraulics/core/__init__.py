"""
Core physics modules for quantum hydraulics simulation.

Contains:
- VortexParticle: Single vortex particle in 3D space
- HydraulicsEngine: First-principles hydraulic computations
- VortexParticleField: 3D vortex particle system with adaptive resolution
- FieldState: Dataclass for passing state between physics and rendering
"""

from quantum_hydraulics.core.particle import VortexParticle
from quantum_hydraulics.core.hydraulics import HydraulicsEngine
from quantum_hydraulics.core.vortex_field import VortexParticleField, FieldState

__all__ = ["VortexParticle", "HydraulicsEngine", "VortexParticleField", "FieldState"]
