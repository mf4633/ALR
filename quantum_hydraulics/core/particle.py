"""
VortexParticle - Canonical definition of a vortex particle.

A vortex particle carries vorticity (circulation) and has a core size (resolution).
The core size sigma determines the spatial resolution of the particle representation.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass(slots=True)
class VortexParticle:
    """
    Single vortex particle in 3D space.

    Carries vorticity (circulation) and has a core size (resolution).
    This is the fundamental computational element in the vortex particle method.

    Attributes
    ----------
    pos : np.ndarray
        3D position vector [x, y, z] in feet
    omega : np.ndarray
        3D vorticity vector [omega_x, omega_y, omega_z] in 1/s
    sigma : float
        Core size (spatial resolution) in feet. Smaller = higher resolution.
    age : float
        Time since particle creation in seconds
    energy : float
        Particle energy, computed as |omega|^2 * sigma^3

    Notes
    -----
    The core size sigma is the key parameter for observation-dependent resolution:
    - Near observation zones: small sigma (high resolution, expensive)
    - Far from observation: large sigma (low resolution, cheap)

    Energy is computed from the vorticity magnitude and core size following
    the vortex particle method convention.
    """

    pos: np.ndarray
    omega: np.ndarray
    sigma: float
    age: float = 0.0
    energy: float = field(init=False)

    def __post_init__(self):
        """Initialize energy from vorticity and core size."""
        self.pos = np.asarray(self.pos, dtype=np.float64)
        self.omega = np.asarray(self.omega, dtype=np.float64)
        self._update_energy()

    def _update_energy(self):
        """Compute particle energy from vorticity and core size."""
        self.energy = np.linalg.norm(self.omega) ** 2 * self.sigma ** 3

    @classmethod
    def create(cls, position, vorticity, core_size: float, age: float = 0.0):
        """
        Factory method to create a vortex particle.

        Parameters
        ----------
        position : array-like
            3D position [x, y, z]
        vorticity : array-like
            3D vorticity [omega_x, omega_y, omega_z]
        core_size : float
            Initial core size sigma
        age : float, optional
            Initial age (default 0.0)

        Returns
        -------
        VortexParticle
            New particle instance
        """
        p = cls(
            pos=np.asarray(position, dtype=np.float64),
            omega=np.asarray(vorticity, dtype=np.float64),
            sigma=core_size,
            age=age,
        )
        return p

    def advect(self, velocity: np.ndarray, dt: float):
        """
        Advect particle by given velocity for timestep dt.

        Parameters
        ----------
        velocity : np.ndarray
            3D velocity vector [vx, vy, vz]
        dt : float
            Timestep in seconds
        """
        self.pos = self.pos + velocity * dt
        self.age += dt

    def update_sigma(self, new_sigma: float):
        """
        Update core size and recompute energy.

        Parameters
        ----------
        new_sigma : float
            New core size
        """
        self.sigma = new_sigma
        self._update_energy()

    def copy(self) -> "VortexParticle":
        """Create a deep copy of this particle."""
        return VortexParticle(
            pos=self.pos.copy(),
            omega=self.omega.copy(),
            sigma=self.sigma,
            age=self.age,
        )

    @property
    def circulation(self) -> float:
        """Return scalar circulation magnitude."""
        return np.linalg.norm(self.omega) * self.sigma ** 2

    @property
    def vorticity_magnitude(self) -> float:
        """Return vorticity magnitude."""
        return np.linalg.norm(self.omega)

    def __repr__(self) -> str:
        return (
            f"VortexParticle(pos=[{self.pos[0]:.2f}, {self.pos[1]:.2f}, {self.pos[2]:.2f}], "
            f"|omega|={self.vorticity_magnitude:.4f}, sigma={self.sigma:.4f})"
        )
