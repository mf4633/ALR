"""
Pier Vortex Shedding — Immersed Boundary Model.

Models the no-slip boundary condition at pier surfaces by injecting
vortex particles, creating horseshoe vortex and von Karman vortex street.

Optional module: only activates when PierBody objects are provided
to VortexParticleField or SWMM2DPostProcessor.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple


STROUHAL = 0.20  # Strouhal number for circular cylinders (Re > 300)


@dataclass
class PierBody:
    """
    Circular pier that sheds vortex particles.

    Parameters
    ----------
    x : float
        Pier center x-coordinate (ft)
    y : float
        Pier center y-coordinate (ft)
    diameter : float
        Pier diameter (ft)
    """
    x: float
    y: float
    diameter: float
    _phase: float = field(default=0.0, repr=False)
    _sign: int = field(default=1, repr=False)

    @property
    def radius(self) -> float:
        return self.diameter / 2.0

    def strouhal_frequency(self, V_approach: float) -> float:
        """Shedding frequency (Hz) from Strouhal number."""
        if self.diameter <= 0 or V_approach <= 0:
            return 0.0
        return STROUHAL * V_approach / self.diameter

    def shed_particles(
        self,
        V_approach: float,
        depth: float,
        dt: float,
        n_surface: int = 12,
        n_horseshoe: int = 6,
    ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Generate shed vortex particles for this timestep.

        Particles are shed at the Strouhal frequency with alternating sign.
        Surface particles model the no-slip boundary layer separation.
        Horseshoe particles model the base vortex wrapping around the pier.

        Parameters
        ----------
        V_approach : float
            Approach velocity (ft/s)
        depth : float
            Water depth (ft)
        dt : float
            Timestep (s)
        n_surface : int
            Number of surface particles per shed event
        n_horseshoe : int
            Number of horseshoe (base) particles per shed event

        Returns
        -------
        (positions, vorticities, sigmas) or None if no shedding this step
        """
        if V_approach <= 0 or depth <= 0:
            return None

        freq = self.strouhal_frequency(V_approach)
        if freq <= 0:
            return None

        # Advance shedding phase
        self._phase += 2.0 * np.pi * freq * dt

        # Shed when phase crosses pi (half-cycle)
        if self._phase < np.pi:
            return None

        # Reset phase, flip sign
        self._phase -= np.pi
        self._sign *= -1

        R = self.radius
        sigma = 0.3 * self.diameter  # core size scales with pier

        # ── Surface particles (downstream half of pier) ───────────────
        # Angles from pi/4 to 3pi/4 (downstream separation zone)
        theta = np.linspace(np.pi / 4, 3 * np.pi / 4, n_surface)
        z_surface = np.linspace(0.15 * depth, 0.85 * depth, n_surface)

        # Place just outside pier surface
        offset = R + 0.1 * self.diameter
        sx = self.x + offset * np.cos(theta)
        sy = self.y + offset * np.sin(theta)
        sz = z_surface

        surface_pos = np.column_stack([sx, sy, sz])

        # Vorticity: no-slip → gamma = -V_surface, oriented in z
        gamma = self._sign * V_approach
        surface_omega = np.zeros((n_surface, 3))
        surface_omega[:, 2] = gamma  # vertical vorticity (2D wake)

        surface_sigma = np.full(n_surface, sigma)

        # ── Horseshoe vortex particles (at pier base) ─────────────────
        theta_hs = np.linspace(0, np.pi, n_horseshoe, endpoint=False)
        hx = self.x + R * 1.2 * np.cos(theta_hs)
        hy = self.y + R * 1.2 * np.sin(theta_hs)
        hz = np.full(n_horseshoe, 0.1 * depth)  # near bed

        horseshoe_pos = np.column_stack([hx, hy, hz])

        # Horseshoe vorticity: ring-like (omega_y, omega_z components)
        hs_strength = V_approach * self.diameter / max(depth, 0.5)
        horseshoe_omega = np.zeros((n_horseshoe, 3))
        horseshoe_omega[:, 1] = hs_strength * np.sin(theta_hs)  # spanwise
        horseshoe_omega[:, 2] = hs_strength * np.cos(theta_hs)  # vertical

        horseshoe_sigma = np.full(n_horseshoe, sigma * 1.2)  # slightly larger

        # ── Combine ───────────────────────────────────────────────────
        positions = np.vstack([surface_pos, horseshoe_pos])
        vorticities = np.vstack([surface_omega, horseshoe_omega])
        sigmas = np.concatenate([surface_sigma, horseshoe_sigma])

        return positions, vorticities, sigmas

    def is_inside(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Return boolean mask of points inside the pier."""
        return (x - self.x) ** 2 + (y - self.y) ** 2 < self.radius ** 2

    def reflect_particles(self, positions: np.ndarray) -> np.ndarray:
        """Push particles that ended up inside the pier back outside."""
        dx = positions[:, 0] - self.x
        dy = positions[:, 1] - self.y
        dist = np.sqrt(dx ** 2 + dy ** 2)

        inside = dist < self.radius
        if not inside.any():
            return positions

        # Reflect to just outside
        positions = positions.copy()
        scale = (self.radius + 0.05 * self.diameter) / np.maximum(dist[inside], 1e-6)
        positions[inside, 0] = self.x + dx[inside] * scale
        positions[inside, 1] = self.y + dy[inside] * scale

        return positions
