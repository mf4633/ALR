"""
QuantumNode - Unified SWMM node analysis with vortex particle method.

Provides physics-based turbulence analysis for SWMM junction nodes,
computing scour risk, bed shear stress, and turbulent kinetic energy.
"""

import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass, field

from quantum_hydraulics.core.particle import VortexParticle
from quantum_hydraulics.core.hydraulics import HydraulicsEngine


@dataclass
class NodeMetrics:
    """Engineering metrics computed for a SWMM node."""

    max_velocity: float = 0.0
    mean_velocity: float = 0.0
    bed_shear_stress: float = 0.0
    scour_risk_index: float = 0.0
    tke: float = 0.0
    n_particles: int = 0
    froude_number: float = 0.0
    reynolds_number: float = 0.0

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "max_velocity": self.max_velocity,
            "mean_velocity": self.mean_velocity,
            "bed_shear_stress": self.bed_shear_stress,
            "scour_risk_index": self.scour_risk_index,
            "tke": self.tke,
            "n_particles": self.n_particles,
            "froude_number": self.froude_number,
            "reynolds_number": self.reynolds_number,
        }


class QuantumNode:
    """
    Quantum-inspired turbulence analysis for a SWMM junction node.

    Uses adaptive Lagrangian refinement (ALR) with vortex particles
    to compute turbulence metrics for engineering design.

    Parameters
    ----------
    node_id : str
        SWMM node identifier
    width : float
        Effective channel width at node in feet
    length : float, optional
        Analysis length in feet (default 30.0)
    roughness_ks : float, optional
        Equivalent sand roughness in feet (default 0.1)
    observation_radius : float, optional
        Observation zone radius in feet (default 15.0)
    side_slope : float, optional
        Side slope H:V (default 2.0)

    Attributes
    ----------
    particles : List[VortexParticle]
        Active vortex particles in the node
    metrics : NodeMetrics
        Latest computed engineering metrics
    """

    # Physical constants
    RHO = 1.94  # slug/ft3 (water density)
    NU = 1.1e-5  # ft2/s (kinematic viscosity at 60F)
    G = 32.2  # ft/s2 (gravity)
    KAPPA = 0.41  # von Karman constant

    # Scour threshold (critical shear stress for sand, psf)
    CRITICAL_SHEAR = 0.15

    def __init__(
        self,
        node_id: str,
        width: float,
        length: float = 30.0,
        roughness_ks: float = 0.1,
        observation_radius: float = 15.0,
        side_slope: float = 2.0,
    ):
        self.node_id = node_id
        self.width = width
        self.length = length
        self.ks = roughness_ks
        self.obs_radius = observation_radius
        self.side_slope = side_slope

        self.particles: List[VortexParticle] = []
        self.metrics = NodeMetrics()

        # Internal state
        self._depth = 0.0
        self._inflow = 0.0
        self._v_mean = 0.0
        self._u_star = 0.0

    def _log_law_velocity(self, z: float) -> float:
        """
        Compute velocity at height z using law of the wall.

        Parameters
        ----------
        z : float
            Height above bed in feet

        Returns
        -------
        float
            Velocity at height z in ft/s
        """
        z_0 = self.ks / 30.0
        z_0 = max(z_0, 1e-6)
        z = max(z, z_0)
        return (self._u_star / self.KAPPA) * np.log(z / z_0)

    def update_from_swmm(self, depth: float, inflow: float):
        """
        Update node state from SWMM simulation results.

        Parameters
        ----------
        depth : float
            Water depth at node in feet
        inflow : float
            Total inflow to node in cfs
        """
        self._depth = max(0.1, depth)
        self._inflow = max(0.0, inflow)

        # Compute mean velocity
        area = self.width * self._depth + self.side_slope * self._depth ** 2
        self._v_mean = self._inflow / area if area > 0 else 0.0

        # Friction velocity (approximation: u* ~ 0.1 * V_mean for rough channels)
        self._u_star = self._v_mean * 0.1

    def update_and_evolve(self, depth: float, inflow: float, dt: float = 0.1):
        """
        Update from SWMM and evolve particle field.

        Parameters
        ----------
        depth : float
            Water depth in feet
        inflow : float
            Inflow in cfs
        dt : float
            Timestep in seconds
        """
        self.update_from_swmm(depth, inflow)

        # Advect existing particles
        for p in self.particles:
            u_z = self._log_law_velocity(p.pos[2])
            p.advect(np.array([u_z, 0, 0]), dt)

            # Boundary damping
            if abs(p.pos[1]) > self.width / 2:
                p.pos[1] *= 0.9

        # Cull particles that left domain
        self.particles = [p for p in self.particles if p.pos[0] < self.length]

        # Inject new particles at inlet
        for _ in range(20):
            z_rand = np.random.uniform(0, self._depth)

            # Adaptive sigma: smaller near bed for higher resolution
            sigma = 0.05 + (z_rand / self._depth) * 0.4

            strength = self._log_law_velocity(z_rand)

            p_new = VortexParticle.create(
                position=np.array([0.0, np.random.uniform(-self.width / 2, self.width / 2), z_rand]),
                vorticity=np.array([strength, 0, 0]),  # Streamwise vorticity
                core_size=sigma,
            )
            self.particles.append(p_new)

    def compute_turbulence(self, n_particles: int = 500):
        """
        Compute turbulence field and update metrics.

        Parameters
        ----------
        n_particles : int
            Target number of particles for analysis
        """
        if self._v_mean <= 0:
            self.metrics = NodeMetrics()
            return

        # Ensure we have enough particles
        while len(self.particles) < n_particles:
            z_rand = np.random.uniform(0, self._depth)
            sigma = 0.05 + (z_rand / self._depth) * 0.4
            strength = self._log_law_velocity(z_rand)

            p_new = VortexParticle.create(
                position=np.array([
                    np.random.uniform(0, self.length),
                    np.random.uniform(-self.width / 2, self.width / 2),
                    z_rand
                ]),
                vorticity=np.array([strength, 0, 0]),
                core_size=sigma,
            )
            self.particles.append(p_new)

        self._compute_metrics()

    def _compute_metrics(self):
        """Compute engineering metrics from particle field."""
        if not self.particles:
            self.metrics = NodeMetrics()
            return

        # Velocity magnitudes from vorticity strengths
        v_mags = np.array([p.vorticity_magnitude for p in self.particles])
        energies = np.array([p.energy for p in self.particles])

        max_v = float(np.max(v_mags))
        mean_v = float(np.mean(v_mags))

        # Bed shear stress: tau = rho * u*^2
        # Use max velocity to estimate u* near critical regions
        u_star_max = max_v * 0.1
        tau = self.RHO * u_star_max ** 2

        # Scour risk index (0-1, based on critical shear)
        scour_risk = min(1.0, tau / self.CRITICAL_SHEAR)

        # Turbulent kinetic energy
        tke = float(np.mean(energies))

        # Froude and Reynolds numbers
        if self._depth > 0:
            froude = self._v_mean / np.sqrt(self.G * self._depth)
        else:
            froude = 0.0

        r_h = (self.width * self._depth) / (self.width + 2 * self._depth) if self._depth > 0 else 0
        reynolds = self._v_mean * r_h / self.NU if r_h > 0 else 0

        self.metrics = NodeMetrics(
            max_velocity=max_v,
            mean_velocity=mean_v,
            bed_shear_stress=tau,
            scour_risk_index=scour_risk,
            tke=tke,
            n_particles=len(self.particles),
            froude_number=froude,
            reynolds_number=reynolds,
        )

    def get_metrics(self) -> Dict:
        """
        Get current engineering metrics as dictionary.

        Returns
        -------
        dict
            Dictionary with all metrics
        """
        return self.metrics.to_dict()

    def get_engineering_metrics(self) -> Dict:
        """Alias for get_metrics() for compatibility."""
        return self.get_metrics()

    def get_scour_assessment(self) -> str:
        """
        Get text assessment of scour risk.

        Returns
        -------
        str
            Assessment string
        """
        risk = self.metrics.scour_risk_index

        if risk > 0.7:
            return "CRITICAL - Scour protection REQUIRED"
        elif risk > 0.5:
            return "HIGH - Scour protection recommended"
        elif risk > 0.3:
            return "MODERATE - Monitor conditions"
        else:
            return "LOW - Acceptable"

    def get_velocity_assessment(self) -> str:
        """
        Get text assessment of velocity.

        Returns
        -------
        str
            Assessment string
        """
        v = self.metrics.max_velocity

        if v > 15:
            return "EXTREME - Energy dissipation REQUIRED"
        elif v > 10:
            return "HIGH - Energy dissipation required"
        elif v > 6:
            return "ELEVATED - Consider energy dissipation"
        else:
            return "ACCEPTABLE"

    def clear(self):
        """Clear all particles."""
        self.particles.clear()
        self.metrics = NodeMetrics()

    def __repr__(self) -> str:
        return (
            f"QuantumNode(id='{self.node_id}', "
            f"particles={len(self.particles)}, "
            f"scour_risk={self.metrics.scour_risk_index:.2f})"
        )
