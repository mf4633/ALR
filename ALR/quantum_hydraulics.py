"""
quantum_hydraulics.py
=====================
Adaptive Lagrangian Refinement (ALR) for PCSWMM.
Professional Version for ICWMM 2026.
"""

import numpy as np
from typing import Dict, List, Optional
import time

class VortexParticle:
    __slots__ = ['pos', 'strength', 'sigma', 'energy']
    def __init__(self, position: np.ndarray, strength: float, core_size: float):
        self.pos = np.asarray(position, dtype=np.float32)
        self.strength = strength
        self.sigma = core_size
        self.energy = strength**2 / (2 * np.pi * core_size**2)

class QuantumNode:
    def __init__(self, node_id: str, width: float, length: float, roughness_ks: float):
        self.node_id = node_id
        self.width = width
        self.length = length
        self.ks = roughness_ks
        self.particles: List[VortexParticle] = []
        self.rho = 1.94  # slug/ft3
        self.metrics = {}

    def _get_log_law_velocity(self, z: float, u_star: float) -> float:
        """Law of the Wall implementation for bed-shear accuracy."""
        z_0 = self.ks / 30.0
        return (u_star / 0.41) * np.log(max(z, z_0) / z_0)

    def update_and_evolve(self, depth: float, inflow: float, dt: float):
        """Evolves the turbulent field using Lagrangian advection."""
        depth = max(0.1, depth)
        v_mean = inflow / (self.width * depth) if inflow > 0 else 0
        u_star = v_mean * 0.1 # Friction velocity approx

        # 1. Advect existing particles
        for p in self.particles:
            u_z = self._get_log_law_velocity(p.pos[2], u_star)
            p.pos[0] += u_z * dt
            # Boundary damping
            if abs(p.pos[1]) > self.width/2: p.pos[1] *= 0.9

        # 2. Cull and Inject
        self.particles = [p for p in self.particles if p.pos[0] < self.length]
        for _ in range(20): # Injection rate
            z_rand = np.random.uniform(0, depth)
            # Quanta shrinking: smaller sigma near bed for higher resolution
            sigma = 0.05 + (z_rand / depth) * 0.4
            p_new = VortexParticle(
                position=np.array([0.0, np.random.uniform(-self.width/2, self.width/2), z_rand]),
                strength=self._get_log_law_velocity(z_rand, u_star),
                core_size=sigma
            )
            self.particles.append(p_new)

    def compute_metrics(self):
        """Extracts design values for riprap and scour assessment."""
        if not self.particles: return
        v_mags = [p.strength for p in self.particles]
        max_v = np.max(v_mags)
        tau = self.rho * (0.1 * max_v)**2
        
        self.metrics = {
            'max_velocity': max_v,
            'bed_shear_stress': tau,
            'scour_risk_index': min(1.0, tau / 0.15), # 0.15 psf threshold
            'tke': np.mean([p.energy for p in self.particles]),
            'n_particles': len(self.particles)
        }

    def get_metrics(self) -> Dict:
        return self.metrics if self.metrics else {k: 0.0 for k in ['max_velocity', 'bed_shear_stress', 'scour_risk_index', 'tke', 'n_particles']}