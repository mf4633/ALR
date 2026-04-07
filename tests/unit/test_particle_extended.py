"""
Extended tests for quantum_hydraulics.core.particle module.

Covers: vorticity_magnitude property (only gap in existing coverage).
"""

import pytest
import numpy as np

from quantum_hydraulics.core.particle import VortexParticle


class TestVorticityMagnitude:
    """Tests for the vorticity_magnitude property."""

    def test_basic_magnitude(self):
        p = VortexParticle.create(
            position=[0, 0, 0],
            vorticity=[3.0, 4.0, 0.0],
            core_size=0.5,
        )
        assert abs(p.vorticity_magnitude - 5.0) < 1e-10

    def test_unit_vorticity(self):
        p = VortexParticle.create(
            position=[0, 0, 0],
            vorticity=[1.0, 0.0, 0.0],
            core_size=0.5,
        )
        assert abs(p.vorticity_magnitude - 1.0) < 1e-10

    def test_3d_vorticity(self):
        p = VortexParticle.create(
            position=[0, 0, 0],
            vorticity=[1.0, 2.0, 3.0],
            core_size=0.5,
        )
        expected = np.sqrt(1 + 4 + 9)
        assert abs(p.vorticity_magnitude - expected) < 1e-10

    def test_zero_vorticity(self):
        p = VortexParticle.create(
            position=[0, 0, 0],
            vorticity=[0.0, 0.0, 0.0],
            core_size=0.5,
        )
        assert p.vorticity_magnitude == 0.0

    def test_consistent_with_circulation(self):
        """circulation = |omega| * sigma^2, so |omega| = circulation / sigma^2."""
        p = VortexParticle.create(
            position=[0, 0, 0],
            vorticity=[2.0, 3.0, 6.0],
            core_size=0.5,
        )
        assert abs(p.vorticity_magnitude * p.sigma**2 - p.circulation) < 1e-10

    def test_consistent_with_energy(self):
        """energy = |omega|^2 * sigma^3."""
        p = VortexParticle.create(
            position=[0, 0, 0],
            vorticity=[2.0, 3.0, 6.0],
            core_size=0.5,
        )
        expected_energy = p.vorticity_magnitude**2 * p.sigma**3
        assert abs(p.energy - expected_energy) < 1e-10
