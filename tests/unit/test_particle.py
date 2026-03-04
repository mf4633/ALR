"""
Unit tests for VortexParticle class.
"""

import pytest
import numpy as np
from quantum_hydraulics.core.particle import VortexParticle


class TestParticleCreation:
    """Test particle creation and initialization."""

    def test_create_basic(self):
        """Test basic particle creation."""
        p = VortexParticle.create([1, 2, 3], [0.1, 0.2, 0.3], 0.5)

        assert p.pos.shape == (3,)
        assert p.omega.shape == (3,)
        assert p.sigma == 0.5
        assert p.age == 0.0

    def test_position_stored_correctly(self):
        """Position should be stored as numpy array."""
        p = VortexParticle.create([1.5, 2.5, 3.5], [0, 0, 0], 1.0)

        np.testing.assert_array_almost_equal(p.pos, [1.5, 2.5, 3.5])

    def test_vorticity_stored_correctly(self):
        """Vorticity should be stored as numpy array."""
        p = VortexParticle.create([0, 0, 0], [1.1, 2.2, 3.3], 1.0)

        np.testing.assert_array_almost_equal(p.omega, [1.1, 2.2, 3.3])

    def test_initial_age_zero(self):
        """Initial age should be zero."""
        p = VortexParticle.create([0, 0, 0], [1, 0, 0], 0.5)
        assert p.age == 0.0

    def test_custom_initial_age(self):
        """Should accept custom initial age."""
        p = VortexParticle.create([0, 0, 0], [1, 0, 0], 0.5, age=1.5)
        assert p.age == 1.5


class TestParticleEnergy:
    """Test energy computation."""

    def test_energy_formula(self):
        """Energy = |omega|^2 * sigma^3"""
        omega = np.array([3.0, 4.0, 0.0])  # |omega| = 5
        sigma = 2.0

        p = VortexParticle.create([0, 0, 0], omega, sigma)

        expected = 5.0 ** 2 * 2.0 ** 3  # 25 * 8 = 200
        assert abs(p.energy - expected) < 1e-10

    def test_energy_updates_with_sigma(self):
        """Energy should update when sigma changes."""
        p = VortexParticle.create([0, 0, 0], [1, 0, 0], 1.0)
        e1 = p.energy

        p.update_sigma(2.0)
        e2 = p.energy

        # sigma doubled, so energy should be 8x (sigma^3)
        assert abs(e2 / e1 - 8.0) < 1e-10

    def test_energy_positive(self):
        """Energy should always be positive."""
        p = VortexParticle.create([0, 0, 0], [-1, -2, -3], 0.5)
        assert p.energy > 0


class TestParticleAdvection:
    """Test particle advection."""

    def test_advection_updates_position(self):
        """Advection should update position correctly."""
        p = VortexParticle.create([0, 0, 0], [1, 0, 0], 0.5)

        velocity = np.array([1.0, 2.0, 3.0])
        dt = 0.1

        p.advect(velocity, dt)

        expected_pos = np.array([0.1, 0.2, 0.3])
        np.testing.assert_array_almost_equal(p.pos, expected_pos)

    def test_advection_updates_age(self):
        """Advection should update age."""
        p = VortexParticle.create([0, 0, 0], [1, 0, 0], 0.5)

        p.advect(np.array([1, 0, 0]), 0.5)

        assert abs(p.age - 0.5) < 1e-10

    def test_multiple_advections(self):
        """Multiple advections should accumulate."""
        p = VortexParticle.create([0, 0, 0], [1, 0, 0], 0.5)

        for _ in range(10):
            p.advect(np.array([1, 0, 0]), 0.1)

        assert abs(p.pos[0] - 1.0) < 1e-10
        assert abs(p.age - 1.0) < 1e-10


class TestParticleCirculation:
    """Test circulation property."""

    def test_circulation_formula(self):
        """Circulation = |omega| * sigma^2"""
        p = VortexParticle.create([0, 0, 0], [3, 4, 0], 2.0)

        expected = 5.0 * 4.0  # |omega|=5, sigma^2=4
        assert abs(p.circulation - expected) < 1e-10


class TestParticleCopy:
    """Test particle copy functionality."""

    def test_copy_creates_independent(self):
        """Copy should be independent of original."""
        p1 = VortexParticle.create([1, 2, 3], [0.1, 0.2, 0.3], 0.5)
        p2 = p1.copy()

        # Modify original
        p1.pos[0] = 100
        p1.omega[0] = 100

        # Copy should be unchanged
        assert p2.pos[0] == 1
        assert p2.omega[0] == 0.1

    def test_copy_preserves_values(self):
        """Copy should preserve all values."""
        p1 = VortexParticle.create([1, 2, 3], [0.1, 0.2, 0.3], 0.5, age=1.5)
        p2 = p1.copy()

        np.testing.assert_array_equal(p2.pos, p1.pos)
        np.testing.assert_array_equal(p2.omega, p1.omega)
        assert p2.sigma == p1.sigma
        assert p2.age == p1.age


class TestParticleRepr:
    """Test string representation."""

    def test_repr_contains_info(self):
        """Repr should contain useful info."""
        p = VortexParticle.create([1.5, 2.5, 3.5], [0.1, 0.2, 0.3], 0.5)
        s = repr(p)

        assert "VortexParticle" in s
        assert "1.5" in s or "1.50" in s
        assert "0.5" in s or "0.50" in s
