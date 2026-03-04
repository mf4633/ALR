"""
Unit tests for VortexParticleField class.
"""

import pytest
import numpy as np
from quantum_hydraulics.core.hydraulics import HydraulicsEngine
from quantum_hydraulics.core.vortex_field import VortexParticleField, FieldState


@pytest.fixture
def hydraulics():
    """Create standard hydraulics for tests."""
    return HydraulicsEngine(Q=600, width=30, depth=5, slope=0.002, roughness_ks=0.15)


@pytest.fixture
def field(hydraulics):
    """Create standard field for tests."""
    return VortexParticleField(hydraulics, length=100, n_particles=500)


class TestFieldCreation:
    """Test field creation and initialization."""

    def test_creates_particles(self, hydraulics):
        """Field should create particles."""
        field = VortexParticleField(hydraulics, n_particles=100)
        assert len(field.particles) > 0

    def test_particle_count_approximate(self, hydraulics):
        """Particle count should be approximately as requested."""
        field = VortexParticleField(hydraulics, n_particles=500)
        # Allow 20% variation due to scale weighting
        assert abs(len(field.particles) - 500) < 150

    def test_domain_dimensions(self, hydraulics):
        """Domain should match hydraulics."""
        field = VortexParticleField(hydraulics, length=150)

        assert field.L == 150
        assert field.W == hydraulics.width
        assert field.H == hydraulics.depth


class TestParticlePositions:
    """Test particle position constraints."""

    def test_particles_in_domain(self, field):
        """All particles should be within domain."""
        for p in field.particles:
            assert 0 <= p.pos[0] <= field.L
            assert 0 <= p.pos[1] <= field.W
            assert 0 <= p.pos[2] <= field.H

    def test_particles_avoid_boundaries(self, field):
        """Particles should avoid immediate boundaries."""
        for p in field.particles:
            # Should be seeded away from boundaries
            assert p.pos[0] > 0.05 * field.L
            assert p.pos[0] < 0.95 * field.L


class TestAdaptiveResolution:
    """Test observation-dependent resolution."""

    def test_observation_reduces_sigma(self, field):
        """Sigma should be smaller near observation zone."""
        # At observation center
        sigma_obs = field.get_adaptive_core_size(field.obs_center)

        # Far from observation
        pos_far = np.array([0, 0, field.H / 2])
        sigma_far = field.get_adaptive_core_size(pos_far)

        assert sigma_obs < sigma_far

    def test_sigma_within_bounds(self, field):
        """Sigma should stay within min/max bounds."""
        for p in field.particles:
            sigma = field.get_adaptive_core_size(p.pos)
            assert field.min_sigma <= sigma <= field.max_sigma

    def test_observation_toggle(self, field):
        """Toggling observation should affect sigma."""
        pos = field.obs_center.copy()

        field.observation_active = True
        sigma_on = field.get_adaptive_core_size(pos)

        field.observation_active = False
        sigma_off = field.get_adaptive_core_size(pos)

        # With observation off, should return base sigma everywhere
        assert sigma_off == field.base_sigma
        assert sigma_on < sigma_off


class TestFieldEvolution:
    """Test field time evolution."""

    def test_step_moves_particles(self, field):
        """Step should move particles."""
        initial_positions = [p.pos.copy() for p in field.particles]

        field.step(dt=0.1)

        final_positions = [p.pos for p in field.particles]

        # At least some should have moved
        moved = any(
            not np.allclose(i, f)
            for i, f in zip(initial_positions, final_positions)
        )
        assert moved

    def test_step_updates_ages(self, field):
        """Step should update particle ages."""
        initial_ages = [p.age for p in field.particles]

        field.step(dt=0.1)

        final_ages = [p.age for p in field.particles]

        # All ages should increase
        for i, f in zip(initial_ages, final_ages):
            assert f > i

    def test_particles_stay_in_domain(self, field):
        """Particles should stay in domain after stepping."""
        for _ in range(10):
            field.step(dt=0.1)

        for p in field.particles:
            assert 0 <= p.pos[0] <= field.L
            assert 0 <= p.pos[1] <= field.W
            assert 0 <= p.pos[2] <= field.H

    def test_step_preserves_particle_count(self, field):
        """Particle count should be roughly preserved."""
        n_initial = len(field.particles)

        for _ in range(10):
            field.step(dt=0.1)

        n_final = len(field.particles)

        # Should not change drastically
        assert abs(n_final - n_initial) < 0.2 * n_initial


class TestFieldState:
    """Test FieldState extraction."""

    def test_get_state_returns_fieldstate(self, field):
        """get_state should return FieldState object."""
        state = field.get_state()
        assert isinstance(state, FieldState)

    def test_state_particle_count(self, field):
        """State should have correct particle count."""
        state = field.get_state()
        assert state.n_particles == len(field.particles)

    def test_state_positions_shape(self, field):
        """State positions should have correct shape."""
        state = field.get_state()
        assert state.positions.shape == (len(field.particles), 3)

    def test_state_energies_positive(self, field):
        """State energies should all be positive."""
        state = field.get_state()
        assert np.all(state.energies > 0)

    def test_state_contains_domain_info(self, field):
        """State should contain domain information."""
        state = field.get_state()

        assert state.domain_length == field.L
        assert state.domain_width == field.W
        assert state.domain_depth == field.H


class TestObservationZone:
    """Test observation zone manipulation."""

    def test_set_observation(self, field):
        """Should be able to set observation zone."""
        new_center = np.array([50, 15, 2.5])
        new_radius = 10.0

        field.set_observation(new_center, new_radius)

        np.testing.assert_array_equal(field.obs_center, new_center)
        assert field.obs_radius == new_radius

    def test_toggle_observation(self, field):
        """Should be able to toggle observation."""
        initial = field.observation_active

        field.toggle_observation()

        assert field.observation_active != initial

        field.toggle_observation()

        assert field.observation_active == initial

    def test_toggle_with_value(self, field):
        """Should be able to set specific observation state."""
        field.toggle_observation(True)
        assert field.observation_active == True

        field.toggle_observation(False)
        assert field.observation_active == False


class TestHydraulicsUpdate:
    """Test updating hydraulics."""

    def test_update_hydraulics(self, field):
        """Should be able to update hydraulics."""
        new_hydraulics = HydraulicsEngine(
            Q=1000, width=40, depth=8, slope=0.003, roughness_ks=0.2
        )

        field.update_hydraulics(new_hydraulics)

        assert field.W == 40
        assert field.H == 8
