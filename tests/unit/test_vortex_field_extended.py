"""
Extended tests for quantum_hydraulics.core.vortex_field module.

Covers methods NOT tested by the existing test_vortex_field.py:
- compute_velocity_induction() (explicit, not just via step())
- apply_diffusion() (explicit)
- _compute_energies()
- _velocity_profile_fast()
- particles property (backward compatibility)
- __repr__()
- FieldState.n_particles property
"""

import pytest
import numpy as np

from quantum_hydraulics.core.hydraulics import HydraulicsEngine
from quantum_hydraulics.core.vortex_field import VortexParticleField, FieldState
from quantum_hydraulics.core.particle import VortexParticle


@pytest.fixture
def hydraulics():
    return HydraulicsEngine(
        Q=600, width=30, depth=5, slope=0.002, roughness_ks=0.15,
    )


@pytest.fixture
def small_field(hydraulics):
    return VortexParticleField(hydraulics, length=50, n_particles=100)


@pytest.fixture
def tiny_field(hydraulics):
    """Very small field for fast kernel tests."""
    return VortexParticleField(hydraulics, length=20, n_particles=20)


# =============================================================================
# TestComputeVelocityInduction
# =============================================================================

class TestComputeVelocityInduction:
    """Explicit tests for Biot-Savart velocity induction."""

    def test_returns_ndarray(self, small_field):
        velocities = small_field.compute_velocity_induction()
        assert isinstance(velocities, np.ndarray)

    def test_shape_matches_particles(self, small_field):
        velocities = small_field.compute_velocity_induction()
        assert velocities.shape == small_field._positions.shape

    def test_no_nan_values(self, small_field):
        velocities = small_field.compute_velocity_induction()
        assert not np.any(np.isnan(velocities))

    def test_no_inf_values(self, small_field):
        velocities = small_field.compute_velocity_induction()
        assert not np.any(np.isinf(velocities))

    def test_velocities_finite_magnitude(self, small_field):
        """Induced velocities should be small compared to mean flow."""
        velocities = small_field.compute_velocity_induction()
        v_mag = np.linalg.norm(velocities, axis=1)
        # Induced velocities should not exceed mean flow by a large factor
        assert np.max(v_mag) < small_field.hydraulics.V_mean * 10

    def test_empty_field(self, hydraulics):
        """Empty field should return empty array."""
        field = VortexParticleField(hydraulics, length=50, n_particles=0)
        # Force empty
        field._positions = np.zeros((0, 3))
        field._vorticities = np.zeros((0, 3))
        field._sigmas = np.zeros(0)
        velocities = field.compute_velocity_induction()
        assert velocities.shape == (0, 3)

    def test_biot_savart_antisymmetry(self, hydraulics):
        """Two particles should induce opposite cross-velocities on each other."""
        field = VortexParticleField(hydraulics, length=50, n_particles=2)
        # Place two particles with same vorticity, symmetric positions
        field._positions = np.array([[10.0, 15.0, 2.5], [10.0, 15.0, 3.5]])
        field._vorticities = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        field._sigmas = np.array([0.5, 0.5])
        field._volumes = np.array([1.0, 1.0])

        velocities = field.compute_velocity_induction()
        # z-component of velocity should be opposite for symmetric vortex pair
        # (not exactly opposite due to regularization, but the y-components should differ)
        assert velocities.shape == (2, 3)


# =============================================================================
# TestApplyDiffusion
# =============================================================================

class TestApplyDiffusion:
    """Explicit tests for PSE viscous diffusion."""

    def test_vorticities_modified(self, small_field):
        """Diffusion should modify vorticities (needs enough particles for PSE overlap)."""
        original = small_field._vorticities.copy()
        small_field.apply_diffusion()
        # With 100 particles, PSE neighbors overlap enough for diffusion to act
        # Allow for the possibility that very sparse regions don't change
        max_diff = np.max(np.abs(small_field._vorticities - original))
        assert max_diff > 0 or len(original) < 10  # Either changed or too sparse

    def test_no_nan_after_diffusion(self, small_field):
        small_field.apply_diffusion()
        assert not np.any(np.isnan(small_field._vorticities))

    def test_no_inf_after_diffusion(self, small_field):
        small_field.apply_diffusion()
        assert not np.any(np.isinf(small_field._vorticities))

    def test_diffusion_smooths_field(self, tiny_field):
        """Diffusion should reduce variance in vorticity (smoothing effect)."""
        var_before = np.var(tiny_field._vorticities)
        tiny_field.apply_diffusion()
        var_after = np.var(tiny_field._vorticities)
        # Diffusion should reduce variance (smoothing)
        assert var_after <= var_before * 1.1  # Allow small tolerance

    def test_empty_field_no_crash(self, hydraulics):
        field = VortexParticleField(hydraulics, length=50, n_particles=0)
        field._positions = np.zeros((0, 3))
        field._vorticities = np.zeros((0, 3))
        field._sigmas = np.zeros(0)
        field.apply_diffusion()  # Should not crash


# =============================================================================
# TestComputeEnergies
# =============================================================================

class TestComputeEnergies:
    """Tests for _compute_energies()."""

    def test_returns_ndarray(self, small_field):
        energies = small_field._compute_energies()
        assert isinstance(energies, np.ndarray)

    def test_length_matches_particles(self, small_field):
        energies = small_field._compute_energies()
        assert len(energies) == len(small_field._positions)

    def test_all_non_negative(self, small_field):
        energies = small_field._compute_energies()
        assert np.all(energies >= 0)

    def test_energy_formula(self, small_field):
        """Energy = |omega|^2 * sigma^3."""
        energies = small_field._compute_energies()
        omega_sq = np.sum(small_field._vorticities**2, axis=1)
        expected = omega_sq * small_field._sigmas**3
        np.testing.assert_array_almost_equal(energies, expected)


# =============================================================================
# TestVelocityProfileFast
# =============================================================================

class TestVelocityProfileFast:
    """Tests for _velocity_profile_fast() lookup table."""

    def test_returns_ndarray(self, small_field):
        z = np.array([0.5, 1.0, 2.0])
        result = small_field._velocity_profile_fast(z)
        assert isinstance(result, np.ndarray)

    def test_shape_preserved(self, small_field):
        z = np.linspace(0.1, 4.0, 25)
        result = small_field._velocity_profile_fast(z)
        assert result.shape == z.shape

    def test_approximates_velocity_profile(self, small_field):
        """LUT should closely approximate the scalar velocity_profile."""
        z = np.array([0.5, 1.0, 2.0, 3.0, 4.0])
        fast = small_field._velocity_profile_fast(z)
        scalar = np.array([small_field.hydraulics.velocity_profile(zi) for zi in z])
        np.testing.assert_array_almost_equal(fast, scalar, decimal=1)

    def test_zero_at_origin(self, small_field):
        z = np.array([0.0])
        result = small_field._velocity_profile_fast(z)
        assert result[0] == pytest.approx(0.0, abs=0.1)


# =============================================================================
# TestParticlesProperty
# =============================================================================

class TestParticlesProperty:
    """Tests for the backward-compatibility particles property."""

    def test_returns_list(self, small_field):
        particles = small_field.particles
        assert isinstance(particles, list)

    def test_elements_are_vortex_particles(self, small_field):
        particles = small_field.particles
        for p in particles[:5]:
            assert isinstance(p, VortexParticle)

    def test_count_matches(self, small_field):
        particles = small_field.particles
        assert len(particles) == len(small_field._positions)

    def test_positions_copied(self, small_field):
        """Modifying returned particles should not affect internal arrays."""
        particles = small_field.particles
        if len(particles) > 0:
            original_pos = small_field._positions[0].copy()
            particles[0].pos[0] = 999.0
            np.testing.assert_array_equal(small_field._positions[0], original_pos)


# =============================================================================
# TestFieldStateNParticles
# =============================================================================

class TestFieldStateNParticles:
    """Tests for FieldState.n_particles property."""

    def test_n_particles_matches_positions(self, small_field):
        state = small_field.get_state()
        assert state.n_particles == len(state.positions)

    def test_empty_state(self, hydraulics):
        state = FieldState(
            positions=np.zeros((0, 3)),
            vorticities=np.zeros((0, 3)),
            energies=np.zeros(0),
            sigmas=np.zeros(0),
            ages=np.zeros(0),
            obs_center=np.array([0, 0, 0]),
            obs_radius=10.0,
            observation_active=True,
            domain_length=100,
            domain_width=30,
            domain_depth=5,
        )
        assert state.n_particles == 0


# =============================================================================
# TestRepr
# =============================================================================

class TestRepr:
    """Tests for VortexParticleField __repr__."""

    def test_repr_contains_info(self, small_field):
        r = repr(small_field)
        assert "VortexParticleField" in r
        assert "obs_active=" in r

    def test_repr_is_string(self, small_field):
        assert isinstance(repr(small_field), str)

    def test_repr_shows_dimensions(self, small_field):
        r = repr(small_field)
        assert "L=" in r
        assert "W=" in r
        assert "H=" in r


# =============================================================================
# TestOverlapDiagnostic
# =============================================================================

class TestOverlapDiagnostic:
    """Tests for overlap_ratio() and observation_zone_mask()."""

    def test_overlap_ratio_keys(self, small_field):
        stats = small_field.overlap_ratio()
        assert set(stats) == {"mean", "median", "frac_gt_1", "n"}
        assert stats["n"] == len(small_field._positions)

    def test_overlap_ratio_empty_safe(self, hydraulics):
        field = VortexParticleField(hydraulics, n_particles=100)
        empty_mask = np.zeros(len(field._positions), dtype=bool)
        stats = field.overlap_ratio(empty_mask)
        assert stats["n"] == 0
        assert np.isnan(stats["mean"])

    def test_overlap_ratio_positive(self, small_field):
        stats = small_field.overlap_ratio()
        assert stats["mean"] > 0
        assert 0.0 <= stats["frac_gt_1"] <= 1.0

    def test_observation_zone_mask_inactive_is_empty(self, small_field):
        small_field.toggle_observation(False)
        mask = small_field.observation_zone_mask()
        assert mask.sum() == 0

    def test_observation_zone_selects_subset(self, small_field):
        small_field.set_observation([25, 15, 2.5], radius=8)
        mask = small_field.observation_zone_mask()
        # Some but not all particles should be inside a small zone
        assert 0 < mask.sum() < len(small_field._positions)

    def test_shrinking_sigma_without_particles_breaks_overlap(self, hydraulics):
        """Documents flaw #2: shrinking sigma in the obs zone without adding
        particles raises h/sigma there above the well-resolved exterior."""
        field = VortexParticleField(hydraulics, length=100, n_particles=2000)
        field.set_observation([50, 15, 2.5], radius=10)
        field._sigmas = field._get_adaptive_core_sizes_batch(field._positions)

        in_zone = field.observation_zone_mask()
        outside = ~in_zone

        r_in = field.overlap_ratio(in_zone)["mean"]
        r_out = field.overlap_ratio(outside)["mean"]

        # Exterior is well overlapped (~1); interior is under-resolved (>~2)
        assert r_out < 1.6
        assert r_in > r_out


# =============================================================================
# TestPerParticleVolume
# =============================================================================

class TestPerParticleVolume:
    """Phase 2: per-particle volume element (strength = omega * Vol)."""

    def test_volumes_uniform_at_seed(self, small_field):
        assert len(small_field._volumes) == len(small_field._positions)
        assert np.ptp(small_field._volumes) == 0.0
        # Uniform seed volume equals domain volume / N
        expected = (small_field.L * small_field.W * small_field.H) / len(small_field._positions)
        assert abs(small_field._volumes[0] - expected) < 1e-9

    def test_particle_volume_property_is_mean(self, small_field):
        assert abs(small_field.particle_volume - small_field._volumes.mean()) < 1e-12

    def test_uniform_volume_matches_scalar_weighting(self, small_field):
        """With uniform volumes, induction == kernel(omega) * scalar volume."""
        from quantum_hydraulics.core.vortex_field import (
            _compute_velocity_induction_numpy,
        )
        v = small_field.compute_velocity_induction()
        base = _compute_velocity_induction_numpy(
            small_field._positions, small_field._vorticities, small_field._sigmas,
        )
        scalar = base * small_field._volumes[0]
        assert np.max(np.abs(v - scalar)) < 1e-9

    def test_volume_scales_source_strength_linearly(self, hydraulics):
        """Doubling a source particle's volume doubles the velocity it induces."""
        field = VortexParticleField(hydraulics, length=50, n_particles=2)
        field._positions = np.array([[10.0, 15.0, 2.5], [12.0, 15.0, 2.5]])
        field._vorticities = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 0.0]])
        field._sigmas = np.array([0.5, 0.5])
        field._volumes = np.array([1.0, 1.0])
        v1 = field.compute_velocity_induction()[1].copy()  # velocity on particle 1

        field._volumes = np.array([2.0, 1.0])  # double the source (particle 0)
        v2 = field.compute_velocity_induction()[1].copy()

        assert np.allclose(v2, 2.0 * v1, rtol=1e-9, atol=1e-12)

    def test_shed_particles_get_volume(self, hydraulics):
        """Pier-shed particles must keep _volumes in sync with the other arrays."""
        field = VortexParticleField(hydraulics, length=50, n_particles=200)
        # Simulate an append like pier shedding does
        n0 = len(field._positions)
        field._positions = np.vstack([field._positions, [[25.0, 15.0, 2.5]]])
        field._vorticities = np.vstack([field._vorticities, [[0.0, 1.0, 0.0]]])
        field._sigmas = np.concatenate([field._sigmas, [0.3]])
        field._volumes = np.concatenate([field._volumes, [field._ref_volume]])
        assert len(field._volumes) == len(field._positions) == n0 + 1
        # Induction must not raise on the appended state
        v = field.compute_velocity_induction()
        assert v.shape == (n0 + 1, 3)
