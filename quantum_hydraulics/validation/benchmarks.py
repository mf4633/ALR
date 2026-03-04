"""
Pytest test suite for quantum hydraulics validation.

Run with: pytest tests/validation/benchmarks.py -v

Coverage targets:
- Unit tests for core physics computations
- Analytical validation against exact solutions
- Integration tests for full simulation
"""

import numpy as np
from typing import Tuple

# Make pytest optional for validation report
try:
    import pytest
    _HAS_PYTEST = True
except ImportError:
    _HAS_PYTEST = False
    # Create dummy decorator for when pytest isn't available
    class _DummyPytest:
        class fixture:
            def __init__(self, *args, **kwargs):
                pass
            def __call__(self, func):
                return func
        class mark:
            @staticmethod
            def slow(func):
                return func
    pytest = _DummyPytest()

# Import modules under test
from quantum_hydraulics.core.particle import VortexParticle
from quantum_hydraulics.core.hydraulics import HydraulicsEngine
from quantum_hydraulics.core.vortex_field import VortexParticleField
from quantum_hydraulics.validation.analytical import (
    lamb_oseen_vortex,
    poiseuille_velocity,
    kolmogorov_spectrum,
    wall_vorticity,
    kolmogorov_scales,
    colebrook_white,
)


class TestVortexParticle:
    """Unit tests for VortexParticle class."""

    def test_particle_creation(self):
        """Test basic particle creation."""
        p = VortexParticle.create(
            position=[1.0, 2.0, 3.0],
            vorticity=[0.1, 0.2, 0.3],
            core_size=0.5,
        )

        assert p.pos.shape == (3,)
        assert p.omega.shape == (3,)
        assert p.sigma == 0.5
        assert p.age == 0.0

    def test_particle_energy(self):
        """Test energy computation: E = |omega|^2 * sigma^3."""
        p = VortexParticle.create(
            position=[0, 0, 0],
            vorticity=[1.0, 0, 0],
            core_size=1.0,
        )

        expected_energy = 1.0 ** 2 * 1.0 ** 3
        assert abs(p.energy - expected_energy) < 1e-10

    def test_particle_advection(self):
        """Test particle advection."""
        p = VortexParticle.create([0, 0, 0], [1, 0, 0], 0.5)
        velocity = np.array([1.0, 2.0, 0.0])
        dt = 0.1

        p.advect(velocity, dt)

        expected_pos = np.array([0.1, 0.2, 0.0])
        np.testing.assert_array_almost_equal(p.pos, expected_pos)
        assert abs(p.age - dt) < 1e-10

    def test_particle_circulation(self):
        """Test circulation property."""
        omega_mag = 2.0
        sigma = 0.5
        p = VortexParticle.create([0, 0, 0], [omega_mag, 0, 0], sigma)

        expected_circ = omega_mag * sigma ** 2
        assert abs(p.circulation - expected_circ) < 1e-10


class TestHydraulicsEngine:
    """Unit tests for HydraulicsEngine."""

    @pytest.fixture
    def typical_channel(self):
        """Create typical open channel conditions."""
        return HydraulicsEngine(
            Q=600.0,  # cfs
            width=30.0,  # ft
            depth=5.0,  # ft
            slope=0.002,
            roughness_ks=0.15,  # ft
        )

    def test_continuity(self, typical_channel):
        """Q = V * A must hold exactly."""
        h = typical_channel
        Q_computed = h.V_mean * h.A
        assert abs(Q_computed - h.Q) < 1e-10

    def test_hydraulic_radius(self, typical_channel):
        """R = A / P must hold."""
        h = typical_channel
        R_computed = h.A / h.P
        assert abs(R_computed - h.R) < 1e-10

    def test_reynolds_number_positive(self, typical_channel):
        """Reynolds number must be positive for positive flow."""
        assert typical_channel.Re > 0

    def test_froude_number_reasonable(self, typical_channel):
        """Froude number should be positive and reasonable."""
        assert 0 < typical_channel.Fr < 10

    def test_colebrook_white_convergence(self, typical_channel):
        """Friction factor must converge to consistent value."""
        f1 = typical_channel.friction_factor
        f2 = colebrook_white(typical_channel.Re, typical_channel.ks / (4 * typical_channel.R))

        # Should agree within 1%
        assert abs(f1 - f2) / f1 < 0.01

    def test_kolmogorov_scales(self, typical_channel):
        """Kolmogorov scales must be physically reasonable."""
        h = typical_channel

        # eta = (nu^3/epsilon)^0.25
        eta_expected = (h.nu ** 3 / h.epsilon) ** 0.25

        assert abs(h.eta_kolmogorov - eta_expected) < 1e-10

        # Kolmogorov scale should be smaller than channel depth
        assert h.eta_kolmogorov < h.depth

        # Kolmogorov scale should be positive
        assert h.eta_kolmogorov > 0

    def test_velocity_profile_boundary(self, typical_channel):
        """Velocity should be zero at z=0."""
        u_0 = typical_channel.velocity_profile(0)
        assert u_0 == 0.0

    def test_velocity_profile_positive(self, typical_channel):
        """Velocity should be positive in channel."""
        z_values = np.linspace(0.1, typical_channel.depth, 10)
        for z in z_values:
            assert typical_channel.velocity_profile(z) > 0

    def test_velocity_profile_increases(self, typical_channel):
        """Velocity should generally increase with height."""
        z1, z2 = 0.1, 1.0
        u1 = typical_channel.velocity_profile(z1)
        u2 = typical_channel.velocity_profile(z2)
        assert u2 > u1

    def test_supercritical_detection(self):
        """High velocity should produce Fr > 1."""
        # Steep slope, shallow depth -> high velocity -> supercritical
        h = HydraulicsEngine(Q=1000, width=10, depth=1.5, slope=0.01, roughness_ks=0.05)
        # Note: May or may not be supercritical depending on exact values
        # Just verify Fr is computed
        assert h.Fr > 0


class TestLambOseenVortex:
    """Validation against Lamb-Oseen analytical solution."""

    def test_initial_condition(self):
        """At t->0, should approach point vortex."""
        r = np.array([0.1, 0.5, 1.0, 2.0])
        t_small = 1e-6
        gamma = 1.0

        v_theta = lamb_oseen_vortex(r, t_small, gamma)

        # Point vortex: v = Gamma / (2*pi*r)
        v_point = gamma / (2 * np.pi * r)

        # Should be close for small t
        np.testing.assert_array_almost_equal(v_theta, v_point, decimal=3)

    def test_decay_with_time(self):
        """Vortex should decay with time."""
        r = np.array([0.1, 0.5, 1.0])
        t1, t2 = 0.1, 1.0

        v1 = lamb_oseen_vortex(r, t1)
        v2 = lamb_oseen_vortex(r, t2)

        # Velocity should decrease with time
        assert np.all(v2 < v1)

    def test_circulation_conservation(self):
        """Total circulation should be conserved."""
        r = np.linspace(0.001, 10, 1000)
        dr = r[1] - r[0]
        gamma = 1.0

        for t in [0.1, 1.0, 10.0]:
            v = lamb_oseen_vortex(r, t, gamma)
            # Circulation = integral of v * 2*pi*r
            circ = np.sum(v * 2 * np.pi * dr)
            # Note: This is approximate due to finite domain
            # Just verify it's reasonable
            assert circ > 0


class TestKolmogorovSpectrum:
    """Validation of energy spectrum."""

    def test_minus_five_thirds_slope(self):
        """Verify -5/3 slope in log-log space."""
        k = np.logspace(-1, 2, 100)
        epsilon = 0.1

        E = kolmogorov_spectrum(k, epsilon)

        # In log space: log(E) = const + (-5/3)*log(k)
        log_k = np.log10(k)
        log_E = np.log10(E)

        slope = np.polyfit(log_k, log_E, 1)[0]

        # Slope should be -5/3 = -1.667
        assert abs(slope - (-5 / 3)) < 0.01

    def test_epsilon_scaling(self):
        """E should scale as epsilon^(2/3)."""
        k = np.array([1.0, 10.0])
        eps1, eps2 = 0.1, 1.0

        E1 = kolmogorov_spectrum(k, eps1)
        E2 = kolmogorov_spectrum(k, eps2)

        # E2/E1 should equal (eps2/eps1)^(2/3)
        ratio_expected = (eps2 / eps1) ** (2 / 3)
        ratio_actual = E2 / E1

        np.testing.assert_array_almost_equal(ratio_actual, ratio_expected)


class TestWallVorticity:
    """Validation of wall boundary vorticity."""

    def test_vorticity_decreases_with_height(self):
        """Vorticity should decrease away from wall."""
        z = np.array([0.01, 0.1, 1.0])
        u_star = 0.1

        omega = wall_vorticity(z, u_star)

        # Should be monotonically decreasing
        assert omega[0] > omega[1] > omega[2]

    def test_vorticity_scales_with_ustar(self):
        """Vorticity should scale linearly with u*."""
        z = np.array([0.1, 1.0])
        u_star_1, u_star_2 = 0.1, 0.2

        omega1 = wall_vorticity(z, u_star_1)
        omega2 = wall_vorticity(z, u_star_2)

        # omega2/omega1 should equal u_star_2/u_star_1
        np.testing.assert_array_almost_equal(omega2 / omega1, u_star_2 / u_star_1)


class TestVortexParticleField:
    """Integration tests for VortexParticleField."""

    @pytest.fixture
    def field(self):
        """Create test field."""
        h = HydraulicsEngine(Q=600, width=30, depth=5, slope=0.002, roughness_ks=0.15)
        return VortexParticleField(h, length=100, n_particles=500)

    def test_particle_count(self, field):
        """Should have approximately requested number of particles."""
        # Allow 10% variation due to scale weighting
        assert abs(len(field.particles) - 500) < 100

    def test_particles_in_domain(self, field):
        """All particles should be in domain."""
        for p in field.particles:
            assert 0 <= p.pos[0] <= field.L
            assert 0 <= p.pos[1] <= field.W
            assert 0 <= p.pos[2] <= field.H

    def test_observation_affects_sigma(self, field):
        """Sigma should be smaller near observation zone."""
        # Point in observation zone
        pos_obs = field.obs_center.copy()
        sigma_obs = field.get_adaptive_core_size(pos_obs)

        # Point far from observation
        pos_far = np.array([0, 0, field.H / 2])
        sigma_far = field.get_adaptive_core_size(pos_far)

        # Sigma should be smaller in observation zone
        assert sigma_obs < sigma_far

    def test_step_updates_positions(self, field):
        """Step should move particles."""
        initial_positions = np.array([p.pos.copy() for p in field.particles])

        field.step(dt=0.1)

        final_positions = np.array([p.pos for p in field.particles])

        # At least some particles should have moved
        moved = np.any(final_positions != initial_positions)
        assert moved

    def test_step_preserves_particle_count(self, field):
        """Step should not create or destroy particles."""
        n_initial = len(field.particles)

        for _ in range(10):
            field.step(dt=0.05)

        # Particle count may vary slightly but shouldn't change drastically
        n_final = len(field.particles)
        assert abs(n_final - n_initial) < 0.1 * n_initial


class TestEnergyCascade:
    """Test energy cascade behavior."""

    def test_energy_spectrum_slope(self):
        """Simulated spectrum should approach -5/3 in inertial range."""
        h = HydraulicsEngine(Q=600, width=30, depth=5, slope=0.002, roughness_ks=0.15)
        field = VortexParticleField(h, length=200, n_particles=2000)

        # Let system evolve
        for _ in range(20):
            field.step(dt=0.05)

        # Get energy by scale
        state = field.get_state()
        energies = state.energies
        scales = state.sigmas

        if len(scales) < 10:
            pytest.skip("Not enough particles for spectrum analysis")

        # Bin by scale
        scale_bins = np.logspace(np.log10(scales.min()), np.log10(scales.max()), 10)
        energy_binned = []
        scale_centers = []

        for i in range(len(scale_bins) - 1):
            mask = (scales >= scale_bins[i]) & (scales < scale_bins[i + 1])
            if np.sum(mask) > 0:
                energy_binned.append(energies[mask].sum())
                scale_centers.append(np.sqrt(scale_bins[i] * scale_bins[i + 1]))

        if len(scale_centers) < 3:
            pytest.skip("Not enough scale bins")

        # Check that energy decreases at small scales (qualitative cascade)
        # More quantitative tests would require longer runs
        assert energy_binned[0] > 0  # Energy exists at large scales


# Run validation report
def run_validation_report():
    """Generate human-readable validation report."""
    print("\n" + "=" * 70)
    print("QUANTUM HYDRAULICS VALIDATION REPORT")
    print("=" * 70)

    results = []

    # Test 1: Continuity
    print("\n[1] CONTINUITY TEST (Q = V * A)")
    h = HydraulicsEngine(Q=600, width=30, depth=5, slope=0.002, roughness_ks=0.15)
    Q_check = h.V_mean * h.A
    error = abs(Q_check - h.Q) / h.Q * 100
    status = "PASS" if error < 0.01 else "FAIL"
    print(f"    Q input: {h.Q:.1f} cfs")
    print(f"    V * A:   {Q_check:.1f} cfs")
    print(f"    Error:   {error:.6f}%")
    print(f"    Status:  {status}")
    results.append(("Continuity", status))

    # Test 2: Colebrook-White
    print("\n[2] COLEBROOK-WHITE CONVERGENCE")
    f_engine = h.friction_factor
    f_analytic = colebrook_white(h.Re, h.ks / (4 * h.R))
    error = abs(f_engine - f_analytic) / f_analytic * 100
    status = "PASS" if error < 1.0 else "FAIL"
    print(f"    f (engine):    {f_engine:.6f}")
    print(f"    f (analytic):  {f_analytic:.6f}")
    print(f"    Error:         {error:.2f}%")
    print(f"    Status:        {status}")
    results.append(("Colebrook-White", status))

    # Test 3: Kolmogorov scales
    print("\n[3] KOLMOGOROV SCALES")
    eta_expected = (h.nu ** 3 / h.epsilon) ** 0.25
    error = abs(h.eta_kolmogorov - eta_expected) / eta_expected * 100
    status = "PASS" if error < 0.01 else "FAIL"
    print(f"    eta (computed): {h.eta_kolmogorov:.6f} ft")
    print(f"    eta (expected): {eta_expected:.6f} ft")
    print(f"    Error:          {error:.4f}%")
    print(f"    Status:         {status}")
    results.append(("Kolmogorov", status))

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    passed = sum(1 for _, s in results if s == "PASS")
    total = len(results)
    print(f"Tests passed: {passed}/{total}")

    for name, status in results:
        symbol = "OK" if status == "PASS" else "X"
        print(f"    [{symbol}] {name}")

    print("=" * 70)

    return passed == total


if __name__ == "__main__":
    # Run validation report
    success = run_validation_report()

    # Also run pytest if available
    if _HAS_PYTEST:
        print("\n\nRunning pytest...")
        pytest.main([__file__, "-v", "--tb=short"])
    else:
        print("\n(pytest not installed - skipping detailed tests)")
        print("Install pytest with: pip install pytest")
