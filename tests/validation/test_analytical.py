"""
Validation tests against analytical solutions.

Tests verify that the numerical methods reproduce known analytical results.
"""

import pytest
import numpy as np
from quantum_hydraulics.validation.analytical import (
    lamb_oseen_vortex,
    lamb_oseen_vorticity,
    poiseuille_velocity,
    kolmogorov_spectrum,
    kolmogorov_scales,
    wall_vorticity,
    log_law_velocity,
    colebrook_white,
)


class TestLambOseen:
    """Validation against Lamb-Oseen exact solution."""

    def test_initial_approaches_point_vortex(self):
        """At t->0, should approach point vortex solution."""
        r = np.array([0.1, 0.5, 1.0, 2.0])
        t_small = 1e-8
        gamma = 1.0

        v_theta = lamb_oseen_vortex(r, t_small, gamma)
        v_point = gamma / (2 * np.pi * r)

        # Should be very close for small t
        np.testing.assert_array_almost_equal(v_theta, v_point, decimal=4)

    def test_velocity_decays_with_time(self):
        """Velocity should decrease as vortex diffuses."""
        r = np.array([0.1, 0.5, 1.0])
        t1, t2 = 0.1, 1.0

        v1 = lamb_oseen_vortex(r, t1)
        v2 = lamb_oseen_vortex(r, t2)

        assert np.all(v2 < v1)

    def test_vorticity_gaussian(self):
        """Vorticity should be Gaussian distributed."""
        r = np.linspace(0, 5, 100)
        t = 1.0
        gamma = 1.0

        omega = lamb_oseen_vorticity(r, t, gamma)

        # Peak should be at r=0
        assert omega[0] == omega.max()

        # Should decay to near zero at large r
        assert omega[-1] < 0.01 * omega[0]

    def test_core_size_grows(self):
        """Core size should grow with sqrt(t)."""
        # Core size ~ sqrt(4*nu*t), so half-max radius should scale with sqrt(t)
        t1, t2 = 1.0, 4.0  # t2 = 4*t1, so core should be 2x larger

        r = np.linspace(0, 1, 100)

        omega1 = lamb_oseen_vorticity(r, t1)
        omega2 = lamb_oseen_vorticity(r, t2)

        # Find half-max radii
        r_half1 = r[np.argmin(np.abs(omega1 - omega1.max() / 2))]
        r_half2 = r[np.argmin(np.abs(omega2 - omega2.max() / 2))]

        # Ratio should be sqrt(4) = 2
        assert abs(r_half2 / r_half1 - 2.0) < 0.2


class TestPoiseuille:
    """Validation against Poiseuille exact solution."""

    def test_parabolic_profile(self):
        """Velocity profile should be parabolic."""
        h = 1.0
        dp_dx = -1.0
        mu = 1.0

        y = np.linspace(-h, h, 100)
        u = poiseuille_velocity(y, h, dp_dx, mu)

        # Maximum at centerline
        center_idx = len(y) // 2
        assert u[center_idx] == u.max()

        # Zero at walls
        assert u[0] == 0.0
        assert u[-1] == 0.0

    def test_symmetry(self):
        """Profile should be symmetric about centerline."""
        h = 1.0
        dp_dx = -1.0
        mu = 1.0

        y = np.linspace(-h, h, 101)  # Odd number for exact centerline
        u = poiseuille_velocity(y, h, dp_dx, mu)

        # Check symmetry
        n = len(y) // 2
        np.testing.assert_array_almost_equal(u[:n], u[-1:-(n+1):-1])


class TestKolmogorov:
    """Validation of Kolmogorov spectrum."""

    def test_minus_five_thirds_slope(self):
        """Spectrum should have -5/3 slope in log-log space."""
        k = np.logspace(-1, 2, 100)
        epsilon = 0.1

        E = kolmogorov_spectrum(k, epsilon)

        # Linear fit in log space
        log_k = np.log10(k)
        log_E = np.log10(E)
        slope = np.polyfit(log_k, log_E, 1)[0]

        # Should be -5/3 = -1.667
        assert abs(slope - (-5.0 / 3.0)) < 0.001

    def test_epsilon_scaling(self):
        """E should scale as epsilon^(2/3)."""
        k = np.array([1.0, 10.0])
        eps1, eps2 = 0.1, 1.0

        E1 = kolmogorov_spectrum(k, eps1)
        E2 = kolmogorov_spectrum(k, eps2)

        ratio_expected = (eps2 / eps1) ** (2.0 / 3.0)
        ratio_actual = E2[0] / E1[0]

        assert abs(ratio_actual - ratio_expected) < 1e-10

    def test_kolmogorov_scale_formulas(self):
        """Test Kolmogorov scale formulas."""
        epsilon = 0.1
        nu = 1e-5

        eta, tau, v = kolmogorov_scales(epsilon, nu)

        # Verify formulas
        assert abs(eta - (nu ** 3 / epsilon) ** 0.25) < 1e-12
        assert abs(tau - (nu / epsilon) ** 0.5) < 1e-12
        assert abs(v - (nu * epsilon) ** 0.25) < 1e-12


class TestWallVorticity:
    """Validation of wall vorticity from log law."""

    def test_vorticity_decreases_with_height(self):
        """Vorticity should decrease away from wall."""
        z = np.array([0.01, 0.1, 1.0, 10.0])
        u_star = 0.1

        omega = wall_vorticity(z, u_star)

        # Should be monotonically decreasing
        assert np.all(np.diff(omega) < 0)

    def test_vorticity_scales_with_ustar(self):
        """Vorticity should scale linearly with friction velocity."""
        z = np.array([0.1, 1.0])
        u_star_1, u_star_2 = 0.1, 0.2

        omega1 = wall_vorticity(z, u_star_1)
        omega2 = wall_vorticity(z, u_star_2)

        np.testing.assert_array_almost_equal(omega2 / omega1, u_star_2 / u_star_1)

    def test_vorticity_formula(self):
        """omega = u* / (kappa * z)"""
        z = 1.0
        u_star = 0.1
        kappa = 0.41

        omega = wall_vorticity(np.array([z]), u_star, kappa)

        expected = u_star / (kappa * z)
        assert abs(omega[0] - expected) < 1e-10


class TestLogLaw:
    """Validation of log-law velocity profile."""

    def test_log_law_formula(self):
        """u = (u*/kappa) * ln(z/z0)"""
        z = np.array([0.1, 1.0, 10.0])
        u_star = 0.1
        z0 = 0.01
        kappa = 0.41

        u = log_law_velocity(z, u_star, z0, kappa)

        expected = (u_star / kappa) * np.log(z / z0)
        np.testing.assert_array_almost_equal(u, expected)

    def test_velocity_increases_with_height(self):
        """Velocity should increase with height."""
        z = np.array([0.1, 1.0, 10.0])
        u = log_law_velocity(z, u_star=0.1)

        assert np.all(np.diff(u) > 0)


class TestColebrookWhite:
    """Validation of Colebrook-White solver."""

    def test_laminar_regime(self):
        """For Re < 2300, f = 64/Re."""
        Re = 1000
        f = colebrook_white(Re, 0.001)

        expected = 64.0 / Re
        assert abs(f - expected) < 1e-10

    def test_smooth_turbulent(self):
        """For smooth pipe, should approach Blasius formula."""
        Re = 100000
        f = colebrook_white(Re, 1e-6)  # Very smooth

        # Blasius: f = 0.316 * Re^(-0.25)
        f_blasius = 0.316 * Re ** (-0.25)

        # Should be close (within 20%)
        assert abs(f - f_blasius) / f_blasius < 0.2

    def test_convergence(self):
        """Should converge for typical conditions."""
        # Should not raise or return NaN
        f = colebrook_white(Re=50000, epsilon_D=0.01)

        assert not np.isnan(f)
        assert f > 0
        assert f < 1

    def test_roughness_increases_friction(self):
        """Increasing roughness should increase friction."""
        Re = 100000

        f_smooth = colebrook_white(Re, 0.0001)
        f_rough = colebrook_white(Re, 0.01)

        assert f_rough > f_smooth
