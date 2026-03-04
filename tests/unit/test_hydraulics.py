"""
Unit tests for HydraulicsEngine.

Tests conservation laws and first-principles computations.
"""

import pytest
import numpy as np
from quantum_hydraulics.core.hydraulics import HydraulicsEngine


class TestContinuity:
    """Test mass conservation: Q = V * A"""

    def test_continuity_holds_exactly(self):
        """Q = V * A must hold to machine precision."""
        h = HydraulicsEngine(Q=600, width=30, depth=5, slope=0.002, roughness_ks=0.15)
        Q_computed = h.V_mean * h.A
        assert abs(Q_computed - h.Q) < 1e-10

    def test_continuity_various_flows(self):
        """Test continuity across range of flows."""
        for Q in [100, 500, 1000, 2000]:
            h = HydraulicsEngine(Q=Q, width=30, depth=5, slope=0.002, roughness_ks=0.15)
            assert abs(h.V_mean * h.A - Q) < 1e-10

    def test_continuity_various_depths(self):
        """Test continuity across range of depths."""
        for depth in [1, 3, 5, 10]:
            h = HydraulicsEngine(Q=600, width=30, depth=depth, slope=0.002, roughness_ks=0.15)
            assert abs(h.V_mean * h.A - 600) < 1e-10


class TestColebrookWhite:
    """Test Colebrook-White friction factor convergence."""

    def test_convergence(self):
        """Friction factor iteration must converge."""
        h = HydraulicsEngine(Q=600, width=30, depth=5, slope=0.002, roughness_ks=0.15)

        # Verify by re-solving
        from quantum_hydraulics.validation.analytical import colebrook_white
        f_check = colebrook_white(h.Re, h.ks / (4 * h.R))

        assert abs(h.friction_factor - f_check) / f_check < 0.01

    def test_friction_positive(self):
        """Friction factor must be positive."""
        h = HydraulicsEngine(Q=600, width=30, depth=5, slope=0.002, roughness_ks=0.15)
        assert h.friction_factor > 0

    def test_friction_reasonable_range(self):
        """Friction factor should be in reasonable range for turbulent flow."""
        h = HydraulicsEngine(Q=600, width=30, depth=5, slope=0.002, roughness_ks=0.15)
        # Typical range 0.01 - 0.1 for open channels
        assert 0.005 < h.friction_factor < 0.2


class TestKolmogorovScales:
    """Test Kolmogorov microscale computations."""

    def test_eta_formula(self):
        """eta = (nu^3/epsilon)^0.25 must be computed correctly."""
        h = HydraulicsEngine(Q=600, width=30, depth=5, slope=0.002, roughness_ks=0.15)
        eta_expected = (h.nu ** 3 / h.epsilon) ** 0.25
        assert abs(h.eta_kolmogorov - eta_expected) < 1e-12

    def test_eta_smaller_than_depth(self):
        """Kolmogorov scale must be much smaller than depth."""
        h = HydraulicsEngine(Q=600, width=30, depth=5, slope=0.002, roughness_ks=0.15)
        assert h.eta_kolmogorov < h.depth / 100

    def test_tau_formula(self):
        """tau = sqrt(nu/epsilon) must be computed correctly."""
        h = HydraulicsEngine(Q=600, width=30, depth=5, slope=0.002, roughness_ks=0.15)
        tau_expected = np.sqrt(h.nu / h.epsilon)
        assert abs(h.tau_kolmogorov - tau_expected) < 1e-12


class TestVelocityProfile:
    """Test velocity profile computations."""

    def test_zero_at_bed(self):
        """Velocity must be zero at z=0."""
        h = HydraulicsEngine(Q=600, width=30, depth=5, slope=0.002, roughness_ks=0.15)
        assert h.velocity_profile(0) == 0.0

    def test_positive_in_flow(self):
        """Velocity must be positive above bed."""
        h = HydraulicsEngine(Q=600, width=30, depth=5, slope=0.002, roughness_ks=0.15)
        for z in [0.1, 0.5, 1.0, 2.0, h.depth]:
            assert h.velocity_profile(z) > 0

    def test_increases_with_height(self):
        """Velocity should increase with height (in general)."""
        h = HydraulicsEngine(Q=600, width=30, depth=5, slope=0.002, roughness_ks=0.15)
        u1 = h.velocity_profile(0.5)
        u2 = h.velocity_profile(1.0)
        assert u2 > u1


class TestFroudeNumber:
    """Test Froude number calculations."""

    def test_froude_positive(self):
        """Froude number must be positive."""
        h = HydraulicsEngine(Q=600, width=30, depth=5, slope=0.002, roughness_ks=0.15)
        assert h.Fr > 0

    def test_subcritical_detection(self):
        """Normal conditions should be subcritical."""
        h = HydraulicsEngine(Q=600, width=30, depth=5, slope=0.002, roughness_ks=0.15)
        # This should be subcritical
        assert h.Fr < 1.0

    def test_supercritical_steep(self):
        """Steep slopes may produce supercritical flow."""
        # Very steep, shallow - may or may not be supercritical
        h = HydraulicsEngine(Q=500, width=10, depth=1.0, slope=0.02, roughness_ks=0.05)
        # Just verify Fr is computed
        assert h.Fr > 0


class TestReynoldsNumber:
    """Test Reynolds number calculations."""

    def test_reynolds_positive(self):
        """Reynolds number must be positive for positive flow."""
        h = HydraulicsEngine(Q=600, width=30, depth=5, slope=0.002, roughness_ks=0.15)
        assert h.Re > 0

    def test_turbulent_flow(self):
        """Typical channel flow should be turbulent."""
        h = HydraulicsEngine(Q=600, width=30, depth=5, slope=0.002, roughness_ks=0.15)
        # Re > 4000 is fully turbulent
        assert h.Re > 4000


class TestBedShearStress:
    """Test bed shear stress calculations."""

    def test_shear_positive(self):
        """Bed shear stress must be positive."""
        h = HydraulicsEngine(Q=600, width=30, depth=5, slope=0.002, roughness_ks=0.15)
        tau = h.bed_shear_stress()
        assert tau > 0

    def test_shear_formula(self):
        """tau = rho * u_star^2"""
        h = HydraulicsEngine(Q=600, width=30, depth=5, slope=0.002, roughness_ks=0.15)
        tau_expected = h.rho * h.u_star ** 2
        tau_actual = h.bed_shear_stress()
        assert abs(tau_actual - tau_expected) < 1e-10
