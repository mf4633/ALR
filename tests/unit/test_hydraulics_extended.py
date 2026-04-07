"""
Extended tests for quantum_hydraulics.core.hydraulics module.

Covers methods NOT tested by the existing test_hydraulics.py:
- velocity_profile_vectorized()
- get_summary()
- get_summary_object()
- __repr__()
- HydraulicsSummary dataclass
"""

import pytest
import numpy as np

from quantum_hydraulics.core.hydraulics import HydraulicsEngine, HydraulicsSummary


@pytest.fixture
def engine():
    return HydraulicsEngine(
        Q=600, width=30, depth=5, slope=0.002, roughness_ks=0.15,
    )


# =============================================================================
# TestVelocityProfileVectorized
# =============================================================================

class TestVelocityProfileVectorized:
    """Tests for the vectorized velocity profile method."""

    def test_returns_ndarray(self, engine):
        z = np.array([0.5, 1.0, 2.0, 3.0])
        result = engine.velocity_profile_vectorized(z)
        assert isinstance(result, np.ndarray)

    def test_shape_preserved(self, engine):
        z = np.linspace(0.1, 4.5, 50)
        result = engine.velocity_profile_vectorized(z)
        assert result.shape == z.shape

    def test_matches_scalar_version(self, engine):
        """Vectorized should match the scalar velocity_profile()."""
        z_values = np.array([0.0, 0.01, 0.1, 0.5, 1.0, 2.0, 3.0, 4.0, 4.9])
        vectorized = engine.velocity_profile_vectorized(z_values)
        scalar = np.array([engine.velocity_profile(z) for z in z_values])
        np.testing.assert_array_almost_equal(vectorized, scalar, decimal=6)

    def test_zero_at_bed(self, engine):
        z = np.array([0.0])
        result = engine.velocity_profile_vectorized(z)
        assert result[0] == 0.0

    def test_all_non_negative(self, engine):
        z = np.linspace(-0.5, 6.0, 100)
        result = engine.velocity_profile_vectorized(z)
        assert np.all(result >= 0.0)

    def test_monotonic_increase(self, engine):
        """Velocity should generally increase with height."""
        z = np.linspace(0.01, 4.5, 50)
        result = engine.velocity_profile_vectorized(z)
        # Allow small non-monotonicity at the inner/outer layer transition
        assert result[-1] > result[0]

    def test_empty_array(self, engine):
        z = np.array([])
        result = engine.velocity_profile_vectorized(z)
        assert len(result) == 0

    def test_single_value(self, engine):
        z = np.array([2.0])
        result = engine.velocity_profile_vectorized(z)
        expected = engine.velocity_profile(2.0)
        assert abs(result[0] - expected) < 1e-10


# =============================================================================
# TestGetSummary
# =============================================================================

class TestGetSummary:
    """Tests for get_summary() returning a dict."""

    def test_returns_dict(self, engine):
        s = engine.get_summary()
        assert isinstance(s, dict)

    def test_expected_keys(self, engine):
        s = engine.get_summary()
        expected_keys = {
            "flow_regime", "uniformity", "Q", "V", "A", "R",
            "Re", "Fr", "f", "u_star", "Sf", "TKE", "epsilon",
            "eta", "Re_t",
        }
        assert set(s.keys()) == expected_keys

    def test_values_match_attributes(self, engine):
        s = engine.get_summary()
        assert s["Q"] == engine.Q
        assert s["V"] == engine.V_mean
        assert s["A"] == engine.A
        assert s["R"] == engine.R
        assert s["Re"] == engine.Re
        assert s["Fr"] == engine.Fr
        assert s["f"] == engine.friction_factor
        assert s["u_star"] == engine.u_star
        assert s["Sf"] == engine.Sf
        assert s["TKE"] == engine.TKE
        assert s["epsilon"] == engine.epsilon
        assert s["eta"] == engine.eta_kolmogorov
        assert s["Re_t"] == engine.Re_turbulent

    def test_subcritical_regime(self, engine):
        s = engine.get_summary()
        assert s["flow_regime"] == "SUBCRITICAL"

    def test_supercritical_regime(self):
        steep = HydraulicsEngine(Q=500, width=5, depth=1, slope=0.1, roughness_ks=0.01)
        s = steep.get_summary()
        assert s["flow_regime"] == "SUPERCRITICAL"


# =============================================================================
# TestGetSummaryObject
# =============================================================================

class TestGetSummaryObject:
    """Tests for get_summary_object() returning HydraulicsSummary."""

    def test_returns_summary_object(self, engine):
        s = engine.get_summary_object()
        assert isinstance(s, HydraulicsSummary)

    def test_fields_match_dict(self, engine):
        obj = engine.get_summary_object()
        d = engine.get_summary()
        assert obj.Q == d["Q"]
        assert obj.V == d["V"]
        assert obj.A == d["A"]
        assert obj.Re == d["Re"]
        assert obj.Fr == d["Fr"]
        assert obj.f == d["f"]
        assert obj.flow_regime == d["flow_regime"]
        assert obj.uniformity == d["uniformity"]

    def test_flow_regime_string(self, engine):
        s = engine.get_summary_object()
        assert s.flow_regime in ("SUBCRITICAL", "SUPERCRITICAL")

    def test_uniformity_string(self, engine):
        s = engine.get_summary_object()
        assert s.uniformity in ("UNIFORM", "GRADUALLY VARIED")


# =============================================================================
# TestRepr
# =============================================================================

class TestRepr:
    """Tests for HydraulicsEngine __repr__."""

    def test_repr_contains_key_info(self, engine):
        r = repr(engine)
        assert "HydraulicsEngine" in r
        assert "Q=" in r
        assert "V=" in r
        assert "Fr=" in r
        assert "Re=" in r

    def test_repr_is_string(self, engine):
        assert isinstance(repr(engine), str)


# =============================================================================
# TestTrapezoidalGeometry
# =============================================================================

class TestTrapezoidalGeometry:
    """Tests for trapezoidal channel geometry edge cases."""

    def test_rectangular_channel(self):
        e = HydraulicsEngine(Q=600, width=30, depth=5, slope=0.002,
                             roughness_ks=0.15, side_slope=0.0)
        assert abs(e.A - 150.0) < 0.01
        assert abs(e.T - 30.0) < 0.01

    def test_trapezoidal_area(self):
        e = HydraulicsEngine(Q=600, width=20, depth=5, slope=0.002,
                             roughness_ks=0.15, side_slope=3.0)
        expected_A = 20 * 5 + 3 * 25  # width*depth + z*depth^2
        assert abs(e.A - expected_A) < 0.01

    def test_top_width(self):
        e = HydraulicsEngine(Q=600, width=20, depth=5, slope=0.002,
                             roughness_ks=0.15, side_slope=3.0)
        expected_T = 20 + 2 * 3 * 5
        assert abs(e.T - expected_T) < 0.01
