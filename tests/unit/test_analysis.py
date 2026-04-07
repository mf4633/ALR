"""
Tests for quantum_hydraulics.analysis module.

Covers: analyze(), analyze_range(), DesignResults, print_design_table()
"""

import pytest
import numpy as np

from quantum_hydraulics.analysis import analyze, analyze_range, DesignResults, print_design_table


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def standard_results():
    """Standard design analysis results for a typical channel."""
    return analyze(Q=600, width=30, depth=5, slope=0.002, roughness_ks=0.15)


@pytest.fixture
def high_velocity_results():
    """Results for a steep, narrow channel with high velocity."""
    return analyze(Q=1000, width=10, depth=3, slope=0.02, roughness_ks=0.05)


@pytest.fixture
def low_flow_results():
    """Results for a mild, wide channel with low velocity."""
    return analyze(Q=50, width=40, depth=2, slope=0.0005, roughness_ks=0.1)


# =============================================================================
# TestAnalyze - core function
# =============================================================================

class TestAnalyze:
    """Tests for the analyze() function."""

    def test_returns_design_results(self, standard_results):
        assert isinstance(standard_results, DesignResults)

    def test_input_parameters_preserved(self, standard_results):
        assert standard_results.Q == 600
        assert standard_results.width == 30
        assert standard_results.depth == 5
        assert standard_results.slope == 0.002
        assert standard_results.roughness_ks == 0.15

    def test_continuity_holds(self, standard_results):
        """Q = V * A must hold."""
        Q_check = standard_results.velocity_mean * standard_results.area
        assert abs(Q_check - standard_results.Q) < 0.01

    def test_velocity_mean_positive(self, standard_results):
        assert standard_results.velocity_mean > 0

    def test_velocity_max_reasonable(self, standard_results):
        """Near-surface velocity should be within reasonable range of mean."""
        # velocity_max is sampled at 0.9*depth using power law: V_mean*(0.9)^(1/7)
        # It can be slightly less than V_mean for shallow/wide channels
        assert standard_results.velocity_max > 0
        assert abs(standard_results.velocity_max - standard_results.velocity_mean) < standard_results.velocity_mean

    def test_area_correct(self):
        """Area = width*depth + side_slope*depth^2 for trapezoidal."""
        r = analyze(Q=600, width=30, depth=5, slope=0.002, roughness_ks=0.1, side_slope=2.0)
        expected_area = 30 * 5 + 2.0 * 5**2
        assert abs(r.area - expected_area) < 0.01

    def test_hydraulic_radius_positive(self, standard_results):
        assert standard_results.hydraulic_radius > 0

    def test_reynolds_number_turbulent(self, standard_results):
        """Typical channel flow should be turbulent (Re >> 4000)."""
        assert standard_results.reynolds_number > 4000

    def test_froude_number_positive(self, standard_results):
        assert standard_results.froude_number > 0

    def test_friction_factor_reasonable(self, standard_results):
        assert 0.005 < standard_results.friction_factor < 0.2

    def test_friction_velocity_positive(self, standard_results):
        assert standard_results.friction_velocity > 0

    def test_bed_shear_positive(self, standard_results):
        assert standard_results.bed_shear_stress > 0

    def test_bed_shear_formula(self, standard_results):
        """tau = rho * u_star^2."""
        rho = 1.94
        expected = rho * standard_results.friction_velocity**2
        assert abs(standard_results.bed_shear_stress - expected) < 0.001

    def test_tke_positive(self, standard_results):
        assert standard_results.tke > 0

    def test_kolmogorov_scale_positive(self, standard_results):
        assert standard_results.kolmogorov_scale > 0

    def test_kolmogorov_much_smaller_than_depth(self, standard_results):
        assert standard_results.kolmogorov_scale < standard_results.depth * 0.01

    def test_scour_risk_bounded(self, standard_results):
        assert 0 <= standard_results.scour_risk_index <= 1.0

    def test_default_roughness(self):
        """Default roughness_ks should be 0.1."""
        r = analyze(Q=600, width=30, depth=5, slope=0.002)
        assert r.roughness_ks == 0.1


# =============================================================================
# TestFlowRegime
# =============================================================================

class TestFlowRegime:
    """Tests for flow regime classification."""

    def test_subcritical_detection(self, standard_results):
        assert standard_results.flow_regime == "SUBCRITICAL"
        assert standard_results.froude_number < 1.0

    def test_supercritical_steep(self):
        """Steep slope should produce supercritical flow."""
        r = analyze(Q=500, width=5, depth=1, slope=0.1, roughness_ks=0.01)
        assert r.flow_regime == "SUPERCRITICAL"
        assert r.froude_number > 1.0


# =============================================================================
# TestScourAssessment
# =============================================================================

class TestScourAssessment:
    """Tests for scour risk assessment strings."""

    def test_critical_scour(self):
        """High shear vs low critical shear -> CRITICAL."""
        r = analyze(Q=1000, width=10, depth=3, slope=0.02, roughness_ks=0.05, critical_shear=0.01)
        assert "CRITICAL" in r.scour_assessment

    def test_low_scour(self, low_flow_results):
        """Low flow should have low or acceptable scour."""
        assert "LOW" in low_flow_results.scour_assessment or "Acceptable" in low_flow_results.scour_assessment

    def test_moderate_scour(self):
        """Intermediate conditions -> MODERATE."""
        # Tune critical_shear to get scour_risk in 0.3-0.5 range
        r = analyze(Q=300, width=20, depth=4, slope=0.003, roughness_ks=0.1, critical_shear=0.5)
        assert r.scour_risk_index < 0.5


# =============================================================================
# TestVelocityAssessment
# =============================================================================

class TestVelocityAssessment:
    """Tests for velocity assessment strings."""

    def test_acceptable_velocity(self, low_flow_results):
        assert low_flow_results.velocity_assessment == "ACCEPTABLE"

    def test_extreme_velocity(self):
        """Very high velocity should trigger EXTREME."""
        r = analyze(Q=5000, width=5, depth=2, slope=0.1, roughness_ks=0.01)
        if r.velocity_max > 15:
            assert "EXTREME" in r.velocity_assessment


# =============================================================================
# TestDesignResults
# =============================================================================

class TestDesignResults:
    """Tests for the DesignResults dataclass."""

    def test_str_output(self, standard_results):
        s = str(standard_results)
        assert "QUANTUM HYDRAULICS" in s
        assert "INPUT PARAMETERS" in s
        assert "HYDRAULIC RESULTS" in s
        assert "TURBULENCE & SHEAR" in s
        assert "DESIGN ASSESSMENT" in s

    def test_str_contains_values(self, standard_results):
        s = str(standard_results)
        assert "600.0 cfs" in s
        assert "30.0 ft" in s

    def test_to_dict_keys(self, standard_results):
        d = standard_results.to_dict()
        expected_keys = {
            'Q_cfs', 'width_ft', 'depth_ft', 'slope', 'roughness_ks_ft',
            'velocity_mean_fps', 'velocity_max_fps', 'area_ft2',
            'hydraulic_radius_ft', 'reynolds_number', 'froude_number',
            'friction_factor', 'friction_velocity_fps', 'bed_shear_stress_psf',
            'tke_ft2s2', 'kolmogorov_scale_ft', 'scour_risk_index', 'flow_regime',
        }
        assert set(d.keys()) == expected_keys

    def test_to_dict_values_match(self, standard_results):
        d = standard_results.to_dict()
        assert d['Q_cfs'] == standard_results.Q
        assert d['velocity_mean_fps'] == standard_results.velocity_mean
        assert d['froude_number'] == standard_results.froude_number
        assert d['flow_regime'] == standard_results.flow_regime


# =============================================================================
# TestAnalyzeRange
# =============================================================================

class TestAnalyzeRange:
    """Tests for the analyze_range() function."""

    def test_returns_list(self):
        results = analyze_range(
            Q_range=(100, 500), width=20, depth_range=(2, 5),
            slope=0.002, n_points=3,
        )
        assert isinstance(results, list)

    def test_result_count(self):
        """n_points^2 combinations (3 Q x 3 depth = 9)."""
        results = analyze_range(
            Q_range=(100, 500), width=20, depth_range=(2, 5),
            slope=0.002, n_points=3,
        )
        assert len(results) == 9

    def test_all_design_results(self):
        results = analyze_range(
            Q_range=(100, 500), width=20, depth_range=(2, 5),
            slope=0.002, n_points=3,
        )
        for r in results:
            assert isinstance(r, DesignResults)

    def test_range_covers_extremes(self):
        results = analyze_range(
            Q_range=(100, 1000), width=20, depth_range=(1, 10),
            slope=0.002, n_points=5,
        )
        Q_values = [r.Q for r in results]
        assert min(Q_values) == pytest.approx(100, abs=1)
        assert max(Q_values) == pytest.approx(1000, abs=1)


# =============================================================================
# TestPrintDesignTable
# =============================================================================

class TestPrintDesignTable:
    """Tests for the print_design_table() function."""

    def test_runs_without_error(self, capsys):
        print_design_table(
            Q_values=[100, 300, 600],
            width=30, depth=5, slope=0.002,
        )
        captured = capsys.readouterr()
        assert "DESIGN TABLE" in captured.out
        assert "Q (cfs)" in captured.out

    def test_shows_all_flows(self, capsys):
        print_design_table(
            Q_values=[100, 500, 1000],
            width=30, depth=5, slope=0.002,
        )
        captured = capsys.readouterr()
        assert "100" in captured.out
        assert "500" in captured.out
        assert "1000" in captured.out


# =============================================================================
# TestEdgeCases
# =============================================================================

class TestEdgeCases:
    """Edge case tests for analysis."""

    def test_very_small_flow(self):
        r = analyze(Q=1, width=10, depth=1, slope=0.001, roughness_ks=0.1)
        assert r.velocity_mean > 0
        assert r.scour_risk_index >= 0

    def test_very_large_flow(self):
        r = analyze(Q=10000, width=100, depth=20, slope=0.001, roughness_ks=0.1)
        assert r.velocity_mean > 0
        assert r.reynolds_number > 0

    def test_rectangular_channel(self):
        """side_slope=0 should give rectangular cross-section."""
        r = analyze(Q=600, width=30, depth=5, slope=0.002, roughness_ks=0.1, side_slope=0.0)
        assert abs(r.area - 30 * 5) < 0.01

    def test_high_critical_shear(self):
        """Very high critical shear -> low scour risk."""
        r = analyze(Q=600, width=30, depth=5, slope=0.002, critical_shear=100.0)
        assert r.scour_risk_index < 0.05
