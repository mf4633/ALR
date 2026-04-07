"""
Tests for quantum_hydraulics.integration.swmm_node module.

Covers: QuantumNode, NodeMetrics, SedimentProperties, scour risk,
sediment transport, assessments, and energy dissipation recommendations.
"""

import pytest
import numpy as np

from quantum_hydraulics.integration.swmm_node import (
    QuantumNode,
    NodeMetrics,
    SedimentProperties,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def node():
    """Standard QuantumNode for testing."""
    return QuantumNode(
        node_id="TEST_NODE",
        width=20.0,
        length=30.0,
        roughness_ks=0.1,
        observation_radius=15.0,
    )


@pytest.fixture
def active_node(node):
    """Node with state updated from SWMM values."""
    node.update_from_swmm(depth=5.0, inflow=400.0)
    return node


@pytest.fixture
def evolved_node(active_node):
    """Node that has been evolved with particles."""
    np.random.seed(42)
    active_node.update_and_evolve(depth=5.0, inflow=400.0, dt=0.1)
    return active_node


# =============================================================================
# TestSedimentProperties
# =============================================================================

class TestSedimentProperties:
    """Tests for SedimentProperties factory methods and dataclass."""

    def test_sand_defaults(self):
        s = SedimentProperties.sand()
        assert s.name == "sand"
        assert s.critical_shear_psf == 0.10
        assert s.d50_mm == 0.5
        assert s.density_slugs_ft3 == 5.14

    def test_fine_sand(self):
        s = SedimentProperties.fine_sand()
        assert s.name == "fine_sand"
        assert s.critical_shear_psf < SedimentProperties.sand().critical_shear_psf

    def test_coarse_sand(self):
        s = SedimentProperties.coarse_sand()
        assert s.critical_shear_psf > SedimentProperties.sand().critical_shear_psf

    def test_gravel(self):
        s = SedimentProperties.gravel()
        assert s.d50_mm > SedimentProperties.coarse_sand().d50_mm

    def test_silt(self):
        s = SedimentProperties.silt()
        assert s.d50_mm < SedimentProperties.fine_sand().d50_mm

    def test_clay(self):
        s = SedimentProperties.clay()
        assert s.d50_mm < SedimentProperties.silt().d50_mm
        assert s.critical_shear_psf > SedimentProperties.silt().critical_shear_psf  # clay is cohesive

    def test_custom_sediment(self):
        s = SedimentProperties(
            name="custom_rock", critical_shear_psf=2.0,
            d50_mm=50.0, density_slugs_ft3=5.5,
        )
        assert s.name == "custom_rock"
        assert s.critical_shear_psf == 2.0


# =============================================================================
# TestNodeMetrics
# =============================================================================

class TestNodeMetrics:
    """Tests for NodeMetrics dataclass."""

    def test_default_values(self):
        m = NodeMetrics()
        assert m.max_velocity == 0.0
        assert m.mean_velocity == 0.0
        assert m.bed_shear_stress == 0.0
        assert m.scour_risk_index == 0.0
        assert m.n_particles == 0

    def test_to_dict(self):
        m = NodeMetrics(max_velocity=5.0, mean_velocity=3.0, n_particles=100)
        d = m.to_dict()
        assert d["max_velocity"] == 5.0
        assert d["mean_velocity"] == 3.0
        assert d["n_particles"] == 100

    def test_to_dict_all_keys(self):
        d = NodeMetrics().to_dict()
        expected_keys = {
            "max_velocity", "mean_velocity", "bed_shear_stress",
            "scour_risk_index", "tke", "n_particles", "froude_number",
            "reynolds_number", "sediment_transport_rate",
            "scour_depth_potential", "shields_parameter", "excess_shear_ratio",
        }
        assert set(d.keys()) == expected_keys


# =============================================================================
# TestQuantumNodeCreation
# =============================================================================

class TestQuantumNodeCreation:
    """Tests for QuantumNode initialization."""

    def test_basic_creation(self, node):
        assert node.node_id == "TEST_NODE"
        assert node.width == 20.0
        assert node.length == 30.0
        assert node.ks == 0.1

    def test_default_sediment_is_sand(self, node):
        assert node.sediment.name == "sand"

    def test_custom_sediment(self):
        gravel = SedimentProperties.gravel()
        n = QuantumNode(node_id="GRAVEL", width=15.0, sediment=gravel)
        assert n.sediment.name == "gravel"

    def test_initial_particle_count(self, node):
        assert node.n_particles == 0

    def test_initial_metrics_zeroed(self, node):
        assert node.metrics.max_velocity == 0.0
        assert node.metrics.scour_risk_index == 0.0

    def test_empty_arrays(self, node):
        assert node._positions.shape == (0, 3)
        assert node._vorticities.shape == (0, 3)
        assert len(node._sigmas) == 0
        assert len(node._ages) == 0

    def test_default_parameters(self):
        n = QuantumNode(node_id="DEFAULTS", width=10.0)
        assert n.length == 30.0
        assert n.ks == 0.1
        assert n.obs_radius == 15.0
        assert n.side_slope == 2.0


# =============================================================================
# TestUpdateFromSWMM
# =============================================================================

class TestUpdateFromSWMM:
    """Tests for update_from_swmm()."""

    def test_depth_stored(self, node):
        node.update_from_swmm(depth=5.0, inflow=400.0)
        assert node._depth == 5.0

    def test_inflow_stored(self, node):
        node.update_from_swmm(depth=5.0, inflow=400.0)
        assert node._inflow == 400.0

    def test_mean_velocity_computed(self, node):
        node.update_from_swmm(depth=5.0, inflow=400.0)
        # Area = 20*5 + 2*25 = 150, V = 400/150
        expected_v = 400.0 / (20 * 5 + 2 * 5**2)
        assert abs(node._v_mean - expected_v) < 0.01

    def test_friction_velocity_computed(self, node):
        node.update_from_swmm(depth=5.0, inflow=400.0)
        assert node._u_star > 0

    def test_minimum_depth_enforced(self, node):
        """Depth should be clamped to at least 0.1."""
        node.update_from_swmm(depth=0.0, inflow=10.0)
        assert node._depth == 0.1

    def test_negative_inflow_zeroed(self, node):
        node.update_from_swmm(depth=3.0, inflow=-50.0)
        assert node._inflow == 0.0

    def test_velocity_lut_built(self, node):
        node.update_from_swmm(depth=5.0, inflow=400.0)
        assert node._velocity_lut_z is not None
        assert node._velocity_lut_v is not None


# =============================================================================
# TestLogLawVelocity
# =============================================================================

class TestLogLawVelocity:
    """Tests for log-law velocity methods."""

    def test_raw_velocity_positive(self, active_node):
        v = active_node._log_law_velocity_raw(1.0)
        assert v > 0

    def test_raw_velocity_increases_with_height(self, active_node):
        v1 = active_node._log_law_velocity_raw(0.5)
        v2 = active_node._log_law_velocity_raw(2.0)
        assert v2 > v1

    def test_lut_velocity_matches_raw(self, active_node):
        """Lookup table interpolation should approximate raw computation."""
        z = 1.5
        v_raw = active_node._log_law_velocity_raw(z)
        v_lut = active_node._log_law_velocity(z)
        assert abs(v_raw - v_lut) < 0.1  # LUT interpolation tolerance

    def test_vectorized_shape(self, active_node):
        z = np.array([0.5, 1.0, 2.0, 3.0])
        v = active_node._log_law_velocity_vectorized(z)
        assert v.shape == (4,)

    def test_vectorized_monotonic(self, active_node):
        z = np.linspace(0.1, 4.0, 20)
        v = active_node._log_law_velocity_vectorized(z)
        # Should be monotonically increasing
        assert np.all(np.diff(v) >= -1e-10)


# =============================================================================
# TestUpdateAndEvolve
# =============================================================================

class TestUpdateAndEvolve:
    """Tests for update_and_evolve()."""

    def test_particles_injected(self, node):
        np.random.seed(42)
        node.update_and_evolve(depth=5.0, inflow=400.0, dt=0.1)
        assert node.n_particles > 0

    def test_particle_count_grows(self, node):
        np.random.seed(42)
        node.update_and_evolve(depth=5.0, inflow=400.0, dt=0.1)
        n1 = node.n_particles
        node.update_and_evolve(depth=5.0, inflow=400.0, dt=0.1)
        n2 = node.n_particles
        assert n2 >= n1

    def test_positions_shape(self, evolved_node):
        assert evolved_node._positions.shape[1] == 3

    def test_vorticities_shape(self, evolved_node):
        assert evolved_node._vorticities.shape[1] == 3

    def test_sigmas_length_matches(self, evolved_node):
        assert len(evolved_node._sigmas) == evolved_node.n_particles

    def test_ages_length_matches(self, evolved_node):
        assert len(evolved_node._ages) == evolved_node.n_particles

    def test_particles_within_domain(self, evolved_node):
        """Particles should be within domain bounds."""
        assert np.all(evolved_node._positions[:, 0] < evolved_node.length)

    def test_multiple_steps_stable(self, node):
        """Multiple evolution steps should not crash or produce NaN."""
        np.random.seed(42)
        for _ in range(10):
            node.update_and_evolve(depth=5.0, inflow=400.0, dt=0.1)
        assert not np.any(np.isnan(node._positions))
        assert not np.any(np.isnan(node._vorticities))


# =============================================================================
# TestInjectParticles
# =============================================================================

class TestInjectParticles:
    """Tests for particle injection."""

    def test_inject_at_inlet(self, active_node):
        np.random.seed(42)
        active_node._inject_particles(50)
        assert active_node.n_particles == 50
        # All injected at x=0
        assert np.all(active_node._positions[:, 0] == 0.0)

    def test_inject_domain_particles(self, active_node):
        np.random.seed(42)
        active_node._inject_domain_particles(100)
        assert active_node.n_particles == 100
        # Distributed throughout domain
        assert np.max(active_node._positions[:, 0]) > 0.0

    def test_inject_with_zero_depth(self, node):
        """Should not inject particles when depth is zero."""
        node._depth = 0.0
        node._inject_particles(50)
        assert node.n_particles == 0

    def test_adaptive_sigma_bed_vs_surface(self, active_node):
        """Particles near bed should have smaller sigma."""
        np.random.seed(42)
        active_node._inject_particles(100)
        # sigma = 0.05 + (z/depth) * 0.4, so bed (z~0) -> small sigma
        bed_mask = active_node._positions[:, 2] < 1.0
        surface_mask = active_node._positions[:, 2] > 3.0
        if np.any(bed_mask) and np.any(surface_mask):
            avg_bed_sigma = np.mean(active_node._sigmas[bed_mask])
            avg_surface_sigma = np.mean(active_node._sigmas[surface_mask])
            assert avg_bed_sigma < avg_surface_sigma


# =============================================================================
# TestComputeTurbulence
# =============================================================================

class TestComputeTurbulence:
    """Tests for compute_turbulence()."""

    def test_populates_metrics(self, active_node):
        np.random.seed(42)
        active_node.compute_turbulence(n_particles=50)
        assert active_node.metrics.n_particles >= 50

    def test_metrics_positive(self, active_node):
        np.random.seed(42)
        active_node.compute_turbulence(n_particles=50)
        assert active_node.metrics.mean_velocity >= 0
        assert active_node.metrics.tke >= 0

    def test_zero_inflow_gives_zero_metrics(self, node):
        node.update_from_swmm(depth=5.0, inflow=0.0)
        node.compute_turbulence(n_particles=50)
        assert node.metrics.max_velocity == 0.0

    def test_froude_computed(self, active_node):
        np.random.seed(42)
        active_node.compute_turbulence(n_particles=50)
        assert active_node.metrics.froude_number >= 0

    def test_reynolds_computed(self, active_node):
        np.random.seed(42)
        active_node.compute_turbulence(n_particles=50)
        assert active_node.metrics.reynolds_number >= 0


# =============================================================================
# TestScourRisk
# =============================================================================

class TestScourRisk:
    """Tests for _compute_scour_risk()."""

    def test_zero_shear_zero_risk(self, node):
        risk, shields, excess = node._compute_scour_risk(0.0)
        assert risk < 0.1  # logistic at x=-1 is small

    def test_critical_shear_half_risk(self, node):
        """At tau = tau_c, risk should be ~0.5 (logistic midpoint)."""
        tau_c = node.sediment.critical_shear_psf
        risk, shields, excess = node._compute_scour_risk(tau_c)
        assert abs(risk - 0.5) < 0.05

    def test_high_shear_high_risk(self, node):
        tau_c = node.sediment.critical_shear_psf
        risk, shields, excess = node._compute_scour_risk(tau_c * 3.0)
        assert risk > 0.9

    def test_excess_ratio(self, node):
        tau_c = node.sediment.critical_shear_psf
        risk, shields, excess = node._compute_scour_risk(tau_c * 2.0)
        assert abs(excess - 2.0) < 0.01

    def test_shields_parameter_positive(self, node):
        risk, shields, excess = node._compute_scour_risk(0.5)
        assert shields > 0

    def test_risk_bounded_zero_one(self, node):
        """Risk should always be in [0, 1]."""
        for tau in [0.0, 0.01, 0.1, 0.5, 1.0, 5.0, 100.0]:
            risk, _, _ = node._compute_scour_risk(tau)
            assert 0.0 <= risk <= 1.0

    def test_zero_critical_shear(self):
        """Should handle zero critical shear without error."""
        sed = SedimentProperties("none", 0.0, 0.5, 5.14)
        n = QuantumNode(node_id="ZERO", width=10.0, sediment=sed)
        risk, shields, excess = n._compute_scour_risk(0.5)
        assert risk == 0.0


# =============================================================================
# TestSedimentTransport
# =============================================================================

class TestSedimentTransport:
    """Tests for _compute_sediment_transport()."""

    def test_no_transport_below_critical(self, node):
        """No transport when shear < critical."""
        tau_c = node.sediment.critical_shear_psf
        rate, depth = node._compute_sediment_transport(tau_c * 0.5)
        assert rate == 0.0
        assert depth == 0.0

    def test_transport_above_critical(self, node):
        tau_c = node.sediment.critical_shear_psf
        rate, depth = node._compute_sediment_transport(tau_c * 2.0)
        assert rate > 0.0
        assert depth > 0.0

    def test_transport_increases_with_shear(self, node):
        tau_c = node.sediment.critical_shear_psf
        rate1, _ = node._compute_sediment_transport(tau_c * 1.5)
        rate2, _ = node._compute_sediment_transport(tau_c * 3.0)
        assert rate2 > rate1

    def test_scour_depth_capped(self, node):
        """Scour depth should not exceed 10 ft/year."""
        _, depth = node._compute_sediment_transport(100.0)
        assert depth <= 10.0

    def test_exactly_at_critical(self, node):
        """At exactly critical shear, no transport."""
        tau_c = node.sediment.critical_shear_psf
        rate, depth = node._compute_sediment_transport(tau_c)
        assert rate == 0.0


# =============================================================================
# TestAssessments
# =============================================================================

class TestAssessments:
    """Tests for text assessment methods."""

    def test_scour_assessment_critical(self, node):
        node.metrics = NodeMetrics(scour_risk_index=0.9, excess_shear_ratio=3.0)
        assert "CRITICAL" in node.get_scour_assessment()

    def test_scour_assessment_high(self, node):
        node.metrics = NodeMetrics(scour_risk_index=0.7, excess_shear_ratio=1.8)
        assert "HIGH" in node.get_scour_assessment()

    def test_scour_assessment_moderate(self, node):
        node.metrics = NodeMetrics(scour_risk_index=0.5, excess_shear_ratio=1.2)
        assert "MODERATE" in node.get_scour_assessment()

    def test_scour_assessment_low(self, node):
        node.metrics = NodeMetrics(scour_risk_index=0.1, excess_shear_ratio=0.3)
        assert "LOW" in node.get_scour_assessment()

    def test_velocity_assessment_extreme(self, node):
        node.metrics = NodeMetrics(max_velocity=20.0)
        assert "EXTREME" in node.get_velocity_assessment()

    def test_velocity_assessment_high(self, node):
        node.metrics = NodeMetrics(max_velocity=12.0)
        assert "HIGH" in node.get_velocity_assessment()

    def test_velocity_assessment_elevated(self, node):
        node.metrics = NodeMetrics(max_velocity=8.0)
        assert "ELEVATED" in node.get_velocity_assessment()

    def test_velocity_assessment_acceptable(self, node):
        node.metrics = NodeMetrics(max_velocity=4.0)
        assert "ACCEPTABLE" in node.get_velocity_assessment()

    def test_sediment_assessment_severe(self, node):
        node.metrics = NodeMetrics(scour_depth_potential=3.0, sediment_transport_rate=0.01)
        assert "SEVERE" in node.get_sediment_transport_assessment()

    def test_sediment_assessment_minimal(self, node):
        node.metrics = NodeMetrics(scour_depth_potential=0.0, sediment_transport_rate=0.0)
        assert "MINIMAL" in node.get_sediment_transport_assessment()


# =============================================================================
# TestEnergyDissipation
# =============================================================================

class TestEnergyDissipation:
    """Tests for get_energy_dissipation_recommendation()."""

    def test_no_dissipation_needed(self, node):
        node.metrics = NodeMetrics(max_velocity=4.0)
        node._depth = 5.0
        rec = node.get_energy_dissipation_recommendation()
        assert rec["recommended_dissipator"] == "None required"
        assert rec["riprap_d50_inches"] == 0.0

    def test_class_ii_riprap(self, node):
        node.metrics = NodeMetrics(max_velocity=8.0)
        node._depth = 5.0
        rec = node.get_energy_dissipation_recommendation()
        assert "Class II" in rec["recommended_dissipator"]
        assert rec["riprap_d50_inches"] > 0

    def test_class_iii_or_basin(self, node):
        node.metrics = NodeMetrics(max_velocity=12.0)
        node._depth = 5.0
        rec = node.get_energy_dissipation_recommendation()
        assert "Class III" in rec["recommended_dissipator"] or "stilling basin" in rec["recommended_dissipator"]

    def test_stilling_basin(self, node):
        node.metrics = NodeMetrics(max_velocity=20.0)
        node._depth = 5.0
        rec = node.get_energy_dissipation_recommendation()
        assert "Stilling basin" in rec["recommended_dissipator"]

    def test_energy_head_positive(self, node):
        node.metrics = NodeMetrics(max_velocity=10.0)
        node._depth = 5.0
        rec = node.get_energy_dissipation_recommendation()
        assert rec["energy_to_dissipate_ft"] > 0

    def test_energy_head_formula(self, node):
        node.metrics = NodeMetrics(max_velocity=10.0)
        node._depth = 5.0
        rec = node.get_energy_dissipation_recommendation()
        expected = 10.0**2 / (2 * 32.2)
        assert abs(rec["energy_to_dissipate_ft"] - expected) < 0.01

    def test_apron_length_scales_with_depth(self, node):
        node.metrics = NodeMetrics(max_velocity=12.0)
        node._depth = 5.0
        rec = node.get_energy_dissipation_recommendation()
        assert rec["apron_length_ft"] > 0

    def test_recommendation_keys(self, node):
        node.metrics = NodeMetrics(max_velocity=10.0)
        node._depth = 5.0
        rec = node.get_energy_dissipation_recommendation()
        expected_keys = {
            "recommended_dissipator", "riprap_d50_inches",
            "apron_length_ft", "energy_to_dissipate_ft", "max_velocity_fps",
        }
        assert set(rec.keys()) == expected_keys


# =============================================================================
# TestClearAndRepr
# =============================================================================

class TestClearAndRepr:
    """Tests for clear() and __repr__()."""

    def test_clear_removes_particles(self, evolved_node):
        assert evolved_node.n_particles > 0
        evolved_node.clear()
        assert evolved_node.n_particles == 0

    def test_clear_resets_metrics(self, evolved_node):
        evolved_node.clear()
        assert evolved_node.metrics.max_velocity == 0.0

    def test_clear_resets_arrays(self, evolved_node):
        evolved_node.clear()
        assert evolved_node._positions.shape == (0, 3)
        assert evolved_node._vorticities.shape == (0, 3)
        assert len(evolved_node._sigmas) == 0

    def test_repr(self, node):
        r = repr(node)
        assert "TEST_NODE" in r
        assert "QuantumNode" in r

    def test_repr_with_particles(self, evolved_node):
        r = repr(evolved_node)
        assert "particles=" in r


# =============================================================================
# TestParticlesProperty
# =============================================================================

class TestParticlesProperty:
    """Tests for the backward-compatibility particles property."""

    def test_returns_list(self, evolved_node):
        particles = evolved_node.particles
        assert isinstance(particles, list)

    def test_particle_count_matches(self, evolved_node):
        assert len(evolved_node.particles) == evolved_node.n_particles

    def test_particles_are_vortex_particles(self, evolved_node):
        from quantum_hydraulics.core.particle import VortexParticle
        for p in evolved_node.particles[:5]:
            assert isinstance(p, VortexParticle)

    def test_n_particles_property(self, evolved_node):
        assert evolved_node.n_particles == len(evolved_node._positions)


# =============================================================================
# TestGetMetrics
# =============================================================================

class TestGetMetrics:
    """Tests for get_metrics() and get_engineering_metrics()."""

    def test_get_metrics_returns_dict(self, node):
        d = node.get_metrics()
        assert isinstance(d, dict)

    def test_get_engineering_metrics_alias(self, node):
        """get_engineering_metrics should return same as get_metrics."""
        assert node.get_engineering_metrics() == node.get_metrics()

    def test_metrics_after_turbulence(self, active_node):
        np.random.seed(42)
        active_node.compute_turbulence(n_particles=50)
        d = active_node.get_metrics()
        assert d["n_particles"] >= 50
