"""
Pytest configuration and fixtures for quantum_hydraulics tests.
"""

import pytest
import numpy as np


@pytest.fixture
def standard_hydraulics():
    """Create standard hydraulics engine for testing."""
    from quantum_hydraulics.core.hydraulics import HydraulicsEngine
    return HydraulicsEngine(
        Q=600,
        width=30,
        depth=5,
        slope=0.002,
        roughness_ks=0.15,
    )


@pytest.fixture
def small_field(standard_hydraulics):
    """Create small vortex field for fast tests."""
    from quantum_hydraulics.core.vortex_field import VortexParticleField
    return VortexParticleField(
        standard_hydraulics,
        length=50,
        n_particles=100,
    )


@pytest.fixture
def medium_field(standard_hydraulics):
    """Create medium vortex field for comprehensive tests."""
    from quantum_hydraulics.core.vortex_field import VortexParticleField
    return VortexParticleField(
        standard_hydraulics,
        length=100,
        n_particles=500,
    )


@pytest.fixture
def random_seed():
    """Set random seed for reproducibility."""
    np.random.seed(42)
    yield
    np.random.seed(None)


def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
