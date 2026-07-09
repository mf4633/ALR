"""
Tests for the Tier-2 vortex-particle bed-shear augmentation (swmm_2d).

The physically load-bearing property is the constant-stress-layer limit: a
mean-shear-only vortex field must not amplify bed shear. The prior
implementation seeded velocity (not vorticity) into the wrong component and
floored the result at 1.2x u*, so it always reported ~1.44x amplification
regardless of the flow. These tests lock in the corrected behaviour:

  * uniform mean-shear -> amplification -> 1.0 (no spurious amplification),
  * amplification is converged (independent of particle count), not a
    1/sqrt(N) placement artifact,
  * amplification is never below 1.0 and never pinned at the old 1.44 floor.
"""

import numpy as np
import pytest

from quantum_hydraulics.integration.swmm_2d import Mesh2DResults
from quantum_hydraulics.integration.swmm_node import SedimentProperties


def _uniform_mesh(v=6.0, depth=5.0, n=200, seed=0):
    """A hydraulically-uniform mesh with a few high-velocity hotspots."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 40, n)
    y = rng.uniform(-10, 10, n)
    d = np.full(n, depth)
    vx = np.full(n, v)
    vy = np.zeros(n)
    mesh = Mesh2DResults(np.arange(n), x, y, d, vx, vy,
                         roughness_ks=0.1, sediment=SedimentProperties.sand())
    mesh.compute_tier1()
    return mesh


def test_uniform_meanshear_gives_unit_amplification():
    """Constant-stress-layer limit: no amplification for mean-shear-only flow."""
    mesh = _uniform_mesh()
    hs = mesh.get_hotspots(top_n=5, metric="v_mag")
    res = mesh.compute_tier2(hs, n_particles=300, cell_size=5.0)
    for r in res:
        assert r.amplification_factor == pytest.approx(1.0, abs=1e-3)


def test_amplification_never_below_one():
    mesh = _uniform_mesh(v=4.0)
    hs = mesh.get_hotspots(top_n=8, metric="v_mag")
    res = mesh.compute_tier2(hs, n_particles=300, cell_size=5.0)
    assert all(r.amplification_factor >= 1.0 - 1e-9 for r in res)


def test_amplification_is_converged_not_sampling_artifact():
    """Amplification must not drift with particle count (no 1/sqrt(N) noise)."""
    mesh = _uniform_mesh()
    hs = mesh.get_hotspots(top_n=1, metric="v_mag")
    amps = [mesh.compute_tier2(hs, n_particles=N, cell_size=5.0)[0].amplification_factor
            for N in (150, 300, 600, 1000)]
    assert max(amps) - min(amps) < 1e-3


def test_not_pinned_at_legacy_floor():
    """The old hard-coded 1.2*u* floor (=> 1.44x shear) must be gone."""
    mesh = _uniform_mesh()
    hs = mesh.get_hotspots(top_n=5, metric="v_mag")
    res = mesh.compute_tier2(hs, n_particles=300, cell_size=5.0)
    assert all(abs(r.amplification_factor - 1.44) > 0.01 for r in res)
