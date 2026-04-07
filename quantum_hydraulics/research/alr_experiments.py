"""
ALR (Adaptive Lagrangian Refinement) Experiments for ICWMM 2026.

Five self-contained experiments demonstrating observation-dependent resolution:
  1. Convergence study — metrics converge as observation radius grows
  2. Cost-benefit analysis — ALR accuracy at reduced particle count
  3. Sigma field visualization — adaptive resolution concentration
  4. Engineering relevance — scour at a bridge pier (Tier 1 vs Tier 2)
  5. Multi-zone independence — two observation zones, independent metrics
"""

import time
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

from quantum_hydraulics.core.hydraulics import HydraulicsEngine
from quantum_hydraulics.core.vortex_field import VortexParticleField, FieldState
from quantum_hydraulics.integration.swmm_2d import (
    SWMM2DPostProcessor,
    Mesh2DResults,
    RHO, NU, G,
)
from quantum_hydraulics.integration.swmm_node import SedimentProperties


# ── Shared scenario parameters ─────────────────────────────────────────────

CHANNEL_LENGTH = 200.0   # ft
CHANNEL_WIDTH = 40.0     # ft
DEPTH = 4.0              # ft
SLOPE = 0.002
ROUGHNESS_KS = 0.1       # ft
Q = CHANNEL_WIDTH * DEPTH * 4.0  # ~640 cfs at 4 fps approach

# Pier location
PIER_X = 100.0
PIER_Y = 20.0            # centerline
PIER_DIAMETER = 3.0       # ft

# Observation zone centered on pier wake
OBS_CENTER = np.array([PIER_X + 15.0, PIER_Y, DEPTH / 2.0])
N_STEPS = 30
DT = 0.05


# ── Result containers ──────────────────────────────────────────────────────

@dataclass
class ConvergenceResult:
    obs_radii: List[float]
    mean_sigma: List[float]
    mean_vorticity: List[float]
    mean_enstrophy: List[float]
    n_particles: List[int]

@dataclass
class CostBenefitResult:
    particle_counts: List[int]
    errors_sigma: List[float]       # relative to baseline
    errors_vorticity: List[float]
    errors_enstrophy: List[float]
    wall_times: List[float]
    baseline_sigma: float
    baseline_vorticity: float
    baseline_enstrophy: float

@dataclass
class SigmaFieldResult:
    x_grid: np.ndarray
    y_grid: np.ndarray
    sigma_pier: np.ndarray       # observation at pier wake
    sigma_entrance: np.ndarray   # observation at channel entrance
    sigma_off: np.ndarray        # observation off (uniform)
    enhancement_at_center: float

@dataclass
class ScourResult:
    tier1_shear_pier: float
    tier1_shear_approach: float
    tier2_shear_pier: float
    tier2_scour_risk: float
    tier2_shields: float
    amplification: float
    n_hotspots: int

@dataclass
class MultiZoneResult:
    zone_a_sigma: float
    zone_b_sigma: float
    midpoint_sigma: float
    zone_a_vorticity: float
    zone_b_vorticity_base: float    # Zone B at original position
    zone_b_vorticity_moved: float   # Zone B at shifted position


# ── Helpers ────────────────────────────────────────────────────────────────

def _create_engine():
    """Create the standard channel HydraulicsEngine."""
    return HydraulicsEngine(
        Q=Q,
        width=CHANNEL_WIDTH,
        depth=DEPTH,
        slope=SLOPE,
        roughness_ks=ROUGHNESS_KS,
    )


def _run_field(field: VortexParticleField, n_steps: int = N_STEPS, dt: float = DT):
    """Advance field n_steps."""
    for _ in range(n_steps):
        field.step(dt)


def _measure_box(field: VortexParticleField,
                 x_min: float, x_max: float,
                 y_min: float, y_max: float) -> dict:
    """Extract metrics within a spatial measurement box."""
    pos = field._positions
    sig = field._sigmas
    vor = field._vorticities
    # Enstrophy (|omega|^2) — sigma-independent, physically meaningful
    enstrophy = np.sum(vor ** 2, axis=1)

    mask = (
        (pos[:, 0] >= x_min) & (pos[:, 0] <= x_max) &
        (pos[:, 1] >= y_min) & (pos[:, 1] <= y_max)
    )
    n_in = int(mask.sum())
    if n_in == 0:
        return {"mean_sigma": 0.0, "mean_vorticity": 0.0, "mean_enstrophy": 0.0, "n": 0}

    return {
        "mean_sigma": float(sig[mask].mean()),
        "mean_vorticity": float(np.sqrt(enstrophy[mask]).mean()),
        "mean_enstrophy": float(enstrophy[mask].mean()),
        "n": n_in,
    }


# Measurement box around pier wake
BOX_X = (PIER_X - 10.0, PIER_X + 40.0)
BOX_Y = (PIER_Y - 10.0, PIER_Y + 10.0)


# ── Experiment 1: Convergence Study ────────────────────────────────────────

def run_convergence(radii=None, n_particles=2000, verbose=False) -> ConvergenceResult:
    """
    Show that ALR metrics at the observation zone converge as obs_radius
    increases (approaching uniform high-res over the measurement box).
    """
    if radii is None:
        radii = [5.0, 10.0, 15.0, 25.0, 50.0, 100.0]

    engine = _create_engine()
    result = ConvergenceResult(
        obs_radii=radii,
        mean_sigma=[], mean_vorticity=[], mean_enstrophy=[], n_particles=[],
    )

    for r in radii:
        np.random.seed(42)
        vf = VortexParticleField(engine, length=CHANNEL_LENGTH, n_particles=n_particles)
        vf.set_observation(OBS_CENTER, r)
        _run_field(vf)

        m = _measure_box(vf, *BOX_X, *BOX_Y)
        result.mean_sigma.append(m["mean_sigma"])
        result.mean_vorticity.append(m["mean_vorticity"])
        result.mean_enstrophy.append(m["mean_enstrophy"])
        result.n_particles.append(m["n"])

        if verbose:
            print(f"    obs_radius={r:5.0f}  sigma={m['mean_sigma']:.4f}  "
                  f"vort={m['mean_vorticity']:.4f}  enstrophy={m['mean_enstrophy']:.4f}  "
                  f"n_in_box={m['n']}")

    return result


# ── Experiment 2: Cost-Benefit Analysis ────────────────────────────────────

def run_cost_benefit(particle_counts=None, baseline_particles=6000,
                     verbose=False) -> CostBenefitResult:
    """
    Compare ALR at various particle counts against a uniform high-res baseline.

    Baseline: observation_active=False, all sigmas forced to min_sigma.
    ALR runs: observation at pier wake, varying particle count.
    """
    if particle_counts is None:
        particle_counts = [200, 500, 1000, 2000, 4000]

    engine = _create_engine()

    # ── Baseline: uniform high-res ────────────────────────────────────
    if verbose:
        print("    Running baseline (uniform high-res)...")
    np.random.seed(42)
    vf_base = VortexParticleField(engine, length=CHANNEL_LENGTH,
                                   n_particles=baseline_particles)
    vf_base.toggle_observation(False)
    # Force all sigmas to minimum (highest resolution everywhere)
    vf_base._sigmas[:] = vf_base.min_sigma
    _run_field(vf_base)
    base_m = _measure_box(vf_base, *BOX_X, *BOX_Y)

    result = CostBenefitResult(
        particle_counts=particle_counts,
        errors_sigma=[], errors_vorticity=[], errors_enstrophy=[],
        wall_times=[],
        baseline_sigma=base_m["mean_sigma"],
        baseline_vorticity=base_m["mean_vorticity"],
        baseline_enstrophy=base_m["mean_enstrophy"],
    )

    if verbose:
        print(f"    Baseline: sigma={base_m['mean_sigma']:.4f}  "
              f"vort={base_m['mean_vorticity']:.4f}  "
              f"enstrophy={base_m['mean_enstrophy']:.4f}")

    # ── ALR runs ──────────────────────────────────────────────────────
    for np_count in particle_counts:
        np.random.seed(42)
        t0 = time.perf_counter()

        vf = VortexParticleField(engine, length=CHANNEL_LENGTH, n_particles=np_count)
        vf.set_observation(OBS_CENTER, 25.0)
        _run_field(vf)

        wall = time.perf_counter() - t0
        m = _measure_box(vf, *BOX_X, *BOX_Y)

        # Relative errors (avoid div/0)
        def _rel_err(val, ref):
            return abs(val - ref) / max(abs(ref), 1e-12)

        result.errors_sigma.append(_rel_err(m["mean_sigma"], base_m["mean_sigma"]))
        result.errors_vorticity.append(_rel_err(m["mean_vorticity"], base_m["mean_vorticity"]))
        result.errors_enstrophy.append(_rel_err(m["mean_enstrophy"], base_m["mean_enstrophy"]))
        result.wall_times.append(wall)

        if verbose:
            print(f"    N={np_count:5d}  err_sigma={result.errors_sigma[-1]:.3f}  "
                  f"err_vort={result.errors_vorticity[-1]:.3f}  "
                  f"time={wall:.2f}s")

    return result


# ── Experiment 3: Sigma Field Visualization ────────────────────────────────

def run_sigma_field(n_particles=2000, grid_res=60, verbose=False) -> SigmaFieldResult:
    """
    Compute sigma fields for three configurations:
      A) Observation at pier wake (obs_radius=25)
      B) Observation at channel entrance (obs_radius=25)
      C) Observation off (uniform)
    """
    engine = _create_engine()

    x_grid = np.linspace(0, CHANNEL_LENGTH, grid_res)
    y_grid = np.linspace(0, CHANNEL_WIDTH, grid_res // 2)
    X, Y = np.meshgrid(x_grid, y_grid)

    configs = [
        ("pier", OBS_CENTER, 25.0, True),
        ("entrance", np.array([10.0, CHANNEL_WIDTH / 2.0, DEPTH / 2.0]), 25.0, True),
        ("off", OBS_CENTER, 25.0, False),
    ]

    sigma_fields = {}
    for name, center, radius, active in configs:
        np.random.seed(42)
        vf = VortexParticleField(engine, length=CHANNEL_LENGTH, n_particles=n_particles)
        if active:
            vf.set_observation(center, radius)
        else:
            vf.toggle_observation(False)

        # Compute sigma at each grid point
        Z = np.zeros_like(X)
        for i in range(len(x_grid)):
            for j in range(len(y_grid)):
                pos = np.array([x_grid[i], y_grid[j], DEPTH / 2.0])
                Z[j, i] = vf.get_adaptive_core_size(pos)
        sigma_fields[name] = Z

        if verbose:
            print(f"    {name:10s}  sigma_range=[{Z.min():.4f}, {Z.max():.4f}]")

    # Enhancement at obs center (pier config)
    np.random.seed(42)
    vf_check = VortexParticleField(engine, length=CHANNEL_LENGTH, n_particles=100)
    vf_check.set_observation(OBS_CENTER, 25.0)
    sigma_at_center = vf_check.get_adaptive_core_size(OBS_CENTER)
    sigma_at_corner = vf_check.get_adaptive_core_size(
        np.array([0.0, 0.0, DEPTH / 2.0])
    )
    enhancement = sigma_at_corner / sigma_at_center if sigma_at_center > 0 else 0.0

    return SigmaFieldResult(
        x_grid=x_grid,
        y_grid=y_grid,
        sigma_pier=sigma_fields["pier"],
        sigma_entrance=sigma_fields["entrance"],
        sigma_off=sigma_fields["off"],
        enhancement_at_center=enhancement,
    )


# ── Experiment 4: Engineering Relevance (Scour) ───────────────────────────

def _generate_single_pier_scenario(processor):
    """
    Synthetic 2D mesh: single circular pier at (100, 20) in a 200x40 ft channel.

    Returns dict {time_label: Mesh2DResults} with 3 timesteps (rising, peak, falling).
    """
    cell_size = 5.0
    xs = np.arange(0, CHANNEL_LENGTH + cell_size, cell_size)
    ys = np.arange(0, CHANNEL_WIDTH + cell_size, cell_size)
    xx, yy = np.meshgrid(xs, ys, indexing="ij")
    x_flat = xx.ravel()
    y_flat = yy.ravel()
    n_cells = len(x_flat)
    cell_ids = np.arange(n_cells)

    approach_v = 4.0
    approach_depth = DEPTH
    pier_half = PIER_DIAMETER / 2.0

    # Effective width loss from single pier
    open_width = CHANNEL_WIDTH - PIER_DIAMETER
    constriction_ratio = CHANNEL_WIDTH / open_width

    def compute_field(scale_v, scale_d):
        depth = np.full(n_cells, approach_depth * scale_d)
        vx = np.full(n_cells, approach_v * scale_v)
        vy = np.zeros(n_cells)

        for i in range(n_cells):
            cx, cy = x_flat[i], y_flat[i]
            dist_to_pier = np.sqrt((cx - PIER_X) ** 2 + (cy - PIER_Y) ** 2)

            # Inside pier
            if dist_to_pier <= pier_half + 0.5:
                depth[i] = 0.0
                vx[i] = 0.0
                vy[i] = 0.0
                continue

            # Constriction zone (pier influence region)
            if abs(cx - PIER_X) < 20.0:
                # Accelerated flow from blockage
                proximity = max(0.0, 1.0 - dist_to_pier / 20.0)
                vx[i] = approach_v * scale_v * (1.0 + (constriction_ratio - 1.0) * proximity)
                # Lateral deflection near pier
                if dist_to_pier < 10.0 and dist_to_pier > pier_half + 0.5:
                    angle = np.arctan2(cy - PIER_Y, cx - PIER_X)
                    deflection = 0.5 * approach_v * scale_v * np.exp(
                        -(dist_to_pier - pier_half) / 3.0
                    )
                    vy[i] = deflection * np.sin(angle)

            # Wake zone downstream
            elif PIER_X + 5.0 < cx < PIER_X + 60.0 and abs(cy - PIER_Y) < 8.0:
                dist_downstream = cx - PIER_X
                lateral_dist = abs(cy - PIER_Y)
                recovery = min(1.0, dist_downstream / 50.0)
                wake_width = pier_half + 2.0 * (1.0 - recovery)
                if lateral_dist < wake_width:
                    vx[i] = approach_v * scale_v * (0.3 + 0.7 * recovery)
                    sign = 1.0 if cy > PIER_Y else -1.0
                    vy[i] = sign * 0.4 * approach_v * scale_v * np.exp(
                        -dist_downstream / 20.0
                    )

        return depth, vx, vy

    timesteps = {}
    for label, sv, sd in [("rising", 0.7, 0.75), ("peak", 1.0, 1.0), ("falling", 0.5, 0.6)]:
        depth, vx, vy = compute_field(sv, sd)
        timesteps[label] = processor.load_arrays(label, cell_ids, x_flat, y_flat, depth, vx, vy)

    return timesteps, {
        "n_cells": n_cells,
        "pier_x": PIER_X, "pier_y": PIER_Y,
        "cell_size": cell_size,
        "x": x_flat, "y": y_flat,
    }


def run_scour(top_n=15, tier2_particles=300, verbose=False) -> ScourResult:
    """
    Single-pier scour: compare Tier 1 (vectorized) vs Tier 2 (vortex particle).
    """
    processor = SWMM2DPostProcessor(roughness_ks=ROUGHNESS_KS, cell_size=5.0)
    timesteps, meta = _generate_single_pier_scenario(processor)

    analysis = processor.analyze(
        timesteps,
        top_n_hotspots=top_n,
        tier2_particles=tier2_particles,
        compute_gradients=True,
    )

    peak_time = analysis["peak_time"]
    peak_metrics = analysis["tier1"][peak_time]
    tier2 = analysis["tier2"][peak_time]

    # Identify approach vs pier-adjacent cells
    x = meta["x"]
    y = meta["y"]
    approach_mask = (x < PIER_X - 30.0) & (peak_metrics.v_mag > 0.1)
    pier_mask = (
        np.sqrt((x - PIER_X) ** 2 + (y - PIER_Y) ** 2) < 15.0
    ) & (peak_metrics.depth > 0.01)

    tier1_shear_approach = float(peak_metrics.bed_shear[approach_mask].mean()) if approach_mask.any() else 0.0
    tier1_shear_pier = float(peak_metrics.bed_shear[pier_mask].mean()) if pier_mask.any() else 0.0

    # Best Tier 2 hotspot (highest amplification)
    if tier2:
        best = max(tier2, key=lambda r: r.amplification_factor)
        tier2_shear = best.quantum_bed_shear
        tier2_risk = best.quantum_scour_risk
        tier2_shields = best.quantum_shields
        amplification = best.amplification_factor
    else:
        tier2_shear = tier2_risk = tier2_shields = amplification = 0.0

    if verbose:
        print(f"    Tier 1 approach shear:  {tier1_shear_approach:.4f} psf")
        print(f"    Tier 1 pier shear:      {tier1_shear_pier:.4f} psf")
        print(f"    Tier 2 pier shear:      {tier2_shear:.4f} psf")
        print(f"    Tier 2 scour risk:      {tier2_risk:.3f}")
        print(f"    Tier 2 Shields:         {tier2_shields:.4f}")
        print(f"    Amplification:          {amplification:.2f}x")
        print(f"    Hotspots analyzed:      {len(tier2)}")

    return ScourResult(
        tier1_shear_pier=tier1_shear_pier,
        tier1_shear_approach=tier1_shear_approach,
        tier2_shear_pier=tier2_shear,
        tier2_scour_risk=tier2_risk,
        tier2_shields=tier2_shields,
        amplification=amplification,
        n_hotspots=len(tier2),
    )


# ── Experiment 5: Multi-Zone Independence ─────────────────────────────────

def run_multi_zone(n_particles=2000, verbose=False) -> MultiZoneResult:
    """
    Two observation zones on a 400-ft channel.
    Zone A at x=100, Zone B at x=300.
    Verify that metrics at each zone are independent.
    """
    length = 400.0
    engine = HydraulicsEngine(
        Q=Q, width=CHANNEL_WIDTH, depth=DEPTH,
        slope=SLOPE, roughness_ks=ROUGHNESS_KS,
    )

    zone_a_center = np.array([100.0, CHANNEL_WIDTH / 2.0, DEPTH / 2.0])
    zone_b_center = np.array([300.0, CHANNEL_WIDTH / 2.0, DEPTH / 2.0])
    zone_radius = 20.0

    # ── Run with both zones ───────────────────────────────────────────
    np.random.seed(42)
    vf = VortexParticleField(engine, length=length, n_particles=n_particles)
    vf.set_observation_zones([
        (zone_a_center, zone_radius),
        (zone_b_center, zone_radius),
    ])
    _run_field(vf)

    m_a = _measure_box(vf, 80.0, 120.0, 10.0, 30.0)
    m_b = _measure_box(vf, 280.0, 320.0, 10.0, 30.0)
    m_mid = _measure_box(vf, 190.0, 210.0, 10.0, 30.0)

    # ── Run with Zone B shifted (should not affect Zone A) ────────────
    zone_b_shifted = np.array([350.0, CHANNEL_WIDTH / 2.0, DEPTH / 2.0])
    np.random.seed(42)
    vf2 = VortexParticleField(engine, length=length, n_particles=n_particles)
    vf2.set_observation_zones([
        (zone_a_center, zone_radius),
        (zone_b_shifted, zone_radius),
    ])
    _run_field(vf2)

    m_b_moved = _measure_box(vf2, 280.0, 320.0, 10.0, 30.0)

    if verbose:
        print(f"    Zone A sigma:    {m_a['mean_sigma']:.4f}")
        print(f"    Zone B sigma:    {m_b['mean_sigma']:.4f}")
        print(f"    Midpoint sigma:  {m_mid['mean_sigma']:.4f}")
        print(f"    Zone A vort:     {m_a['mean_vorticity']:.4f}")
        print(f"    Zone B vort:     {m_b['mean_vorticity']:.4f}")
        print(f"    Zone B moved:    {m_b_moved['mean_vorticity']:.4f}")

    return MultiZoneResult(
        zone_a_sigma=m_a["mean_sigma"],
        zone_b_sigma=m_b["mean_sigma"],
        midpoint_sigma=m_mid["mean_sigma"],
        zone_a_vorticity=m_a["mean_vorticity"],
        zone_b_vorticity_base=m_b["mean_vorticity"],
        zone_b_vorticity_moved=m_b_moved["mean_vorticity"],
    )
