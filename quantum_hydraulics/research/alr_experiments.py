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
    # Valid accuracy-vs-cost experiment: in-zone induced-velocity RMS relative
    # error against a converged, deterministic reference (fixed core size), for
    # uniform vs observation-concentrated particle PLACEMENT. See
    # docs/DESIGN_costbenefit_fix.md.
    particle_counts: List[int]
    errors_uniform: List[float]           # in-zone RMS rel err, uniform placement
    errors_concentrated: List[float]      # in-zone RMS rel err, concentrated
    errors_out_uniform: List[float]       # out-of-zone control, uniform
    errors_out_concentrated: List[float]  # out-of-zone, concentrated (the price)
    wall_times: List[float]
    reduction_factor: float               # (K_uniform/K_concentrated)^2, in-zone
    out_of_zone_penalty: float            # (K_conc_out/K_unif_out)^2

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

# ── Valid cost-benefit: induced-velocity accuracy vs particle placement ────
#
# The coherent seed defines a DETERMINISTIC vorticity field omega=(0,du/dz,0) on
# the interior support box; its induced velocity at a probe is a fixed integral.
# Holding the core size fixed (overlap-maintaining), the particle field is a
# Monte-Carlo quadrature that converges to that integral as ~K/sqrt(N). Placing
# more particles in the observation zone (importance sampling) lowers K in-zone
# at the cost of raising it out-of-zone. This is the honest, measurable ALR
# trade-off (the old experiment compared a trivial box-mean of the seed against a
# silently-coarse baseline -- see docs/DESIGN_costbenefit_fix.md).

_CB_SIGMA_REF = DEPTH / 5.0                    # fixed physical core size (=base_sigma)
_CB_S2 = 2.0 * _CB_SIGMA_REF ** 2              # symmetrized smoothing, both cores = ref
_CB_FLO, _CB_FHI = 0.1, 0.9                    # interior support (matches _seed_coherent)
_CB_PROBES = np.array([                        # probes INSIDE the observation zone
    [OBS_CENTER[0], OBS_CENTER[1], 1.0], [OBS_CENTER[0], OBS_CENTER[1], 2.0],
    [OBS_CENTER[0], OBS_CENTER[1], 3.0], [OBS_CENTER[0] - 5, OBS_CENTER[1], 2.0],
    [OBS_CENTER[0] + 5, OBS_CENTER[1], 2.0], [OBS_CENTER[0], OBS_CENTER[1] - 5, 2.0],
    [OBS_CENTER[0], OBS_CENTER[1] + 5, 2.0],
])
_CB_PROBES_OUT = np.array([                    # control probes far outside the zone
    [30.0, OBS_CENTER[1], 2.0], [30.0, OBS_CENTER[1] - 5, 2.0], [30.0, OBS_CENTER[1], 3.0],
])


def _cb_support(engine):
    L, W, H = CHANNEL_LENGTH, engine.width, engine.depth
    return (_CB_FLO * L, _CB_FHI * L, _CB_FLO * W, _CB_FHI * W, _CB_FLO * H, _CB_FHI * H)


def _cb_omega_y(engine, z):
    dz = 0.01 * engine.depth
    up = engine.velocity_profile_vectorized(z + dz)
    dn = engine.velocity_profile_vectorized(np.maximum(z - dz, 1e-6))
    return (up - dn) / (2.0 * dz)


def _cb_induced(probes, pos, om, vol, s2):
    out = np.zeros((len(probes), 3))
    strength = om * vol[:, None]
    for p in range(len(probes)):
        r = pos - probes[p]
        r2 = np.einsum("ij,ij->i", r, r)
        K = 1.0 / (4.0 * np.pi * ((r2 + s2) ** 1.5 + 1e-12))
        out[p] = (K[:, None] * np.cross(strength, r)).sum(axis=0)
    return out


def _cb_reference(engine, probes, ncells=(260, 110, 38)):
    """Deterministic grid quadrature of the induced velocity (the ground truth)."""
    xlo, xhi, ylo, yhi, zlo, zhi = _cb_support(engine)
    v_dom = CHANNEL_LENGTH * engine.width * engine.depth
    nx, ny, nz = ncells
    xs = np.linspace(xlo, xhi, nx, endpoint=False) + 0.5 * (xhi - xlo) / nx
    ys = np.linspace(ylo, yhi, ny, endpoint=False) + 0.5 * (yhi - ylo) / ny
    zs = np.linspace(zlo, zhi, nz, endpoint=False) + 0.5 * (zhi - zlo) / nz
    vol_cell = v_dom / (nx * ny * nz)
    wy = _cb_omega_y(engine, zs)
    XX, YY = np.meshgrid(xs, ys, indexing="ij")
    xy = np.column_stack([XX.ravel(), YY.ravel()])
    out = np.zeros((len(probes), 3))
    for k in range(nz):
        z, w = zs[k], wy[k]
        for p in range(len(probes)):
            rx = xy[:, 0] - probes[p, 0]
            ry = xy[:, 1] - probes[p, 1]
            rz = z - probes[p, 2]
            K = 1.0 / (4.0 * np.pi * ((rx * rx + ry * ry + rz * rz + _CB_S2) ** 1.5 + 1e-12))
            sx = w * vol_cell
            out[p, 0] += (K * sx * rz).sum()
            out[p, 2] += (K * sx * (-rx)).sum()
    return out


def _cb_uniform(engine, N, seed):
    xlo, xhi, ylo, yhi, zlo, zhi = _cb_support(engine)
    v_dom = CHANNEL_LENGTH * engine.width * engine.depth
    rng = np.random.default_rng(seed)
    pos = np.column_stack([rng.uniform(xlo, xhi, N), rng.uniform(ylo, yhi, N),
                           rng.uniform(zlo, zhi, N)])
    om = np.zeros((N, 3)); om[:, 1] = _cb_omega_y(engine, pos[:, 2])
    return pos, om, np.full(N, v_dom / N)


def _cb_concentrated(engine, N, seed, beta=0.85, hx=15.0, hy=10.0):
    """Importance-sampled placement: a fraction beta of particles in a box around
    the observation center, with unbiased per-particle volumes Vol=1/(N p(x))."""
    xlo, xhi, ylo, yhi, zlo, zhi = _cb_support(engine)
    v_dom = CHANNEL_LENGTH * engine.width * engine.depth
    v_supp = (xhi - xlo) * (yhi - ylo) * (zhi - zlo)
    rng = np.random.default_rng(seed)
    bxlo, bxhi = max(xlo, OBS_CENTER[0] - hx), min(xhi, OBS_CENTER[0] + hx)
    bylo, byhi = max(ylo, OBS_CENTER[1] - hy), min(yhi, OBS_CENTER[1] + hy)
    v_box = (bxhi - bxlo) * (byhi - bylo) * (zhi - zlo)
    inb = rng.random(N) < beta
    nb = int(inb.sum())
    pos = np.empty((N, 3))
    pos[inb, 0] = rng.uniform(bxlo, bxhi, nb); pos[inb, 1] = rng.uniform(bylo, byhi, nb)
    pos[~inb, 0] = rng.uniform(xlo, xhi, N - nb); pos[~inb, 1] = rng.uniform(ylo, yhi, N - nb)
    pos[:, 2] = rng.uniform(zlo, zhi, N)
    inbox = ((pos[:, 0] >= bxlo) & (pos[:, 0] <= bxhi) &
             (pos[:, 1] >= bylo) & (pos[:, 1] <= byhi))
    pdf = (1 - beta) / v_supp + np.where(inbox, beta / v_box, 0.0)
    vol = (v_dom / v_supp) / (N * pdf)
    om = np.zeros((N, 3)); om[:, 1] = _cb_omega_y(engine, pos[:, 2])
    return pos, om, vol


def _cb_rms_rel(vest, ref):
    return np.sqrt(np.mean(((vest - ref) ** 2).sum(1)) / np.mean((ref ** 2).sum(1)))


def run_cost_benefit(particle_counts=None, seeds=12, verbose=False) -> CostBenefitResult:
    """
    Valid accuracy-vs-cost experiment for the coherent-seeded induced-velocity
    field: in-zone RMS relative error vs a converged deterministic reference,
    comparing uniform vs observation-concentrated particle placement (fixed core
    size). Reports the measured reduction factor (concentration vs uniform).
    """
    if particle_counts is None:
        particle_counts = [500, 1000, 2000, 4000, 8000, 16000]

    engine = _create_engine()
    ref = _cb_reference(engine, _CB_PROBES)
    ref_out = _cb_reference(engine, _CB_PROBES_OUT)
    if verbose:
        print("    Built deterministic induced-velocity reference "
              f"(|v| in-zone ~{np.linalg.norm(ref, axis=1).mean():.3f})")

    eu, ec, euo, eco, wall = [], [], [], [], []
    for N in particle_counts:
        t0 = time.perf_counter()
        du = [_cb_rms_rel(_cb_induced(_CB_PROBES, *_cb_uniform(engine, N, 20000 + s), _CB_S2), ref)
              for s in range(seeds)]
        dc = [_cb_rms_rel(_cb_induced(_CB_PROBES, *_cb_concentrated(engine, N, 20000 + s), _CB_S2), ref)
              for s in range(seeds)]
        duo = [_cb_rms_rel(_cb_induced(_CB_PROBES_OUT, *_cb_uniform(engine, N, 20000 + s), _CB_S2), ref_out)
               for s in range(seeds)]
        dco = [_cb_rms_rel(_cb_induced(_CB_PROBES_OUT, *_cb_concentrated(engine, N, 20000 + s), _CB_S2), ref_out)
               for s in range(seeds)]
        rms = lambda a: float(np.sqrt(np.mean(np.array(a) ** 2)))
        eu.append(rms(du)); ec.append(rms(dc)); euo.append(rms(duo)); eco.append(rms(dco))
        wall.append(time.perf_counter() - t0)
        if verbose:
            print(f"    N={N:6d}  in-zone uniform={eu[-1]:.3f} concentrated={ec[-1]:.3f}  "
                  f"out uniform={euo[-1]:.3f} concentrated={eco[-1]:.3f}")

    counts = np.array(particle_counts, dtype=float)
    K = lambda errs: float(np.median(np.array(errs) * np.sqrt(counts)))
    reduction = (K(eu) / max(K(ec), 1e-12)) ** 2                 # in-zone benefit
    penalty = (K(eco) / max(K(euo), 1e-12)) ** 2                 # out-of-zone cost

    return CostBenefitResult(
        particle_counts=list(particle_counts),
        errors_uniform=eu, errors_concentrated=ec,
        errors_out_uniform=euo, errors_out_concentrated=eco,
        wall_times=wall, reduction_factor=reduction,
        out_of_zone_penalty=penalty,
    )


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
