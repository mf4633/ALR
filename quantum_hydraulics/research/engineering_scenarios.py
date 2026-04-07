"""
Synthetic Engineering Scenarios for Quantum Hydraulics.

Four scenario generators following the established pattern:
  generate_XXX_scenario(processor) → ({time_label: Mesh2DResults}, metadata)

Each returns 3 timesteps and a metadata dict with geometry info and masks.
"""

import numpy as np
from quantum_hydraulics.integration.swmm_2d import SWMM2DPostProcessor


# ══════════════════════════════════════════════════════════════════════════
# 1. BANK EROSION — Trapezoidal channel with sloped banks
# ══════════════════════════════════════════════════════════════════════════

def generate_bank_erosion_scenario(processor):
    """
    Trapezoidal channel: 300 ft long, 30 ft bottom, 3:1 side slopes, 4 ft depth.

    Bank cells have wedge-shaped depth (linear taper to zero at top of bank).
    Velocity on banks scales with sqrt(depth/max_depth).

    Returns 3 timesteps: low_flow (2 ft), bankfull (4 ft), overbank (5 ft).
    """
    cell_size = 5.0
    length = 300.0
    bottom_w = 30.0
    side_slope = 3.0   # H:V
    max_depth = 4.0
    bank_height = max_depth
    bank_run = side_slope * bank_height  # 12 ft horizontal per bank

    # Grid covers full trapezoidal cross-section
    total_w = bottom_w + 2 * bank_run  # 54 ft
    y_min = -bank_run   # -12
    y_max = bottom_w + bank_run  # 42

    xs = np.arange(0, length + cell_size, cell_size)
    ys = np.arange(y_min, y_max + cell_size, cell_size)
    xx, yy = np.meshgrid(xs, ys, indexing="ij")
    x_flat = xx.ravel()
    y_flat = yy.ravel()
    n_cells = len(x_flat)
    cell_ids = np.arange(n_cells)

    approach_v = 4.0  # fps at bankfull

    # Masks for bank vs channel cells
    left_bank_mask = y_flat < 0
    right_bank_mask = y_flat > bottom_w
    bank_mask = left_bank_mask | right_bank_mask
    channel_mask = ~bank_mask

    def compute_field(scale_v, scale_d):
        depth_full = max_depth * scale_d
        depth = np.zeros(n_cells)
        vx = np.zeros(n_cells)
        vy = np.zeros(n_cells)

        for i in range(n_cells):
            cy = y_flat[i]

            if cy < 0:
                # Left bank: depth tapers from full at y=0 to 0 at y=-bank_run
                d = depth_full * (1.0 + cy / bank_run)  # cy is negative
            elif cy > bottom_w:
                # Right bank: depth tapers from full at y=bottom_w to 0
                d = depth_full * (1.0 - (cy - bottom_w) / bank_run)
            else:
                # Channel bottom: full depth
                d = depth_full

            d = max(0.0, d)
            depth[i] = d

            if d > 0.05:
                # Velocity scales with sqrt(d/d_full) on banks
                v_scale = np.sqrt(d / max(depth_full, 0.1))
                vx[i] = approach_v * scale_v * v_scale

        return depth, vx, vy

    timesteps = {}
    stages = [("low_flow", 0.5, 0.5), ("bankfull", 1.0, 1.0), ("overbank", 1.1, 1.25)]
    for label, sv, sd in stages:
        depth, vx, vy = compute_field(sv, sd)
        timesteps[label] = processor.load_arrays(label, cell_ids, x_flat, y_flat, depth, vx, vy)

    return timesteps, {
        "n_cells": n_cells, "length": length,
        "bottom_w": bottom_w, "bank_run": bank_run,
        "side_slope": side_slope, "max_depth": max_depth,
        "bank_mask": bank_mask, "channel_mask": channel_mask,
        "x": x_flat, "y": y_flat,
    }


# ══════════════════════════════════════════════════════════════════════════
# 2. BED DEGRADATION — Grade change (slope steepening)
# ══════════════════════════════════════════════════════════════════════════

def generate_degradation_scenario(processor):
    """
    Rectangular channel: 500 ft long, 40 ft wide.
    Upstream slope 0.001 (V~3 fps), downstream slope 0.003 (V~5.2 fps).
    Transition zone x=225-275.

    The steeper downstream reach has higher transport capacity than the upstream
    reach can supply, causing bed degradation.
    """
    cell_size = 5.0
    length = 500.0
    width = 40.0
    depth_base = 3.0
    v_upstream = 3.0    # fps (based on slope=0.001)
    v_downstream = 5.2  # fps (based on slope=0.003)
    transition_start = 225.0
    transition_end = 275.0

    xs = np.arange(0, length + cell_size, cell_size)
    ys = np.arange(0, width + cell_size, cell_size)
    xx, yy = np.meshgrid(xs, ys, indexing="ij")
    x_flat = xx.ravel()
    y_flat = yy.ravel()
    n_cells = len(x_flat)
    cell_ids = np.arange(n_cells)

    upstream_mask = x_flat < transition_start
    downstream_mask = x_flat > transition_end
    transition_mask = ~upstream_mask & ~downstream_mask

    def compute_field(scale_v, scale_d):
        depth = np.full(n_cells, depth_base * scale_d)
        vx = np.zeros(n_cells)
        vy = np.zeros(n_cells)

        for i in range(n_cells):
            cx = x_flat[i]
            if cx < transition_start:
                vx[i] = v_upstream * scale_v
            elif cx > transition_end:
                vx[i] = v_downstream * scale_v
            else:
                # Linear interpolation in transition
                frac = (cx - transition_start) / (transition_end - transition_start)
                vx[i] = (v_upstream + frac * (v_downstream - v_upstream)) * scale_v

        return depth, vx, vy

    timesteps = {}
    for label, sv, sd in [("low", 0.6, 0.7), ("design", 1.0, 1.0), ("flood", 1.3, 1.15)]:
        depth, vx, vy = compute_field(sv, sd)
        timesteps[label] = processor.load_arrays(label, cell_ids, x_flat, y_flat, depth, vx, vy)

    return timesteps, {
        "n_cells": n_cells, "length": length, "width": width,
        "v_upstream": v_upstream, "v_downstream": v_downstream,
        "transition_start": transition_start, "transition_end": transition_end,
        "upstream_mask": upstream_mask, "downstream_mask": downstream_mask,
        "x": x_flat, "y": y_flat,
    }


# ══════════════════════════════════════════════════════════════════════════
# 3. CULVERT OUTLET — Jet expansion into receiving channel
# ══════════════════════════════════════════════════════════════════════════

def generate_culvert_outlet_scenario(processor):
    """
    Culvert (6 ft wide) discharging into 40 ft wide channel.
    Jet velocity 10 fps, Gaussian lateral spread, plunge zone at x=5-20.

    Fine grid (2 ft cells) to capture jet structure.
    """
    cell_size = 2.0
    x_start = -10.0
    x_end = 100.0
    width = 40.0
    culvert_center = 20.0
    culvert_half_w = 3.0  # 6 ft wide
    v_jet = 10.0
    jet_depth = 4.0
    tailwater_depth = 2.0

    xs = np.arange(x_start, x_end + cell_size, cell_size)
    ys = np.arange(0, width + cell_size, cell_size)
    xx, yy = np.meshgrid(xs, ys, indexing="ij")
    x_flat = xx.ravel()
    y_flat = yy.ravel()
    n_cells = len(x_flat)
    cell_ids = np.arange(n_cells)

    def compute_field(scale_v, scale_d):
        depth = np.full(n_cells, tailwater_depth * scale_d)
        vx = np.zeros(n_cells)
        vy = np.zeros(n_cells)

        for i in range(n_cells):
            cx, cy = x_flat[i], y_flat[i]
            dy = cy - culvert_center

            if cx <= 0:
                # Inside/near culvert: full jet in culvert opening
                if abs(dy) <= culvert_half_w:
                    vx[i] = v_jet * scale_v
                    depth[i] = jet_depth * scale_d
                else:
                    vx[i] = 0.3 * scale_v  # ambient
            else:
                # Downstream: Gaussian jet expansion
                b0 = culvert_half_w
                b_x = b0 + 0.12 * cx  # spreading rate
                v_core = v_jet * scale_v * b0 / max(b_x, b0)
                vx[i] = max(0.3 * scale_v, v_core * np.exp(-dy ** 2 / (2 * b_x ** 2)))

                # Slight lateral divergence
                if abs(dy) > 0.5:
                    vy[i] = 0.05 * vx[i] * np.sign(dy) * np.exp(-cx / 30.0)

                # Plunge zone: enhanced depth (scour hole)
                if 5.0 <= cx <= 25.0 and abs(dy) < b_x * 1.5:
                    plunge_factor = 1.0 + 0.4 * np.exp(-((cx - 12.0) / 6.0) ** 2)
                    depth[i] = tailwater_depth * scale_d * plunge_factor

        return depth, vx, vy

    timesteps = {}
    for label, sv, sd in [("partial", 0.6, 0.7), ("design", 1.0, 1.0), ("flood", 1.3, 1.2)]:
        depth, vx, vy = compute_field(sv, sd)
        timesteps[label] = processor.load_arrays(label, cell_ids, x_flat, y_flat, depth, vx, vy)

    # Masks
    jet_mask = (x_flat <= 2.0) & (np.abs(y_flat - culvert_center) <= culvert_half_w)
    plunge_mask = (x_flat >= 5.0) & (x_flat <= 25.0) & (np.abs(y_flat - culvert_center) < 10.0)

    return timesteps, {
        "n_cells": n_cells, "culvert_center": culvert_center,
        "culvert_half_w": culvert_half_w, "v_jet": v_jet,
        "tailwater_depth": tailwater_depth,
        "jet_mask": jet_mask, "plunge_mask": plunge_mask,
        "x": x_flat, "y": y_flat,
    }


# ══════════════════════════════════════════════════════════════════════════
# 4. CHANNEL BEND — 90-degree bend with outer bank amplification
# ══════════════════════════════════════════════════════════════════════════

def generate_bend_scenario(processor):
    """
    Straight approach (200 ft) → 90-degree bend (R=100 ft) → straight exit.
    Channel width W=30 ft.  Forced-vortex velocity distribution in bend.

    Uses rectangular grid; cells outside channel get depth=0.
    Bend center at (200, 0). Approach along x-axis (y=-15 to 15).
    Bend sweeps from theta=0 (east) to theta=pi/2 (north).
    Exit along y-axis from bend center.
    """
    cell_size = 5.0
    approach_length = 200.0
    R = 100.0           # centerline bend radius
    W = 30.0            # channel width
    exit_length = 150.0
    depth_base = 4.0
    approach_v = 4.0

    R_inner = R - W / 2  # 85
    R_outer = R + W / 2  # 115

    # Grid covering the full domain
    x_max = approach_length + R_outer + 10
    y_max = R_outer + exit_length + 10
    xs = np.arange(-10, x_max + cell_size, cell_size)
    ys = np.arange(-W / 2 - 10, y_max + cell_size, cell_size)
    xx, yy = np.meshgrid(xs, ys, indexing="ij")
    x_flat = xx.ravel()
    y_flat = yy.ravel()
    n_cells = len(x_flat)
    cell_ids = np.arange(n_cells)

    bend_center_x = approach_length
    bend_center_y = 0.0

    def compute_field(scale_v, scale_d):
        depth = np.zeros(n_cells)
        vx = np.zeros(n_cells)
        vy = np.zeros(n_cells)

        for i in range(n_cells):
            cx, cy = x_flat[i], y_flat[i]

            # Zone 1: Straight approach (x < bend_center_x, channel at y=-W/2 to W/2)
            if cx < bend_center_x and abs(cy) <= W / 2:
                depth[i] = depth_base * scale_d
                vx[i] = approach_v * scale_v
                continue

            # Zone 2: Bend (90-degree arc)
            dx = cx - bend_center_x
            dy = cy - bend_center_y
            r = np.sqrt(dx ** 2 + dy ** 2)
            theta = np.arctan2(dy, dx)  # 0 = east, pi/2 = north

            if R_inner <= r <= R_outer and 0 <= theta <= np.pi / 2 + 0.05:
                depth[i] = depth_base * scale_d
                # Forced vortex: V(r) = V_mean * r / R
                v_tangential = approach_v * scale_v * r / R
                # Tangential direction (perpendicular to radius, CCW)
                vx[i] = -v_tangential * np.sin(theta)
                vy[i] = v_tangential * np.cos(theta)
                continue

            # Zone 3: Straight exit (heading +y from bend, x = bend_center_x - W/2 to + W/2...
            # actually from bend end: at theta=pi/2, channel center is at (bend_center_x, R))
            # Exit runs from (bend_center_x - W/2, R) northward
            exit_x_center = bend_center_x
            if (abs(cx - exit_x_center) <= W / 2 and
                    R_inner <= cy <= R_inner + exit_length and
                    cx >= bend_center_x - W / 2):
                # Only cells above the bend exit point
                if cy >= R_inner:
                    depth[i] = depth_base * scale_d
                    vy[i] = approach_v * scale_v
                    # Slight recovery from bend
                    recovery = min(1.0, (cy - R_inner) / 50.0)
                    vy[i] *= (1.0 + 0.1 * (1.0 - recovery))  # slight deceleration

        return depth, vx, vy

    timesteps = {}
    for label, sv, sd in [("rising", 0.7, 0.75), ("peak", 1.0, 1.0), ("falling", 0.5, 0.6)]:
        depth, vx, vy = compute_field(sv, sd)
        timesteps[label] = processor.load_arrays(label, cell_ids, x_flat, y_flat, depth, vx, vy)

    # Compute masks on peak timestep
    approach_mask = (x_flat < bend_center_x - 20) & (np.abs(y_flat) <= W / 2)
    # Bend cells
    dx = x_flat - bend_center_x
    dy = y_flat - bend_center_y
    r_arr = np.sqrt(dx ** 2 + dy ** 2)
    theta_arr = np.arctan2(dy, dx)
    bend_mask = (r_arr >= R_inner) & (r_arr <= R_outer) & (theta_arr >= 0) & (theta_arr <= np.pi / 2 + 0.05)
    outer_mask = bend_mask & (r_arr > R)
    inner_mask = bend_mask & (r_arr <= R)

    return timesteps, {
        "n_cells": n_cells, "R": R, "W": W,
        "approach_v": approach_v, "depth": depth_base,
        "bend_center": (bend_center_x, bend_center_y),
        "approach_mask": approach_mask, "bend_mask": bend_mask,
        "outer_mask": outer_mask, "inner_mask": inner_mask,
        "x": x_flat, "y": y_flat,
    }
