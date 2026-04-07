"""
Quantum Hydraulics — 2D Mesh Post-Processor
============================================

Post-processes 2D SWMM (PCSWMM) mesh results for scour analysis.
Runs a two-tier analysis: fast vectorized scan at every cell, then
full vortex particle turbulence analysis at the hotspots.

Built-in synthetic test: bridge pier scenario (300x50 ft channel,
4 piers, 671 cells, 3 timesteps). Validates the full 2D pipeline.

Usage:
  python run_headless_2d.py                    # synthetic bridge pier test
  python run_headless_2d.py data.csv           # any PCSWMM 2D export
  python run_headless_2d.py --top 10 --verbose
  python run_headless_2d.py --json

Exit code 0 = all checks pass, 1 = failure.
"""

import sys
import os
import json
import time
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from quantum_hydraulics.integration.swmm_2d import (
    SWMM2DPostProcessor,
    Mesh2DResults,
    RHO, NU, G, KAPPA,
)
from quantum_hydraulics.integration.swmm_node import SedimentProperties


# ── Synthetic bridge pier scenario ──────────────────────────────────────────

def generate_bridge_pier_scenario(processor):
    """
    Generate a synthetic 2D mesh for a channel with bridge piers.

    Channel: 300 ft long, 50 ft wide, 5-ft cells
    Bridge at x=130-170, 4 piers (3 ft wide each) at y=10,20,30,40
    Approach: V=3.5 fps, depth=2.0 ft
    Constriction: V~4.6 fps (continuity: 50/38 width ratio)
    Wake zone downstream with reduced velocity

    Returns dict {time_label: Mesh2DResults} with 3 timesteps.
    """
    cell_size = 5.0
    channel_length = 300.0
    channel_width = 50.0

    # Grid
    xs = np.arange(0, channel_length + cell_size, cell_size)
    ys = np.arange(0, channel_width + cell_size, cell_size)
    nx, ny = len(xs), len(ys)
    xx, yy = np.meshgrid(xs, ys, indexing="ij")
    x_flat = xx.ravel()
    y_flat = yy.ravel()
    n_cells = len(x_flat)
    cell_ids = np.arange(n_cells)

    # Pier geometry
    bridge_x_min, bridge_x_max = 130.0, 170.0
    pier_y_centers = [10.0, 20.0, 30.0, 40.0]
    pier_half_width = 1.5  # 3 ft total per pier

    approach_v = 3.5       # fps
    approach_depth = 2.0   # ft
    total_width = 50.0
    pier_blockage = 4 * 3.0  # 12 ft
    open_width = total_width - pier_blockage  # 38 ft
    constriction_ratio = total_width / open_width  # ~1.316

    def compute_field(scale_v, scale_d):
        """Compute depth and velocity fields for one timestep."""
        depth = np.full(n_cells, approach_depth * scale_d)
        vx = np.full(n_cells, approach_v * scale_v)
        vy = np.zeros(n_cells)

        for i in range(n_cells):
            cx, cy = x_flat[i], y_flat[i]

            # Check if cell is inside a pier
            in_pier = False
            if bridge_x_min <= cx <= bridge_x_max:
                for py in pier_y_centers:
                    if abs(cy - py) <= pier_half_width:
                        in_pier = True
                        break

            if in_pier:
                depth[i] = 0.0
                vx[i] = 0.0
                vy[i] = 0.0
                continue

            # Bridge constriction zone (between piers)
            if bridge_x_min <= cx <= bridge_x_max:
                # Accelerated flow between piers
                vx[i] = approach_v * scale_v * constriction_ratio
                # Slight depth reduction (continuity)
                depth[i] = approach_depth * scale_d / constriction_ratio

                # Lateral velocity near pier edges
                for py in pier_y_centers:
                    dist_to_pier = abs(cy - py)
                    if dist_to_pier < 5.0 and dist_to_pier > pier_half_width:
                        # Flow deflection near piers
                        sign = 1.0 if cy > py else -1.0
                        vy[i] = sign * 0.4 * approach_v * scale_v * np.exp(
                            -(dist_to_pier - pier_half_width) / 2.0
                        )

            # Wake zone (downstream of bridge)
            elif bridge_x_max < cx < bridge_x_max + 50.0:
                dist_downstream = cx - bridge_x_max

                # Check if in wake shadow of a pier
                in_wake = False
                for py in pier_y_centers:
                    if abs(cy - py) < pier_half_width + 2.0:
                        in_wake = True
                        # Reduced velocity in wake, recovering downstream
                        recovery = min(1.0, dist_downstream / 40.0)
                        vx[i] = approach_v * scale_v * (0.4 + 0.6 * recovery)
                        # Lateral spreading in wake
                        sign = 1.0 if cy > py else -1.0
                        vy[i] = sign * 0.3 * approach_v * scale_v * np.exp(
                            -dist_downstream / 15.0
                        )
                        break

                if not in_wake:
                    # Gradual expansion back to approach velocity
                    recovery = min(1.0, dist_downstream / 30.0)
                    vx[i] = approach_v * scale_v * (constriction_ratio - (constriction_ratio - 1.0) * recovery)

        return depth, vx, vy

    # 3 timesteps: rising, peak, falling
    timesteps = {}
    for label, sv, sd in [("rising", 0.7, 0.75), ("peak", 1.0, 1.0), ("falling", 0.5, 0.6)]:
        depth, vx, vy = compute_field(sv, sd)
        timesteps[label] = processor.load_arrays(
            label, cell_ids, x_flat, y_flat, depth, vx, vy
        )

    return timesteps, {"n_cells": n_cells, "nx": nx, "ny": ny, "cell_size": cell_size,
                        "bridge_x": (bridge_x_min, bridge_x_max),
                        "pier_y_centers": pier_y_centers}


# ── Check infrastructure ────────────────────────────────────────────────────

class CheckResult:
    def __init__(self, name, passed, detail, values=None):
        self.name = name
        self.passed = passed
        self.detail = detail
        self.values = values or {}

    def __str__(self):
        tag = "PASS" if self.passed else "FAIL"
        return f"  [{tag}] {self.name}: {self.detail}"


# ── Validation ──────────────────────────────────────────────────────────────

def run_synthetic_checks(top_n=20, tier2_particles=300, verbose=False):
    """Run full validation against synthetic bridge pier scenario."""
    results = []
    t0 = time.perf_counter()

    processor = SWMM2DPostProcessor(roughness_ks=0.1, cell_size=5.0)

    # Generate scenario
    timesteps, meta = generate_bridge_pier_scenario(processor)
    n_cells = meta["n_cells"]
    bridge_x_min, bridge_x_max = meta["bridge_x"]

    results.append(CheckResult(
        "Scenario generated",
        len(timesteps) == 3 and n_cells > 0,
        f"{n_cells} cells, {len(timesteps)} timesteps",
    ))

    if verbose:
        print(f"\n  Synthetic bridge pier: {n_cells} cells, {len(timesteps)} timesteps")

    # ── Run analysis ────────────────────────────────────────────────────
    analysis = processor.analyze(
        timesteps,
        top_n_hotspots=top_n,
        tier2_particles=tier2_particles,
        compute_gradients=True,
    )

    peak_time = analysis["peak_time"]
    peak_metrics = analysis["tier1"][peak_time]
    tier2 = analysis["tier2"][peak_time]

    results.append(CheckResult(
        "Peak timestep identified",
        peak_time == "peak",
        f"peak_time='{peak_time}'",
    ))

    if verbose:
        print(f"  Peak timestep: '{peak_time}'")
        print(f"  Tier 1: {peak_metrics.n_cells} cells analyzed")
        print(f"  Tier 2: {len(tier2)} hotspot cells analyzed")

    # ── Tier 1 checks ──────────────────────────────────────────────────

    # Identify approach vs constriction cells (excluding piers)
    approach_mask = (peak_metrics.x < bridge_x_min - 20) & (peak_metrics.v_mag > 0.1)
    constrict_mask = (
        (peak_metrics.x >= bridge_x_min) &
        (peak_metrics.x <= bridge_x_max) &
        (peak_metrics.depth > 0.01)  # not pier cells
    )

    v_approach_mean = peak_metrics.v_mag[approach_mask].mean()
    v_constrict_mean = peak_metrics.v_mag[constrict_mask].mean()

    results.append(CheckResult(
        "Constriction velocity > approach velocity",
        v_constrict_mean > v_approach_mean,
        f"constrict={v_constrict_mean:.2f} fps, approach={v_approach_mean:.2f} fps",
    ))

    # Froude subcritical everywhere
    wet_mask = peak_metrics.depth > 0.01
    max_fr = peak_metrics.froude[wet_mask].max()
    results.append(CheckResult(
        "Froude subcritical (all wet cells)",
        max_fr < 1.0,
        f"max Fr={max_fr:.4f}",
    ))

    # Reynolds turbulent where there's flow
    flow_mask = wet_mask & (peak_metrics.v_mag > 0.1)
    min_re = peak_metrics.reynolds[flow_mask].min()
    results.append(CheckResult(
        "Reynolds turbulent (all flowing cells)",
        min_re > 2300,
        f"min Re={min_re:,.0f}",
    ))

    # Friction factor in Moody bounds
    f_flow = peak_metrics.friction_factor[flow_mask]
    results.append(CheckResult(
        "Colebrook-White in Moody bounds",
        np.all((f_flow > 0.005) & (f_flow < 0.08)),
        f"f range [{f_flow.min():.5f}, {f_flow.max():.5f}]",
    ))

    # Bed shear at constriction > approach
    tau_approach = peak_metrics.bed_shear[approach_mask].mean()
    tau_constrict = peak_metrics.bed_shear[constrict_mask].mean()
    results.append(CheckResult(
        "Bed shear: constriction > approach",
        tau_constrict > tau_approach,
        f"constrict={tau_constrict:.4f} psf, approach={tau_approach:.4f} psf",
    ))

    # Scour risk higher at constriction
    risk_approach = peak_metrics.scour_risk[approach_mask].mean()
    risk_constrict = peak_metrics.scour_risk[constrict_mask].mean()
    results.append(CheckResult(
        "Scour risk: constriction > approach",
        risk_constrict >= risk_approach,
        f"constrict={risk_constrict:.4f}, approach={risk_approach:.4f}",
    ))

    # ── Velocity gradients ──────────────────────────────────────────────
    grad_computed = peak_metrics.grad_mag is not None
    results.append(CheckResult(
        "Velocity gradients computed",
        grad_computed,
        f"{'yes' if grad_computed else 'no'}",
    ))

    if grad_computed:
        # Gradient should be highest near pier edges (bridge zone)
        bridge_zone = (
            (peak_metrics.x >= bridge_x_min - 10) &
            (peak_metrics.x <= bridge_x_max + 10) &
            (peak_metrics.depth > 0.01)
        )
        upstream_zone = peak_metrics.x < bridge_x_min - 30
        grad_bridge = peak_metrics.grad_mag[bridge_zone].mean()
        grad_upstream = peak_metrics.grad_mag[upstream_zone].mean()

        results.append(CheckResult(
            "Velocity gradient: bridge zone > upstream",
            grad_bridge > grad_upstream,
            f"bridge={grad_bridge:.4f}, upstream={grad_upstream:.4f}",
        ))

    # ── Hotspot detection ───────────────────────────────────────────────
    hotspot_idx = analysis["hotspot_indices"]
    hotspot_x = peak_metrics.x[hotspot_idx]
    # Most hotspots should be in or near the bridge
    near_bridge = np.sum(
        (hotspot_x >= bridge_x_min - 10) & (hotspot_x <= bridge_x_max + 30)
    )
    results.append(CheckResult(
        f"Hotspots near bridge ({near_bridge}/{len(hotspot_idx)})",
        near_bridge > len(hotspot_idx) * 0.5,
        f"{near_bridge} of {len(hotspot_idx)} hotspots in bridge zone",
    ))

    # ── Tier 2 checks ──────────────────────────────────────────────────
    results.append(CheckResult(
        "Tier 2 analyses completed",
        len(tier2) > 0,
        f"{len(tier2)} cells analyzed with vortex particles",
    ))

    if tier2:
        # Amplification >= 1.0 (quantum shear >= tier1 shear)
        amps = [r.amplification_factor for r in tier2]
        min_amp = min(amps)
        results.append(CheckResult(
            "Tier 2 amplification >= 1.0",
            min_amp >= 0.99,  # tiny tolerance
            f"min amplification={min_amp:.3f}, mean={np.mean(amps):.3f}",
        ))

        # TKE positive
        tkes = [r.tke for r in tier2]
        results.append(CheckResult(
            "Tier 2 TKE > 0",
            all(t > 0 for t in tkes),
            f"min TKE={min(tkes):.6f}",
        ))

        # Particles injected
        results.append(CheckResult(
            "Tier 2 particles injected",
            all(r.n_particles > 0 for r in tier2),
            f"n_particles={tier2[0].n_particles}",
        ))

    # ── Runtime ─────────────────────────────────────────────────────────
    elapsed = time.perf_counter() - t0
    results.append(CheckResult(
        "Runtime",
        elapsed < 60.0,
        f"{elapsed:.1f}s",
    ))

    return results, analysis, peak_metrics, tier2, meta


def run_csv_analysis(csv_path, top_n, ks, sediment_name, verbose=False):
    """Run analysis on a CSV file."""
    results = []

    sediment_factory = {
        "sand": SedimentProperties.sand,
        "fine_sand": SedimentProperties.fine_sand,
        "coarse_sand": SedimentProperties.coarse_sand,
        "gravel": SedimentProperties.gravel,
        "silt": SedimentProperties.silt,
        "clay": SedimentProperties.clay,
    }
    sediment = sediment_factory.get(sediment_name, SedimentProperties.sand)()

    processor = SWMM2DPostProcessor(roughness_ks=ks, sediment=sediment, cell_size=5.0)

    if not os.path.exists(csv_path):
        results.append(CheckResult("CSV file exists", False, f"Not found: {csv_path}"))
        return results, None, None, None

    results.append(CheckResult("CSV file exists", True, os.path.basename(csv_path)))

    timesteps = processor.load_csv(csv_path)
    results.append(CheckResult(
        "CSV loaded",
        len(timesteps) > 0,
        f"{len(timesteps)} timesteps, {next(iter(timesteps.values())).n_cells} cells",
    ))

    analysis = processor.analyze(timesteps, top_n_hotspots=top_n, compute_gradients=True)

    peak_time = analysis["peak_time"]
    peak_metrics = analysis["tier1"][peak_time]
    tier2 = analysis["tier2"].get(peak_time, [])

    # Basic validation
    wet = peak_metrics.depth > 0.01
    if np.any(wet):
        results.append(CheckResult(
            "Wet cells found",
            True,
            f"{np.sum(wet)} of {peak_metrics.n_cells} cells",
        ))

    if tier2:
        results.append(CheckResult(
            "Tier 2 completed",
            True,
            f"{len(tier2)} hotspot cells",
        ))

    return results, analysis, peak_metrics, tier2


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Quantum Hydraulics -- 2D Mesh Post-Processor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_headless_2d.py                           # synthetic bridge pier test
  python run_headless_2d.py exported_2d_results.csv   # PCSWMM export
  python run_headless_2d.py --top 10 --verbose
  python run_headless_2d.py --json
        """,
    )
    parser.add_argument("csv_file", nargs="?", default=None,
                        help="CSV with 2D results (omit for synthetic test)")
    parser.add_argument("--top", type=int, default=20,
                        help="Number of hotspot cells for Tier 2 (default 20)")
    parser.add_argument("--ks", type=float, default=0.1,
                        help="Roughness ks in ft (default 0.1)")
    parser.add_argument("--sediment", default="sand",
                        choices=["sand", "fine_sand", "coarse_sand", "gravel", "silt", "clay"])
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    synthetic = args.csv_file is None

    if not args.json:
        print()
        print("=" * 72)
        print("QUANTUM HYDRAULICS -- 2D MESH POST-PROCESSOR")
        print("=" * 72)
        if synthetic:
            print("Mode: Synthetic bridge pier validation")
            print("Channel: 300 x 50 ft, 4 piers @ 3 ft, 5-ft cells")
        else:
            print(f"Input: {args.csv_file}")
        print(f"Tier 2 hotspots: {args.top} | ks={args.ks} ft | sediment={args.sediment}")
        print("-" * 72)

    if synthetic:
        check_results, analysis, peak_metrics, tier2, meta = run_synthetic_checks(
            top_n=args.top, verbose=args.verbose,
        )
    else:
        check_results, analysis, peak_metrics, tier2 = run_csv_analysis(
            args.csv_file, args.top, args.ks, args.sediment, args.verbose,
        )

    n_pass = sum(1 for r in check_results if r.passed)
    n_fail = sum(1 for r in check_results if not r.passed)
    all_pass = n_fail == 0

    if args.json:
        output = {
            "mode": "synthetic_bridge_pier" if synthetic else os.path.basename(args.csv_file),
            "passed": n_pass,
            "failed": n_fail,
            "all_pass": all_pass,
            "checks": [
                {"name": r.name, "passed": r.passed, "detail": r.detail, **r.values}
                for r in check_results
            ],
        }
        if tier2:
            output["tier2_hotspots"] = [r.to_dict() for r in tier2[:10]]
        print(json.dumps(output, indent=2,
                          default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    else:
        # Print checks
        print()
        for r in check_results:
            if args.verbose or not r.passed:
                print(r)
            else:
                print(f"  [{'PASS' if r.passed else 'FAIL'}] {r.name}")

        # Tier 1 summary
        if peak_metrics is not None:
            wet = peak_metrics.depth > 0.01
            print()
            print("  TIER 1 SUMMARY (peak timestep):")
            print(f"    Wet cells: {np.sum(wet)}")
            print(f"    V_mag range: [{peak_metrics.v_mag[wet].min():.2f}, "
                  f"{peak_metrics.v_mag[wet].max():.2f}] fps")
            print(f"    Bed shear range: [{peak_metrics.bed_shear[wet].min():.4f}, "
                  f"{peak_metrics.bed_shear[wet].max():.4f}] psf")
            print(f"    Froude range: [{peak_metrics.froude[wet].min():.4f}, "
                  f"{peak_metrics.froude[wet].max():.4f}]")

        # Tier 2 table
        if tier2:
            print()
            print(f"  TIER 2 HOTSPOTS (top {len(tier2)} by velocity):")
            print(f"  {'Cell':>6} {'X':>7} {'Y':>7} {'T1_tau':>8} {'Q_tau':>8} "
                  f"{'Amplif':>7} {'Q_Scour':>8} {'TKE':>10}")
            print(f"  {'':>6} {'(ft)':>7} {'(ft)':>7} {'(psf)':>8} {'(psf)':>8} "
                  f"{'':>7} {'Risk':>8} {'':>10}")
            for r in tier2:
                print(f"  {r.cell_id:>6} {r.x:>7.1f} {r.y:>7.1f} "
                      f"{r.tier1_bed_shear:>8.4f} {r.quantum_bed_shear:>8.4f} "
                      f"{r.amplification_factor:>7.2f} {r.quantum_scour_risk:>8.4f} "
                      f"{r.tke:>10.6f}")

        print()
        print("-" * 72)
        print(f"  {n_pass}/{n_pass + n_fail} checks passed", end="")
        if all_pass:
            print(" -- ALL PASS")
        else:
            print(f" -- {n_fail} FAILED")
            for r in check_results:
                if not r.passed:
                    print(f"         X {r.name}: {r.detail}")

        print("=" * 72)
        print()

    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
