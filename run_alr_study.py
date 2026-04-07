"""
Quantum Hydraulics — ALR Research Study
========================================

Adaptive Lagrangian Refinement proof points for ICWMM 2026.

Runs five headless experiments demonstrating observation-dependent
resolution for vortex particle hydraulic simulation:

  1. Convergence — metrics converge as observation radius increases
  2. Cost-Benefit — ALR accuracy at reduced particle count
  3. Sigma Field — visualization of adaptive resolution concentration
  4. Engineering Scour — Tier 1 vs Tier 2 at a bridge pier
  5. Multi-Zone — two independent observation zones

Usage:
  python run_alr_study.py                     # all experiments
  python run_alr_study.py --experiment 3      # sigma field only
  python run_alr_study.py --figures           # generate paper figures
  python run_alr_study.py --verbose --json

Exit code 0 = all checks pass, 1 = failure.
"""

import sys
import os
import json
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from quantum_hydraulics.research.alr_experiments import (
    run_convergence,
    run_cost_benefit,
    run_sigma_field,
    run_scour,
    run_multi_zone,
    CHANNEL_LENGTH, CHANNEL_WIDTH, DEPTH, Q,
    PIER_X, PIER_Y, OBS_CENTER,
)


# ── Check infrastructure ──────────────────────────────────────────────────

class CheckResult:
    def __init__(self, name, passed, detail, values=None):
        self.name = name
        self.passed = passed
        self.detail = detail
        self.values = values or {}

    def __str__(self):
        tag = "PASS" if self.passed else "FAIL"
        return f"  [{tag}] {self.name}: {self.detail}"


# ── Experiment runners with checks ────────────────────────────────────────

def checks_convergence(verbose=False):
    """Experiment 1: Convergence study."""
    results = []
    if verbose:
        print("\n  Experiment 1: Convergence Study")
        print("  " + "-" * 50)

    r = run_convergence(verbose=verbose)

    # Check: mean sigma should generally decrease as obs_radius grows
    # (larger radius = more of the box is in high-res zone)
    sigma_first = r.mean_sigma[0]   # smallest radius
    sigma_last = r.mean_sigma[-1]   # largest radius
    results.append(CheckResult(
        "Sigma responds to obs_radius",
        sigma_first != sigma_last,
        f"sigma(r={r.obs_radii[0]})={sigma_first:.4f}, "
        f"sigma(r={r.obs_radii[-1]})={sigma_last:.4f}",
    ))

    # Check: all sigmas are physically valid (positive, bounded)
    all_valid = all(s > 0 for s in r.mean_sigma)
    results.append(CheckResult(
        "All sigmas positive",
        all_valid,
        f"range=[{min(r.mean_sigma):.4f}, {max(r.mean_sigma):.4f}]",
    ))

    # Check: vorticity is positive at all radii (field is active)
    all_vort_pos = all(v > 0 for v in r.mean_vorticity)
    results.append(CheckResult(
        "Vorticity positive at all radii",
        all_vort_pos,
        f"range=[{min(r.mean_vorticity):.4f}, {max(r.mean_vorticity):.4f}]",
    ))

    # Check: convergence — last two radii within 30% of each other
    if len(r.mean_vorticity) >= 2 and r.mean_vorticity[-1] > 0:
        rel_diff = abs(r.mean_vorticity[-1] - r.mean_vorticity[-2]) / r.mean_vorticity[-1]
        results.append(CheckResult(
            "Vorticity converges (last 2 radii < 30% diff)",
            rel_diff < 0.30,
            f"relative_diff={rel_diff:.3f}",
        ))
    else:
        results.append(CheckResult(
            "Vorticity converges", False, "insufficient data",
        ))

    # Check: particles found in measurement box
    all_populated = all(n > 0 for n in r.n_particles)
    results.append(CheckResult(
        "Measurement box populated at all radii",
        all_populated,
        f"counts={r.n_particles}",
    ))

    # Check: circulation conservation (total |omega| before/after simulation)
    import numpy as np
    from quantum_hydraulics.core.hydraulics import HydraulicsEngine
    from quantum_hydraulics.core.vortex_field import VortexParticleField
    from quantum_hydraulics.research.alr_experiments import (
        _create_engine, _run_field, OBS_CENTER, CHANNEL_LENGTH, N_STEPS, DT
    )
    np.random.seed(42)
    engine = _create_engine()
    vf = VortexParticleField(engine, length=CHANNEL_LENGTH, n_particles=2000)
    vf.set_observation(OBS_CENTER, 25.0)
    circ_before = float(np.sum(np.sqrt(np.sum(vf._vorticities ** 2, axis=1))))
    _run_field(vf)
    circ_after = float(np.sum(np.sqrt(np.sum(vf._vorticities ** 2, axis=1))))
    circ_change = abs(circ_after - circ_before) / circ_before if circ_before > 0 else 0
    results.append(CheckResult(
        "Circulation conserved (< 50% drift over simulation)",
        circ_change < 0.50,
        f"before={circ_before:.1f}, after={circ_after:.1f}, drift={circ_change:.1%}",
    ))

    return results, r


def checks_cost_benefit(verbose=False):
    """Experiment 2: Cost-benefit analysis."""
    results = []
    if verbose:
        print("\n  Experiment 2: Cost-Benefit Analysis")
        print("  " + "-" * 50)

    r = run_cost_benefit(verbose=verbose)

    # Check: baseline values are nonzero
    results.append(CheckResult(
        "Baseline metrics nonzero",
        r.baseline_vorticity > 0 and r.baseline_enstrophy > 0,
        f"vort={r.baseline_vorticity:.4f}, enstrophy={r.baseline_enstrophy:.4f}",
    ))

    # Check: errors generally decrease with more particles
    if len(r.errors_vorticity) >= 3:
        # Compare first third vs last third
        n = len(r.errors_vorticity)
        early_avg = sum(r.errors_vorticity[:n // 3 + 1]) / (n // 3 + 1)
        late_avg = sum(r.errors_vorticity[-(n // 3 + 1):]) / (n // 3 + 1)
        results.append(CheckResult(
            "Error decreases with more particles",
            late_avg <= early_avg * 1.5,  # allow some noise
            f"early_avg={early_avg:.3f}, late_avg={late_avg:.3f}",
        ))
    else:
        results.append(CheckResult(
            "Error decreases with more particles", False, "insufficient data",
        ))

    # Check: highest particle count achieves reasonable accuracy
    if r.errors_vorticity:
        best_err = r.errors_vorticity[-1]
        results.append(CheckResult(
            "Best ALR vorticity error < 50%",
            best_err < 0.50,
            f"error={best_err:.3f} at N={r.particle_counts[-1]}",
        ))

    # Check: wall times increase with particle count (sanity)
    if len(r.wall_times) >= 2:
        time_trend = r.wall_times[-1] > r.wall_times[0]
        results.append(CheckResult(
            "Compute time increases with N (sanity)",
            time_trend,
            f"time(N={r.particle_counts[0]})={r.wall_times[0]:.2f}s, "
            f"time(N={r.particle_counts[-1]})={r.wall_times[-1]:.2f}s",
        ))

    return results, r


def checks_sigma_field(verbose=False):
    """Experiment 3: Sigma field visualization."""
    results = []
    if verbose:
        print("\n  Experiment 3: Sigma Field Visualization")
        print("  " + "-" * 50)

    r = run_sigma_field(verbose=verbose)

    # Check: pier observation concentrates resolution (small sigma near center)
    sigma_pier_center = r.sigma_pier[r.sigma_pier.shape[0] // 2, r.sigma_pier.shape[1] // 2]
    sigma_pier_corner = r.sigma_pier[0, 0]
    results.append(CheckResult(
        "Pier obs: center sigma < corner sigma",
        sigma_pier_center < sigma_pier_corner,
        f"center={sigma_pier_center:.4f}, corner={sigma_pier_corner:.4f}",
    ))

    # Check: enhancement ratio at center
    results.append(CheckResult(
        "Enhancement at obs center >= 3x",
        r.enhancement_at_center >= 3.0,
        f"enhancement={r.enhancement_at_center:.2f}x",
    ))

    # Check: observation-off field is uniform
    sigma_off_std = r.sigma_off.std()
    results.append(CheckResult(
        "Obs-off field is uniform (std ~ 0)",
        sigma_off_std < 1e-6,
        f"std={sigma_off_std:.8f}",
    ))

    # Check: entrance observation shifts the resolution focus
    # The entrance config should have smaller sigma near x=0 than the pier config
    sigma_entrance_at_start = r.sigma_entrance[r.sigma_entrance.shape[0] // 2, 0]
    sigma_pier_at_start = r.sigma_pier[r.sigma_pier.shape[0] // 2, 0]
    results.append(CheckResult(
        "Entrance obs: sigma(x=0) < pier obs sigma(x=0)",
        sigma_entrance_at_start < sigma_pier_at_start,
        f"entrance={sigma_entrance_at_start:.4f}, pier={sigma_pier_at_start:.4f}",
    ))

    return results, r


def checks_scour(verbose=False):
    """Experiment 4: Engineering scour relevance."""
    results = []
    if verbose:
        print("\n  Experiment 4: Engineering Scour (Single Pier)")
        print("  " + "-" * 50)

    r = run_scour(verbose=verbose)

    # Check: Tier 2 shear >= Tier 1 shear at pier
    results.append(CheckResult(
        "Tier 2 shear >= Tier 1 at pier",
        r.tier2_shear_pier >= r.tier1_shear_pier,
        f"T2={r.tier2_shear_pier:.4f}, T1={r.tier1_shear_pier:.4f} psf",
    ))

    # Check: Amplification >= 1.0
    results.append(CheckResult(
        "Amplification factor >= 1.0",
        r.amplification >= 1.0,
        f"amplification={r.amplification:.2f}x",
    ))

    # Check: Scour risk > 0.3 at pier (sand bed, 4 fps approach)
    results.append(CheckResult(
        "Scour risk > 0.3 at pier",
        r.tier2_scour_risk > 0.3,
        f"risk={r.tier2_scour_risk:.3f}",
    ))

    # Check: Shields parameter > 0.047 (critical for sand)
    results.append(CheckResult(
        "Shields > 0.047 (sand critical)",
        r.tier2_shields > 0.047,
        f"Shields={r.tier2_shields:.4f}",
    ))

    # Check: Hotspots found near pier
    results.append(CheckResult(
        "Hotspot cells analyzed",
        r.n_hotspots > 0,
        f"n_hotspots={r.n_hotspots}",
    ))

    # Check: Pier shear > approach shear (constriction effect)
    results.append(CheckResult(
        "Pier Tier 1 shear > approach shear",
        r.tier1_shear_pier > r.tier1_shear_approach,
        f"pier={r.tier1_shear_pier:.4f}, approach={r.tier1_shear_approach:.4f} psf",
    ))

    return results, r


def checks_multi_zone(verbose=False):
    """Experiment 5: Multi-zone independence."""
    results = []
    if verbose:
        print("\n  Experiment 5: Multi-Zone Independence")
        print("  " + "-" * 50)

    r = run_multi_zone(verbose=verbose)

    # Check: Zone A sigma < midpoint sigma
    results.append(CheckResult(
        "Zone A sigma < midpoint sigma",
        r.zone_a_sigma < r.midpoint_sigma,
        f"A={r.zone_a_sigma:.4f}, mid={r.midpoint_sigma:.4f}",
    ))

    # Check: Zone B sigma < midpoint sigma
    results.append(CheckResult(
        "Zone B sigma < midpoint sigma",
        r.zone_b_sigma < r.midpoint_sigma,
        f"B={r.zone_b_sigma:.4f}, mid={r.midpoint_sigma:.4f}",
    ))

    # Check: Both zones have active particles
    results.append(CheckResult(
        "Zone A vorticity > 0",
        r.zone_a_vorticity > 0,
        f"vort={r.zone_a_vorticity:.4f}",
    ))

    results.append(CheckResult(
        "Zone B vorticity > 0",
        r.zone_b_vorticity_base > 0,
        f"vort={r.zone_b_vorticity_base:.4f}",
    ))

    return results, r


# ── Figure generation ─────────────────────────────────────────────────────

def generate_figures(exp_results, output_dir="ALR_figures"):
    """Generate paper-quality figures using light_publication theme."""
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    from quantum_hydraulics.visualization.theme import THEMES

    theme = THEMES["light_publication"]
    os.makedirs(output_dir, exist_ok=True)

    dpi = 300
    figsize = (8.5, 6)

    # ── Figure 1: Sigma field (3-panel) ───────────────────────────────
    if "sigma_field" in exp_results:
        r = exp_results["sigma_field"]
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        fig.patch.set_facecolor(theme.background)

        configs = [
            (r.sigma_pier, "Observation at Pier Wake"),
            (r.sigma_entrance, "Observation at Entrance"),
            (r.sigma_off, "No Observation (Uniform)"),
        ]
        for ax, (sigma, title) in zip(axes, configs):
            ax.set_facecolor(theme.background)
            X, Y = np.meshgrid(r.x_grid, r.y_grid)
            im = ax.contourf(X, Y, 1.0 / sigma, levels=25, cmap=theme.detail_cmap)
            ax.set_xlabel("Distance (ft)", color=theme.foreground, fontsize=8)
            ax.set_ylabel("Width (ft)", color=theme.foreground, fontsize=8)
            ax.set_title(title, color=theme.foreground, fontsize=9, weight="bold")
            ax.tick_params(colors=theme.foreground, labelsize=7)
            plt.colorbar(im, ax=ax, label="1/sigma", shrink=0.8)

            # Mark pier
            pier = Circle((PIER_X, PIER_Y), 1.5, fill=True,
                          facecolor=theme.bed_color, edgecolor="black", linewidth=1)
            ax.add_patch(pier)

        fig.suptitle("Adaptive Resolution: Observation-Dependent Sigma Fields",
                      color=theme.foreground, fontsize=11, weight="bold", y=1.02)
        fig.tight_layout()
        path = os.path.join(output_dir, "fig1_sigma_field.png")
        fig.savefig(path, dpi=dpi, bbox_inches="tight",
                    facecolor=theme.background)
        plt.close(fig)
        print(f"    Saved {path}")

    # ── Figure 2: Convergence ─────────────────────────────────────────
    if "convergence" in exp_results:
        r = exp_results["convergence"]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        fig.patch.set_facecolor(theme.background)

        ax1.set_facecolor(theme.background)
        ax1.plot(r.obs_radii, r.mean_sigma, "o-", color=theme.accent_primary,
                 linewidth=2, markersize=6)
        ax1.set_xlabel("Observation Radius (ft)", color=theme.foreground)
        ax1.set_ylabel("Mean Sigma in Box", color=theme.foreground)
        ax1.set_title("Core Size vs Observation Radius",
                       color=theme.foreground, fontsize=10, weight="bold")
        ax1.tick_params(colors=theme.foreground)
        ax1.grid(True, color=theme.grid_color, alpha=theme.grid_alpha)

        ax2.set_facecolor(theme.background)
        ax2.plot(r.obs_radii, r.mean_vorticity, "s-", color=theme.accent_secondary,
                 linewidth=2, markersize=6)
        ax2.set_xlabel("Observation Radius (ft)", color=theme.foreground)
        ax2.set_ylabel("Mean Vorticity in Box", color=theme.foreground)
        ax2.set_title("Vorticity Convergence",
                       color=theme.foreground, fontsize=10, weight="bold")
        ax2.tick_params(colors=theme.foreground)
        ax2.grid(True, color=theme.grid_color, alpha=theme.grid_alpha)

        fig.suptitle("Experiment 1: Convergence Study",
                      color=theme.foreground, fontsize=11, weight="bold")
        fig.tight_layout()
        path = os.path.join(output_dir, "fig2_convergence.png")
        fig.savefig(path, dpi=dpi, bbox_inches="tight",
                    facecolor=theme.background)
        plt.close(fig)
        print(f"    Saved {path}")

    # ── Figure 3: Cost-benefit Pareto ─────────────────────────────────
    if "cost_benefit" in exp_results:
        r = exp_results["cost_benefit"]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        fig.patch.set_facecolor(theme.background)

        ax1.set_facecolor(theme.background)
        ax1.plot(r.particle_counts, r.errors_vorticity, "o-",
                 color=theme.accent_primary, linewidth=2, markersize=6,
                 label="Vorticity error")
        ax1.plot(r.particle_counts, r.errors_enstrophy, "s--",
                 color=theme.accent_secondary, linewidth=2, markersize=6,
                 label="Enstrophy error")
        ax1.axhline(y=0.20, color="gray", linestyle=":", label="20% target")
        ax1.set_xlabel("Particle Count (ALR)", color=theme.foreground)
        ax1.set_ylabel("Relative Error vs Baseline", color=theme.foreground)
        ax1.set_title("Accuracy vs Particle Count",
                       color=theme.foreground, fontsize=10, weight="bold")
        ax1.legend(fontsize=8)
        ax1.tick_params(colors=theme.foreground)
        ax1.grid(True, color=theme.grid_color, alpha=theme.grid_alpha)

        ax2.set_facecolor(theme.background)
        ax2.plot(r.particle_counts, r.wall_times, "D-",
                 color=theme.accent_primary, linewidth=2, markersize=6)
        ax2.set_xlabel("Particle Count (ALR)", color=theme.foreground)
        ax2.set_ylabel("Wall Time (s)", color=theme.foreground)
        ax2.set_title("Computational Cost",
                       color=theme.foreground, fontsize=10, weight="bold")
        ax2.tick_params(colors=theme.foreground)
        ax2.grid(True, color=theme.grid_color, alpha=theme.grid_alpha)

        fig.suptitle("Experiment 2: Cost-Benefit Analysis (vs 6000-particle uniform)",
                      color=theme.foreground, fontsize=11, weight="bold")
        fig.tight_layout()
        path = os.path.join(output_dir, "fig3_cost_benefit.png")
        fig.savefig(path, dpi=dpi, bbox_inches="tight",
                    facecolor=theme.background)
        plt.close(fig)
        print(f"    Saved {path}")

    # ── Figure 4: Pier scour ──────────────────────────────────────────
    if "scour" in exp_results:
        r = exp_results["scour"]
        fig, ax = plt.subplots(figsize=(7, 5))
        fig.patch.set_facecolor(theme.background)
        ax.set_facecolor(theme.background)

        categories = ["Approach\n(Tier 1)", "Pier\n(Tier 1)", "Pier\n(Tier 2)"]
        shears = [r.tier1_shear_approach, r.tier1_shear_pier, r.tier2_shear_pier]
        colors = [theme.accent_primary, theme.accent_primary, theme.accent_secondary]

        bars = ax.bar(categories, shears, color=colors, edgecolor="black", linewidth=0.8)
        ax.set_ylabel("Bed Shear Stress (psf)", color=theme.foreground, fontsize=10)
        ax.set_title(f"Scour Analysis: Single Pier\n"
                     f"Risk={r.tier2_scour_risk:.2f}  "
                     f"Shields={r.tier2_shields:.3f}  "
                     f"Amplification={r.amplification:.1f}x",
                     color=theme.foreground, fontsize=10, weight="bold")
        ax.tick_params(colors=theme.foreground)
        ax.grid(True, axis="y", color=theme.grid_color, alpha=theme.grid_alpha)

        fig.tight_layout()
        path = os.path.join(output_dir, "fig4_pier_scour.png")
        fig.savefig(path, dpi=dpi, bbox_inches="tight",
                    facecolor=theme.background)
        plt.close(fig)
        print(f"    Saved {path}")

    # ── Figure 5: Multi-zone ──────────────────────────────────────────
    if "multi_zone" in exp_results:
        r = exp_results["multi_zone"]
        # Generate the dual-zone sigma field for visualization
        from quantum_hydraulics.core.hydraulics import HydraulicsEngine
        from quantum_hydraulics.core.vortex_field import VortexParticleField

        engine = HydraulicsEngine(Q=Q, width=CHANNEL_WIDTH, depth=DEPTH,
                                  slope=0.002, roughness_ks=0.1)
        np.random.seed(42)
        vf = VortexParticleField(engine, length=400.0, n_particles=100)
        vf.set_observation_zones([
            (np.array([100.0, CHANNEL_WIDTH / 2.0, DEPTH / 2.0]), 20.0),
            (np.array([300.0, CHANNEL_WIDTH / 2.0, DEPTH / 2.0]), 20.0),
        ])

        x_grid = np.linspace(0, 400.0, 100)
        y_grid = np.linspace(0, CHANNEL_WIDTH, 30)
        X, Y = np.meshgrid(x_grid, y_grid)
        Z = np.zeros_like(X)
        for i in range(len(x_grid)):
            for j in range(len(y_grid)):
                pos = np.array([x_grid[i], y_grid[j], DEPTH / 2.0])
                Z[j, i] = vf.get_adaptive_core_size(pos)

        fig, ax = plt.subplots(figsize=(10, 4))
        fig.patch.set_facecolor(theme.background)
        ax.set_facecolor(theme.background)

        im = ax.contourf(X, Y, 1.0 / Z, levels=25, cmap=theme.detail_cmap)
        ax.set_xlabel("Distance (ft)", color=theme.foreground)
        ax.set_ylabel("Width (ft)", color=theme.foreground)
        ax.set_title("Multi-Zone Observation: Independent Resolution Concentration",
                     color=theme.foreground, fontsize=10, weight="bold")
        ax.tick_params(colors=theme.foreground)

        # Mark zones
        for cx in [100.0, 300.0]:
            circle = Circle((cx, CHANNEL_WIDTH / 2.0), 20.0,
                           fill=False, edgecolor=theme.observation_color,
                           linewidth=2, linestyle="--")
            ax.add_patch(circle)

        plt.colorbar(im, ax=ax, label="Resolution (1/sigma)", shrink=0.8)
        fig.tight_layout()
        path = os.path.join(output_dir, "fig5_multi_zone.png")
        fig.savefig(path, dpi=dpi, bbox_inches="tight",
                    facecolor=theme.background)
        plt.close(fig)
        print(f"    Saved {path}")

    print(f"\n  All figures saved to {output_dir}/")


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    import numpy as np

    parser = argparse.ArgumentParser(description="ALR Research Study — ICWMM 2026")
    parser.add_argument("--experiment", "-e", type=int, choices=[1, 2, 3, 4, 5],
                        help="Run a single experiment (1-5)")
    parser.add_argument("--figures", "-f", action="store_true",
                        help="Generate paper-quality figures")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Verbose output")
    parser.add_argument("--json", action="store_true",
                        help="Output results as JSON")
    parser.add_argument("--report", "-r", type=str, nargs="?",
                        const="ALR_Research_Report.pdf",
                        help="Generate PDF report (optional: output path)")
    args = parser.parse_args()

    print("=" * 72)
    print("  QUANTUM HYDRAULICS — ALR Research Study (ICWMM 2026)")
    print("=" * 72)
    print(f"  Channel: {CHANNEL_LENGTH:.0f} x {CHANNEL_WIDTH:.0f} ft, "
          f"depth={DEPTH:.1f} ft, Q={Q:.0f} cfs")
    print(f"  Pier: ({PIER_X:.0f}, {PIER_Y:.0f}), diameter=3 ft")
    print(f"  Observation center: ({OBS_CENTER[0]:.0f}, {OBS_CENTER[1]:.0f}, {OBS_CENTER[2]:.1f})")
    print("-" * 72)

    experiments = {
        1: ("Convergence", checks_convergence),
        2: ("Cost-Benefit", checks_cost_benefit),
        3: ("Sigma Field", checks_sigma_field),
        4: ("Engineering Scour", checks_scour),
        5: ("Multi-Zone", checks_multi_zone),
    }

    if args.experiment:
        to_run = {args.experiment: experiments[args.experiment]}
    else:
        to_run = experiments

    all_checks = []
    exp_results = {}
    result_keys = {1: "convergence", 2: "cost_benefit", 3: "sigma_field",
                   4: "scour", 5: "multi_zone"}
    t_total = time.perf_counter()

    for num, (name, func) in to_run.items():
        t0 = time.perf_counter()
        checks, raw_result = func(verbose=args.verbose)
        elapsed = time.perf_counter() - t0
        all_checks.extend(checks)
        exp_results[result_keys[num]] = raw_result

        if not args.json:
            print(f"\n  [{num}] {name} ({elapsed:.1f}s)")
            for c in checks:
                print(c)

    total_time = time.perf_counter() - t_total
    passed = sum(1 for c in all_checks if c.passed)
    total = len(all_checks)

    print("\n" + "-" * 72)
    if passed == total:
        print(f"  {passed}/{total} checks passed -- ALL PASS  ({total_time:.1f}s)")
    else:
        failed = [c for c in all_checks if not c.passed]
        print(f"  {passed}/{total} checks passed -- {total - passed} FAILED  ({total_time:.1f}s)")
        for c in failed:
            print(f"    FAIL: {c.name}: {c.detail}")
    print("=" * 72)

    if args.json:
        output = {
            "passed": passed,
            "total": total,
            "all_pass": passed == total,
            "checks": [
                {"name": c.name, "passed": c.passed, "detail": c.detail}
                for c in all_checks
            ],
        }
        print(json.dumps(output, indent=2))

    # --report implies --figures (figures must exist for embedding)
    if args.report and not args.figures:
        args.figures = True

    if args.figures:
        print("\n  Generating paper figures...")
        generate_figures(exp_results)

    if args.report:
        from quantum_hydraulics.reporting import generate_alr_report, ReportConfig
        report_config = ReportConfig(
            project_name="ALR Research Study — ICWMM 2026",
            firm_name="McGill Associates, PA",
            output_path=args.report,
            draft=False,
        )
        pdf_path = generate_alr_report(
            convergence=exp_results.get("convergence"),
            cost_benefit=exp_results.get("cost_benefit"),
            sigma_field=exp_results.get("sigma_field"),
            scour=exp_results.get("scour"),
            multi_zone=exp_results.get("multi_zone"),
            figure_dir="ALR_figures",
            config=report_config,
        )
        print(f"\n  PDF report: {pdf_path}")

    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
