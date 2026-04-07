"""
Quantum Hydraulics — Engineering Scenario Validation
=====================================================

Four common PE scenarios with pass/fail checks:
  1. Bank Erosion — trapezoidal channel with sloped banks
  2. Bed Degradation — grade change causing transport deficit
  3. Culvert Outlet — jet expansion into receiving channel
  4. Channel Bend — outer bank shear amplification

Usage:
  python run_engineering_scenarios.py --verbose
  python run_engineering_scenarios.py -s 1 --verbose      # bank erosion only
  python run_engineering_scenarios.py --figures --report

Exit code 0 = all checks pass, 1 = failure.
"""

import sys
import os
import json
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from quantum_hydraulics.integration.swmm_2d import SWMM2DPostProcessor
from quantum_hydraulics.integration.swmm_node import SedimentProperties
from quantum_hydraulics.research.engineering_scenarios import (
    generate_bank_erosion_scenario,
    generate_degradation_scenario,
    generate_culvert_outlet_scenario,
    generate_bend_scenario,
)
from quantum_hydraulics.research.engineering_metrics import (
    compute_bank_shear,
    compute_degradation,
    compute_culvert_outlet,
    compute_bend_metrics,
    PERMISSIBLE_SHEAR_PSF,
)


class CheckResult:
    def __init__(self, name, passed, detail, values=None):
        self.name = name
        self.passed = passed
        self.detail = detail
        self.values = values or {}

    def __str__(self):
        tag = "PASS" if self.passed else "FAIL"
        return f"  [{tag}] {self.name}: {self.detail}"


# ── 1. Bank Erosion ───────────────────────────────────────────────────────

def checks_bank_erosion(verbose=False):
    results = []
    if verbose:
        print("\n  Scenario 1: Bank Erosion — Trapezoidal Channel")
        print("  " + "-" * 50)

    processor = SWMM2DPostProcessor(roughness_ks=0.1, cell_size=5.0)
    timesteps, meta = generate_bank_erosion_scenario(processor)

    results.append(CheckResult(
        "Scenario generated",
        len(timesteps) == 3,
        f"{meta['n_cells']} cells, {len(timesteps)} timesteps",
    ))

    # Analyze bankfull
    bankfull = timesteps["bankfull"]
    t1 = bankfull.compute_tier1()

    # Bank cells should have lower depth than channel cells
    bank_depths = bankfull.depth[meta["bank_mask"]]
    chan_depths = bankfull.depth[meta["channel_mask"]]
    wet_bank = bank_depths[bank_depths > 0.05]
    results.append(CheckResult(
        "Bank cells have lower depth than channel",
        wet_bank.mean() < chan_depths.mean() if len(wet_bank) > 0 else False,
        f"bank_mean={wet_bank.mean():.2f}, channel_mean={chan_depths.mean():.2f}" if len(wet_bank) > 0 else "no wet banks",
    ))

    # Bank shear for bare soil
    ba_bare = compute_bank_shear(t1, meta["bank_mask"], bank_material="bare_soil")
    results.append(CheckResult(
        "Bank shear computed (K=0.75)",
        ba_bare.max_bank_shear > 0,
        f"max={ba_bare.max_bank_shear:.4f} psf, mean={ba_bare.mean_bank_shear:.4f} psf",
    ))

    # Bare soil should fail at bankfull (tau_perm = 0.02 psf)
    results.append(CheckResult(
        "Bare soil fails at bankfull (FOS < 1.0)",
        ba_bare.factor_of_safety < 1.0,
        f"FOS={ba_bare.factor_of_safety:.3f}, assessment={ba_bare.assessment}",
    ))

    # Grass (good) should hold at bankfull (tau_perm = 0.35 psf)
    ba_grass = compute_bank_shear(t1, meta["bank_mask"], bank_material="grass_good")
    results.append(CheckResult(
        "Good grass holds at bankfull (FOS >= 1.0)",
        ba_grass.factor_of_safety >= 1.0,
        f"FOS={ba_grass.factor_of_safety:.3f}, assessment={ba_grass.assessment}",
    ))

    # Bank shear increases from low_flow to bankfull
    low_t1 = timesteps["low_flow"].compute_tier1()
    ba_low = compute_bank_shear(low_t1, meta["bank_mask"])
    results.append(CheckResult(
        "Bank shear increases with stage",
        ba_bare.max_bank_shear > ba_low.max_bank_shear,
        f"low={ba_low.max_bank_shear:.4f}, bankfull={ba_bare.max_bank_shear:.4f} psf",
    ))

    if verbose:
        print(f"    Bare soil:  max_shear={ba_bare.max_bank_shear:.4f} psf, "
              f"FOS={ba_bare.factor_of_safety:.3f} -- {ba_bare.assessment}")
        print(f"    Good grass: max_shear={ba_grass.max_bank_shear:.4f} psf, "
              f"FOS={ba_grass.factor_of_safety:.3f} -- {ba_grass.assessment}")

    return results, {"bare": ba_bare, "grass": ba_grass, "low": ba_low, "meta": meta}


# ── 2. Bed Degradation ───────────────────────────────────────────────────

def checks_degradation(verbose=False):
    results = []
    if verbose:
        print("\n  Scenario 2: Bed Degradation — Grade Change")
        print("  " + "-" * 50)

    processor = SWMM2DPostProcessor(roughness_ks=0.1, cell_size=5.0,
                                     sediment=SedimentProperties.sand())
    timesteps, meta = generate_degradation_scenario(processor)

    design = timesteps["design"]
    t1 = design.compute_tier1()

    # Downstream velocity > upstream
    v_up = t1.v_mag[meta["upstream_mask"]].mean()
    v_dn = t1.v_mag[meta["downstream_mask"]].mean()
    results.append(CheckResult(
        "Downstream velocity > upstream",
        v_dn > v_up,
        f"upstream={v_up:.2f} fps, downstream={v_dn:.2f} fps",
    ))

    deg = compute_degradation(t1, meta["upstream_mask"], meta["downstream_mask"],
                               channel_width=meta["width"])

    # Downstream transport > upstream
    results.append(CheckResult(
        "Downstream transport > upstream",
        deg.downstream_transport > deg.upstream_transport,
        f"up={deg.upstream_transport:.6f}, down={deg.downstream_transport:.6f} lb/ft/s",
    ))

    # Deficit is positive (degrading)
    results.append(CheckResult(
        "Transport deficit positive (degrading)",
        deg.transport_deficit > 0,
        f"deficit={deg.transport_deficit:.6f} lb/ft/s",
    ))

    # Assessment is DEGRADING
    results.append(CheckResult(
        "Assessment is DEGRADING",
        deg.assessment == "DEGRADING",
        f"assessment={deg.assessment}",
    ))

    # Annual degradation physically reasonable
    results.append(CheckResult(
        "Annual degradation 0-10 ft/yr",
        0 <= deg.annual_degradation_ft <= 10,
        f"degradation={deg.annual_degradation_ft:.3f} ft/yr",
    ))

    if verbose:
        print(f"    V upstream:  {deg.upstream_mean_v:.2f} fps")
        print(f"    V downstream: {deg.downstream_mean_v:.2f} fps")
        print(f"    Transport deficit: {deg.transport_deficit:.6f} lb/ft/s")
        print(f"    Annual degradation: {deg.annual_degradation_ft:.3f} ft/yr")
        print(f"    Assessment: {deg.assessment}")

    return results, deg


# ── 3. Culvert Outlet ─────────────────────────────────────────────────────

def checks_culvert_outlet(verbose=False):
    results = []
    if verbose:
        print("\n  Scenario 3: Culvert Outlet — Jet Expansion")
        print("  " + "-" * 50)

    processor = SWMM2DPostProcessor(roughness_ks=0.1, cell_size=2.0,
                                     sediment=SedimentProperties.sand())
    timesteps, meta = generate_culvert_outlet_scenario(processor)

    design = timesteps["design"]
    t1 = design.compute_tier1()

    # Jet velocity at outlet > downstream velocity
    jet_v = t1.v_mag[meta["jet_mask"]].max() if meta["jet_mask"].any() else 0
    far_mask = (meta["x"] > 80)
    far_v = t1.v_mag[far_mask].mean() if far_mask.any() else 0
    results.append(CheckResult(
        "Jet velocity > far-field velocity",
        jet_v > far_v,
        f"jet={jet_v:.2f} fps, far={far_v:.2f} fps",
    ))

    co = compute_culvert_outlet(t1, meta["jet_mask"], meta["plunge_mask"],
                                 tailwater_depth=meta["tailwater_depth"])

    # Max shear in plunge zone
    results.append(CheckResult(
        "Plunge shear > 0",
        co.max_plunge_shear > 0,
        f"plunge_shear={co.max_plunge_shear:.4f} psf",
    ))

    # Riprap sizing reasonable
    results.append(CheckResult(
        "Required riprap D50 in range (1-24 in)",
        1.0 <= co.required_riprap_d50_in <= 24.0,
        f"D50={co.required_riprap_d50_in:.1f} inches",
    ))

    # Apron length reasonable
    results.append(CheckResult(
        "Apron length in range (2-30 ft)",
        2.0 <= co.required_apron_length_ft <= 30.0,
        f"apron={co.required_apron_length_ft:.1f} ft",
    ))

    # Scour risk > 0 at plunge
    results.append(CheckResult(
        "Scour risk > 0 at plunge zone",
        co.scour_risk > 0,
        f"risk={co.scour_risk:.3f}",
    ))

    if verbose:
        print(f"    Jet exit velocity: {co.jet_exit_velocity:.2f} fps")
        print(f"    Max plunge shear: {co.max_plunge_shear:.4f} psf")
        print(f"    Required riprap D50: {co.required_riprap_d50_in:.1f} in")
        print(f"    Required apron: {co.required_apron_length_ft:.1f} ft")
        print(f"    Scour risk: {co.scour_risk:.3f}")
        print(f"    Assessment: {co.assessment}")

    return results, co


# ── 4. Channel Bend ───────────────────────────────────────────────────────

def checks_channel_bend(verbose=False):
    results = []
    if verbose:
        print("\n  Scenario 4: Channel Bend — Outer Bank Erosion")
        print("  " + "-" * 50)

    processor = SWMM2DPostProcessor(roughness_ks=0.1, cell_size=5.0)
    timesteps, meta = generate_bend_scenario(processor)

    peak = timesteps["peak"]
    t1 = peak.compute_tier1()

    # Check we have cells in each zone
    n_approach = meta["approach_mask"].sum()
    n_outer = meta["outer_mask"].sum()
    n_inner = meta["inner_mask"].sum()
    results.append(CheckResult(
        "All zones populated",
        n_approach > 0 and n_outer > 0 and n_inner > 0,
        f"approach={n_approach}, outer={n_outer}, inner={n_inner}",
    ))

    # Outer velocity > inner velocity in bend
    v_outer = t1.v_mag[meta["outer_mask"]].mean() if n_outer > 0 else 0
    v_inner = t1.v_mag[meta["inner_mask"]].mean() if n_inner > 0 else 0
    results.append(CheckResult(
        "Outer bank velocity > inner bank velocity",
        v_outer > v_inner,
        f"outer={v_outer:.2f}, inner={v_inner:.2f} fps",
    ))

    ba = compute_bend_metrics(
        t1, meta["approach_mask"], meta["outer_mask"], meta["inner_mask"],
        R_centerline=meta["R"], channel_width=meta["W"], depth=meta["depth"],
    )

    # Amplification > 1.0
    results.append(CheckResult(
        "Outer bank shear amplification > 1.0",
        ba.amplification_factor > 1.0,
        f"amplification={ba.amplification_factor:.3f}x",
    ))

    # Outer shear > approach shear
    results.append(CheckResult(
        "Outer bend shear > approach shear",
        ba.outer_mean_shear > ba.approach_mean_shear,
        f"outer={ba.outer_mean_shear:.4f}, approach={ba.approach_mean_shear:.4f} psf",
    ))

    # Bend scour depth reasonable
    results.append(CheckResult(
        "Bend scour depth >= 0",
        ba.bend_scour_depth_ft >= 0,
        f"scour_depth={ba.bend_scour_depth_ft:.2f} ft",
    ))

    if verbose:
        print(f"    Approach shear: {ba.approach_mean_shear:.4f} psf")
        print(f"    Outer shear:  {ba.outer_mean_shear:.4f} psf")
        print(f"    Inner shear:  {ba.inner_mean_shear:.4f} psf")
        print(f"    Amplification: {ba.amplification_factor:.3f}x")
        print(f"    R/W ratio: {ba.r_over_w:.2f}")
        print(f"    Bend scour: {ba.bend_scour_depth_ft:.2f} ft")
        print(f"    Assessment: {ba.assessment}")

    return results, ba


# ── Figure generation ─────────────────────────────────────────────────────

def generate_figures(exp_results, output_dir="Engineering_figures"):
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from quantum_hydraulics.visualization.theme import THEMES

    theme = THEMES["light_publication"]
    os.makedirs(output_dir, exist_ok=True)
    dpi = 300

    # ── Fig 1: Bank shear plan view ───────────────────────────────────
    if "bank" in exp_results:
        ba = exp_results["bank"]["bare"]
        meta = exp_results["bank"]["meta"]

        fig, ax = plt.subplots(figsize=(8, 4))
        fig.patch.set_facecolor(theme.background)
        ax.set_facecolor(theme.background)

        wet = ba.bank_shear > 0
        sc = ax.scatter(meta["x"][wet], meta["y"][wet],
                        c=ba.bank_shear[wet], cmap=theme.energy_cmap,
                        s=8, edgecolors="none")
        plt.colorbar(sc, ax=ax, label="Bank Shear (psf)")
        ax.axhline(y=0, color="gray", linewidth=0.5, linestyle="--")
        ax.axhline(y=meta["bottom_w"], color="gray", linewidth=0.5, linestyle="--")
        ax.set_xlabel("Distance (ft)", color=theme.foreground)
        ax.set_ylabel("Width (ft)", color=theme.foreground)
        ax.set_title("Bank Shear Stress at Bankfull",
                      color=theme.foreground, fontsize=11, weight="bold")
        ax.tick_params(colors=theme.foreground)
        fig.tight_layout()
        path = os.path.join(output_dir, "fig1_bank_shear_plan.png")
        fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor=theme.background)
        plt.close(fig)
        print(f"    Saved {path}")

    # ── Fig 2: Bank stages bar chart ──────────────────────────────────
    if "bank" in exp_results:
        ba_bare = exp_results["bank"]["bare"]
        ba_low = exp_results["bank"]["low"]

        fig, ax = plt.subplots(figsize=(6, 5))
        fig.patch.set_facecolor(theme.background)
        ax.set_facecolor(theme.background)

        stages = ["Low Flow", "Bankfull"]
        shears = [ba_low.max_bank_shear, ba_bare.max_bank_shear]
        bars = ax.bar(stages, shears, color=[theme.accent_primary, theme.accent_secondary],
                      edgecolor="black", linewidth=0.8)
        ax.axhline(y=PERMISSIBLE_SHEAR_PSF["bare_soil"], color="red",
                    linestyle="--", linewidth=1.5, label="Bare soil (0.02 psf)")
        ax.axhline(y=PERMISSIBLE_SHEAR_PSF["grass_good"], color="green",
                    linestyle="--", linewidth=1.5, label="Good grass (0.35 psf)")
        ax.set_ylabel("Max Bank Shear (psf)", color=theme.foreground)
        ax.set_title("Bank Shear vs Flow Stage",
                      color=theme.foreground, fontsize=11, weight="bold")
        ax.legend(fontsize=9)
        ax.tick_params(colors=theme.foreground)
        ax.grid(True, axis="y", color=theme.grid_color, alpha=theme.grid_alpha)
        fig.tight_layout()
        path = os.path.join(output_dir, "fig2_bank_stages.png")
        fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor=theme.background)
        plt.close(fig)
        print(f"    Saved {path}")

    # ── Fig 3: Degradation profile ────────────────────────────────────
    if "degradation" in exp_results:
        deg = exp_results["degradation"]

        fig, ax = plt.subplots(figsize=(8, 4))
        fig.patch.set_facecolor(theme.background)
        ax.set_facecolor(theme.background)

        x_vals = [0, 225, 275, 500]
        tr_vals = [deg.upstream_transport, deg.upstream_transport,
                   deg.downstream_transport, deg.downstream_transport]
        ax.fill_between(x_vals, tr_vals, alpha=0.3, color=theme.accent_primary)
        ax.plot(x_vals, tr_vals, "o-", color=theme.accent_primary, linewidth=2, markersize=6)
        ax.axvline(x=250, color="gray", linestyle=":", label="Grade break")
        ax.set_xlabel("Distance (ft)", color=theme.foreground)
        ax.set_ylabel("Transport Capacity (lb/ft/s)", color=theme.foreground)
        ax.set_title(f"Bed Degradation: Transport Deficit = {deg.transport_deficit:.6f} lb/ft/s\n"
                      f"Annual Degradation = {deg.annual_degradation_ft:.3f} ft/yr -- {deg.assessment}",
                      color=theme.foreground, fontsize=10, weight="bold")
        ax.legend(fontsize=9)
        ax.tick_params(colors=theme.foreground)
        ax.grid(True, color=theme.grid_color, alpha=theme.grid_alpha)
        fig.tight_layout()
        path = os.path.join(output_dir, "fig3_degradation_profile.png")
        fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor=theme.background)
        plt.close(fig)
        print(f"    Saved {path}")

    # ── Fig 4: Culvert outlet plan view ───────────────────────────────
    if "culvert" in exp_results:
        co = exp_results["culvert"]
        # Re-run scenario to get mesh data for plotting
        proc = SWMM2DPostProcessor(roughness_ks=0.1, cell_size=2.0)
        ts, cmeta = generate_culvert_outlet_scenario(proc)
        t1 = ts["design"].compute_tier1()

        fig, ax = plt.subplots(figsize=(10, 4))
        fig.patch.set_facecolor(theme.background)
        ax.set_facecolor(theme.background)

        wet = t1.v_mag > 0.1
        sc = ax.scatter(cmeta["x"][wet], cmeta["y"][wet],
                        c=t1.v_mag[wet], cmap=theme.velocity_cmap,
                        s=4, edgecolors="none")
        plt.colorbar(sc, ax=ax, label="Velocity (ft/s)")

        # Mark culvert
        from matplotlib.patches import Rectangle
        culv = Rectangle((-10, cmeta["culvert_center"] - cmeta["culvert_half_w"]),
                          10, 2 * cmeta["culvert_half_w"],
                          linewidth=2, edgecolor="black", facecolor="gray", alpha=0.5)
        ax.add_patch(culv)
        ax.set_xlabel("Distance from Outlet (ft)", color=theme.foreground)
        ax.set_ylabel("Width (ft)", color=theme.foreground)
        ax.set_title(f"Culvert Outlet Jet Expansion — V_jet={co.jet_exit_velocity:.1f} fps",
                      color=theme.foreground, fontsize=11, weight="bold")
        ax.tick_params(colors=theme.foreground)
        fig.tight_layout()
        path = os.path.join(output_dir, "fig4_culvert_plan.png")
        fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor=theme.background)
        plt.close(fig)
        print(f"    Saved {path}")

    # ── Fig 5: Culvert centerline shear ───────────────────────────────
    if "culvert" in exp_results:
        proc = SWMM2DPostProcessor(roughness_ks=0.1, cell_size=2.0)
        ts, cmeta = generate_culvert_outlet_scenario(proc)
        t1 = ts["design"].compute_tier1()

        # Centerline cells
        center_mask = np.abs(cmeta["y"] - cmeta["culvert_center"]) < 1.5
        cx = cmeta["x"][center_mask]
        cs = t1.bed_shear[center_mask]
        order = np.argsort(cx)

        fig, ax = plt.subplots(figsize=(8, 4))
        fig.patch.set_facecolor(theme.background)
        ax.set_facecolor(theme.background)
        ax.plot(cx[order], cs[order], "-", color=theme.accent_secondary, linewidth=2)
        ax.axvspan(5, 25, alpha=0.15, color="red", label="Plunge zone")
        ax.axhline(y=0.10, color="gray", linestyle="--", label="Critical shear (sand)")
        ax.set_xlabel("Distance from Outlet (ft)", color=theme.foreground)
        ax.set_ylabel("Bed Shear Stress (psf)", color=theme.foreground)
        ax.set_title("Centerline Bed Shear — Culvert Outlet",
                      color=theme.foreground, fontsize=11, weight="bold")
        ax.legend(fontsize=9)
        ax.tick_params(colors=theme.foreground)
        ax.grid(True, color=theme.grid_color, alpha=theme.grid_alpha)
        fig.tight_layout()
        path = os.path.join(output_dir, "fig5_culvert_scour.png")
        fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor=theme.background)
        plt.close(fig)
        print(f"    Saved {path}")

    # ── Fig 6: Bend shear plan view ───────────────────────────────────
    if "bend" in exp_results:
        proc = SWMM2DPostProcessor(roughness_ks=0.1, cell_size=5.0)
        ts, bmeta = generate_bend_scenario(proc)
        t1 = ts["peak"].compute_tier1()

        fig, ax = plt.subplots(figsize=(7, 7))
        fig.patch.set_facecolor(theme.background)
        ax.set_facecolor(theme.background)

        wet = t1.v_mag > 0.1
        sc = ax.scatter(bmeta["x"][wet], bmeta["y"][wet],
                        c=t1.bed_shear[wet], cmap=theme.energy_cmap,
                        s=10, edgecolors="none")
        plt.colorbar(sc, ax=ax, label="Bed Shear (psf)")
        ax.set_xlabel("X (ft)", color=theme.foreground)
        ax.set_ylabel("Y (ft)", color=theme.foreground)
        ax.set_title(f"Channel Bend Shear — Amplification = "
                      f"{exp_results['bend'].amplification_factor:.2f}x",
                      color=theme.foreground, fontsize=11, weight="bold")
        ax.set_aspect("equal")
        ax.tick_params(colors=theme.foreground)
        fig.tight_layout()
        path = os.path.join(output_dir, "fig6_bend_shear.png")
        fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor=theme.background)
        plt.close(fig)
        print(f"    Saved {path}")

    print(f"\n  All figures saved to {output_dir}/")


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    import numpy as np

    parser = argparse.ArgumentParser(description="Engineering Scenario Validation")
    parser.add_argument("--scenario", "-s", type=int, choices=[1, 2, 3, 4],
                        help="Run a single scenario (1-4)")
    parser.add_argument("--figures", "-f", action="store_true")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--report", "-r", type=str, nargs="?",
                        const="Engineering_Scenarios_Report.pdf")
    args = parser.parse_args()

    print("=" * 72)
    print("  QUANTUM HYDRAULICS — Engineering Scenario Validation")
    print("=" * 72)

    scenarios = {
        1: ("Bank Erosion", checks_bank_erosion),
        2: ("Bed Degradation", checks_degradation),
        3: ("Culvert Outlet", checks_culvert_outlet),
        4: ("Channel Bend", checks_channel_bend),
    }

    if args.scenario:
        to_run = {args.scenario: scenarios[args.scenario]}
    else:
        to_run = scenarios

    all_checks = []
    exp_results = {}
    result_keys = {1: "bank", 2: "degradation", 3: "culvert", 4: "bend"}
    t_total = time.perf_counter()

    for num, (name, func) in to_run.items():
        t0 = time.perf_counter()
        checks, raw = func(verbose=args.verbose)
        elapsed = time.perf_counter() - t0
        all_checks.extend(checks)
        exp_results[result_keys[num]] = raw

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
        out = {"passed": passed, "total": total, "all_pass": passed == total,
               "checks": [{"name": c.name, "passed": c.passed, "detail": c.detail}
                           for c in all_checks]}
        print(json.dumps(out, indent=2))

    if args.report and not args.figures:
        args.figures = True

    if args.figures:
        print("\n  Generating figures...")
        generate_figures(exp_results)

    if args.report:
        from quantum_hydraulics.reporting import generate_engineering_report, ReportConfig
        config = ReportConfig(
            project_name="Engineering Scenario Assessment",
            firm_name="McGill Associates, PA",
            output_path=args.report,
            draft=True,
        )
        pdf = generate_engineering_report(
            bank_erosion=exp_results.get("bank"),
            degradation=exp_results.get("degradation"),
            culvert_outlet=exp_results.get("culvert"),
            bend=exp_results.get("bend"),
            figure_dir="Engineering_figures",
            config=config,
        )
        print(f"\n  PDF report: {pdf}")

    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
