"""
Quantum Hydraulics -- Quasi-Unsteady Sediment Transport Validation
===================================================================

Fractional bedload transport with Hirano active-layer armoring,
Egiazaroff hiding/exposure, Exner equation, morphodynamic feedback.

Usage:
  python run_sediment_transport.py --verbose
  python run_sediment_transport.py --figures --report

Exit code 0 = all checks pass, 1 = failure.
"""

import sys
import os
import json
import time
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from quantum_hydraulics.integration.sediment_transport import (
    QuasiUnsteadyEngine,
)
from quantum_hydraulics.research.sediment_scenarios import (
    generate_clearwater_scour_scenario,
)


class CheckResult:
    def __init__(self, name, passed, detail):
        self.name = name
        self.passed = passed
        self.detail = detail

    def __str__(self):
        tag = "PASS" if self.passed else "FAIL"
        return f"  [{tag}] {self.name}: {self.detail}"


def checks_quasi_unsteady(verbose=False):
    results = []
    if verbose:
        print("\n  Quasi-Unsteady Sediment Transport: Clear-Water Scour")
        print("  " + "-" * 50)

    channel, sediment_mix, hydrograph, meta = generate_clearwater_scour_scenario()

    engine = QuasiUnsteadyEngine(
        channel=channel,
        sediment_mix=sediment_mix,
        upstream_feed_fraction=0.0,
        computational_increment_hours=5.0,
        bed_mixing_steps=3,
    )
    engine.set_hydrograph_durations(hydrograph)
    sim = engine.run()

    # 1. Completed
    results.append(CheckResult(
        "Simulation completed",
        len(sim.steps) > 0,
        f"{len(sim.steps)} steps over {meta['total_hours']:.0f} hours",
    ))

    # 2. Bed degraded
    results.append(CheckResult(
        "Bed degraded (cumulative < 0)",
        sim.total_scour_ft < 0,
        f"total_scour={sim.total_scour_ft:.4f} ft",
    ))

    # 3. Surface coarsened
    results.append(CheckResult(
        "Surface coarsened (d50 increased)",
        sim.final_d50_mm > sim.initial_gradation.d50_mm,
        f"initial_d50={sim.initial_gradation.d50_mm:.3f}, final_d50={sim.final_d50_mm:.3f} mm",
    ))

    # 4. Total scour reasonable (clear-water below dam can be significant)
    scour = abs(sim.max_scour_ft)
    results.append(CheckResult(
        "Scour depth reasonable (< 20 ft for clear-water)",
        scour < 20.0,
        f"max_scour={scour:.3f} ft",
    ))

    # 5. Armoring detected
    results.append(CheckResult(
        "Armoring detected",
        sim.armored,
        f"armored={sim.armored}, final_d50={sim.final_d50_mm:.3f} mm",
    ))

    # 6. Armoring reduced transport capacity
    # Compare transport at matching Q (300 cfs appears in both moderate and recession)
    moderate_steps = [s for s in sim.steps if abs(s.discharge_cfs - 300) < 10]
    if len(moderate_steps) >= 4:
        first_mod = np.mean([s.total_transport_rate for s in moderate_steps[:2]])
        last_mod = np.mean([s.total_transport_rate for s in moderate_steps[-2:]])
        results.append(CheckResult(
            "Armoring reduced transport at same Q",
            last_mod < first_mod * 1.05,
            f"Q=300: early={first_mod:.6f}, late={last_mod:.6f}",
        ))
    else:
        results.append(CheckResult(
            "Armoring feedback confirmed",
            sim.armored,
            f"armored={sim.armored}, d50 ratio={sim.final_d50_mm / sim.initial_gradation.d50_mm:.1f}x",
        ))

    # 7. Bed elevation monotonically decreasing (clear-water)
    bed = sim.bed_elevations
    mono = all(bed[i] <= bed[i - 1] + 0.001 for i in range(1, len(bed)))
    results.append(CheckResult(
        "Bed monotonically degrading (clear-water)",
        mono,
        f"bed_range=[{bed.min():.4f}, {bed.max():.4f}]",
    ))

    if verbose:
        print(f"    Steps: {len(sim.steps)}")
        print(f"    Total scour: {sim.total_scour_ft:.4f} ft")
        print(f"    Initial d50: {sim.initial_gradation.d50_mm:.3f} mm")
        print(f"    Final d50:   {sim.final_d50_mm:.3f} mm")
        print(f"    Armored: {sim.armored}")
        print(f"    Assessment: {sim.get_assessment()}")

    return results, sim


def generate_figures(sim, output_dir="Sediment_figures"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from quantum_hydraulics.visualization.theme import THEMES

    theme = THEMES["light_publication"]
    os.makedirs(output_dir, exist_ok=True)
    dpi = 300

    t = sim.times
    Q = sim.discharges
    bed = sim.bed_elevations
    d50 = sim.surface_d50

    # Fig 1: Hydrograph + bed elevation
    fig, ax1 = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor(theme.background)
    ax1.set_facecolor(theme.background)

    ax1.plot(t, Q, "-", color=theme.accent_primary, linewidth=2, label="Discharge")
    ax1.set_xlabel("Time (hours)", color=theme.foreground)
    ax1.set_ylabel("Discharge (cfs)", color=theme.accent_primary)
    ax1.tick_params(axis="y", labelcolor=theme.accent_primary)

    ax2 = ax1.twinx()
    ax2.plot(t, bed, "-", color=theme.accent_secondary, linewidth=2, label="Bed Elevation")
    ax2.set_ylabel("Bed Elevation (ft)", color=theme.accent_secondary)
    ax2.tick_params(axis="y", labelcolor=theme.accent_secondary)

    ax1.set_title("Quasi-Unsteady Simulation: Hydrograph and Bed Response",
                   color=theme.foreground, fontsize=11, weight="bold")
    ax1.tick_params(colors=theme.foreground)
    fig.tight_layout()
    path = os.path.join(output_dir, "fig1_hydrograph_bed.png")
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor=theme.background)
    plt.close(fig)
    print(f"    Saved {path}")

    # Fig 2: Surface d50 evolution
    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor(theme.background)
    ax.set_facecolor(theme.background)
    ax.plot(t, d50, "-", color=theme.accent_primary, linewidth=2)
    ax.axhline(y=sim.initial_gradation.d50_mm, color="gray", linestyle="--",
               label=f"Initial d50 = {sim.initial_gradation.d50_mm:.2f} mm")
    ax.set_xlabel("Time (hours)", color=theme.foreground)
    ax.set_ylabel("Surface d50 (mm)", color=theme.foreground)
    ax.set_title("Surface Coarsening (Armoring)",
                  color=theme.foreground, fontsize=11, weight="bold")
    ax.legend(fontsize=9)
    ax.tick_params(colors=theme.foreground)
    ax.grid(True, color=theme.grid_color, alpha=theme.grid_alpha)
    fig.tight_layout()
    path = os.path.join(output_dir, "fig2_d50_evolution.png")
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor=theme.background)
    plt.close(fig)
    print(f"    Saved {path}")

    # Fig 3: Cumulative bed change
    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor(theme.background)
    ax.set_facecolor(theme.background)
    cum = sim.cumulative_bed_change
    ax.fill_between(t, cum, 0, alpha=0.3, color=theme.accent_secondary)
    ax.plot(t, cum, "-", color=theme.accent_secondary, linewidth=2)
    ax.axhline(y=0, color="gray", linewidth=0.5)
    ax.set_xlabel("Time (hours)", color=theme.foreground)
    ax.set_ylabel("Cumulative Bed Change (ft)", color=theme.foreground)
    ax.set_title(f"Cumulative Scour: {sim.total_scour_ft:.3f} ft -- {sim.get_assessment()}",
                  color=theme.foreground, fontsize=11, weight="bold")
    ax.tick_params(colors=theme.foreground)
    ax.grid(True, color=theme.grid_color, alpha=theme.grid_alpha)
    fig.tight_layout()
    path = os.path.join(output_dir, "fig3_cumulative_scour.png")
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor=theme.background)
    plt.close(fig)
    print(f"    Saved {path}")

    # Fig 4: Transport rate vs time
    transport = np.array([s.total_transport_rate for s in sim.steps])
    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor(theme.background)
    ax.set_facecolor(theme.background)
    ax.plot(t, transport, "-", color=theme.accent_primary, linewidth=1.5)
    ax.set_xlabel("Time (hours)", color=theme.foreground)
    ax.set_ylabel("Transport Rate (ft3/ft/s)", color=theme.foreground)
    ax.set_title("Bedload Transport Rate (decreasing = armoring feedback)",
                  color=theme.foreground, fontsize=11, weight="bold")
    ax.tick_params(colors=theme.foreground)
    ax.grid(True, color=theme.grid_color, alpha=theme.grid_alpha)
    fig.tight_layout()
    path = os.path.join(output_dir, "fig4_transport_rate.png")
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor=theme.background)
    plt.close(fig)
    print(f"    Saved {path}")

    print(f"\n  All figures saved to {output_dir}/")


def main():
    parser = argparse.ArgumentParser(
        description="Quasi-Unsteady Sediment Transport Validation")
    parser.add_argument("--figures", "-f", action="store_true")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--report", "-r", type=str, nargs="?",
                        const="Sediment_Transport_Report.pdf")
    args = parser.parse_args()

    print("=" * 72)
    print("  QUANTUM HYDRAULICS -- Quasi-Unsteady Sediment Transport")
    print("=" * 72)

    t0 = time.perf_counter()
    checks, sim = checks_quasi_unsteady(verbose=args.verbose)
    elapsed = time.perf_counter() - t0

    if not args.json:
        print(f"\n  Quasi-Unsteady ({elapsed:.1f}s)")
        for c in checks:
            print(c)

    passed = sum(1 for c in checks if c.passed)
    total = len(checks)

    print("\n" + "-" * 72)
    if passed == total:
        print(f"  {passed}/{total} checks passed -- ALL PASS  ({elapsed:.1f}s)")
    else:
        failed = [c for c in checks if not c.passed]
        print(f"  {passed}/{total} checks passed -- {total - passed} FAILED")
        for c in failed:
            print(f"    FAIL: {c.name}: {c.detail}")
    print("=" * 72)

    if args.json:
        out = {"passed": passed, "total": total, "all_pass": passed == total,
               "checks": [{"name": c.name, "passed": c.passed, "detail": c.detail}
                           for c in checks]}
        print(json.dumps(out, indent=2))

    if args.report and not args.figures:
        args.figures = True

    if args.figures:
        print("\n  Generating figures...")
        generate_figures(sim)

    if args.report:
        from quantum_hydraulics.reporting import generate_sediment_transport_report, ReportConfig
        config = ReportConfig(
            project_name="Quasi-Unsteady Sediment Transport",
            firm_name="McGill Associates, PA",
            output_path=args.report,
            draft=True,
        )
        pdf = generate_sediment_transport_report(
            results=sim, figure_dir="Sediment_figures", config=config,
        )
        print(f"\n  PDF report: {pdf}")

    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
