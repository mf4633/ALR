"""
JWMM Worked Example -- Turbulence-Aware Post-Processing of EPA SWMM Output
=========================================================================

A self-contained, reproducible worked example for the SWMM post-processing
paper. Reads an EPA SWMM model, runs the vortex-particle (ALR) turbulence
screen at every junction, and produces a ranked results table + a
publication-quality figure.

Why this script exists (vs. run_headless_swmm.py):
  * It reads SWMM output through ``swmm.toolkit`` *directly*, so it does NOT
    depend on ``pyswmm`` -- whose legacy ``julian`` dependency fails to build
    on current Linux/macOS toolchains. ``pip install swmm-toolkit matplotlib``
    is all that is required, and it runs the SWMM engine here too, so the
    result is byte-for-byte reproducible on any machine.
  * It reports the *continuous, discriminating* hydraulic screen (bed shear,
    Shields parameter, velocity, Meyer-Peter-Muller scour-depth potential)
    rather than only the logistic scour-risk index, which saturates to 1.0
    for any node in a typical urban conduit network (see PAPER_SWMM_EXAMPLE.md).

Usage:
    python run_swmm_example.py                       # bundled test model
    python run_swmm_example.py path/to/model.inp     # run + analyze any model
    python run_swmm_example.py path/to/model.out     # analyze existing output
    python run_swmm_example.py --bed gravel          # override bed material
    python run_swmm_example.py --no-figure           # table only

Outputs (written next to the input file's directory, or --outdir):
    swmm_example_table.md   -- ranked results table (Markdown)
    swmm_example_table.csv  -- same, machine-readable
    swmm_example.png        -- two-panel publication figure
"""

import os
import sys
import csv
import argparse

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from swmm.toolkit import solver, output
from swmm.toolkit.shared_enum import ElementType, NodeAttribute, Time

from quantum_hydraulics.integration.swmm_node import QuantumNode, SedimentProperties


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_INP = os.path.join(SCRIPT_DIR, "QuantumTest_Simple.inp")

N_PARTICLES = 300
MIN_DEPTH = 0.1   # ft
MIN_INFLOW = 0.1  # cfs

# SWMM cross-section shapes -> which geom field is the effective width
WIDTH_FROM_GEOM2 = {"RECT_OPEN", "RECT_CLOSED", "TRAPEZOIDAL", "IRREGULAR"}
WIDTH_FROM_GEOM1 = {"CIRCULAR", "FORCE_MAIN", "FILLED_CIRCULAR"}

BED_MATERIALS = {
    "fine_sand": SedimentProperties.fine_sand,
    "sand": SedimentProperties.sand,
    "coarse_sand": SedimentProperties.coarse_sand,
    "gravel": SedimentProperties.gravel,
    "silt": SedimentProperties.silt,
    "clay": SedimentProperties.clay,
}


# ── SWMM .inp geometry parser (topology + cross-sections) ────────────────────

def parse_inp(inp_path):
    """Return (link_topo, xsections) from a SWMM .inp file.

    link_topo : {link_id: {"from": node, "to": node}}
    xsections : {link_id: {"shape": str, "width": float}}
    """
    link_topo, xsections = {}, {}
    if not inp_path or not os.path.exists(inp_path):
        return link_topo, xsections

    section = None
    with open(inp_path) as f:
        for line in f:
            s = line.strip()
            if s.startswith("[") and s.endswith("]"):
                section = s.upper()
                continue
            if not s or s.startswith(";"):
                continue
            parts = s.split()
            if section == "[CONDUITS]" and len(parts) >= 3:
                link_topo[parts[0]] = {"from": parts[1], "to": parts[2]}
            elif section == "[XSECTIONS]" and len(parts) >= 4:
                shape = parts[1].upper()
                geom1, geom2 = float(parts[2]), float(parts[3])
                if shape in WIDTH_FROM_GEOM2:
                    width = geom2
                elif shape in WIDTH_FROM_GEOM1:
                    width = geom1
                else:
                    width = max(geom1, geom2)
                xsections[parts[0]] = {"shape": shape, "width": width}
    return link_topo, xsections


# ── SWMM .out reader (swmm-toolkit direct; no pyswmm dependency) ──────────────

def read_swmm_output(out_path):
    """Read per-node peak depth/inflow and per-link peak velocity from a .out.

    Returns
    -------
    nodes : dict {node_id: {"depth": [...], "inflow": [...]}}
    link_peak_vel : dict {link_id: peak_velocity_fps}
    """
    from swmm.toolkit.shared_enum import LinkAttribute

    handle = output.init()
    try:
        output.open(handle, out_path)
        n_periods = output.get_times(handle, Time.NUM_PERIODS)
        last = n_periods - 1
        counts = output.get_proj_size(handle)  # [subcatch, node, link, pollut, ...]
        n_nodes, n_links = counts[1], counts[2]

        nodes = {}
        for i in range(n_nodes):
            name = output.get_elem_name(handle, ElementType.NODE, i)
            depth = list(output.get_node_series(
                handle, i, NodeAttribute.INVERT_DEPTH, 0, last))
            inflow = list(output.get_node_series(
                handle, i, NodeAttribute.TOTAL_INFLOW, 0, last))
            nodes[name] = {"depth": depth, "inflow": inflow}

        link_peak_vel = {}
        for i in range(n_links):
            name = output.get_elem_name(handle, ElementType.LINK, i)
            vel = output.get_link_series(
                handle, i, LinkAttribute.FLOW_VELOCITY, 0, last)
            link_peak_vel[name] = max(vel) if vel else 0.0
    finally:
        output.close(handle)
    return nodes, link_peak_vel


# ── Node geometry / incoming-velocity assignment ─────────────────────────────

def build_node_geometry(node_ids, link_topo, xsections, link_peak_vel):
    """For each node, find its worst incoming link's width, shape, velocity."""
    incoming, outgoing = {}, {}
    for lid, topo in link_topo.items():
        xs = xsections.get(lid, {})
        entry = (lid, link_peak_vel.get(lid, 0.0),
                 xs.get("width", 5.0), xs.get("shape", "UNKNOWN"))
        incoming.setdefault(topo["to"], []).append(entry)
        outgoing.setdefault(topo["from"], []).append(entry)

    geom = {}
    for nid in node_ids:
        links = incoming.get(nid) or outgoing.get(nid)
        if links:
            lid, vel, width, shape = max(links, key=lambda e: e[1])
            geom[nid] = {"link": lid, "vel": vel, "width": width, "shape": shape}
        else:
            geom[nid] = {"link": "?", "vel": 0.0, "width": 5.0, "shape": "?"}
    return geom


# ── The screen ───────────────────────────────────────────────────────────────

def screen_nodes(nodes, geom, sediment, verbose=False):
    """Run the vortex-particle turbulence screen over every node's timeseries.

    Returns a list of per-node result dicts, ranked by peak bed shear.
    """
    results = []
    for nid, ts in nodes.items():
        g = geom[nid]
        if g["vel"] <= 0 and max(ts["inflow"], default=0) < MIN_INFLOW:
            continue  # dry / no meaningful flow

        qnode = QuantumNode(
            node_id=nid, width=g["width"], length=30.0,
            roughness_ks=0.10, sediment=sediment,
        )

        peak = {"depth": 0.0, "inflow": 0.0, "shear": 0.0, "shields": 0.0,
                "excess": 0.0, "velocity": 0.0, "risk": 0.0, "scour_depth": 0.0}
        n_analyzed = 0
        for depth, inflow in zip(ts["depth"], ts["inflow"]):
            peak["depth"] = max(peak["depth"], depth)
            peak["inflow"] = max(peak["inflow"], inflow)
            if depth > MIN_DEPTH and inflow > MIN_INFLOW:
                qnode.update_from_swmm(depth, inflow)
                qnode.compute_turbulence(n_particles=N_PARTICLES)
                m = qnode.metrics
                peak["shear"] = max(peak["shear"], m.bed_shear_stress)
                peak["shields"] = max(peak["shields"], m.shields_parameter)
                peak["excess"] = max(peak["excess"], m.excess_shear_ratio)
                peak["velocity"] = max(peak["velocity"], m.max_velocity)
                peak["risk"] = max(peak["risk"], m.scour_risk_index)
                peak["scour_depth"] = max(peak["scour_depth"],
                                          m.scour_depth_potential)
                n_analyzed += 1

        rec = {"node": nid, "width": g["width"], "shape": g["shape"],
               "incoming_link": g["link"], "n_analyzed": n_analyzed, **peak}
        results.append(rec)
        if verbose:
            print(f"  {nid:<6} tau={peak['shear']:.3f} psf  "
                  f"Shields={peak['shields']:.2f}  v={peak['velocity']:.2f} fps  "
                  f"scour_depth={peak['scour_depth']:.2f} ft/yr")

    results.sort(key=lambda r: r["shear"], reverse=True)
    return results


def classify(shields):
    """Physical screening class from the Shields parameter (incipient ~0.03-0.06)."""
    if shields > 1.0:
        return "SEVERE"
    if shields > 0.25:
        return "HIGH"
    if shields > 0.06:
        return "MODERATE"
    if shields > 0.03:
        return "INCIPIENT"
    return "STABLE"


# ── Outputs ──────────────────────────────────────────────────────────────────

def write_table(results, sediment, outdir):
    headers = ["Rank", "Node", "Width_ft", "Peak_Q_cfs", "Depth_ft",
               "V_max_fps", "BedShear_psf", "Shields", "ScourDepth_ft_yr",
               "RiskIndex", "Screen"]
    rows = []
    for i, r in enumerate(results, 1):
        rows.append([
            i, r["node"], f"{r['width']:.1f}", f"{r['inflow']:.1f}",
            f"{r['depth']:.2f}", f"{r['velocity']:.2f}", f"{r['shear']:.3f}",
            f"{r['shields']:.2f}", f"{r['scour_depth']:.2f}",
            f"{r['risk']:.3f}", classify(r["shields"]),
        ])

    csv_path = os.path.join(outdir, "swmm_example_table.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(headers)
        w.writerows(rows)

    md_path = os.path.join(outdir, "swmm_example_table.md")
    with open(md_path, "w") as f:
        f.write(f"# SWMM Turbulence Screen -- bed material: {sediment.name}\n\n")
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("|" + "|".join(["---"] * len(headers)) + "|\n")
        for row in rows:
            f.write("| " + " | ".join(str(c) for c in row) + " |\n")
        f.write(
            "\n*Nodes ranked by peak bed shear. The Shields parameter is the "
            "physical incipient-motion discriminator (motion begins near "
            "0.03-0.06). RiskIndex is the logistic scour-severity flag; it "
            "saturates to 1.0 above the design shear and is a screening flag, "
            "not a ranking.*\n")
    return md_path, csv_path


def make_figure(results, nodes, sediment, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [r["node"] for r in results]
    shear = [r["shear"] for r in results]
    shields = [r["shields"] for r in results]
    risk = [r["risk"] for r in results]
    x = np.arange(len(labels))

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.2))

    # Panel A: inflow hydrographs driving the screen (from the SWMM run)
    for nid, ts in nodes.items():
        if max(ts["inflow"], default=0) >= MIN_INFLOW:
            axL.plot(ts["inflow"], lw=1.5, label=nid)
    axL.set_xlabel("Reporting step")
    axL.set_ylabel("Total inflow (cfs)")
    axL.set_title("(a) SWMM node inflow hydrographs")
    axL.legend(fontsize=8, ncol=2, frameon=False)
    axL.grid(alpha=0.3)

    # Panel B: ranked bed shear + Shields, with the saturating risk index
    bars = axR.bar(x, shear, color="#4C72B0", alpha=0.85, label="Peak bed shear")
    axR.set_ylabel("Peak bed shear (psf)", color="#4C72B0")
    axR.tick_params(axis="y", labelcolor="#4C72B0")
    axR.set_xticks(x)
    axR.set_xticklabels(labels)
    axR.set_title("(b) Turbulence screen, ranked")

    ax2 = axR.twinx()
    ax2.plot(x, shields, "o-", color="#C44E52", label="Shields parameter")
    ax2.plot(x, risk, "s--", color="#8C8C8C",
             label="Risk index (saturates)")
    ax2.axhline(0.06, color="#55A868", ls=":", lw=1.2,
                label="Incipient motion (~0.06)")
    ax2.set_ylabel("Shields param.  /  risk index", color="#C44E52")
    ax2.tick_params(axis="y", labelcolor="#C44E52")

    lines1, lab1 = axR.get_legend_handles_labels()
    lines2, lab2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, lab1 + lab2, fontsize=7.5,
               loc="upper right", frameon=False)

    fig.suptitle(
        f"Turbulence-aware SWMM screen (bed = {sediment.name})",
        fontsize=12, y=1.02)
    fig.tight_layout()
    path = os.path.join(outdir, "swmm_example.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", nargs="?", default=DEFAULT_INP,
                    help="SWMM .inp (will be run) or .out (read directly)")
    ap.add_argument("--bed", default="sand", choices=sorted(BED_MATERIALS),
                    help="Bed material for the scour screen (default: sand)")
    ap.add_argument("--outdir", default=None,
                    help="Directory for outputs (default: ./swmm_example_output)")
    ap.add_argument("--no-figure", action="store_true")
    ap.add_argument("--verbose", "-v", action="store_true")
    args = ap.parse_args()

    model = os.path.abspath(args.model)
    if not os.path.exists(model):
        sys.exit(f"ERROR: model not found: {model}")

    outdir = args.outdir or os.path.join(SCRIPT_DIR, "swmm_example_output")
    os.makedirs(outdir, exist_ok=True)
    sediment = BED_MATERIALS[args.bed]()

    # Resolve .inp / .out. If given an .inp, run the SWMM engine to produce a
    # version-matched .out (avoids "Error 440" from stale/foreign .out files).
    if model.lower().endswith(".inp"):
        inp_path = model
        out_path = os.path.join(outdir, os.path.splitext(
            os.path.basename(model))[0] + ".out")
        rpt_path = os.path.splitext(out_path)[0] + ".rpt"
        print(f"Running SWMM engine: {os.path.basename(inp_path)} ...")
        solver.swmm_run(inp_path, rpt_path, out_path)
    else:
        out_path = model
        inp_path = os.path.splitext(model)[0] + ".inp"

    link_topo, xsections = parse_inp(inp_path)
    if not link_topo:
        print("WARNING: no .inp geometry found -- widths default to 5 ft.")

    nodes, link_peak_vel = read_swmm_output(out_path)
    geom = build_node_geometry(list(nodes), link_topo, xsections, link_peak_vel)

    print(f"\nScreening {len(nodes)} nodes on a {sediment.name} bed "
          f"({N_PARTICLES} particles/node)...")
    results = screen_nodes(nodes, geom, sediment, verbose=args.verbose)
    if not results:
        sys.exit("ERROR: no nodes with analyzable flow.")

    md_path, csv_path = write_table(results, sediment, outdir)

    # Console summary
    print("\n" + "=" * 74)
    print(f"{'Rank':<5}{'Node':<7}{'V_max':>8}{'Shear':>9}{'Shields':>9}"
          f"{'Scour':>9}{'Risk':>7}  Screen")
    print(f"{'':5}{'':7}{'(fps)':>8}{'(psf)':>9}{'':9}{'(ft/yr)':>9}{'':7}")
    print("-" * 74)
    for i, r in enumerate(results, 1):
        print(f"{i:<5}{r['node']:<7}{r['velocity']:>8.2f}{r['shear']:>9.3f}"
              f"{r['shields']:>9.2f}{r['scour_depth']:>9.2f}{r['risk']:>7.3f}"
              f"  {classify(r['shields'])}")
    print("=" * 74)
    print(f"\nTable written: {md_path}")
    print(f"               {csv_path}")

    if not args.no_figure:
        try:
            fig_path = make_figure(results, nodes, sediment, outdir)
            print(f"Figure written: {fig_path}")
        except Exception as e:  # matplotlib optional
            print(f"(figure skipped: {e})")

    # Honest headline the paper should use
    n_sat = sum(1 for r in results if r["risk"] >= 0.999)
    print(f"\nNote: {n_sat}/{len(results)} nodes saturate the logistic risk "
          f"index at 1.0; the Shields parameter and bed shear provide the "
          f"discriminating rank (worst: {results[0]['node']}).")


if __name__ == "__main__":
    main()
