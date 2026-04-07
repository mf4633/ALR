"""
Quantum Hydraulics — SWMM Output Post-Processor
================================================

Pure post-processor: reads an existing EPA SWMM .out file,
auto-detects the critical junctions, and runs vortex particle
turbulence analysis where it matters most.

Auto-discovery workflow:
  1. Scan all links for peak velocity from the .out file
  2. Parse the .inp file for cross-section geometry and link topology
  3. Rank nodes by worst incoming link velocity
  4. Run Quantum analysis on the top nodes (or all above threshold)

Point it at any .out file from PCSWMM, EPA SWMM, or PySWMM.

Usage:
  python run_headless_swmm.py                              # default test model
  python run_headless_swmm.py path/to/project.out          # any SWMM output
  python run_headless_swmm.py model.out --top 5            # analyze top 5 nodes
  python run_headless_swmm.py model.out --nodes J1,J3      # specific nodes only
  python run_headless_swmm.py model.out --all              # every junction
  python run_headless_swmm.py model.out --verbose --json

Exit code 0 = all checks pass, 1 = failure.
"""

import sys
import os
import json
import time
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pyswmm import Output
from swmm.toolkit.shared_enum import NodeAttribute, LinkAttribute
from quantum_hydraulics.integration.swmm_node import QuantumNode, SedimentProperties


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT = os.path.join(SCRIPT_DIR, "QuantumTest_Simple.out")

N_PARTICLES = 300
MIN_DEPTH = 0.1   # ft
MIN_INFLOW = 0.1  # cfs

# SWMM cross-section shapes where geom2 = width
WIDTH_FROM_GEOM2 = {
    "RECT_OPEN", "RECT_CLOSED", "TRAPEZOIDAL", "IRREGULAR",
}
# Shapes where geom1 = diameter (width = geom1)
WIDTH_FROM_GEOM1 = {
    "CIRCULAR", "FORCE_MAIN", "FILLED_CIRCULAR",
}


class CheckResult:
    def __init__(self, name, passed, detail, values=None):
        self.name = name
        self.passed = passed
        self.detail = detail
        self.values = values or {}

    def __str__(self):
        tag = "PASS" if self.passed else "FAIL"
        return f"  [{tag}] {self.name}: {self.detail}"


# ── .inp parser ─────────────────────────────────────────────────────────────

def parse_inp(inp_path):
    """
    Parse SWMM .inp file for link topology and cross-section geometry.

    Returns
    -------
    link_topo : dict
        {link_id: {"from": node_id, "to": node_id}}
    xsections : dict
        {link_id: {"shape": str, "geom1": float, "geom2": float, "width": float}}
    """
    link_topo = {}
    xsections = {}

    if not os.path.exists(inp_path):
        return link_topo, xsections

    with open(inp_path) as f:
        lines = f.readlines()

    current_section = None

    for line in lines:
        stripped = line.strip()

        # Track section headers
        if stripped.startswith("[") and stripped.endswith("]"):
            current_section = stripped.upper()
            continue

        # Skip comments and blanks
        if not stripped or stripped.startswith(";;"):
            continue

        parts = stripped.split()

        if current_section == "[CONDUITS]" and len(parts) >= 3:
            link_id, from_node, to_node = parts[0], parts[1], parts[2]
            link_topo[link_id] = {"from": from_node, "to": to_node}

        elif current_section == "[XSECTIONS]" and len(parts) >= 4:
            link_id = parts[0]
            shape = parts[1].upper()
            geom1 = float(parts[2])
            geom2 = float(parts[3])

            # Determine effective width
            if shape in WIDTH_FROM_GEOM2:
                width = geom2
            elif shape in WIDTH_FROM_GEOM1:
                width = geom1
            else:
                # Fallback: use the larger dimension
                width = max(geom1, geom2)

            xsections[link_id] = {
                "shape": shape,
                "geom1": geom1,
                "geom2": geom2,
                "width": width,
            }

    return link_topo, xsections


# ── Auto-discovery ──────────────────────────────────────────────────────────

def discover_critical_nodes(out_file, inp_file=None, top_n=None,
                             requested_nodes=None, analyze_all=False, verbose=False):
    """
    Auto-discover which nodes to analyze and their channel widths.

    Strategy:
      1. Read peak velocity at every link from .out
      2. Map links to downstream nodes via .inp topology
      3. Assign channel width from .inp cross-sections
      4. Rank nodes by worst incoming link velocity
      5. Return top N (or requested subset, or all)

    Returns
    -------
    node_config : dict
        {node_id: {"width": float, "length": float, "ks": float,
                    "reason": str, "peak_link_vel": float}}
    scan_summary : dict
        Summary of the scan for reporting
    """
    # ── Read link peaks from .out ───────────────────────────────────────
    link_peaks = {}
    node_peaks = {}

    with Output(out_file) as out:
        all_nodes = list(out.nodes)
        all_links = list(out.links)

        for lid in all_links:
            vel_ts = out.link_series(lid, LinkAttribute.FLOW_VELOCITY)
            flow_ts = out.link_series(lid, LinkAttribute.FLOW_RATE)
            depth_ts = out.link_series(lid, LinkAttribute.FLOW_DEPTH)
            link_peaks[lid] = {
                "max_vel": max(vel_ts.values()) if vel_ts else 0,
                "max_flow": max(flow_ts.values()) if flow_ts else 0,
                "max_depth": max(depth_ts.values()) if depth_ts else 0,
            }

        for nid in all_nodes:
            depth_ts = out.node_series(nid, NodeAttribute.INVERT_DEPTH)
            inflow_ts = out.node_series(nid, NodeAttribute.TOTAL_INFLOW)
            node_peaks[nid] = {
                "max_depth": max(depth_ts.values()) if depth_ts else 0,
                "max_inflow": max(inflow_ts.values()) if inflow_ts else 0,
            }

    # ── Parse .inp for topology and geometry ────────────────────────────
    if inp_file is None:
        # Try to find .inp next to the .out
        inp_file = out_file.replace(".out", ".inp")

    link_topo, xsections = parse_inp(inp_file)
    has_inp = bool(link_topo)

    # ── Map: for each node, find the worst incoming link ────────────────
    # "incoming" = links whose "to" node is this node
    node_worst_link = {}  # {node_id: {"link": lid, "vel": float, "width": float}}

    if has_inp:
        # Build incoming and outgoing link maps
        node_incoming = {}  # {node_id: [(lid, vel, width, shape), ...]}
        node_outgoing = {}

        for lid, topo in link_topo.items():
            lp = link_peaks.get(lid, {})
            vel = lp.get("max_vel", 0)
            xs = xsections.get(lid, {})
            width = xs.get("width", 5.0)
            shape = xs.get("shape", "UNKNOWN")
            entry = (lid, vel, width, shape)

            node_incoming.setdefault(topo["to"], []).append(entry)
            node_outgoing.setdefault(topo["from"], []).append(entry)

        # For each node, pick the worst incoming link.
        # If no incoming link (headwater), use the outgoing link for width.
        for nid in all_nodes:
            incoming = node_incoming.get(nid, [])
            outgoing = node_outgoing.get(nid, [])

            if incoming:
                best = max(incoming, key=lambda x: x[1])
                node_worst_link[nid] = {
                    "link": best[0], "vel": best[1],
                    "width": best[2], "shape": best[3],
                }
            elif outgoing:
                # Headwater node — use outgoing link geometry, vel = 0
                best = max(outgoing, key=lambda x: x[1])
                node_worst_link[nid] = {
                    "link": best[0] + " (out)", "vel": best[1],
                    "width": best[2], "shape": best[3],
                }
    else:
        # No .inp — estimate from node peaks
        for nid in all_nodes:
            np_ = node_peaks.get(nid, {})
            depth = np_.get("max_depth", 1.0)
            node_worst_link[nid] = {
                "link": "?",
                "vel": np_.get("max_inflow", 0) / max(depth * 3.0, 1.0),  # rough V estimate
                "width": max(3.0, depth * 3.0),
                "shape": "ESTIMATED",
            }

    # ── Rank and select ─────────────────────────────────────────────────
    # Exclude outfalls (nodes with no incoming link or zero inflow that look like outlets)
    candidate_nodes = []
    for nid in all_nodes:
        np_ = node_peaks.get(nid, {})
        wl = node_worst_link.get(nid)

        # Skip nodes with no meaningful flow
        if np_.get("max_inflow", 0) < MIN_INFLOW:
            continue

        candidate_nodes.append({
            "node_id": nid,
            "peak_inflow": np_.get("max_inflow", 0),
            "peak_depth": np_.get("max_depth", 0),
            "peak_link_vel": wl["vel"] if wl else 0,
            "width": wl["width"] if wl else 5.0,
            "incoming_link": wl["link"] if wl else "?",
            "shape": wl["shape"] if wl else "?",
        })

    # Sort by peak incoming link velocity (highest first)
    candidate_nodes.sort(key=lambda x: x["peak_link_vel"], reverse=True)

    # Select
    if requested_nodes:
        selected = [c for c in candidate_nodes if c["node_id"] in requested_nodes]
    elif analyze_all:
        selected = candidate_nodes
    else:
        n = top_n or min(10, len(candidate_nodes))
        selected = candidate_nodes[:n]

    # Build node_config
    node_config = {}
    for c in selected:
        node_config[c["node_id"]] = {
            "width": c["width"],
            "length": 30.0,
            "ks": 0.10,
            "reason": f"v={c['peak_link_vel']:.1f} fps via {c['incoming_link']}",
            "peak_link_vel": c["peak_link_vel"],
        }

    scan_summary = {
        "total_nodes": len(all_nodes),
        "total_links": len(all_links),
        "candidates_with_flow": len(candidate_nodes),
        "selected_for_analysis": len(selected),
        "has_inp": has_inp,
        "all_candidates": candidate_nodes,
    }

    return node_config, scan_summary


# ── Quantum analysis ────────────────────────────────────────────────────────

def run_quantum_postprocess(out_file, node_config, verbose=False):
    """Read SWMM .out and run Quantum analysis."""
    results = []

    # ── Read timeseries ─────────────────────────────────────────────────
    node_timeseries = {}

    with Output(out_file) as out:
        for nid in node_config:
            depth_ts = out.node_series(nid, NodeAttribute.INVERT_DEPTH)
            inflow_ts = out.node_series(nid, NodeAttribute.TOTAL_INFLOW)

            series = []
            for t in depth_ts:
                series.append((t, depth_ts[t], inflow_ts.get(t, 0.0)))
            node_timeseries[nid] = series

        n_steps = len(next(iter(node_timeseries.values())))

    results.append(CheckResult(
        "Timeseries extracted",
        n_steps > 0,
        f"{n_steps} reporting steps per node",
    ))

    # ── Initialize QuantumNodes ─────────────────────────────────────────
    qnodes = {}
    for nid, cfg in node_config.items():
        qnodes[nid] = QuantumNode(
            node_id=nid,
            width=cfg["width"],
            length=cfg["length"],
            roughness_ks=cfg["ks"],
            sediment=SedimentProperties.sand(),
        )

    # ── Run analysis ────────────────────────────────────────────────────
    t0 = time.perf_counter()
    analysis_count = 0

    peak = {nid: {"depth": 0.0, "inflow": 0.0, "scour_risk": 0.0,
                   "max_velocity": 0.0, "bed_shear": 0.0,
                   "excess_shear": 0.0, "shields": 0.0}
            for nid in node_config}

    quantum_results = {nid: [] for nid in node_config}

    for nid in node_config:
        qnode = qnodes[nid]

        for (t, depth, inflow) in node_timeseries[nid]:
            peak[nid]["depth"] = max(peak[nid]["depth"], depth)
            peak[nid]["inflow"] = max(peak[nid]["inflow"], inflow)

            if depth > MIN_DEPTH and inflow > MIN_INFLOW:
                qnode.update_from_swmm(depth, inflow)
                qnode.compute_turbulence(n_particles=N_PARTICLES)
                analysis_count += 1

                m = qnode.metrics
                peak[nid]["scour_risk"] = max(peak[nid]["scour_risk"], m.scour_risk_index)
                peak[nid]["max_velocity"] = max(peak[nid]["max_velocity"], m.max_velocity)
                peak[nid]["bed_shear"] = max(peak[nid]["bed_shear"], m.bed_shear_stress)
                peak[nid]["excess_shear"] = max(peak[nid]["excess_shear"], m.excess_shear_ratio)
                peak[nid]["shields"] = max(peak[nid]["shields"], m.shields_parameter)

                quantum_results[nid].append({
                    "time": str(t),
                    "depth_ft": depth,
                    "inflow_cfs": inflow,
                    "max_velocity_fps": m.max_velocity,
                    "bed_shear_psf": m.bed_shear_stress,
                    "scour_risk": m.scour_risk_index,
                    "shields": m.shields_parameter,
                    "excess_shear": m.excess_shear_ratio,
                })

        if verbose:
            p = peak[nid]
            n_a = len(quantum_results[nid])
            assessment = "CRITICAL" if p["scour_risk"] > 0.8 else \
                         "HIGH" if p["scour_risk"] > 0.6 else \
                         "MODERATE" if p["scour_risk"] > 0.4 else "LOW"
            print(f"    {nid:<10} {n_a:>3} steps | "
                  f"Q={p['inflow']:>8.1f} cfs | "
                  f"tau={p['bed_shear']:>7.3f} psf | "
                  f"scour={p['scour_risk']:.3f} ({assessment})")

    elapsed = time.perf_counter() - t0

    results.append(CheckResult(
        "Quantum analysis completed",
        analysis_count > 0,
        f"{analysis_count} analyses in {elapsed:.1f}s",
    ))

    # ── Validation checks ───────────────────────────────────────────────
    for nid in node_config:
        p = peak[nid]

        results.append(CheckResult(
            f"Node {nid} -- scour risk in [0, 1]",
            0.0 <= p["scour_risk"] <= 1.0,
            f"risk = {p['scour_risk']:.4f}",
        ))

        results.append(CheckResult(
            f"Node {nid} -- bed shear > 0",
            p["bed_shear"] > 0,
            f"tau = {p['bed_shear']:.4f} psf",
        ))

        results.append(CheckResult(
            f"Node {nid} -- velocity > 0",
            p["max_velocity"] > 0,
            f"v_max = {p['max_velocity']:.3f} ft/s",
        ))

    results.append(CheckResult("Runtime", elapsed < 300.0, f"{elapsed:.1f}s"))

    return results, peak, quantum_results


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Quantum Hydraulics -- SWMM .out post-processor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_headless_swmm.py                              # default test model
  python run_headless_swmm.py path/to/project.out          # any SWMM output
  python run_headless_swmm.py model.out --top 5            # top 5 by velocity
  python run_headless_swmm.py model.out --nodes J1,J3      # specific nodes
  python run_headless_swmm.py model.out --all              # every junction
        """,
    )
    parser.add_argument("out_file", nargs="?", default=DEFAULT_OUT,
                        help="Path to SWMM .out file (default: QuantumTest_Simple.out)")
    parser.add_argument("--nodes", type=str, default=None,
                        help="Comma-separated node IDs to analyze")
    parser.add_argument("--top", type=int, default=None,
                        help="Analyze top N nodes by peak velocity (default: 10)")
    parser.add_argument("--all", action="store_true", dest="analyze_all",
                        help="Analyze every junction with flow")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    out_file = args.out_file
    requested_nodes = args.nodes.split(",") if args.nodes else None

    # ── Verify .out file ────────────────────────────────────────────────
    if not os.path.exists(out_file):
        print(f"ERROR: .out file not found: {out_file}")
        sys.exit(1)

    if not args.json:
        print()
        print("=" * 72)
        print("QUANTUM HYDRAULICS -- SWMM OUTPUT POST-PROCESSOR")
        print("=" * 72)
        print(f"Input: {out_file}")

    # ── Phase 1: Scan and discover ──────────────────────────────────────
    if not args.json and args.verbose:
        print("\n  Scanning .out file for critical nodes...")

    node_config, scan = discover_critical_nodes(
        out_file,
        top_n=args.top,
        requested_nodes=requested_nodes,
        analyze_all=args.analyze_all,
        verbose=args.verbose,
    )

    if not node_config:
        print("ERROR: No analyzable nodes found")
        sys.exit(1)

    if not args.json:
        inp_status = "(with .inp geometry)" if scan["has_inp"] else "(no .inp -- widths estimated)"
        print(f"Scanned: {scan['total_nodes']} nodes, {scan['total_links']} links {inp_status}")
        print(f"Candidates with flow: {scan['candidates_with_flow']}")
        print(f"Selected for analysis: {scan['selected_for_analysis']}")

        if args.verbose and scan["all_candidates"]:
            print()
            print("  LINK VELOCITY SCAN (ranked by peak incoming velocity):")
            print(f"  {'Node':<10} {'Link':>6} {'V_peak':>8} {'Q_peak':>10} {'Width':>8} {'Shape':<12}")
            print(f"  {'':.<10} {'':>6} {'(fps)':>8} {'(cfs)':>10} {'(ft)':>8} {'':.<12}")
            for c in scan["all_candidates"]:
                marker = " <--" if c["node_id"] in node_config else ""
                print(f"  {c['node_id']:<10} {c['incoming_link']:>6} "
                      f"{c['peak_link_vel']:>8.2f} {c['peak_inflow']:>10.2f} "
                      f"{c['width']:>8.1f} {c['shape']:<12}{marker}")

        print("-" * 72)

    # ── Phase 2: Quantum analysis ───────────────────────────────────────
    if not args.json and args.verbose:
        print("\n  Running Quantum vortex particle analysis...")

    results, peak, quantum_results = run_quantum_postprocess(
        out_file, node_config, verbose=args.verbose,
    )

    n_pass = sum(1 for r in results if r.passed)
    n_fail = sum(1 for r in results if not r.passed)
    all_pass = n_fail == 0

    if args.json:
        output = {
            "input": os.path.basename(out_file),
            "scan": {
                "total_nodes": scan["total_nodes"],
                "total_links": scan["total_links"],
                "has_inp_geometry": scan["has_inp"],
                "candidates": scan["candidates_with_flow"],
                "analyzed": scan["selected_for_analysis"],
            },
            "passed": n_pass,
            "failed": n_fail,
            "all_pass": all_pass,
            "peak_conditions": peak,
            "checks": [
                {"name": r.name, "passed": r.passed, "detail": r.detail, **r.values}
                for r in results
            ],
        }
        print(json.dumps(output, indent=2,
                          default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    else:
        print()
        for r in results:
            if args.verbose or not r.passed:
                print(r)
            else:
                print(f"  [{'PASS' if r.passed else 'FAIL'}] {r.name}")

        if peak:
            print()
            print("  QUANTUM ANALYSIS RESULTS (peak conditions):")
            print(f"  {'Node':<10} {'Depth':>7} {'Inflow':>9} {'V_max':>7} "
                  f"{'Shear':>7} {'Scour':>7} {'Shields':>8} {'Assessment':<12}")
            print(f"  {'':.<10} {'(ft)':>7} {'(cfs)':>9} {'(fps)':>7} "
                  f"{'(psf)':>7} {'Risk':>7} {'':>8} {'':.<12}")
            for nid in node_config:
                p = peak[nid]
                assessment = "CRITICAL" if p["scour_risk"] > 0.8 else \
                             "HIGH" if p["scour_risk"] > 0.6 else \
                             "MODERATE" if p["scour_risk"] > 0.4 else \
                             "LOW-MOD" if p["scour_risk"] > 0.2 else "LOW"
                print(f"  {nid:<10} {p['depth']:>7.2f} {p['inflow']:>9.2f} "
                      f"{p['max_velocity']:>7.3f} {p['bed_shear']:>7.4f} "
                      f"{p['scour_risk']:>7.4f} {p['shields']:>8.4f} {assessment:<12}")

        print()
        print("-" * 72)
        print(f"  {n_pass}/{n_pass + n_fail} checks passed", end="")
        if all_pass:
            print(" -- ALL PASS")
        else:
            print(f" -- {n_fail} FAILED")
            for r in results:
                if not r.passed:
                    print(f"         X {r.name}: {r.detail}")

        print("=" * 72)
        print()

    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
