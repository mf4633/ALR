"""
PCSWMM Integration Script - Auto-detect version.

Automatically uses whatever model is currently open in PCSWMM.
Run this script from PCSWMM's Scripts tab.

Usage:
1. Open your model in PCSWMM
2. Go to Scripts tab
3. Run this script
4. Edit OBSERVATION_ZONES dict below if needed
"""

import sys
import os
from datetime import datetime
from typing import Dict, List, Optional

# Configuration
OBSERVATION_ZONES: Dict[str, Dict] = {
    # Add your critical nodes here
    # 'J3': {
    #     'name': 'Junction 3',
    #     'width': 20.0,
    #     'roughness_ks': 0.10,
    #     'analysis_type': 'junction'
    # },
}

# Thresholds
N_PARTICLES = 500
MIN_DEPTH = 0.1  # ft
MIN_INFLOW = 0.05  # cfs

# Output directory
OUTPUT_DIR = os.path.join(os.path.expanduser("~"), "Documents", "quantum_results")


def get_current_model() -> Optional[str]:
    """Try multiple methods to find currently open model."""
    cwd = os.getcwd()
    inp_files = [f for f in os.listdir(cwd) if f.endswith(".inp")]

    if len(inp_files) == 1:
        return os.path.join(cwd, inp_files[0])
    elif len(inp_files) > 1:
        print(f"Multiple .inp files found in {cwd}, using first")
        return os.path.join(cwd, inp_files[0])

    # Check common locations
    user_docs = os.path.expanduser("~\\Documents")
    pcswmm_paths = [
        os.path.join(user_docs, "PCSWMM Projects"),
        user_docs,
    ]

    for path in pcswmm_paths:
        if os.path.exists(path):
            for root, dirs, files in os.walk(path):
                inp_files = [f for f in files if f.endswith(".inp")]
                if inp_files:
                    recent_inp = max(
                        [os.path.join(root, f) for f in inp_files],
                        key=os.path.getmtime,
                    )
                    mod_time = os.path.getmtime(recent_inp)
                    if (datetime.now().timestamp() - mod_time) < 3600:
                        return recent_inp

    return None


def run_quantum_analysis(
    model_file: Optional[str] = None,
    observation_zones: Optional[Dict] = None,
) -> Dict:
    """
    Run quantum-enhanced hydraulic analysis.

    Parameters
    ----------
    model_file : str, optional
        Path to SWMM .inp file. Auto-detects if None.
    observation_zones : dict, optional
        Node configuration. Uses OBSERVATION_ZONES if None.

    Returns
    -------
    dict
        Results dictionary with 'success', 'results', 'summary_file' keys
    """
    try:
        from pyswmm import Simulation, Nodes
    except ImportError:
        return {"success": False, "error": "PySWMM not available"}

    try:
        from quantum_hydraulics.integration.swmm_node import QuantumNode
    except ImportError:
        return {"success": False, "error": "quantum_hydraulics package not found"}

    if model_file is None:
        model_file = get_current_model()

    if model_file is None or not os.path.exists(model_file):
        return {"success": False, "error": "Could not detect model file"}

    if observation_zones is None:
        observation_zones = OBSERVATION_ZONES

    if not observation_zones:
        return {"success": False, "error": "No observation zones configured"}

    print(f"Using model: {os.path.basename(model_file)}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Initialize quantum nodes
    quantum_nodes = {}
    for node_id, config in observation_zones.items():
        quantum_nodes[node_id] = QuantumNode(
            node_id=node_id,
            width=config.get("width", 20.0),
            length=config.get("length", 30.0),
            roughness_ks=config.get("roughness_ks", 0.1),
            observation_radius=config.get("obs_radius", 15.0),
        )

    all_results = []
    timestep_count = 0
    quantum_analyses = 0

    try:
        with Simulation(model_file) as sim:
            nodes = Nodes(sim)

            for step in sim:
                timestep_count += 1
                current_time = sim.current_time

                for node_id, quantum_node in quantum_nodes.items():
                    try:
                        node = nodes[node_id]
                        depth = node.depth
                        inflow = node.total_inflow

                        if depth > MIN_DEPTH and inflow > MIN_INFLOW:
                            quantum_node.update_from_swmm(depth, inflow)
                            quantum_node.compute_turbulence(n_particles=N_PARTICLES)
                            quantum_analyses += 1

                            metrics = quantum_node.get_metrics()

                            result = {
                                "datetime": current_time,
                                "node_id": node_id,
                                "node_name": observation_zones[node_id].get("name", node_id),
                                "depth_ft": depth,
                                "inflow_cfs": inflow,
                                **metrics,
                            }
                            all_results.append(result)

                    except KeyError:
                        if timestep_count == 1:
                            print(f"Node '{node_id}' not found in model")
                    except Exception as e:
                        if timestep_count == 1:
                            print(f"Error at {node_id}: {e}")

                if timestep_count % 100 == 0:
                    print(f"Timestep {timestep_count} | Analyses: {quantum_analyses}")

    except Exception as e:
        return {"success": False, "error": str(e)}

    if not all_results:
        return {"success": False, "error": "No results generated"}

    # Save results
    try:
        import pandas as pd

        df = pd.DataFrame(all_results)
        csv_file = os.path.join(OUTPUT_DIR, "quantum_results_detailed.csv")
        df.to_csv(csv_file, index=False)
    except ImportError:
        csv_file = None

    # Generate summary
    report_file = os.path.join(OUTPUT_DIR, "quantum_summary_report.txt")
    _write_summary_report(report_file, all_results, observation_zones, model_file)

    return {
        "success": True,
        "results": all_results,
        "summary_file": report_file,
        "csv_file": csv_file,
        "timesteps": timestep_count,
        "analyses": quantum_analyses,
    }


def _write_summary_report(
    report_file: str,
    results: List[Dict],
    observation_zones: Dict,
    model_file: str,
):
    """Write summary report to file."""
    with open(report_file, "w") as f:
        f.write("QUANTUM-ENHANCED HYDRAULIC ANALYSIS\n")
        f.write("=" * 70 + "\n")
        f.write(f"Model: {os.path.basename(model_file)}\n")
        f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Records: {len(results)}\n")
        f.write("=" * 70 + "\n\n")

        for node_id, config in observation_zones.items():
            node_data = [r for r in results if r["node_id"] == node_id]
            if not node_data:
                f.write(f"\nNode {node_id}: No data (insufficient flow)\n")
                continue

            f.write(f"\n{'=' * 70}\n")
            f.write(f"NODE: {config.get('name', node_id)} ({node_id})\n")
            f.write(f"{'=' * 70}\n\n")

            max_depth = max(r["depth_ft"] for r in node_data)
            max_inflow = max(r["inflow_cfs"] for r in node_data)
            max_vel = max(r["max_velocity"] for r in node_data)
            max_scour = max(r["scour_risk_index"] for r in node_data)

            f.write("PEAK CONDITIONS:\n")
            f.write(f"  Max Depth: {max_depth:.2f} ft\n")
            f.write(f"  Max Inflow: {max_inflow:.1f} cfs\n")
            f.write(f"  Max Velocity: {max_vel:.2f} ft/s\n")
            f.write(f"  Peak Scour Risk: {max_scour:.3f}\n\n")

            f.write("RECOMMENDATIONS:\n")
            if max_scour > 0.7:
                f.write("  CRITICAL SCOUR RISK - Protection REQUIRED\n")
            elif max_scour > 0.5:
                f.write("  HIGH SCOUR RISK - Protection recommended\n")
            elif max_scour > 0.3:
                f.write("  MODERATE SCOUR RISK - Monitor\n")
            else:
                f.write("  LOW SCOUR RISK\n")

            if max_vel > 10:
                f.write("  HIGH VELOCITY - Energy dissipation required\n")
            elif max_vel > 6:
                f.write("  ELEVATED VELOCITY - Consider dissipation\n")

            f.write("\n")


def main():
    """Main entry point for PCSWMM script."""
    print("\n" + "=" * 70)
    print("QUANTUM-ENHANCED PCSWMM ANALYSIS")
    print("=" * 70)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    if not OBSERVATION_ZONES:
        print("\nNo observation zones configured!")
        print("Edit OBSERVATION_ZONES dict in this script to add your nodes.")
        print("\nExample:")
        print("OBSERVATION_ZONES = {")
        print("    'J3': {")
        print("        'name': 'Junction 3',")
        print("        'width': 20.0,")
        print("        'roughness_ks': 0.10,")
        print("    },")
        print("}")
        return

    result = run_quantum_analysis()

    if result["success"]:
        print("\n" + "=" * 70)
        print("ANALYSIS COMPLETE!")
        print("=" * 70)
        print(f"Results in: {OUTPUT_DIR}")
    else:
        print(f"\nERROR: {result.get('error', 'Unknown error')}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"\nFATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
