"""
Quantum Hydraulics - PCSWMM Test Script
========================================

This script runs quantum-enhanced turbulence analysis on the test model.

USAGE IN PCSWMM:
1. Open QuantumTest_PCSWMM.inp in PCSWMM
2. Run the simulation once to generate results
3. Go to Scripts tab
4. Load and run this script

The script will analyze critical junctions for:
- Scour risk assessment
- Bed shear stress
- Turbulent kinetic energy (TKE)
- Velocity distribution
- Froude/Reynolds numbers

Results are saved to: ~/Documents/quantum_results/
"""

import sys
import os
from datetime import datetime

# Add quantum_hydraulics to path if running from this directory
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

# ============================================================================
# OBSERVATION ZONES CONFIGURATION
# These are the critical nodes to analyze in QuantumTest_PCSWMM.inp
# ============================================================================
OBSERVATION_ZONES = {
    'CONFLUENCE': {
        'name': 'Main Confluence',
        'description': 'Where tributary meets main stem - HIGH turbulence expected',
        'width': 15.0,           # feet (channel width at junction)
        'length': 40.0,          # feet (analysis domain length)
        'roughness_ks': 0.10,    # feet (equivalent sand roughness)
        'obs_radius': 20.0,      # feet (observation zone radius)
        'analysis_type': 'confluence',
        'expected_risk': 'HIGH'  # Expected scour risk
    },
    'TRIB_JCT': {
        'name': 'Tributary Junction',
        'description': 'Steep tributary - HIGH velocity expected',
        'width': 8.0,
        'length': 30.0,
        'roughness_ks': 0.15,    # Rougher channel
        'obs_radius': 15.0,
        'analysis_type': 'steep_channel',
        'expected_risk': 'MODERATE-HIGH'
    },
    'OUTLET_JCT': {
        'name': 'Outlet Junction',
        'description': 'Pre-outfall junction - MODERATE conditions',
        'width': 20.0,
        'length': 50.0,
        'roughness_ks': 0.08,
        'obs_radius': 25.0,
        'analysis_type': 'outlet',
        'expected_risk': 'MODERATE'
    },
    'MID_JCT': {
        'name': 'Mid-Point Junction',
        'description': 'Transition zone in main channel',
        'width': 16.0,
        'length': 35.0,
        'roughness_ks': 0.10,
        'obs_radius': 18.0,
        'analysis_type': 'junction',
        'expected_risk': 'LOW-MODERATE'
    }
}

# Analysis parameters
N_PARTICLES = 500          # Number of vortex particles per analysis
MIN_DEPTH = 0.1            # Minimum depth (ft) to trigger analysis
MIN_INFLOW = 0.1           # Minimum inflow (cfs) to trigger analysis
REPORT_INTERVAL = 50       # Print status every N timesteps

# Output directory
OUTPUT_DIR = os.path.join(os.path.expanduser("~"), "Documents", "quantum_results")


def print_header():
    """Print analysis header."""
    print("\n" + "=" * 70)
    print("   QUANTUM-ENHANCED HYDRAULIC ANALYSIS")
    print("   Vortex Particle Turbulence Simulation")
    print("=" * 70)
    print(f"   Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   Output: {OUTPUT_DIR}")
    print("=" * 70 + "\n")


def print_zone_config():
    """Print observation zone configuration."""
    print("CONFIGURED OBSERVATION ZONES:")
    print("-" * 50)
    for node_id, config in OBSERVATION_ZONES.items():
        print(f"  {node_id}: {config['name']}")
        print(f"    - Width: {config['width']} ft")
        print(f"    - Roughness: {config['roughness_ks']} ft")
        print(f"    - Expected risk: {config['expected_risk']}")
    print("-" * 50 + "\n")


def run_analysis(model_file=None):
    """
    Run quantum turbulence analysis.

    Parameters
    ----------
    model_file : str, optional
        Path to .inp file. If None, uses QuantumTest_PCSWMM.inp in same directory.

    Returns
    -------
    dict
        Results dictionary
    """
    # Try to import required packages
    try:
        from pyswmm import Simulation, Nodes
        print("PySWMM loaded successfully")
    except ImportError:
        print("ERROR: PySWMM not installed!")
        print("Install with: pip install pyswmm")
        return {"success": False, "error": "PySWMM not available"}

    try:
        from quantum_hydraulics.integration.swmm_node import QuantumNode
        print("Quantum Hydraulics loaded successfully")
    except ImportError:
        print("ERROR: quantum_hydraulics not found!")
        print("Make sure the package is installed or in your Python path")
        return {"success": False, "error": "quantum_hydraulics not available"}

    # Find model file
    if model_file is None:
        model_file = os.path.join(script_dir, "QuantumTest_PCSWMM.inp")

    if not os.path.exists(model_file):
        print(f"ERROR: Model file not found: {model_file}")
        return {"success": False, "error": f"Model not found: {model_file}"}

    print(f"\nUsing model: {os.path.basename(model_file)}")
    print(f"Full path: {model_file}\n")

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Initialize quantum nodes
    quantum_nodes = {}
    for node_id, config in OBSERVATION_ZONES.items():
        quantum_nodes[node_id] = QuantumNode(
            node_id=node_id,
            width=config['width'],
            length=config.get('length', 30.0),
            roughness_ks=config['roughness_ks'],
            observation_radius=config.get('obs_radius', 15.0),
        )
        print(f"Initialized QuantumNode: {node_id}")

    # Storage for results
    all_results = []
    timestep_count = 0
    quantum_analyses = 0
    peak_values = {node_id: {
        'depth': 0, 'inflow': 0, 'scour_risk': 0, 'velocity': 0,
        'excess_shear': 0, 'scour_depth': 0, 'shields': 0
    } for node_id in OBSERVATION_ZONES}

    print("\nStarting simulation...")
    print("-" * 50)

    try:
        with Simulation(model_file) as sim:
            nodes = Nodes(sim)

            # Verify nodes exist
            for node_id in OBSERVATION_ZONES:
                try:
                    _ = nodes[node_id]
                    print(f"  Found node: {node_id}")
                except KeyError:
                    print(f"  WARNING: Node '{node_id}' not found in model!")

            print("\nRunning simulation with quantum analysis...")

            for step in sim:
                timestep_count += 1
                current_time = sim.current_time

                for node_id, quantum_node in quantum_nodes.items():
                    try:
                        node = nodes[node_id]
                        depth = node.depth
                        inflow = node.total_inflow

                        # Update peak tracking
                        peak_values[node_id]['depth'] = max(peak_values[node_id]['depth'], depth)
                        peak_values[node_id]['inflow'] = max(peak_values[node_id]['inflow'], inflow)

                        # Only analyze if sufficient flow
                        if depth > MIN_DEPTH and inflow > MIN_INFLOW:
                            quantum_node.update_from_swmm(depth, inflow)
                            quantum_node.compute_turbulence(n_particles=N_PARTICLES)
                            quantum_analyses += 1

                            metrics = quantum_node.get_metrics()

                            # Update peak tracking
                            peak_values[node_id]['scour_risk'] = max(
                                peak_values[node_id]['scour_risk'],
                                metrics['scour_risk_index']
                            )
                            peak_values[node_id]['velocity'] = max(
                                peak_values[node_id]['velocity'],
                                metrics['max_velocity']
                            )
                            peak_values[node_id]['excess_shear'] = max(
                                peak_values[node_id]['excess_shear'],
                                metrics.get('excess_shear_ratio', 0)
                            )
                            peak_values[node_id]['scour_depth'] = max(
                                peak_values[node_id]['scour_depth'],
                                metrics.get('scour_depth_potential', 0)
                            )
                            peak_values[node_id]['shields'] = max(
                                peak_values[node_id]['shields'],
                                metrics.get('shields_parameter', 0)
                            )

                            result = {
                                'datetime': str(current_time),
                                'node_id': node_id,
                                'node_name': OBSERVATION_ZONES[node_id]['name'],
                                'depth_ft': depth,
                                'inflow_cfs': inflow,
                                **metrics
                            }
                            all_results.append(result)

                    except KeyError:
                        pass  # Node not in model
                    except Exception as e:
                        if timestep_count == 1:
                            print(f"  Error at {node_id}: {e}")

                # Progress report
                if timestep_count % REPORT_INTERVAL == 0:
                    print(f"  Timestep {timestep_count:4d} | Time: {current_time} | Analyses: {quantum_analyses}")

    except Exception as e:
        print(f"\nERROR during simulation: {e}")
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}

    print("-" * 50)
    print(f"\nSimulation complete!")
    print(f"  Total timesteps: {timestep_count}")
    print(f"  Quantum analyses: {quantum_analyses}")
    print(f"  Results records: {len(all_results)}")

    if not all_results:
        print("\nWARNING: No results generated!")
        print("Check that the model has been run and has flow data.")
        return {"success": False, "error": "No results generated"}

    # Save detailed results to CSV
    csv_file = None
    try:
        import pandas as pd
        df = pd.DataFrame(all_results)
        csv_file = os.path.join(OUTPUT_DIR, "quantum_detailed_results.csv")
        df.to_csv(csv_file, index=False)
        print(f"\nDetailed CSV: {csv_file}")
    except ImportError:
        print("\nPandas not available - skipping CSV export")

    # Write summary report
    report_file = os.path.join(OUTPUT_DIR, "quantum_analysis_report.txt")
    write_summary_report(report_file, all_results, peak_values, model_file, timestep_count)
    print(f"Summary report: {report_file}")

    # Print summary to console
    print_summary(peak_values)

    return {
        "success": True,
        "results": all_results,
        "peak_values": peak_values,
        "report_file": report_file,
        "csv_file": csv_file,
        "timesteps": timestep_count,
        "analyses": quantum_analyses
    }


def write_summary_report(report_file, results, peak_values, model_file, timesteps):
    """Write detailed summary report."""
    with open(report_file, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("QUANTUM-ENHANCED HYDRAULIC ANALYSIS REPORT\n")
        f.write("Vortex Particle Turbulence Simulation (Optimized)\n")
        f.write("=" * 70 + "\n\n")

        f.write(f"Model: {os.path.basename(model_file)}\n")
        f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Timesteps: {timesteps}\n")
        f.write(f"Total Analysis Records: {len(results)}\n")
        f.write("\n" + "=" * 70 + "\n\n")

        for node_id, config in OBSERVATION_ZONES.items():
            f.write("-" * 70 + "\n")
            f.write(f"NODE: {config['name']} ({node_id})\n")
            f.write("-" * 70 + "\n")
            f.write(f"Description: {config['description']}\n")
            f.write(f"Channel Width: {config['width']} ft\n")
            f.write(f"Roughness (ks): {config['roughness_ks']} ft\n\n")

            peaks = peak_values[node_id]
            f.write("PEAK CONDITIONS:\n")
            f.write(f"  Max Depth:          {peaks['depth']:.2f} ft\n")
            f.write(f"  Max Inflow:         {peaks['inflow']:.1f} cfs\n")
            f.write(f"  Max Velocity:       {peaks['velocity']:.2f} ft/s\n")
            f.write(f"  Peak Scour Risk:    {peaks['scour_risk']:.3f}\n")
            f.write(f"  Excess Shear Ratio: {peaks.get('excess_shear', 0):.2f} (tau/tau_c)\n")
            f.write(f"  Scour Potential:    {peaks.get('scour_depth', 0):.2f} ft/year\n\n")

            f.write("ASSESSMENT:\n")
            risk = peaks['scour_risk']
            excess = peaks.get('excess_shear', 0)
            if risk > 0.8:
                f.write(f"  *** CRITICAL SCOUR RISK - Protection REQUIRED (tau/tau_c={excess:.1f}) ***\n")
            elif risk > 0.6:
                f.write(f"  ** HIGH SCOUR RISK - Protection recommended (tau/tau_c={excess:.1f}) **\n")
            elif risk > 0.4:
                f.write(f"  * MODERATE SCOUR RISK - Monitor conditions (tau/tau_c={excess:.1f}) *\n")
            elif risk > 0.2:
                f.write(f"  LOW-MODERATE SCOUR RISK - Acceptable with monitoring (tau/tau_c={excess:.1f})\n")
            else:
                f.write(f"  LOW SCOUR RISK - Acceptable (tau/tau_c={excess:.1f})\n")

            vel = peaks['velocity']
            if vel > 15:
                f.write("  *** EXTREME VELOCITY - Energy dissipation REQUIRED ***\n")
            elif vel > 10:
                f.write("  ** HIGH VELOCITY - Energy dissipation required **\n")
            elif vel > 6:
                f.write("  * ELEVATED VELOCITY - Consider energy dissipation *\n")

            scour_depth = peaks.get('scour_depth', 0)
            if scour_depth > 2.0:
                f.write(f"  *** SEVERE EROSION - {scour_depth:.1f} ft/year scour potential ***\n")
            elif scour_depth > 1.0:
                f.write(f"  ** HIGH EROSION - {scour_depth:.1f} ft/year scour potential **\n")
            elif scour_depth > 0.5:
                f.write(f"  * MODERATE EROSION - {scour_depth:.1f} ft/year scour potential *\n")

            f.write(f"\n  Expected Risk (design): {config['expected_risk']}\n")
            f.write("\n")

        f.write("=" * 70 + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * 70 + "\n")


def print_summary(peak_values):
    """Print summary to console."""
    print("\n" + "=" * 70)
    print("ANALYSIS SUMMARY (Optimized Vortex Particle Method)")
    print("=" * 70)

    for node_id, config in OBSERVATION_ZONES.items():
        peaks = peak_values[node_id]
        risk = peaks['scour_risk']
        excess = peaks.get('excess_shear', 0)

        # Determine risk level using new thresholds
        if risk > 0.8:
            risk_str = "CRITICAL"
        elif risk > 0.6:
            risk_str = "HIGH"
        elif risk > 0.4:
            risk_str = "MODERATE"
        elif risk > 0.2:
            risk_str = "LOW-MOD"
        else:
            risk_str = "LOW"

        print(f"\n{config['name']} ({node_id}):")
        print(f"  Peak Depth:       {peaks['depth']:.2f} ft")
        print(f"  Peak Inflow:      {peaks['inflow']:.1f} cfs")
        print(f"  Peak Velocity:    {peaks['velocity']:.2f} ft/s")
        print(f"  Scour Risk:       {risk:.3f} ({risk_str})")
        print(f"  Excess Shear:     {excess:.2f}x critical (tau/tau_c)")
        print(f"  Scour Potential:  {peaks.get('scour_depth', 0):.2f} ft/year")

    print("\n" + "=" * 70)


def main():
    """Main entry point."""
    print_header()
    print_zone_config()

    result = run_analysis()

    if result['success']:
        print("\n" + "=" * 70)
        print("ANALYSIS COMPLETE!")
        print(f"Results saved to: {OUTPUT_DIR}")
        print("=" * 70 + "\n")
    else:
        print(f"\nANALYSIS FAILED: {result.get('error', 'Unknown error')}")

    return result


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\nFATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
