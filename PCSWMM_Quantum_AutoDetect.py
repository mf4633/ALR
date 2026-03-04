"""
PCSWMM Quantum Analysis - AUTO-DETECT VERSION
==============================================
Automatically uses whatever model is currently open in PCSWMM!
No need to edit file paths - just open your model and run this script.

Usage:
1. Open your model in PCSWMM
2. Go to Scripts tab
3. Run this script
4. Edit OBSERVATION_ZONES if needed
"""

import sys
import os
from datetime import datetime
import pandas as pd

# Try to get currently open model from PCSWMM
try:
    # PCSWMM stores current model in environment or can be accessed via swmm5 module
    import swmm5
    # When script runs in PCSWMM, the model is already loaded
    MODEL_FILE = None  # Will be handled by PySWMM context
    print("✓ Will use currently open PCSWMM model")
except:
    MODEL_FILE = None
    print("✓ Will attempt to detect open model")

# PySWMM
try:
    from pyswmm import Simulation, Nodes
    print("✓ PySWMM loaded")
except ImportError:
    print("ERROR: PySWMM not available")
    sys.exit(1)

# Quantum hydraulics
try:
    from quantum_hydraulics import QuantumNode
    print("✓ Quantum hydraulics loaded")
except ImportError:
    print("ERROR: quantum_hydraulics.py not found!")
    sys.exit(1)

# ============================================================================
# CONFIGURATION - EDIT YOUR NODES HERE
# ============================================================================

# Your critical nodes - THESE ARE THE ONLY THINGS TO EDIT!
OBSERVATION_ZONES = {
    'J3': {
        'name': 'Junction 3',
        'width': 20.0,
        'roughness_ks': 0.10,
        'analysis_type': 'junction'
    },
    # Add more nodes here...
}

# Settings
N_PARTICLES = 500
MIN_DEPTH = 0.1  # ft - LOWERED for better detection
MIN_INFLOW = 0.05  # cfs - LOWERED for better detection

# Output to Documents folder
OUTPUT_DIR = os.path.join(os.path.expanduser('~'), 'Documents', 'quantum_results')

# ============================================================================
# AUTO-DETECT MODEL FILE
# ============================================================================

def get_current_model():
    """Try multiple methods to get currently open model"""
    
    # Method 1: Check if .inp file in current directory
    cwd = os.getcwd()
    inp_files = [f for f in os.listdir(cwd) if f.endswith('.inp')]
    
    if len(inp_files) == 1:
        return os.path.join(cwd, inp_files[0])
    elif len(inp_files) > 1:
        print(f"\nMultiple .inp files found in {cwd}:")
        for i, f in enumerate(inp_files, 1):
            print(f"  {i}. {f}")
        print("\nUsing first file. To use different file, open it in PCSWMM first.")
        return os.path.join(cwd, inp_files[0])
    
    # Method 2: Check common PCSWMM project locations
    user_docs = os.path.expanduser('~\\Documents')
    pcswmm_paths = [
        os.path.join(user_docs, 'PCSWMM Projects'),
        os.path.join(user_docs, 'OneDrive - McGill Associates, PA', 'Desktop'),
        user_docs
    ]
    
    for path in pcswmm_paths:
        if os.path.exists(path):
            for root, dirs, files in os.walk(path):
                inp_files = [f for f in files if f.endswith('.inp')]
                if inp_files:
                    # Found some .inp files, check modification time
                    recent_inp = max(
                        [os.path.join(root, f) for f in inp_files],
                        key=os.path.getmtime
                    )
                    mod_time = os.path.getmtime(recent_inp)
                    if (datetime.now().timestamp() - mod_time) < 3600:  # Modified in last hour
                        return recent_inp
    
    return None

# ============================================================================
# MAIN SCRIPT
# ============================================================================

def main():
    print("\n" + "="*70)
    print("QUANTUM-ENHANCED PCSWMM ANALYSIS - AUTO-DETECT")
    print("="*70)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Observation Zones: {len(OBSERVATION_ZONES)}")
    print(f"Particle Resolution: {N_PARTICLES}")
    print("="*70 + "\n")
    
    # Get model file
    model_file = MODEL_FILE if MODEL_FILE else get_current_model()
    
    if model_file and os.path.exists(model_file):
        print(f"✓ Using model: {os.path.basename(model_file)}")
        print(f"  Full path: {model_file}\n")
    else:
        print("ERROR: Could not detect model file!")
        print("\nPlease do ONE of:")
        print("  1. Open your model in PCSWMM before running this script")
        print("  2. Make sure .inp file is in current directory")
        print(f"  3. Current directory: {os.getcwd()}")
        print("\nOR edit this script and set MODEL_FILE manually at line 22")
        return
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Initialize quantum nodes
    print("Initializing quantum nodes...")
    quantum_nodes = {}
    for node_id, config in OBSERVATION_ZONES.items():
        quantum_nodes[node_id] = QuantumNode(
            node_id=node_id,
            width=config['width'],
            length=config.get('length', 30.0),
            roughness_ks=config.get('roughness_ks', 0.1),
            observation_radius=config.get('obs_radius', 15.0)
        )
        print(f"  ✓ {node_id}: {config['name']}")
    
    print(f"\n✓ Initialized {len(quantum_nodes)} quantum nodes\n")
    
    # Storage
    all_results = []
    timestep_count = 0
    quantum_analyses = 0
    
    # Run simulation
    print("Starting SWMM simulation with quantum enhancement...")
    print("-" * 70)
    
    try:
        with Simulation(model_file) as sim:
            nodes = Nodes(sim)
            
            # Get total expected timesteps for progress estimate
            try:
                sim_duration = (sim.end_time - sim.start_time).total_seconds()
                routing_step = sim.flow_routing_step
                total_steps_estimate = int(sim_duration / routing_step)
                print(f"Estimated timesteps: ~{total_steps_estimate:,}\n")
            except:
                print("Running simulation...\n")
            
            for step in sim:
                timestep_count += 1
                current_time = sim.current_time
                
                # Process each observation zone
                for node_id, quantum_node in quantum_nodes.items():
                    try:
                        node = nodes[node_id]
                        depth = node.depth
                        inflow = node.total_inflow
                        
                        # Only analyze if significant flow
                        if depth > MIN_DEPTH and inflow > MIN_INFLOW:
                            quantum_node.update_from_swmm(depth, inflow)
                            quantum_node.compute_turbulence(n_particles=N_PARTICLES)
                            quantum_analyses += 1
                            
                            metrics = quantum_node.get_engineering_metrics()
                            
                            result = {
                                'datetime': current_time,
                                'node_id': node_id,
                                'node_name': OBSERVATION_ZONES[node_id]['name'],
                                'depth_ft': depth,
                                'inflow_cfs': inflow,
                                **metrics
                            }
                            all_results.append(result)
                    
                    except KeyError:
                        if timestep_count == 1:
                            print(f"  ⚠️  Node '{node_id}' not found in model!")
                        continue
                    except Exception as e:
                        if timestep_count == 1:
                            print(f"  ⚠️  Error at {node_id}: {e}")
                        continue
                
                # Progress
                if timestep_count % 100 == 0:
                    print(f"  Timestep {timestep_count:,} | Quantum analyses: {quantum_analyses}")
        
        print("-" * 70)
        print(f"✓ Simulation complete!")
        print(f"  Total timesteps: {timestep_count:,}")
        print(f"  Quantum analyses: {quantum_analyses}")
    
    except Exception as e:
        print(f"\nERROR during simulation: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Check results
    if len(all_results) == 0:
        print("\n⚠️  No results generated!")
        print(f"   Thresholds: depth > {MIN_DEPTH} ft, inflow > {MIN_INFLOW} cfs")
        print("   Check that your nodes actually have flow.")
        print("   Try lowering MIN_DEPTH and MIN_INFLOW at top of script.")
        return
    
    print(f"\nProcessing {len(all_results)} result records...")
    
    # Save results
    df = pd.DataFrame(all_results)
    csv_file = os.path.join(OUTPUT_DIR, 'quantum_results_detailed.csv')
    df.to_csv(csv_file, index=False)
    print(f"✓ Saved: {csv_file}")
    
    # Generate summary report
    report_file = os.path.join(OUTPUT_DIR, 'quantum_summary_report.txt')
    with open(report_file, 'w') as f:
        f.write("QUANTUM-ENHANCED HYDRAULIC ANALYSIS\n")
        f.write("=" * 70 + "\n")
        f.write(f"Model: {os.path.basename(model_file)}\n")
        f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Records: {len(df)}\n")
        f.write("=" * 70 + "\n\n")
        
        for node_id, config in OBSERVATION_ZONES.items():
            node_data = df[df['node_id'] == node_id]
            if len(node_data) == 0:
                f.write(f"\nNode {node_id}: No data (insufficient flow)\n")
                continue
            
            f.write(f"\n{'='*70}\n")
            f.write(f"NODE: {config['name']} ({node_id})\n")
            f.write(f"{'='*70}\n\n")
            
            f.write("PEAK CONDITIONS:\n")
            f.write(f"  Max Depth: {node_data['depth_ft'].max():.2f} ft\n")
            f.write(f"  Max Inflow: {node_data['inflow_cfs'].max():.1f} cfs\n")
            f.write(f"  Max Velocity: {node_data['max_velocity'].max():.2f} ft/s\n")
            f.write(f"  Peak Scour Risk: {node_data['scour_risk_index'].max():.3f}\n\n")
            
            f.write("RECOMMENDATIONS:\n")
            max_scour = node_data['scour_risk_index'].max()
            max_vel = node_data['max_velocity'].max()
            
            if max_scour > 0.7:
                f.write("  ⚠️  CRITICAL SCOUR RISK - Protection REQUIRED\n")
            elif max_scour > 0.5:
                f.write("  ⚠️  HIGH SCOUR RISK - Protection recommended\n")
            elif max_scour > 0.3:
                f.write("  ⚠️  MODERATE SCOUR RISK - Monitor\n")
            else:
                f.write("  ✓  LOW SCOUR RISK\n")
            
            if max_vel > 10:
                f.write("  ⚠️  HIGH VELOCITY - Energy dissipation required\n")
            elif max_vel > 6:
                f.write("  ⚠️  ELEVATED VELOCITY - Consider dissipation\n")
            
            f.write("\n")
    
    print(f"✓ Saved: {report_file}")
    
    # Generate plots if possible
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(len(OBSERVATION_ZONES), 2, 
                                figsize=(12, 4*len(OBSERVATION_ZONES)))
        
        if len(OBSERVATION_ZONES) == 1:
            axes = axes.reshape(1, -1)
        
        for i, (node_id, config) in enumerate(OBSERVATION_ZONES.items()):
            node_data = df[df['node_id'] == node_id]
            if len(node_data) == 0:
                continue
            
            axes[i, 0].plot(node_data['datetime'], node_data['max_velocity'], 
                          'b-', linewidth=1.5)
            axes[i, 0].set_ylabel('Velocity (ft/s)')
            axes[i, 0].set_title(f"{config['name']} - Velocity")
            axes[i, 0].grid(True, alpha=0.3)
            
            axes[i, 1].plot(node_data['datetime'], node_data['scour_risk_index'], 
                          'r-', linewidth=1.5)
            axes[i, 1].axhline(y=0.5, color='orange', linestyle='--')
            axes[i, 1].axhline(y=0.7, color='red', linestyle='--')
            axes[i, 1].set_ylabel('Scour Risk')
            axes[i, 1].set_title('Scour Risk Index')
            axes[i, 1].set_ylim(0, 1)
            axes[i, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_file = os.path.join(OUTPUT_DIR, 'quantum_plots.png')
        plt.savefig(plot_file, dpi=150)
        plt.close()
        print(f"✓ Saved: {plot_file}")
    
    except ImportError:
        print("  (Matplotlib not available - skipping plots)")
    except Exception as e:
        print(f"  (Plot generation failed: {e})")
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE!")
    print("="*70)
    print(f"Results in: {OUTPUT_DIR}")
    print("\nOpen:")
    print(f"  • quantum_summary_report.txt (recommendations)")
    print(f"  • quantum_results_detailed.csv (full data)")
    print(f"  • quantum_plots.png (visualizations)")
    print("="*70)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\n\nFATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\nScript finished.")
