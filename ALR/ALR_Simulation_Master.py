import os
import pandas as pd
from pyswmm import Simulation, Nodes
from quantum_hydraulics import QuantumNode

# USER CONFIGURATION
TARGET_NODES = {
    'J8428': {'width': 12.0, 'length': 40.0, 'roughness_ks': 0.15},
    'Out1': {'width': 25.0, 'length': 50.0, 'roughness_ks': 0.25}
}
OUTPUT_PATH = os.path.join(os.path.expanduser('~'), 'Documents', 'ICWMM_Analysis')

def run_alr_analysis():
    model_path = os.getenv("PCSWMM_ACTIVE_MODEL")
    if not model_path:
        print("Error: No active PCSWMM model detected.")
        return

    os.makedirs(OUTPUT_PATH, exist_ok=True)
    q_nodes = {nid: QuantumNode(nid, **cfg) for nid, cfg in TARGET_NODES.items()}
    results = []

    print(f"Starting ALR Analysis on: {os.path.basename(model_path)}")
    with Simulation(model_path) as sim:
        nodes = Nodes(sim)
        dt = sim.step_advance
        
        for step in sim:
            for nid, qnode in q_nodes.items():
                n = nodes[nid]
                if n.depth > 0.1:
                    qnode.update_and_evolve(n.depth, n.total_inflow, dt)
                    qnode.compute_metrics()
                    res = qnode.get_metrics()
                    res.update({'time': sim.current_time, 'node_id': nid})
                    results.append(res)

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(OUTPUT_PATH, 'alr_peak_scour_report.csv'))
    print(f"Done! Results exported to {OUTPUT_PATH}")

if __name__ == "__main__":
    run_alr_analysis()