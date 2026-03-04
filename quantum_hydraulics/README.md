# Quantum Hydraulics

A physics-based hydraulic simulation package using vortex particle methods with observation-dependent (quantum-inspired) adaptive resolution.

## Quick Start

```python
from quantum_hydraulics import HydraulicsEngine, VortexParticleField

# Create hydraulics engine
hydraulics = HydraulicsEngine(
    Q=600,           # Discharge (cfs)
    width=30,        # Channel width (ft)
    depth=5,         # Water depth (ft)
    slope=0.002,     # Bed slope
    roughness_ks=0.15  # Roughness height (ft)
)

# Create vortex particle field
field = VortexParticleField(hydraulics, length=200, n_particles=6000)

# Run simulation steps
for _ in range(100):
    field.step(dt=0.05)

# Get results
state = field.get_state()
print(f"Particles: {state.n_particles}")
print(f"Max energy: {state.energies.max():.4f}")
```

## Interactive Demo

```bash
python -m quantum_hydraulics.demos.engineering_demo
```

Or with options:
```bash
python -m quantum_hydraulics.demos.engineering_demo --quick  # Fewer particles
python -m quantum_hydraulics.demos.engineering_demo --theme light_publication
python -m quantum_hydraulics.demos.engineering_demo --export demo.mp4
```

## Installation

```bash
pip install -e .
```

Or install dependencies manually:
```bash
pip install numpy scipy matplotlib
```

Optional for PCSWMM integration:
```bash
pip install pyswmm pandas
```

## Features

### First-Principles Physics
- **Colebrook-White friction** - Not Manning's equation
- **Kolmogorov cascade** - Energy spectrum follows -5/3 law
- **Log-law velocity profile** - Boundary layer theory
- **Biot-Savart velocity induction** - True vortex dynamics

### Observation-Dependent Resolution
The "quantum" part: computation adapts to observation needs.
- Near observation zones: high resolution (small core size sigma)
- Far from observation: coarse approximation (large sigma)
- Computational cost scales with observation, not domain size

### Professional Visualization
- Multiple themes: dark_professional, light_publication, hec_ras_style
- Plan view with particle field
- Velocity profiles
- Energy spectrum
- Resolution map

### PCSWMM Integration
Analyze SWMM junctions with physics-based turbulence:
```python
from quantum_hydraulics import QuantumNode

node = QuantumNode("J3", width=20, roughness_ks=0.1)
node.update_from_swmm(depth=3.5, inflow=150)
node.compute_turbulence()

metrics = node.get_metrics()
print(f"Scour risk: {metrics['scour_risk_index']:.2f}")
```

## Package Structure

```
quantum_hydraulics/
    core/
        particle.py       - VortexParticle class
        hydraulics.py     - HydraulicsEngine (first-principles)
        vortex_field.py   - VortexParticleField (Biot-Savart)
    visualization/
        theme.py          - Color schemes
        renderers.py      - Plot functions
        interactive.py    - Interactive simulator
        export.py         - Animation export
    integration/
        swmm_node.py      - PCSWMM QuantumNode
        pcswmm_script.py  - Auto-detect integration script
    validation/
        analytical.py     - Lamb-Oseen, Poiseuille, Kolmogorov
        benchmarks.py     - Pytest test suite
    demos/
        engineering_demo.py
        conceptual_demo.py
```

## Validation

Run the test suite:
```bash
pytest quantum_hydraulics/validation/benchmarks.py -v
```

Or run the validation report:
```python
from quantum_hydraulics.validation.benchmarks import run_validation_report
run_validation_report()
```

## Documentation

- [LIMITATIONS.md](LIMITATIONS.md) - What this software is NOT
- [PHYSICS_VS_EMPIRICAL.md](../PHYSICS_VS_EMPIRICAL.md) - Technical background

## Requirements

- Python 3.8+
- numpy >= 1.20
- scipy >= 1.7
- matplotlib >= 3.5
- pyswmm >= 1.0 (optional, for PCSWMM)
- pytest >= 7.0 (for testing)

## License

MIT License - See LICENSE file.

## Citation

If you use this software in research, please cite:
```
Flynn, M. (2024). Quantum Hydraulics: Physics-based vortex particle simulation
with observation-dependent resolution. https://github.com/...
```

## Important Notes

**This is research/educational software.** See [LIMITATIONS.md](LIMITATIONS.md) for details on:
- What this software is NOT suitable for
- Known limitations
- Validation status

For regulatory hydraulic analysis (FEMA, etc.), use peer-reviewed software like HEC-RAS.
