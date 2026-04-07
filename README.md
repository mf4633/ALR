# Quantum Hydraulics — Adaptive Lagrangian Refinement

Physics-based vortex particle simulation for open-channel scour screening with PCSWMM integration.

**Replaces Manning's equation with first-principles fluid mechanics** (Colebrook-White friction, Biot-Savart velocity induction, Kolmogorov cascade turbulence) while keeping computation fast enough for routine engineering screening.

[![Tests](https://img.shields.io/badge/tests-109%20passing-brightgreen)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)]()

## Key Innovation

**Observation-dependent resolution** — computational effort concentrates where engineers need measurements (bridge piers, scour-prone junctions) while maintaining coarse approximation elsewhere. 5x resolution enhancement at observation centers. 12x particle reduction with 0.2% vorticity error.

```
sigma = sigma_base / (1 + 4 * exp(-(dist / obs_radius)^2))
```

## What It Does

| Capability | Description |
|-----------|-------------|
| **ALR Vortex Particles** | Symmetrized Biot-Savart kernel (Barba & Rossi 2010) with 0.03% circulation conservation |
| **Scour Screening** | Sediment-dependent severity index, Shields parameter, Meyer-Peter Muller transport |
| **Quasi-Unsteady Sediment** | Fractional transport, Hirano active-layer armoring, Exner morphodynamic feedback |
| **Engineering Scenarios** | Bank erosion, bed degradation, culvert outlet, channel bend |
| **Pier Vortex Shedding** | Strouhal-based boundary condition with horseshoe vortex |
| **Bernoulli Free Surface** | Depth correction at constrictions and bends |
| **Benchmark Validation** | Cross-validated against Manning, Shields, Neill, HEC-18, Laursen, Melville (1997) |
| **PE Reports** | Stampable PDF reports with cover page, tables, figures, PE seal block |
| **PCSWMM Integration** | Post-processes 1D (.out) and 2D mesh exports without additional software |

## Quick Start

```python
from quantum_hydraulics import HydraulicsEngine, VortexParticleField

# Create channel
engine = HydraulicsEngine(Q=600, width=30, depth=5, slope=0.002, roughness_ks=0.15)

# Run vortex particle simulation with adaptive resolution
field = VortexParticleField(engine, length=200, n_particles=2000)
field.set_observation([120, 15, 2.5], radius=25)

for _ in range(50):
    field.step(dt=0.05)

state = field.get_state()
print(f"Particles: {state.n_particles}, Mean sigma: {state.sigmas.mean():.4f}")
```

## Headless Test Runners

Seven self-contained validation suites — no GUI, no PCSWMM required:

```bash
python run_headless_test.py          # 23 core physics checks
python run_headless_2d.py            # 16 mesh checks
python run_alr_study.py --verbose    # 24 ALR checks + figures + PDF report
python run_engineering_scenarios.py  # 21 engineering scenario checks
python run_sediment_transport.py     # 7 quasi-unsteady checks
python run_benchmark_validation.py   # 18 cross-validation checks
python run_headless_swmm.py model.out  # SWMM post-processor
```

**109 total checks, all passing.**

## Generate Reports and Figures

```bash
# ALR research report with paper-quality figures
python run_alr_study.py --figures --report

# Engineering scenario assessment
python run_engineering_scenarios.py --figures --report

# Sediment transport assessment
python run_sediment_transport.py --figures --report

# ICWMM 2026 conference paper (Word document)
python generate_icwmm_paper.py
```

## Quasi-Unsteady Sediment Transport

Step through an arbitrary hydrograph with fractional bedload transport, armoring, and morphodynamic feedback:

```python
from quantum_hydraulics.integration.sediment_transport import (
    QuasiUnsteadyEngine, ChannelReach, GrainSizeDistribution
)

channel = ChannelReach(length_ft=500, width_ft=40, slope=0.002, roughness_ks=0.1)
sediment = GrainSizeDistribution.default_sand_gravel()  # 6 fractions

engine = QuasiUnsteadyEngine(channel, sediment, upstream_feed_fraction=0.0)
engine.set_hydrograph_durations([
    (100, 2000),  # low flow: 100 cfs for 2000 hours
    (600, 100),   # bankfull
    (900, 20),    # flood
    (300, 200),   # recession
])
results = engine.run()

print(f"Scour: {results.total_scour_ft:.1f} ft")
print(f"d50: {results.initial_gradation.d50_mm:.1f} -> {results.final_d50_mm:.1f} mm")
print(f"Armored: {results.armored}")
```

## PE-Stampable PDF Reports

```python
from quantum_hydraulics.reporting import generate_scour_report, ReportConfig

config = ReportConfig(
    project_name="US-19 Bridge Scour Screening",
    project_number="2026-0142",
    client="NCDOT",
    firm_name="McGill Associates, PA",
    pe_name="Michael Flynn, PE",
    pe_license_number="12345",
    pe_state="NC",
)

generate_scour_report(design_results=results, config=config)
```

Generates professional PDFs with cover page, numbered sections, data tables, embedded figures, methodology description, limitations disclaimer, and PE signature/seal block.

## Architecture

```
quantum_hydraulics/
    core/
        hydraulics.py          # Colebrook-White engine
        vortex_field.py        # Biot-Savart + adaptive sigma + multi-zone
        particle.py            # VortexParticle dataclass
        pier_shedding.py       # Strouhal vortex injection
    integration/
        swmm_node.py           # 1D PCSWMM post-processor
        swmm_2d.py             # 2D mesh Tier 1/Tier 2
        sediment_transport.py  # Quasi-unsteady engine
    research/
        alr_experiments.py     # 5 ALR experiments
        engineering_metrics.py # Bank, degradation, culvert, bend, free surface
        engineering_scenarios.py
        sediment_scenarios.py
    reporting/
        report_generator.py    # ReportBuilder + 4 report types
    visualization/
        theme.py               # 4 professional themes
        renderers.py           # 6-panel figure layout
        interactive.py         # Real-time simulator
        export.py              # Animation/frame export
    validation/
        analytical.py          # Lamb-Oseen, Poiseuille, Kolmogorov
```

## Installation

```bash
pip install numpy scipy matplotlib reportlab
pip install pyswmm pandas  # optional, for PCSWMM integration
```

## Physics — Not Empiricism

| This Package | Instead Of |
|-------------|-----------|
| Colebrook-White friction factor | Manning's n |
| Log-law + 1/7th power velocity profile | Bulk average velocity |
| Biot-Savart vortex induction | k-epsilon turbulence model |
| Symmetrized variable-blob kernel | Uniform mesh CFD |
| Meyer-Peter Muller per grain fraction | Single d50 transport |
| Hirano active-layer armoring | Static bed assumption |
| Egiazaroff hiding/exposure | Uniform critical shear |

## Benchmark Validation

Cross-validated against six independent published methods:

| Method | Comparison | Result |
|--------|-----------|--------|
| Manning's equation | Velocity across 5 channel types | 25.5% avg offset (known CW-Manning discrepancy) |
| Shields diagram | Critical Shields vs grain size | Correct trend: decreases with Re* |
| Neill's critical velocity | Transport onset per sediment | Both increase with grain size |
| HEC-18 pier scour | Shear amplification vs CSU depth | r = 0.605 correlation |
| Laursen contraction | Shear vs width ratio | r = 0.998 correlation |
| Melville (1997) | Design curve K_I comparison | Complementary: QH adds geometric amplification |

## Limitations

- **Screening-level tool** — not for FEMA compliance or regulatory submittals
- **Not validated against laboratory flume data** — cross-validated against published design curves only
- **Fixed bed** in Tier 2 — quasi-unsteady armoring operates at reach scale only
- **Bernoulli free surface** — steady, inviscid; no hydraulic jumps or waves
- **Pier shedding** — ad-hoc Strouhal forcing, not resolved immersed boundary

For regulatory work, use HEC-RAS, MIKE, or other agency-accepted software.

## Citation

```bibtex
@article{flynn2026alr,
  title={Adaptive Lagrangian Refinement for Observation-Dependent Hydraulic 
         Simulation Using Vortex Particle Methods},
  author={Flynn, Michael},
  year={2026},
  doi={10.5281/zenodo.19462126},
  note={McGill Associates, PA. Submitted to ASCE Journal of Hydraulic Engineering.}
}
```

## License

MIT License — see [LICENSE](LICENSE).

## Author

**Michael Flynn, PE**
McGill Associates, PA
Asheville, North Carolina
