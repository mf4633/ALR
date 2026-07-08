# Limitations and Disclaimers

## What This Software IS NOT

### NOT a Replacement for HEC-RAS/MIKE for Regulatory Work

This software is **not** suitable for:
- FEMA flood studies
- NFIP compliance analyses
- Regulatory permitting
- Design certification

For regulatory work, use peer-reviewed, agency-accepted software:
- HEC-RAS (USACE)
- MIKE (DHI)
- SWMM (EPA)

### NOT Peer-Reviewed for FEMA Compliance

This software has **not** been:
- Peer-reviewed in academic journals
- Approved by FEMA, USACE, or any regulatory agency
- Validated against field measurements at prototype scale
- Certified for any engineering design application

### NOT Validated Against Field Measurements

The validation suite includes:
- Unit tests against analytical solutions (Lamb-Oseen, Poiseuille)
- Verification of conservation laws (continuity, energy)
- Comparison to theoretical Kolmogorov spectrum

It does **not** include:
- Comparison to laboratory flume experiments
- Validation against field measurements
- Calibration to prototype-scale data

## Known Limitations

### Physical Model Limitations

| Limitation | Impact | Workaround |
|------------|--------|------------|
| No free surface deformation | Cannot model waves, hydraulic jumps properly | Use for gradually varied flow only |
| No sediment transport | Cannot predict erosion/deposition | Use empirical scour methods for design |
| 2D+1D representation | Simplifies true 3D flow | Results are approximate |
| Periodic boundaries | Artificial downstream condition | Use observation zone away from boundaries |
| Fixed bed | Cannot model mobile bed | Bed shear is output only |

### Computational Limitations

| Limitation | Impact | Workaround |
|------------|--------|------------|
| O(N^2) particle interactions | Slow for large particle counts | Use spatial trees (included), limit to ~10k particles |
| No mesh refinement | Cannot resolve sharp gradients | Place observation zone at critical locations |
| Single precision particles | Accumulation errors over time | Reset simulation for long runs |
| No parallel processing | Single-threaded only | Use smaller domains |

### Induced-Velocity Field Is Not Discretization-Independent

By default the vorticity field is seeded as a **random-phase turbulence proxy**
(`seeding="random"`; random orientations per particle), not a coherent resolved
vorticity distribution. The Biot-Savart induction correctly weights each
particle by its volume element (strength = omega * Vol), which fixes the units
and the previous sqrt(N) growth of the induced velocity with particle count.
However, because the *random-phase* seeds average to zero, the net induced
velocity is a random walk whose magnitude scales like 1/sqrt(N); with this seed
the field does **not** converge to a fixed value as particles are added.

A coherent seeding option (`seeding="coherent"`) addresses this: it seeds the
mean-shear vorticity `omega = du/dz` (a deterministic, resolved field), for which
the ensemble-mean induced velocity converges to a stable non-zero limit as N
grows (demonstrated in `run_convergence_study.py`). Combined with conservative
refinement (`enable_refinement=True`), this is the configuration in which the
adaptive-Lagrangian-refinement claim holds. Both options are **off by default**
pending recalibration of the empirical 2D scour model against the new field
(see docs/ALR_REFINEMENT_DESIGN.md, Phase 5).

Consequences:
- The adaptive-resolution ("ALR") behavior concentrates core size (sigma) near
  observation zones, but shrinking sigma without adding particles does not add
  spatial degrees of freedom; it does not produce a mesh-convergent solution.
- Claims of "same answer with fewer particles" hold only for the qualitative,
  volume-weighted statistics, not for a pointwise-convergent velocity field.

**Induction kernel is now unified.** `integration/swmm_node.py` previously had a
second Biot-Savart implementation (`_compute_velocity_induction_fast`) with the
old double-regularized, unsymmetrized kernel; it now delegates to the corrected
core kernel, so the 1D node, 2D mesh, and ALR field share one implementation.
This is output-neutral for the bundled scenarios because the 2D/1D scour path
floors the effective friction velocity at `max(u*_turb, 1.2 * u*_base)` and that
floor dominates there (the reported pier amplification is exactly 1.2^2 = 1.44).
The turbulence-augmentation constants are still empirically tuned, not
first-principles: treat the Tier 2 shear augmentation as a calibrated empirical
correction. The constants that must be refit before enabling refinement /
coherent seeding by default are catalogued in `docs/RECALIBRATION_SPEC.md`.

### Validation Gaps

| Gap | Status | Notes |
|-----|--------|-------|
| Supercritical flow | Untested | May not be accurate for Fr > 1.5 |
| High roughness | Untested | ks/depth > 0.1 not validated |
| Unsteady flow | Untested | Only quasi-steady validation |
| Bends/confluences | Untested | Only straight channels validated |

## Appropriate Use Cases

### Research and Education

- Understanding turbulence physics
- Visualizing vortex dynamics
- Teaching boundary layer theory
- Exploring adaptive resolution concepts

### Preliminary Engineering Analysis

- Screening-level scour assessment
- Relative comparison of alternatives
- Identifying critical locations
- Informing detailed analysis

### PCSWMM Enhancement

- Adding turbulence metrics to SWMM results
- Identifying high-velocity junctions
- Preliminary scour risk screening
- Research applications

## Recommendations

### Always

- Compare results to established methods (HEC-RAS, etc.)
- Use safety factors appropriate for screening-level analysis
- Document that results are from research software
- Have qualified engineer review results

### Never

- Use as sole basis for design
- Submit for regulatory approval without peer review
- Apply to conditions outside validation envelope
- Ignore physical model limitations

## Validation Status

| Test | Status | Notes |
|------|--------|-------|
| Continuity (Q = VA) | PASS | Exact to machine precision |
| Colebrook-White | PASS | <1% error vs analytical |
| Kolmogorov scales | PASS | Exact computation |
| Lamb-Oseen decay | PASS | Qualitative agreement |
| Energy spectrum slope | PARTIAL | Approaches -5/3 |
| Wall vorticity | PARTIAL | Log-law verified |

## Comparison to Other Software

| Feature | Quantum Hydraulics | HEC-RAS | MIKE 3 |
|---------|-------------------|---------|--------|
| Regulatory acceptance | No | Yes | Yes |
| Free surface | Fixed | Dynamic | Dynamic |
| Turbulence | Vortex particles | k-epsilon (optional) | k-epsilon |
| 3D flow | Simplified | 2D/3D | Full 3D |
| Computational cost | Low-Medium | Low | High |
| Adaptive resolution | Yes (observation) | No | Mesh-based |
| Open source | Yes | No | No |

## Contact

For questions about appropriate use, contact the developer or consult a licensed professional engineer.

---

**Bottom line:** This is a research tool demonstrating physics-based hydraulic concepts. It produces scientifically interesting results but is not a replacement for validated engineering software. Use judgment.
