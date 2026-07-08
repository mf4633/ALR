# Review Findings — Round 4 (legacy scripts, validation scenarios, interactive)

Final sweep of the remaining surface. Clean fixes applied (see the commit);
items below need source documents or an author decision.

## Fixed / actioned

- **`Quantum_Fluid.py`, `Quantum_Fluid-G.py`** — added a DEPRECATED/LEGACY header.
  Both are dead standalone scripts (not imported anywhere). `Quantum_Fluid-G.py`
  is the pre-package monolith and still contains all five physics bugs that were
  corrected in `quantum_hydraulics/` (TKE=0.5 V^2, double-regularized kernel,
  no-dt diffusion, missing volume element, discontinuous velocity profile) plus a
  Biot-Savart displacement-sign inconsistency and a stale-KD-tree rebuild. They
  are safe to delete; the header prevents anyone running the buggy versions
  meanwhile.
- **`validation/benchmark_scenarios.py`** — corrected a descriptive `notes`
  string (K2=1.25 -> 1.31, the value the HEC-18 formula actually gives for
  theta=7.5 deg, L/a=4). Notes only; no computed input changed.

## Needs the source documents (not fixed — would be guessing)

1. **`scenario_hec18_example_4` contracted section violates continuity**
   (benchmark_scenarios.py ~lines 220-234). Upstream discharge auto-computes to
   `100*4.3*7.1 = 3053 cfs`, but the bridge section gives `95*4.8*12.8 = 5837 cfs`
   -- ~91% more flow through a 5%-width contraction, with depth *rising* and
   velocity nearly doubling, which is impossible. Any check reading
   `contracted_section` is fed a non-physical discharge. The pier benchmark
   (which uses the approach values V1=7.1/y1=4.3) is unaffected. Fixing requires
   the actual HEC-18 Example 4 values.

2. **`scenario_hecras_example_11` abutment inputs cannot reproduce the asymmetric
   published values** (~lines 160-195). Left and right abutments are given
   identical geometry (spill_through, L'=200, theta=90) but the published results
   are asymmetric (left 10.92 ft, right 15.2 ft). Identical inputs must give
   identical HIRE scour, so one abutment benchmark always misses its band. The
   real Example 11 has asymmetric overbank flow the scenario does not encode.

Note: the transcribed scour/velocity magnitudes in these scenarios could not be
independently confirmed against the original HEC-18/HEC-RAS/FHWA documents; the
two items above are the internal-consistency defects that are demonstrable
arithmetically. A pass against the source documents is recommended.

## Clean

- **`visualization/interactive.py`** — no correctness bugs (constructor arg order,
  slider->parameter mapping with correct closure binding, animation toggle,
  frame counter, and checkbox/observation default all verified correct).
