# Sediment / Morphodynamics Model — Findings

Status: **findings only — not yet fixed in code.** A review of the quasi-unsteady
sediment engine (`integration/sediment_transport.py`) surfaced several real
correctness defects. Two are individually clear and correct to fix, but fixing
them *in isolation* makes the model behave *worse* (see "Why not fixed piecemeal"),
so they are documented here for a coordinated fix validated against graded-bed
data rather than shipped half-done.

## Findings

1. **Exner update divides by width, not length** (`_apply_exner`, ~line 411).
   `dz = (qs_in − qs_out)·dt / ((1−p)·width)`. The transport `qs` is per unit
   width, so sediment continuity over a reach of length L is
   `dz = (qs_in − qs_out)·dt / ((1−p)·L)` — the divisor is the reach **length**
   (`channel.length_ft`, which is currently stored but never used), not the
   width. Overpredicts bed change by `L/W`. Partly masked today by the 1%-of-depth
   per-substep stability clip.

2. **Egiazaroff hiding/exposure factor is mis-formed** (`_compute_fractional_transport`, ~line 391).
   Code: `tau_c_corrected = tau_c_ref · ratio**(xi−1)`. Correct Egiazaroff:
   `tau*_ci = tau*_ref · xi²` with `xi = log(19)/log(19·d_i/d_m)`, i.e.
   `tau_c_corrected = tau_c_ref · xi²`. The implemented form leaves fine grains
   *unprotected* (lowers their critical shear instead of raising it).

3. **Per-fraction calibrated `tau_c_psf` is defined but never used** (field at
   ~line 29). Critical shear is recomputed from a hardwired Shields 0.047
   (~line 390), silently discarding calibrated/cohesive thresholds (3–10× larger
   for the presets).

4. **Active-layer substrate refill is not mass-conserving** (`ActiveLayerModel.update`,
   ~line 156): `surface.percentages += substrate.percentages · removal_ratio · 0.5`
   re-injects fine-rich substrate every scour step, then the array is renormalized,
   so the two-layer sediment balance is not conserved.

5. **Bed-elevation → hydraulics feedback is absent** (documented simplification):
   `compute_hydraulics` uses the fixed slope and never reads `bed_elevation`, so
   the accumulating bed change never alters depth/shear. Likely intentional
   (uniform flow at fixed slope), noted against the module docstring's claim.

## Why not fixed piecemeal

Fixing #2 (Egiazaroff) alone was tested: the bundled clear-water-scour demo then
**fines to the smallest grain size (d50 → 0.25 mm) at every flow and never
armors**, which is unphysical. Removing the substrate refill (#4) as well did not
restore armoring. Diagnosis: the fractional transport is coarse-dominated (the
`√(d_i³)` volume factor amplifies coarse transport), and the *buggy* hiding
function was **compensating** for that — lowering fine critical shear so fines
washed out and the bed appeared to armor. Correcting the hiding function removes
the compensation and exposes the coarse-dominated-transport error underneath.

In other words, the model's plausible armoring behavior depended on two errors
cancelling. A correct model requires fixing the transport/active-layer coupling
**together** with the hiding function, and validating against graded-bed flume
data (e.g. Parker, Wilcock–Crowe, or lab armoring datasets) — not tuning the demo
scenario until `armored == True` reappears.

## Recommended coordinated fix (needs validation data)

1. Length-based Exner (#1) using `channel.length_ft`.
2. Egiazaroff `xi²` hiding (#2).
3. Use the calibrated per-fraction `tau_c_psf` where provided (#3), falling back
   to Shields only when absent.
4. A mass-conserving two-layer (Hirano) active-layer exchange (#4): remove
   transported fractions and entrain substrate at the substrate composition with
   a conserved bookkeeping (track absolute volumes, not renormalized percentages).
5. Re-validate against a graded-bed armoring dataset; only then update the
   `run_sediment_transport.py` checks to assert the (correct) armoring behavior.

Until then the quasi-unsteady sediment engine should be treated as a qualitative
demonstration, not a quantitative morphodynamic predictor.
