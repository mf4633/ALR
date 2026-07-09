# Memo to the Editor — Journal of Water Management Modeling

**Manuscript:** *Turbulence-Aware Post-Processing of EPA SWMM Output Using Adaptive Vortex Particle Methods*
**Author:** Michael Flynn, PE
**Date:** 2026-07-09
**Subject:** Author-initiated corrections prior to publication

---

Dear Editor,

While the manuscript has been under review, I completed a systematic audit of the
open-source reference implementation that accompanies the paper. That audit
identified several places where the manuscript reported numbers produced by
defective code paths rather than by the physics the paper describes. I am
submitting a revision that corrects these, together with this memo documenting
every change and its cause. I would rather correct the record now, before
publication, than have a reader discover it against the released source.

The corrections do not change the **thesis** of the paper — that a vortex-particle
post-processor can occupy the gap between Manning's equation and 3-D CFD for
screening-level work. They do change several **quantitative claims**, and they
require the central Tier-2 "turbulence amplification" result to be reframed. I
summarize the changes below in decreasing order of significance. All revised
numbers are reproducible from the tagged source with a single benchmark runner,
as before.

## 1. The Tier-2 "1.44× amplification" was a code artifact (most significant)

The manuscript reported (Sections 3.4, 3.5, 4.3; Tables 3–4; Conclusion 4) that
the Tier-2 vortex-particle analysis amplifies bed shear by **1.44×** through
Biot–Savart-resolved Reynolds stresses. On audit, this value was not a physical
measurement. Three defects in the Tier-2 routine combined to produce it:

1. the seeding placed the log-law **velocity** `u(z)` into the streamwise
   vorticity component (wrong physical quantity, wrong component) instead of the
   coherent mean-shear vorticity `ω_y = du/dz`;
2. random particle placement produced a 1/√N sampling artifact in the resolved
   Reynolds stress; and
3. the effective friction velocity was floored at `1.2·u*`, so the returned
   shear was pinned at `1.2² = 1.44×` whenever the (artifactual) vortex stress
   fell below the floor — which was every case tested.

With the seeding corrected, the placement made convergent, and the floor
removed, a mean-shear-only vortex field correctly returns **amplification =
1.00×** — the constant-stress-layer limit `−⟨u′w′⟩ → u*²`. In other words, the
Tier-2 step as implemented adds no bed-shear amplification beyond the Tier-1
geometric-blockage estimate, because the coherent near-pier structures that
would create real amplification (horseshoe and shedding vortices) are not
injected into the Tier-2 control volume.

**Revision.** I have withdrawn the Tier-2 shear-amplification claim and reframed
Tier-2 as what it actually computes: a resolved **turbulence-field diagnostic**
(turbulent kinetic energy / turbulence intensity), not a bed-shear multiplier.
The bed-shear screening signal in the paper is carried entirely by **Tier-1**
(Colebrook–White friction plus geometric blockage, ≈1.09× at the benchmark
pier), which is unchanged and which is what actually correlates with the HEC-18
and Laursen benchmarks. The Section 3.5 CFD comparison (Roulund et al., 2005)
is rewritten accordingly: ALR does not attempt to reproduce the near-pier CFD
peak, and the corrected Tier-2 result makes explicit that resolving that peak
requires injected coherent structures with empirically (flume-)calibrated
magnitude — now stated as future work rather than an achieved result.

I regard this as a strengthening, not a weakening: the revised paper no longer
claims a physics-based amplification it did not actually compute, and its
positioning (a Tier-1 geometric + turbulence-intensity screen, explicitly below
CFD) is cleaner and fully supported by the benchmarks that do hold.

## 2. Performance claim reframed (Sections 4.2, 5.4; Abstract; Conclusion 2)

The manuscript stated that a 500-particle ALR run "recovers the vorticity field
of a 6,000-particle uniform baseline to 0.18% error at roughly 19× the
wall-clock speed" (a 12× particle reduction). This paired an accuracy figure and
a speedup that came from a since-replaced experiment and are not reproducible
from the current code.

The corrected cost–benefit experiment measures a well-defined quantity: because
vorticity error scales as 1/√N, observation-concentrated placement reaches the
**same in-zone accuracy with ≈4.7× fewer particles** than uniform placement, with
correspondingly lower wall time under the sub-quadratic 6σ-cutoff scaling. I have
replaced the "12× / 0.18% / 19×" language throughout with this measured 4.7×
particle-reduction result. Circulation conservation (0.03%) and the
observation-radius convergence result are unchanged and still hold.

## 3. Sediment-transport demonstration regenerated (Section 4.4; Table 7; Fig. 5)

The active-layer/Exner sediment routine used in the extensibility demonstration
did not conserve mass; it over-eroded, producing the reported 10.6 ft of
degradation and 9× (0.80 → 7.2 mm) surface coarsening. I rewrote it as a
mass-conserving Hirano active-layer model (mass residual now ~10⁻¹⁵). With
conservation enforced, armoring is **self-limiting**, as it should be. For the
paper's exact scenario (500 × 40 ft channel, zero feed, the 5-step / 2920-h
hydrograph with 900 cfs peak), the corrected engine coarsens the surface
**0.80 → 5.0 mm (6.3×)** and degrades the bed by **~0.31 ft**, with mass
conserved to ~10⁻¹⁵. The former 10.6 ft was an over-erosion symptom of the
non-conserving update; the erosion is transport-limited (the per-step Courant
limiter never binds), not artificially clipped. The qualitative conclusion
(armor formation limits long-term degradation below a dam) is unchanged and, if
anything, better illustrated. Table 7 and Figure 5 are regenerated accordingly.

## 4. Critical-shear physics corrected (Section 2.4; Section 3.2 Shields benchmark)

The critical shear stress used for the Shields parameter and Meyer-Peter-Müller
transport was a hard-coded value that implied a physically impossible Shields
parameter (θ_c ≈ 0.6 for medium sand). It is now derived from the Shields curve
via the Soulsby & Whitehouse (1997) explicit fit, giving θ_c in the standard
0.03–0.06 band. This is an internal-consistency fix with no effect on the
Tier-1 bed-shear benchmarks; it does correct the Shields-diagram discussion,
where θ_c is now shown (correctly) to be non-monotonic in grain size.

## Claims that are unchanged and were re-verified

- Manning vs Colebrook–White velocity offset: 25.5% (Table 1).
- HEC-18 CSU pier-scour correlation: **r = 0.605** (Section 3.2).
- Laursen contraction-scour correlation: **r = 0.998** (Section 3.3).
- Melville geometric-blockage amplification ≈ 1.11× (Section 3.4).
- Circulation conservation 0.03%; observation-radius convergence (Sections 2.2, 4.1).

These rest on the Tier-1 hydraulics and are unaffected by the corrections above.

## Reproducibility

Every revised number is produced by the tagged open-source release via the
one-command benchmark runner. The test suite (288 unit/validation tests) and all
benchmark scripts pass, including new regression tests that lock in the corrected
Tier-2 behaviour (analytical constant-stress-layer limit, convergence, and
absence of the former floor).

I appreciate the opportunity to correct these before publication and am happy to
provide the audit trail (commit history and per-change validation) if useful to
the reviewers.

Sincerely,

Michael Flynn, PE
Asheville, North Carolina
