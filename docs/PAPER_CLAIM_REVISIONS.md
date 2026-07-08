# Draft Paper-Claim Revisions (for author review)

These are **suggested** honest replacements for the quantitative claims that the
code review found are not substantiated by the experiments as written. They are
drafts — accept, edit, or reject. The underlying issue in each case is that the
experiment does not measure what the claim states (details in
`REVIEW_FINDINGS_round2.md`). None of these can be fixed by re-wording alone; the
honest options are (a) revise the experiment and re-measure, or (b) soften the
claim to what the data supports. Both are offered below.

---

## 1. "12x particle reduction at 0.2% vorticity error"

**Problem.** The cost-benefit experiment compares ALR-500 against a "uniform
high-resolution 6,000-particle baseline," but the baseline's core size is reset
to the coarse `base_sigma` every step (the forced `min_sigma` is overwritten), so
it is actually a *coarse* baseline. The "0.2% error" is measured against the
wrong reference. Separately, a uniform 6,000-particle field at `min_sigma` has no
blob overlap, so a converged high-res reference does not exist for this method as
written.

**Option A — revise and re-measure (preferred).** Build a genuine reference:
either a much larger particle count with `sigma` chosen to maintain overlap
(`h/sigma <~ 1`), or an independent analytical/gridded vorticity field; report
the ALR-N error against *that*, with a convergence curve. Only claim the ratio
the corrected study yields.

**Option B — soften to what is supported.** Replace with, e.g.:

> "ALR concentrates core size (and thus resolution) within observation zones,
> reducing the effective degrees of freedom away from them. A formal accuracy
> study against a converged high-resolution reference is left to future work;
> the present results demonstrate the resolution-concentration mechanism and its
> conservation properties, not a quantified accuracy-vs-cost trade-off."

Remove the specific "12x / 0.2%" figures from the abstract, conclusions, and the
generated tables until Option A is done.

---

## 2. "Metrics at each observation zone are independent (multi-zone)"

**Problem.** After moving Zone B, the experiment re-measures the *old Zone B box*,
never Zone A, so the returned data cannot show that Zone A is unaffected — it
measures a box the zone just left (a large change), the opposite of independence.

**Option A — revise.** After moving Zone B, re-measure **Zone A** (unchanged) and
report that its in-zone metrics are unchanged to within tolerance; that is the
actual independence test.

**Option B — soften.** Drop the independence claim, or state only that multiple
observation zones can be specified and each concentrates resolution locally,
without asserting statistical independence of the resulting metrics.

---

## 3. "Metrics converge as observation radius increases"

**Problem.** Mean sigma decreasing/flattening is forced by the
`clip(min_sigma, max_sigma)`; the box vorticity/enstrophy means are radius-
independent by construction with the coherent `omega = du/dz` seed. The figure
looks convergent regardless of any resolution effect.

**Option A — revise.** Report a quantity that genuinely depends on resolution
(e.g. the induced-velocity field at fixed probes, per `run_convergence_study.py`)
and show its behavior versus the observation radius / particle budget.

**Option B — soften.** Describe the panel as showing how core size and in-zone
sampling respond to observation radius, not as a physics "convergence."

---

## 4. Speedup framing

**Current (internally honest but easily misread).** The reported speedup is
ALR-500 vs a uniform-6,000 run — an *internal* particle-count comparison, not a
comparison to traditional methods. Measured tool runtime is ~0.3 s for a full ALR
screen; it is *slower* than the empirical HEC-18 equation (microseconds) and
faster only than equivalent physics-resolving CFD (hours), at screening (not CFD)
fidelity.

**Suggested wording.** State the baseline explicitly: "an N-fold wall-time
reduction relative to a uniform-resolution vortex-particle run of the same
domain" — and, if comparing to other tools, "turbulence-informed scour screening
in ~0.3 s versus hours for equivalent CFD, at screening rather than CFD
fidelity." Avoid an unqualified "N× faster than traditional methods."

---

## Already reconciled in the generators (done)

"circulation conserved to 0.03%" -> total vortex strength conserved to within
0.1%; "Particle Strength Exchange (PSE)" -> relaxation approximation; "Barba &
Rossi 2010" -> "Barba et al. 2005" (author to confirm the exact reference);
"first principles" -> "mechanistic." See the PR-3 commit.
