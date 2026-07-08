# Review Findings — Round 2 (reporting, analysis, research experiments, papers)

Covers the modules not in the first two PRs. **Clean correctness fixes were
applied** (see below). The items in "Requires author decision" are **not fixed**
because they concern the validity of the paper's quantitative claims or the
author's scientific wording — fixing them to "pass" would be fabrication.

## Fixed (this round)

- **`report_generator.py` (stamped PE reports):**
  - Conclusions severity thresholds (0.8/0.2) contradicted the Scour Assessment
    cell (0.7/0.3) — the same PDF could say "protection required" and
    "protection recommended" for the same risk. Aligned to 0.7/0.3.
  - Methodology described a logistic risk model, but the design-results table
    renders a linear `min(1, tau/tau_c)` ratio. Corrected the prose to the
    actual computation and noted the vortex-particle table uses a logistic.
  - Fixed the "non-saturating logistic" misnomer and the riprap D50 `.0f` vs
    `.1f` rounding (sub-inch D50 rendered as "0 in").
- **`analysis.py`:** documented that the `critical_shear = 0.15 psf` default is a
  coarse-gravel value, not sand (sand ~0.01-0.04 psf); using the default for a
  sand bed under-reports scour risk.

## Requires author decision — paper's quantitative claims are not substantiated

These are the important findings. The three headline ALR experiments do not
demonstrate what the papers state, and the paper generators hardcode the
resulting numbers.

1. **Cost-benefit "12x particle reduction at 0.2% vorticity error" is invalid as
   computed** (`research/alr_experiments.py` ~line 206). The "uniform high-res
   baseline" sets `vf_base._sigmas[:] = min_sigma`, but `step()` recomputes
   sigma every step and (observation off) resets it to the *coarse* `base_sigma`
   (~80-800x larger than `min_sigma`). So ALR-500 is compared against a uniform
   *coarse* baseline, not a high-res one. The "0.2% error / 12x" figure — hard-
   coded in `generate_icwmm_paper.py` and `generate_technical_note.py` — does not
   mean what it says. Fixing the mechanic does not rescue the claim: with only
   6000 tiny-sigma blobs the "high-res" field has no particle overlap, so a
   meaningful converged baseline does not exist for this method as written.

2. **Multi-zone "independence" is never tested** (`alr_experiments.py:486-512`).
   After moving Zone B, the code re-measures the *old Zone B box*, never Zone A,
   so the returned data cannot support the "zones are independent" claim; it
   actually measures a large change (dependence).

3. **"Convergence as observation radius grows" is largely tautological**
   (`alr_experiments.py:149-180`). Mean sigma shrinking is guaranteed by the
   `np.clip(min_sigma, max_sigma)`; the box vorticity/enstrophy means are radius-
   independent by construction (the coherent seed is a deterministic `du/dz`).
   The figure looks good regardless of the resolution effect claimed.

4. **Paper generators hardcode now-corrected overclaims** (`generate_icwmm_paper.py`,
   `generate_trr_paper.py`, `generate_technical_note.py`, `generate_scour_validation_paper.py`):
   - "circulation conserved to 0.03%" — that metric sums `|omega|`, not the
     conserved total strength `sum(omega*Vol)`; the code was corrected to measure
     strength (drift ~0%).
   - "symmetrized Particle Strength Exchange (PSE)" — the diffusion is a
     relaxation approximation, not strict PSE.
   - "Barba & Rossi 2010" — inconsistent citation for the algebraic kernel.
   - "first principles / Navier-Stokes" — softened to "mechanistic" in the README
     (Colebrook-White, log-law, MPM are semi-empirical).
   These should be reconciled with the corrected code/README before submission.

## Lower-priority (documented, not fixed)

- **Dormant vacuous validation tests** (`validation/benchmarks.py`). `pyproject.toml`
  sets `testpaths=["tests"]`, so this file is NOT run by `pytest tests/`. Several
  of its tests are tautological (Colebrook and Kolmogorov compare a value to its
  own definition via the same code path) or vacuous (`test_circulation_conservation`
  asserts only `circ>0`; `test_supercritical_detection` asserts only `Fr>0`;
  `test_energy_spectrum_slope` asserts only `energy>0`). Strengthening them to
  assert the real property (circ≈gamma, Fr>1, -5/3 slope) is worthwhile but may
  expose that the crude `epsilon=V^3/R` estimate / this particle method do not
  pass rigorous checks — a validation exercise, not a one-line fix.
- **`analysis.analyze()` and `swmm_node` use different risk models** (linear ratio
  vs logistic); one report can show two different `scour_risk_index` values for
  the same site. Unifying them changes tested behavior (the linear model floors
  at 0 for zero excess; the logistic floors at ~0.076) — a modeling decision.
- **Latent:** `np.random.seed(42)` calls in the experiments are no-ops (the field
  uses an internal `default_rng(42)`); enstrophy is an unweighted per-particle
  mean (correct only while refinement is off / volumes uniform).
