# Verified Design Spec: costbenefit fix

> **STATUS: IMPLEMENTED.** `run_cost_benefit` rewritten to the valid experiment; `run_alr_study.py` cost-benefit checks and all paper generators updated to the honest ~4.7x placement result. 25/25 ALR checks pass.

> Produced by a design+prototype workflow (agents derived, built runnable /tmp prototypes, and measured real results). The adversarial/synth stages were cut short by a session limit, but each design self-verified with runnable evidence. Numbers below are measured, not asserted.

I have all the evidence I need. Here is the complete deliverable.

---

# Valid ALR accuracy-vs-cost experiment — design, measured evidence, honest claim

## Bottom line

The paper's "12× particle reduction at 0.2% vorticity error" is not recoverable — it rests on two independent errors (a coarse baseline **and** a metric that doesn't measure the vortex physics at all). The honest, defensible statement the corrected field supports is:

> With a fixed, overlap-maintaining core size, the coherent-seeded induced-velocity field at fixed probes converges to a well-defined reference as ~K/√N. Concentrating particle *placement* on the observation zone (unbiased importance sampling) reaches any given **in-zone** induced-velocity accuracy with **≈4× fewer particles** than uniform placement — measured 3.9–4.6× — at the cost of **4–6× larger error outside the zone**. The reduction is ~4×, not 12×, and it comes from *placement*, not from shrinking σ.

Crucially, the **current `set_observation` mechanism (shrink σ in-zone, uniform placement) does the opposite of what's claimed**: at fixed N it makes the in-zone induced velocity *less* accurate than a plain fixed-σ field, and leaves in-zone overlap h/σ ≈ 2–6 (badly under-resolved). So `run_cost_benefit` cannot be "fixed" to show a reduction with the current mechanism — the mechanism itself has to change to particle concentration.

Prototype scripts (all run, real output above): `/tmp/alr_proto/proto.py` (reference), `/tmp/alr_proto/exp2.py`, `/tmp/alr_proto/exp3.py`, `/tmp/alr_proto/exp4.py`.

---

## 1. Why the current experiment is invalid (two bugs, not one)

**Bug A — coarse baseline (the one you identified).** `run_cost_benefit` at `alr_experiments.py:204-207` calls `vf_base.toggle_observation(False)` then `vf_base._sigmas[:] = vf_base.min_sigma`, but `step()` recomputes σ every step at `core/vortex_field.py:813`, and with observation inactive `_get_adaptive_core_sizes_batch` returns `base_sigma` (`core/vortex_field.py:631-632`). So after step 1 the "high-res" baseline is actually σ = base_sigma = 0.8 everywhere (coarse), and it's uniformly seeded so blobs barely overlap. Confirmed.

**Bug B — the metric measures nothing dynamical.** `_measure_box` (`alr_experiments.py:116-139`) reports mean σ, mean |ω|, mean enstrophy of the particles in a box. With coherent seeding ω_y = du/dz is a *deterministic function of z* (`core/vortex_field.py:554-555`), so "mean vorticity in the box" is just the average of du/dz over whoever is in the box — it does **not** involve Biot–Savart induction, blob overlap, or particle count in any meaningful way. It converges trivially and is ~N-independent regardless of resolution. That is why "0.2% error at 500 particles" looks great and means nothing. The `errors_vorticity[1]` value that the papers quote (`generate_technical_note.py:190-192`, `generate_trr_paper.py:291`, `generate_icwmm_paper.py:629`) is an artifact of this trivial metric plus the 6000/500 = 12 ratio.

The physically meaningful quantity — the thing a vortex method exists to compute — is the **induced velocity field**. That is what the valid experiment must measure.

---

## 2. Valid experiment design

**Reference (legitimate ground truth).** The coherent seed defines a deterministic vorticity field ω(x′) = (0, du/dz(z′), 0) on the interior support box. Its induced velocity at a probe is a fixed integral

```
v(x0) = ∫ K_s(|x0−x′|) [ω(x′) × (x′−x0)] dV′ ,   K_s(r) = 1/(4π (r²+s²)^{3/2})
```

with the **core size σ held fixed** at an overlap-maintaining value σ_ref (I used σ_ref = base_sigma = H/5 = 0.8 ft; h/σ ≈ 1 at N ≈ 32 k). This integral is N-independent and I evaluate it two ways that agree to ~1%: a deterministic grid quadrature (converged to ~0.1%; `proto.py:grid_reference`) and a huge-N Monte-Carlo average. This is the reference the ALR field is measured against.

**Metric.** RMS relative error of the induced-velocity *vector* over a cluster of fixed probes **inside** the observation zone (`PROBES` within ~8 ft of OBS_CENTER), plus the same for control probes **outside** the zone to quantify the tradeoff.

**Two design decisions that make it well-posed:**
- **Fix σ.** Shrinking σ with N changes the kernel/target (the design doc flags this at `docs/ALR_REFINEMENT_DESIGN.md:186-192`), so the reference wouldn't be fixed. Hold σ = σ_ref; refine only the quadrature. (This mirrors `run_convergence_study.py:30,54`.)
- **Measure at t=0, do not advect 30 steps.** Stepping (the current `_run_field`, `alr_experiments.py:110-113`) makes particle positions and the evolved vorticity themselves N-dependent (chaotic advection), so there is no N-independent reference to converge to. The static induced-velocity quadrature at the seeded coherent field is the clean, well-posed target.

**The ALR lever that actually works: placement, not σ.** Instead of shrinking σ in-zone, place a higher *density* of particles in the zone and give each particle an unbiased importance-sampling volume `Vol_j = 1/(N·p(x_j))` (the field already supports per-particle `_volumes`, `core/vortex_field.py:429,471`). This is genuine adaptive resolution: more degrees of freedom where you observe.

---

## 3. Measured results (real numbers)

**In-zone induced-velocity error vs N, fixed σ_ref (RMS over 7 probes, 20–24 seeds):**

| N | uniform | concentrated (box ±13/±9 ft, 85% in-box) |
|---|--------|-------------|
| 1000 | 0.98 | 0.45 |
| 2000 | 0.67 | 0.32 |
| 4000 | 0.49 | 0.21 |
| 8000 | 0.34 | 0.17 |
| 16000 | 0.22 | 0.10 |
| 32000 | 0.16 | 0.087 |

**Clean 1/√N scaling confirmed** (err·√N ≈ const): K_uniform ≈ 29, K_concentrated ≈ 14 → reduction = (K_u/K_c)² ≈ **4.4×**, and because both scale as 1/√N this factor is **the same at every error level**:

| target in-zone error | uniform N | concentrated N | reduction |
|---|---|---|---|
| 20% | ~21,600 | ~4,900 | 4.4× |
| 10% | ~86,400 | ~19,700 | 4.4× |
| 5% | ~346,000 | ~79,000 | 4.4× |

Sensible concentration settings span **3.9–4.6×** (box half-widths 12–20 ft; the field's own enhancement profile A=4/A=8 used as the placement density gives 2.6×/3.2×). Over-tight boxes (±8 ft) drop to 1.8× because the influence tail (~10–15 ft) gets under-sampled.

**The price of concentrating (out-of-zone control probes):** K rises from ~19 to ~40–53, i.e. **4–6× worse** far from the zone. This is exactly the observation-dependent tradeoff — and it is real and measurable, unlike the original claim.

**The current mechanism actively hurts** (fixed N, in-zone error vs the resolved reference):

| N | fixed σ_ref, uniform | **adaptive σ (obs ON), current** | in-zone h/σ |
|---|---|---|---|
| 2000 | 0.66 | **0.82** | 5.8 (100% > 1) |
| 8000 | 0.36 | **0.45** | 3.5 (98% > 1) |
| 32000 | 0.14 | **0.22** | 2.1 (91% > 1) |

Shrinking σ without adding particles both changes the target integral and violates overlap, so obs-ON is *worse* in-zone than doing nothing. This is the core reason the paper's claim can't be salvaged with the present mechanism.

**Refinement (the design-doc mechanism) restores overlap but is a different target.** `enable_refinement=True` over 5 steps: in-zone h/σ 5.07 → 0.76, n 2000 → 6691. It genuinely adds in-zone DOF — but children inherit the *reduced* in-zone σ (`core/vortex_field.py:1009`), so it resolves the small-σ field, not σ_ref, and it requires time-stepping and ~3.3× more particles. It's a valid tool, but it is not what supports a clean "fewer particles, same answer" quadrature statement.

---

## 4. The honest claim, and what is NOT supportable

**Supportable:** "Observation-concentrated particle placement achieves a given induced-velocity accuracy in the observation zone with ≈4× fewer particles than uniform placement (measured 3.9–4.6×, N-independent), at the cost of ≈4–6× larger error outside the zone. This is demonstrated against a converged, deterministic reference at fixed core size."

**Not supportable, state plainly:**
- **"12×"** — no. The measured placement reduction is ~4×. The 12× was `6000/500` combined with a meaningless metric.
- **"0.2% error"** — no. Pointwise in-zone induced velocity is *expensive*: ~10% error needs ~20 k concentrated / ~86 k uniform particles; 5% needs ~79 k / ~346 k. At N = 500 the in-zone velocity error is order 100%.
- **"fewer particles, same answer" from the current adaptive-σ mechanism** — no; it makes in-zone velocity worse (table above).
- Any claim measured on **mean vorticity/enstrophy in a box** — no; that's a trivial function of the seed, not a resolution metric.

(Minor latent issue worth noting: the field sets `Vol_j = V_domain/N` (`core/vortex_field.py:486`) while particles live only in the 0.8³ interior box, a fixed ~1.95× scale bias on absolute induced velocity. It cancels in relative-error/convergence work but would corrupt any absolute-velocity claim.)

---

## 5. Implementation plan

### A. `run_cost_benefit` — rewrite around induced velocity (`alr_experiments.py:185-250`)

1. **Add a probe induced-velocity helper** (new function in `alr_experiments.py`, near `_measure_box:116`): given a field's `_positions/_vorticities/_volumes` and a probe set + fixed s², return v at each probe (the exact kernel is `proto.py:induced_velocity`).
2. **Build a deterministic reference** once: grid quadrature of v at the in-zone probes at σ_ref (port `proto.py:grid_reference`, ~30 lines) — or a huge-N MC average. Store as the ground truth.
3. **Replace the baseline block** (`alr_experiments.py:198-207`): delete the `toggle_observation(False)` + `_sigmas[:] = min_sigma` hack; there is no "6000-particle uniform baseline" anymore — the reference is the converged integral.
4. **Replace the ALR loop** (`alr_experiments.py:225-243`): for each N, build **two** fields at fixed σ_ref — uniform placement and observation-concentrated placement — measure at **t=0** (do not call `_run_field`; drop the 30-step advection here). Add a concentrated-seeding helper: construct the field, then overwrite `_positions` (density ∝ observation enhancement or a box around OBS_CENTER), `_vorticities` (ω_y = du/dz via `hydraulics.velocity_profile_vectorized`), `_sigmas[:] = σ_ref`, and importance-sampling `_volumes = 1/(N·p)` — exactly the array-overwrite pattern already used in `run_convergence_study.py:52-56`. (Productization option: add a `seeding="concentrated"` branch to `_seed_field` at `core/vortex_field.py:512-517`, but the research-layer helper needs no core change.)
5. **Rewrite `CostBenefitResult`** (`alr_experiments.py:57-66`): replace `errors_sigma/errors_vorticity/errors_enstrophy` + `baseline_*` with `errors_uniform`, `errors_concentrated`, `errors_out_of_zone`, `reduction_factor`, and keep `particle_counts`, `wall_times`.

### B. `run_alr_study.py` checks — replace `checks_cost_benefit` (`run_alr_study.py:152-203`)

- Drop "Baseline metrics nonzero" (162-166) — no baseline metric anymore.
- Replace "Error decreases with more particles" (169-182) with **"In-zone error scales ~1/√N"** (err·√N roughly constant, e.g. CV < 25%) for both curves.
- Replace "Best ALR vorticity error < 50%" (185-191) with **"Concentrated in-zone error < uniform at matched N"** and **"Reduction factor ≥ 3×"** (measured 4.4×; set a conservative floor).
- Add **"Out-of-zone error is worse for concentrated"** (asserts the tradeoff is real, i.e. the concentration is doing something).
- Keep "Compute time increases with N" (193-201).

### C. Downstream consumers (must change together or they'll crash / keep lying)

- **Figure 3** `run_alr_study.py:432-471`: plot `errors_uniform` and `errors_concentrated` vs N (both curves) instead of vorticity/enstrophy; fix the suptitle "vs 6000-particle uniform" (`:464`) to "vs converged induced-velocity reference".
- **`report_generator.py:934-955`**: reads `baseline_vorticity`, `errors_vorticity`, `errors_enstrophy`, `wall_times` — update the table to the new fields and the ~4× reduction language.
- **Paper generators** — the false claims live here and must be rewritten to ~4×/placement, not 12×/σ: `generate_technical_note.py:190-192, 251-253, 328-329`; `generate_trr_paper.py:291-292, 656`; `generate_icwmm_paper.py:300, 616-629, 754`. Each currently quotes `cost.errors_vorticity[1]` as "X% vorticity error at 500 particles" and "12-fold reduction."

### D. Leave the core untouched
`VortexParticleField` needs no change for the experiment (per-particle `_volumes`, `overlap_ratio` at `:869`, and `observation_zone_mask` at `:924` already exist and are used above). The only optional core addition is a `seeding="concentrated"` mode if you want concentration to be a first-class field feature rather than a research-layer array overwrite.