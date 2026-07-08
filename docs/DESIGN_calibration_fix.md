# Verified Design Spec: calibration fix

> **STATUS: IMPLEMENTED (validation) + DOCUMENTED (recalibration).** Added `contraction_scour_equilibrium` + `incipient_tau_c_psf` to engineering_metrics.py and benchmark #8 (transport-continuity vs Laursen, r=1.000, ratio 0.98). Option A (fit alpha to CSU) rejected as circular/unphysical, per this analysis. The GLOBAL Vc-anchoring of the tuned MPM threshold (which changes reported scour) is left as an author calibration decision; the finding (incipient tau_c ~4.4x below the allowable default) is now validated in the suite.

> Produced by a design+prototype workflow (agents derived, built runnable /tmp prototypes, and measured real results). The adversarial/synth stages were cut short by a session limit, but each design self-verified with runnable evidence. Numbers below are measured, not asserted.

I have all the evidence I need. Here is my analysis and recommendation.

---

# Most defensible way to calibrate/validate the engineering scour output

## Bottom line (recommendation)

**Do NOT calibrate the pier shear-amplification factor to CSU pier-scour depth (Option A). It is both circular and physically impossible to do with a well-behaved amplification factor.** The two quantities have incompatible velocity scaling, and any α you fit to bridge them is closure-dependent and takes an unphysical (Froude-decreasing) form.

**Do adopt Option B, in two parts, both backed by real measured agreement:**

1. **Non-circular anchor — calibrate the incipient-motion threshold τ_c against HEC-18 critical velocity `Vc`** (`critical_velocity`, hec18_scour.py:89). These are independent formulations of the *same* physics (Shields/incipient motion), so agreement is a genuine cross-check, not a fit-to-a-fit.
2. **Like-for-like validation — validate the transport-based general-scour output against Laursen contraction scour**, not against CSU. Both are reach-scale sediment-continuity phenomena. When the tool's Meyer-Peter-Müller transport is closed with sediment continuity, its equilibrium contraction depth reproduces Laursen at **r = 0.989**, rising to **r = 1.000 (ratio 0.98)** once τ_c is anchored to `Vc`.

CSU local pier scour should be kept as a **separate design-equation module** (it already is, in hec18_scour.py) and only compared against the FHWA flume *measurements* (scenario 4), which is the only real lab data in the repo. The tool does not compute local pier-scour depth, so it cannot be validated against CSU without fabricating a bridge.

---

## 1. The dimensional mismatch (why the current comparison is invalid)

| Quantity | Produced by | Units | Physical type |
|---|---|---|---|
| `scour_risk_index` | `_compute_scour_risk` (swmm_node.py:546), `_vectorized_scour_risk` (swmm_2d.py:93) | 0–1 | logistic index of τ/τ_c |
| `scour_depth_potential` | `_compute_sediment_transport` (swmm_node.py:575), `_vectorized_meyer_peter_muller` (swmm_2d.py:111) | ft/yr | transport **rate** → general degradation |
| CSU `ys` | `csu_pier_scour` (hec18_scour.py:232) | ft | **equilibrium local** pier-scour depth |

None of these three is dimensionally comparable to CSU `ys`. The current `checks_parametric_sweep` (run_scour_benchmarks.py:569) sidesteps this by correlating the raw **constriction shear** `RHO*e_pier.u_star**2` (line 596) against CSU depth. Measured behavior of that proxy (real modules, /tmp/probe1.py):

- **Pooled Pearson r ≈ 0.36** across the velocity+depth+pier_width sweeps (0.359–0.373 depending on pooling). The per-sweep "|r|>0.7 PASS" checks are an artifact of testing one variable at a time.
- **Depth sweep: r = −0.996 — wrong sign.** CSU scour *rises* with depth (`ys ∝ y1^0.35`) while constriction shear *falls* (deeper → slower → lower τ).
- **Sensitivity is grossly mismatched.** Over the pier-width sweep CSU ranges ×3.86 while constriction τ ranges only ×1.24; over the velocity sweep CSU ranges ×2.45 while τ ranges ×63.6.
- **Constriction amplification is only α = 1.03–1.28×**, versus the physical horseshoe-vortex bed-shear amplification at a pier of ≈2–11× (Hjorth/Melville). Removing a 3-ft pier from a 60-ft channel barely perturbs the section-average shear, so constriction shear is not a physical proxy for the *local* pier-scour driver at all.

---

## 2. Option A (calibrate α = f(Fr, a/y1) to CSU): circular AND unphysical — rejected

Recasting CSU dimensionlessly: **ys/y1 = 2·K·(a/y1)^0.65·Fr^0.43**. A regression of `log(ys/y1)` on `log(Fr)`, `log(a/y1)` over the sweep (/tmp/probe2.py) returns **C = 2.200, p = 0.430, q = 0.650, R² = 1.00000** — i.e. it just re-derives CSU algebraically. That is the definition of circular: any α forced to reproduce CSU merely re-encodes CSU and can never *validate* against it.

Worse, it cannot even be done with a physical α:

- **Velocity scaling is incompatible** (/tmp/probe5.py): CSU depth ∝ V^0.43, but the tool's shear ∝ V^2.0. To make `α·τ` track CSU you need **α ∝ V^−1.57** — an amplification factor that *decreases* with velocity, the opposite of a real pier horseshoe-vortex amplification.
- **The depth→shear bridge needs an unmeasured closure exponent n** (τ_pier/τ_c = (ys/a)^n). Sweeping n = 0.5/1.0/1.5 gives fitted forms `α ∝ Fr^−1.72…−1.29·(a/y)^−0.05…−0.40` spanning **α = 0.4–16×** — i.e. the "answer" is whatever you assume for n. Values α<1 are physically impossible (amplification below the approach shear).

So Option A's only real merit would be giving the *index* the correct CSU functional dependence (fixing the wrong-sign depth response). But that is a design-curve re-encoding, explicitly not validation, and it corrupts the tool's physically-correct τ∝V² shear.

---

## 3. Option B (validate transport-based general scour vs Laursen): defensible — recommended

### 3a. Non-circular threshold anchor: τ_c ↔ Vc

For each sediment, find the velocity at which the tool's `RHO*u_star**2` reaches `critical_shear_psf`, and compare to HEC-18 `Vc` (/tmp/probe2.py):

| Sediment | D50 | tool Vc(τ_c) | HEC-18 Vc | ratio |
|---|---|---|---|---|
| fine_sand | 0.20 mm | 3.24 | 1.27 | 2.56 |
| sand | 0.50 mm | 4.19 | 1.72 | 2.43 |
| coarse_sand | 1.00 mm | 5.13 | 2.17 | 2.36 |
| gravel | 10.0 mm | 7.26 | 4.68 | 1.55 |

The tool's `critical_shear_psf` values (0.06–0.30 psf) sit ~1.5–2.6× too high **in velocity terms** (≈4–6× too high in shear). For 0.5 mm sand the `Vc`-consistent τ_c is **0.0226 psf vs the tool's 0.10 psf (4.4× too high)**, /tmp/probe4.py. This matters because the same τ_c is (mis)used as the MPM incipient threshold in `_compute_sediment_transport` (swmm_node.py:586,604) and `_vectorized_meyer_peter_muller` (swmm_2d.py:120–123): an allowable-level τ_c plugged into a transport law systematically suppresses computed transport. **This is a genuine, independent, non-circular calibration target.**

### 3b. Like-for-like validation: transport-continuity equilibrium vs Laursen

Closing the tool's MPM transport with sediment continuity (transport-in = transport-out through the contraction) yields an *equilibrium* general-scour depth that is directly comparable to Laursen `live_bed_contraction_scour` (hec18_scour.py:400). MPM and Laursen-Manning are independent transport laws, so agreement is meaningful (/tmp/probe3.py, /tmp/probe4.py):

| W2 (ft) | Laursen ys | tool (τ_c=0.10) | tool (τ_c=0.0226, Vc-anchored) |
|---|---|---|---|
| 550 | 0.62 | 0.00 | 0.61 |
| 500 | 1.34 | 0.00 | 1.32 |
| 450 | 2.20 | 0.81 | 2.16 |
| 400 | 3.23 | 1.99 | 3.16 |
| 350 | 4.50 | 3.49 | 4.39 |
| 300 | 6.13 | 5.46 | 5.94 |

- With the original τ_c: **r = 0.989, mean ratio 0.44** (under-predicts, and predicts zero scour at mild contraction because the too-high τ_c is never exceeded).
- After the **non-circular Vc anchor** (3a): **r = 1.000, mean ratio 0.98.**

The two findings reinforce each other: fixing the threshold via the independent `Vc` anchor is exactly what tightens the independent Laursen comparison.

---

## 4. Honest statement of what this establishes — and what it does not

- **Establishes:** (i) internal *threshold* consistency between the tool and HEC-18 incipient-motion physics (τ_c ↔ Vc), and (ii) *design-curve consistency* between the tool's transport-based general scour and Laursen contraction scour, using independent transport laws.
- **Does NOT establish:** flume/field validation. Laursen and CSU are themselves empirical design equations; agreeing with them is agreeing with a curve, not with nature. The only measured data in the repo are the FHWA flume pier-scour depths (benchmark_scenarios.py:334), and those validate *CSU* (already done at CSU-vs-measured r>0.8 in `checks_fhwa_flume`), not the tool.
- **The tool does not compute local pier-scour depth at all.** CSU/Froehlich must remain a separate design-equation deliverable (hec18_scour.py). Do not claim the tool "reproduces CSU."
- **Any Option-A-style calibration of α to CSU would be circular** (fitting one model to another) and should not be presented as validation.

---

## 5. Concrete implementation plan

**Change 1 — Anchor τ_c to incipient motion; separate it from the allowable/midpoint value.**
- File: `quantum_hydraulics/integration/swmm_node.py:172–206` (`SedimentProperties`). Add a field `tau_c_incipient_psf` (Shields/`Vc`-consistent, e.g. ~0.005–0.023 psf for 0.5 mm sand) distinct from `critical_shear_psf` (keep as the allowable used by the logistic midpoint). Provide a classmethod that derives `tau_c_incipient_psf` from `critical_velocity(y_ref, D50)` (hec18_scour.py:89) so the anchor is explicit.
- Use `tau_c_incipient_psf` (not `critical_shear_psf`) as the MPM incipient threshold at: `_compute_sediment_transport` swmm_node.py:586–588 and 604, and `_vectorized_meyer_peter_muller` swmm_2d.py:120–123. Leave the logistic `_compute_scour_risk` (swmm_node.py:555–571) and `_vectorized_scour_risk` (swmm_2d.py:93–100) using the allowable `critical_shear_psf`/`scour_midpoint`.

**Change 2 — Replace the ad-hoc 1.2× floor with a defensible, documented turbulence floor (do NOT tie it to CSU).**
- Files: `swmm_node.py:513` and `swmm_2d.py:449` (`max(u_star_turb, u_star_base * 1.2)`).
- Keep it as a *local turbulence/structure* enhancement, but justify the magnitude physically (constriction hydraulics + a horseshoe-vortex factor) and label it as an index enhancement, not a pier-scour-depth predictor. Do not make it a function fitted to CSU depth.

**Change 3 — Add a transport-continuity closure so the tool can emit an equilibrium general-scour depth (ft), enabling the like-for-like Laursen comparison.**
- Best home: `quantum_hydraulics/research/engineering_metrics.py` alongside `compute_degradation` (engineering_metrics.py:203). Add `compute_contraction_scour_equilibrium(approach, contracted, sediment)` that solves `qs(V2,y2)·W2 = qs(V1,y1)·W1` (the bisection in /tmp/probe4.py). This does not modify the Tier-1/Tier-2 pipeline.

**Change 4 — Fix / replace the misleading benchmark checks.**
- `run_scour_benchmarks.py:569–647` (`checks_parametric_sweep`): stop correlating constriction shear against CSU depth (pooled r≈0.36, wrong-sign depth). Replace with: (a) a **τ_c↔Vc threshold check** per sediment (assert tool Vc(τ_c_incipient) within tolerance of HEC-18 `Vc`), and (b) a **transport-continuity-vs-Laursen check** across a contraction sweep (assert r>0.95 and mean ratio in ~0.8–1.2). Both are backed by the measured numbers above.
- Keep CSU-vs-measured (`checks_fhwa_flume`, run_scour_benchmarks.py:488) unchanged — that is the only real validation and it validates CSU, not the tool.
- Optionally relabel `checks_hec18_example_4`'s "QH detects shear amplification" line (run_scour_benchmarks.py:396) so it is not read as a scour-depth validation.

**Prototypes with the real modules:** `/tmp/probe1.py` (mismatch + constriction proxy), `/tmp/probe2.py` (CSU exponent recovery + closure-dependence + τ_c↔Vc), `/tmp/probe3.py` (transport-continuity vs Laursen), `/tmp/probe4.py` (Vc-anchored recalibration → r=1.000), `/tmp/probe5.py` (pooled r and velocity-scaling incompatibility).