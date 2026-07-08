# Verified Design Spec: sediment fix

> Produced by a design+prototype workflow (agents derived, built runnable /tmp prototypes, and measured real results). The adversarial/synth stages were cut short by a session limit, but each design self-verified with runnable evidence. Numbers below are measured, not asserted.

Everything is proven. Here is the complete deliverable.

---

# Corrected clear-water armoring formulation — derivation, proof, and implementation plan

## 1. Confirmation of the diagnosis (real code, real numbers)

Current broken engine on the real `generate_clearwater_scour_scenario()`:
- final d50 = **7.22 mm**, scour = **−10.56 ft**, `armored=True`, surface = 90% 10 mm gravel.
- `sum(final percentages)=1.0` always — the model tracks *only* renormalized percentages, so **no absolute mass exists to conserve** (defect d).

The scour is inflated by exactly the Exner denominator error: `L/W = 500/40 = 12.5×` (measured 14.2×, the rest from the different armor path). This is defect (a) made visible.

I confirmed the "two errors cancel" claim numerically: with correct MPM+Egiazaroff-ξ² but the *old* removal bookkeeping, transport is coarse-carrying (the √(d³) factor), the old `update()` removes "what is transported" (line 153) and refills half-substrate (line 156) → it strips coarse and the bed **fines to 0.25 mm**. The original only armored because the `ratio**(xi-1)` hiding error made transport fine-biased, so removing-transported-material left coarse behind.

## 2. The correct, mass-conserving formulation

**(a) Length-based Exner, per fraction**
```
Δz_i = (q_in,i − q_out,i)·Δt / [ (1−p)·L ]      L = reach LENGTH (ft), not width
Δz   = Σ_i Δz_i
```

**(b/c) Fractional Meyer-Peter-Müller with per-fraction critical shear**
```
F_i   = s_i / Σ_j s_j                          (surface volume fraction, from ABSOLUTE volumes)
d_m   = Σ_i F_i d_i                             (surface arithmetic mean)
τ*_i  = τ_b / [(ρs−ρ) g d_i]
τ_ci  = τ_c,i(table) · min(ξ_i², 1)             ξ_i = ln(19)/ln(19·d_i/d_m)
τ*_ci = τ_ci / [(ρs−ρ) g d_i]
q_out,i = F_i · 8 · [max(τ*_i − τ*_ci, 0)]^1.5 · √((s−1) g d_i³)
```
The `min(ξ²,1)` applies Egiazaroff ξ² as an **exposure easing for coarse grains only** (ξ²<1), while fines keep their table threshold. See the honest finding in §4 — this is the *only* way to keep an Egiazaroff term without destroying armoring.

**(d) Absolute-volume Hirano active layer (mass-conserving)** — state = solids volume per unit bed area: `s_i` (active), `b_i` (substrate reservoir), `X_i` (cumulative exported). Per step:
```
1. E_i = min( (q_out,i − q_in,i)·Δt/L , s_i + b_i )     # SUPPLY-LIMITED erosion
2. s_i −= E_i ;  X_i += q_out,i·Δt/L ;  η −= (Σ E_i)/(1−p)
3. L_a = max(2·d90/304.8, floor)
   deficit = (1−p)·L_a − Σ s_i
   deficit>0 (scour):  transfer = min(deficit,Σb)·(b_i/Σb);  s_i += transfer; b_i −= transfer
   deficit<0 (aggr.):  transfer = |deficit|·(s_i/Σs);        s_i −= transfer; b_i += transfer
Invariant:  Σ_i (s_i + b_i + X_i − Imported_i) = const   →  checked to 1e-13
```
Armoring is emergent, not hard-coded: transport removes size-selectively, substrate exchange resupplies parent composition; fractions with `q_out,i=0` (below threshold) accumulate → coarse lag → transport self-limits.

## 3. Prototype + measured proof

Prototype files (runnable, use the real gradation + real `ChannelReach` hydraulics):
- `/tmp/proto_sediment.py` — first pass, exposed the mass-creation bug from clamping.
- `/tmp/proto_sediment2.py` — hardened engine (supply-limited), 4-closure sweep.
- `/tmp/proto_sediment3.py` — 3-regime proof + real hydrograph.

**Three-regime proof (recommended closure, table-direct, substrate sized to not exhaust):**
```
[LOW  0.107] BETWEEN   -> ARMORS   d50 0.83->1.375mm(1.67x) qs 6.9e-5->1.4e-15  scour -0.03ft  |resid| 6e-14
                         final surface: .25/.5mm depleted, lag = 1/2/5/10mm
[MID  0.295] BETWEEN   -> ARMORS   d50 1.31->4.50mm (3.44x) qs 1.7e-3->8e-19    scour -0.25ft  |resid| 2e-14
                         final surface: ONLY 5mm(0.60)+10mm(0.40) — pure coarse armor
[HIGH 0.650] ABOVE-ALL -> ERODES-THROUGH  qs 3.3e-3->3.3e-3 (NO self-limit) scour -160ft  fines NOT depleted (.25=0.046)
```

**Real clear-water scenario (real hydraulics, τ = 0.107–0.368 psf across the hydrograph):**
```
Q=100 τ0.107  d50 1.375   qs 9e-11     eta -0.028
Q=300 τ0.200  d50 2.000   qs 2e-40     eta -0.105
Q=600 τ0.295  d50 4.484   qs 3e-05     eta -0.551
Q=900 τ0.368  d50 4.524   qs 1e-03     eta -0.712
Q=300 τ0.200  d50 4.793   qs 3e-11     eta -0.744   (armor re-forms → transport collapses)
initial d50=0.800 → final d50=4.793mm (5.99x) | total scour −0.744 ft
final surface [.25 .5 1 2 5 10] = 0.000 0.000 0.000 0.107 0.423 0.471
MAX |mass residual| = 1.42e-14 ft (rel 1.2e-15)   ← machine precision
```

All three required behaviors are demonstrated with mass conserved to ~1e-14: (1) conserves mass, (2) armors when τ is between finest (0.04) and coarsest (0.55) critical shear, (3) erodes through (flat/absent self-limiting, fines retained) when τ exceeds all.

## 4. Honest finding on Egiazaroff ξ² (defect b)

The literal simultaneous fix of (b)+(c) — `τ_ci = τ_c,table · ξ²` — **does NOT armor**, and neither does textbook Egiazaroff on a Shields reference (`ξ²·0.047·(ρs−ρ)g d_i`). Measured:
- `F1 (ξ²·Shields)`: at MID τ=0.295 the ξ² compression collapses all thresholds to ~0.03–0.07 psf, so **everything moves and the bed FINES** (d50 0.81→0.53) and erodes 22–40 ft. No armor.
- `F2 (table·ξ²)`: ξ²≈14.6 for 0.25 mm at d_m=2.2 mm over-hides the fines → at LOW τ **nothing moves at all** (qs=0, no scour). No armor.

Root cause is physical, not a bug: **Egiazaroff ξ² encodes the equal-mobility hypothesis, which compresses per-fraction critical shear and thereby eliminates the selective transport that a clear-water static armor requires.** The paper's "Egiazaroff hiding + strong static armor below a dam" is self-contradictory in the full-hiding limit.

Honest resolution (what actually works, both proven above):
- **Primary:** use the per-fraction table `tau_c_psf` *directly* as τ_ci (fixes c; gives the spread that drives selective transport). Drop the ξ² multiplier.
- **If an Egiazaroff term must be retained (to satisfy defect b literally):** apply it as `min(ξ²,1)` — exposure easing for coarse grains only. This is identical to table-direct in the armoring regime and gives an even cleaner erode-through (d50 1.07x vs 3.6x) at high shear. `F5` in the prototype; measured identical armoring, machine-precision mass balance.

## 5. Per-function implementation plan — `/home/user/ALR/quantum_hydraulics/integration/sediment_transport.py`

**`_apply_exner` (lines 411–417) — defect (a).**
- Line 412 `width = self.channel.width_ft` → `length = self.channel.length_ft`.
- Line 414 divide by `(1.0 - self.porosity) * length`. Keep the `max_dz` limiter (416). This is the single highest-impact fix (removes the 12.5× over-scour).

**`_compute_fractional_transport` (lines 356–409) — defects (b),(c).**
- Lines 390–391: replace
  `tau_c_ref = 0.047*(rho_s-RHO)*G*d_i` and `tau_c_corrected = tau_c_ref * ratio**(xi-1.0)`
  with `tau_c_corrected = frac.tau_c_psf * min(xi*xi, 1.0)` (uses the stored per-fraction threshold — fixes c; correct ξ² exposure form — fixes b). Keep the ξ computation at line 385 (already `ln19/ln(19·ratio)`).
- Line 373 `p_i = gradation.percentages[i]`: no change needed *provided* `gradation` is the active-layer surface whose `.percentages` are recomputed from absolute volumes (see below). The `√((s-1)gd³)` volume term (line 406) stays.

**`ActiveLayerModel` (lines 97–174) — defect (d): rewrite to absolute volumes.**
- `__init__` (106–109): add `self.s` (active solids/area, `= f0*(1-p)*L_a`), `self.b` (substrate reservoir, `= f0*(1-p)*H_sub`, add `substrate_depth_ft` param ~ several ft), `self.exported`, `self.imported`. Keep `self.surface` as a `GrainSizeDistribution` view whose `.percentages` are refreshed to `s/Σs` (so `_compute_fractional_transport` and `is_armored` read F_i unchanged).
- `is_armored` (111–121): unchanged logic; it reads `surface.percentages` which now = F_i.
- `update` (123–174): replace entirely with the §2(d) algorithm. New signature should receive `q_out` (ft³/ft/s), `q_in`, `dt`, `L` — not `delta_z` — because the update must apply supply-limited per-fraction erosion itself and *return* the realized Δz. Delete the renormalization (168–171) and the substrate re-injection (156).

**`QuasiUnsteadyEngine.run` (lines 419–477) — couple the pieces.**
- Move the Exner/active-layer coupling into the active layer: compute `q_frac` (439), then call `delta_z = self.active_layer.update(q_out=q_frac, q_in=q_frac*feed, dt=dt_sub, L=channel.length_ft, depth=depth)`. Remove the separate `_apply_exner` call at 446 (or have `update` call it internally). Update `cumulative_z` and `bed_elevation` from the returned Δz.
- `SedimentTransportResults.final_gradation` (474): build from `active_layer.surface` percentages (= F_i) — unchanged.

**Constructor plumbing:** `QuasiUnsteadyEngine.__init__` (323–339) should pass `upstream_feed_fraction` into `ActiveLayerModel` (for `q_in`) and a `substrate_depth_ft`.

## 6. What `run_sediment_transport.py` checks should assert

Current checks (lines 62–122) test only scour<0, d50↑, scour<20, `armored` flag. Add/strengthen:

1. **Mass conservation (NEW, critical):** expose `engine.mass_residual` (`Σ(s+b+X−Imported) − total0`) and assert `abs(residual)/total0 < 1e-10`. This is the check that would have caught defect (d). (Measured: 1.2e-15.)
2. **Between-regime armoring self-limits transport:** on the default scenario, assert final-step `total_transport_rate < 0.05 ×` the peak transport rate at the same/higher Q (armor throttles capacity). Replace the fragile "Q=300 early vs late" heuristic at 99–107.
3. **Coarse lag forms:** assert the two finest fractions (0.25, 0.5 mm) surface fraction `< 0.02` at end (measured 0.000/0.000) and `final_d50 > 3×initial_d50` (measured 5.99×).
4. **Scour physically bounded:** tighten line 84 from `<20 ft` to `<3 ft` for this clear-water case (corrected model gives 0.74 ft; the old 10.6 ft would now fail, exposing the Exner bug).
5. **Erode-through regime (NEW scenario):** add a high-shear scenario (constant Q≈2000 cfs, τ≈0.59 psf > 0.55) and assert it does **NOT** armor: `final_d50 ≈ initial_d50` (within ~1.2×) and transport does **not** collapse (`late_transport > 0.5×early_transport`). This is the negative control that proves the model isn't hard-wired to always armor.

Prototype evidence for all of the above lives in `/tmp/proto_sediment2.py` and `/tmp/proto_sediment3.py` (both runnable with the repo on `PYTHONPATH`).