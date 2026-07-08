# Recalibration Specification (Phase 5)

Status: **spec only — no constants changed here.** This catalogues every fitted
constant in the scour/turbulence path, what it controls, what it is currently
set to, what it was (or was not) fitted against, and the reference data needed to
refit it. It exists so the empirical model can be recalibrated against real data
before adaptive refinement and coherent seeding are enabled by default.

Nothing in this document should be "filled in" by guessing. Each constant needs
a documented data source.

## Why recalibration is needed

The turbulence-augmentation and scour-risk constants were tuned so the model
produced reasonable-looking screening numbers with the *original* (random-phase,
buggy-kernel, no-volume) field. Two things now change the field the constants
sit on top of:

1. **Coherent seeding** (`seeding="coherent"`) replaces zero-mean random
   vorticity with the deterministic mean-shear field `omega = du/dz`. The
   induced velocity (hence the turbulence-derived friction velocity `u*_turb`)
   is no longer a random walk — it has a real, larger, non-cancelling value.
2. **Refinement** (`enable_refinement=True`) changes local particle density and
   therefore the induced-velocity statistics in observation zones.

With the current constants, the `1.2x` friction-velocity floor dominates and
masks `u*_turb` entirely (pier amplification is exactly `1.2^2 = 1.44`). Once
`u*_turb` is physically meaningful (coherent seeding), it may exceed the floor,
at which point these constants directly drive reported scour and must be right.

## Constants to refit

| # | Constant | Location | Current | Controls | Fitted to? | Data needed to refit |
|---|----------|----------|---------|----------|-----------|----------------------|
| 1 | Friction-velocity floor `1.2` | `integration/swmm_node.py` (`u_star_effective = max(u*_turb, 1.2*u*_base)`); `integration/swmm_2d.py` `compute_tier2` (`u_star_eff = max(u*_turb, u_star_base*1.2)`) | 1.2 (→ 1.44x shear amplification) | Minimum turbulence amplification of bed shear at a feature | Not documented; appears ad hoc | Measured local-vs-approach bed shear (or scour depth) at bridge piers/contractions: HEC-18 CSU pier-scour equation, Melville & Chiew (1999), Laursen contraction data. Fit the floor (and ideally replace the hard floor with a smooth function of pier geometry / approach Froude). |
| 2 | `scour_steepness` (k) per sediment | `integration/swmm_node.py` `SedimentProperties` factory methods (sand 3.0, fine_sand 3.2, coarse_sand 2.5, gravel 2.0, silt 2.5, clay 1.5) | see values | Slope of the logistic scour-risk curve vs excess shear ratio | Comment says "HEC-18 derived"; actual fit not recorded | Scour-onset / scour-depth vs excess Shields data per grain class (Shields diagram; flume scour datasets). Fit k so the curve matches measured onset sharpness. |
| 3 | `scour_midpoint` (m) per sediment | same factory methods (sand 0.8, fine_sand 0.7, coarse_sand 1.0, gravel 1.2, silt 1.0, clay 1.5) | see values | Excess-shear ratio at which scour risk = 0.5 | Not recorded (sand=0.8 means onset just below nominal critical shear) | Same datasets as #2. Decide and document the intended convention (risk 0.5 at critical shear → m=1.0, or conservative onset → m<1) and fit per grain class. |
| 4 | Tier 2 vorticity proxy | `integration/swmm_2d.py` `compute_tier2` (particle "vorticity" set to the log-law velocity `u_at_z`, not a true vorticity) | log-law u(z) | Magnitude of the induced Reynolds stress → `u*_turb` | Heuristic; magnitude arbitrary | Replace with physical vorticity `omega = du/dz` (now available via coherent seeding) and calibrate resulting `u*_turb` against measured turbulence intensity / Reynolds stress in open-channel and pier flows (e.g. Nezu & Nakagawa profiles; pier-wake measurements). |

## Constants that are standard literature values (do NOT refit)

- Meyer-Peter-Muller bedload: coefficient `8.0` and exponent `1.5`
  (`phi = 8*(tau* - tau*_c)^1.5`) — established MPM constants.
- Critical Shields `~0.047` for the fully-rough regime.
- von Karman `kappa = 0.41`, roughness length `z0 = ks/30`, C_mu = 0.09.
- These should be cross-checked against their sources but not re-fitted.

## Suggested procedure

1. Assemble reference datasets: HEC-18 CSU pier scour, Melville & Chiew design
   curves, Laursen contraction scour, and a flume scour dataset with grain-class
   resolution if available.
2. Switch the Tier 2 vorticity source to `du/dz` (#4) and re-derive `u*_turb`.
3. Fit the pier/contraction amplification (#1) as a function of geometry and
   approach flow, replacing the hard `1.2x` floor.
4. Fit `k`, `m` per grain class (#2, #3) to onset/depth data; document the
   convention.
5. Re-run `run_scour_benchmarks.py` / `run_engineering_scenarios.py` and compare
   to the reference curves; record residuals in `VALIDATION_REPORT.txt`.
6. Only then set `enable_refinement=True` and `seeding="coherent"` as defaults,
   and update the paper's reported numbers.

## Regression guard

`_vectorized_scour_risk` (2D) and `_compute_scour_risk` (1D) must keep using the
same `(k, m)` for a given sediment. When refitting, change both call sites (and
the `SedimentProperties` factory values) together, and add a test asserting the
1D and 2D scour-risk curves agree for a shared sediment and shear.
