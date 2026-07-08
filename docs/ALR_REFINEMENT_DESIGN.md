# Adaptive Lagrangian Refinement — Design & Scoping (Flaw #2)

Status: **design / not yet implemented**. This document scopes the change
needed to make the "adaptive Lagrangian refinement" (ALR) claim hold. It is the
plan behind the paper's central contribution; read it before implementing.

## 1. The problem

The current ALR mechanism reduces the vortex core size `sigma` near observation
zones:

```
sigma = sigma_base / (1 + 4 * exp(-(dist / obs_radius)^2))     # up to 5x smaller
```

but it does **not** change the particle positions or add particles. A vortex
particle method is only accurate when neighbouring blobs overlap, i.e. the local
inter-particle spacing `h` satisfies the *overlap condition*

```
h / sigma  <~  1
```

Shrinking `sigma` while leaving `h` fixed drives `h / sigma` *up*, so the field
becomes **under-resolved exactly where the tool advertises the highest
resolution**. Measured on the default channel (`Q=600, W=30, H=5`, 2000
particles, obs radius 10 ft), via `VortexParticleField.overlap_ratio()`:

| Region | mean `h/sigma` | median | fraction `> 1` | mean `sigma` |
|--------|---------------:|-------:|---------------:|-------------:|
| Outside obs zone | 1.10 | 1.01 | 51 % | 0.880 |
| **Inside obs zone** | **3.19** | **3.03** | **99 %** | 0.295 |

The exterior is properly overlapped (`h/sigma ~ 1`); the interior — the point of
the method — is roughly 3x under-resolved. Reducing `sigma` by ~3x needs the
spacing `h` to shrink ~3x as well to keep overlap, which in 3D means ~`3^3 ≈ 27x`
more particles *in that zone*.

A second, independent problem compounds this (see LIMITATIONS.md): the vorticity
is seeded as a **random-phase turbulence proxy**, so the net induced velocity is
a random walk that scales like `1/sqrt(N)`. Refinement fixes the overlap /
degrees-of-freedom problem but does **not** by itself make the field converge,
because a random-phase field has no converged limit to approach.

## 2. What "done" looks like

1. In every observation zone, `overlap_ratio()["mean"] <~ 1.3` (comparable to
   the exterior), achieved by **adding degrees of freedom**, not by clamping
   `sigma`.
2. Total vortex strength is conserved by the refinement operator: the sum of
   `omega_i * Vol_i` (zeroth moment) and the strength-weighted centroid (first
   moment) are unchanged by a split.
3. A convergence study (`overlap_ratio` and a fixed-probe induced velocity vs.
   particle budget) shows the in-zone field approaching a limit as the budget
   grows — which additionally **requires** the seeding change in §5.
4. No regression when refinement is disabled (default off until validated).

## 3. Algorithm: conservative particle splitting

Refinement operates on the particles already flagged as in-zone
(`observation_zone_mask`). For each in-zone particle whose `h_i / sigma_i`
exceeds a threshold `tau_split` (e.g. 1.4):

**Split 1 -> M** (M = 7 recommended: centre + 6 face neighbours of an octahedral
stencil):

- Children inherit the parent core size `sigma_child = sigma_parent` (already
  reduced by the zone), so overlap is restored by the tighter spacing, not by
  changing `sigma`.
- Children are placed on a stencil of radius `~0.5 * sigma_parent` about the
  parent so that child spacing `~ sigma_child`.
- Strength split: `alpha_parent = omega_parent * Vol_parent` is distributed as
  `alpha_child_k = w_k * alpha_parent` with `sum_k w_k = 1` and the stencil
  chosen symmetric so `sum_k w_k * x_k = x_parent` (first moment preserved). The
  simplest valid choice is `w_k = 1/M` on a centrally symmetric stencil.
- Per-particle volume: `Vol_child = Vol_parent / M`. **This requires moving from
  the current single scalar `particle_volume` to a per-particle `_volumes`
  array** (see §4).

**Merge (coarse zones / population control):** when in a coarse region
`h_i / sigma_i` falls below `tau_merge` (e.g. 0.4), merge clusters back into one
particle carrying the summed strength and volume, placed at the
strength-weighted centroid. Needed to keep the total count bounded.

**Population control:** cap total particles at `N_max`; when exceeded, merge the
lowest-strength clusters first. Log (do not silently drop) any capping.

## 4. Data-model change required

The just-added `particle_volume` is a single scalar (`domain_volume / N`),
correct only for a roughly uniform particle distribution. Refinement makes the
distribution non-uniform, so volume must become **per particle**:

- Replace `self.particle_volume: float` with `self._volumes: np.ndarray (N,)`.
- Seeding sets `_volumes[:] = domain_volume / N` (reproduces today's behaviour).
- Biot-Savart induction multiplies each source term by its own `Vol_j` (premultiply
  strengths `alpha_j = omega_j * Vol_j` once per step, or pass `_volumes` into the
  kernels) instead of scaling the result by a scalar.
- `apply_diffusion`, pier shedding, and `update_hydraulics` must maintain
  `_volumes` alongside positions/vorticities/sigmas/ages.

This is the main invasive part and is why the change is a redesign, not a patch.

## 5. Seeding change (required for convergence, separable from §3)

Even with perfect refinement, a random-phase seed yields a `1/sqrt(N)`
random-walk field. For a discretization-independent result the seed must be a
**coherent, resolved vorticity distribution**, e.g.:

- derive the mean-shear vorticity `omega = d(u)/dz` from the velocity profile
  (deterministic, already available via `HydraulicsEngine.velocity_profile`),
  plus
- coherent structures (pier horseshoe / shed vortices) with prescribed
  circulation, rather than isotropic random `omega`.

Refinement (§3) and coherent seeding (§5) are independent; both are needed for
the ALR claim. §3 restores overlap; §5 gives a field that overlap-restored
refinement can converge to.

## 6. Phased plan

1. **Diagnostics (done).** `overlap_ratio()` and `observation_zone_mask()` on
   `VortexParticleField`, plus a regression test documenting the in-zone overlap
   violation.
2. **Per-particle volumes (done).** Scalar `particle_volume` replaced by
   `_volumes`; induction uses per-particle strength. Verified identical to the
   old scalar path for uniform volumes (~1e-15).
3. **Conservative split/merge (done, opt-in `enable_refinement=False`).**
   Octahedral 1->7 split with volume subdivision; greedy pairwise merge for
   over-dense particles; population cap `refine_n_max`. Measured: in-zone
   `overlap_ratio` mean 3.22 -> 0.70 in one pass and holds ~1.0 over 20 steps
   under the cap; total strength `sum(omega*Vol)` and total volume conserved to
   ~1e-15 / ~1e-12. Unit tests cover conservation, overlap restoration, cap, and
   multi-step stability.
4. **Coherent seeding (done, opt-in `seeding="coherent"`).** Seed the mean-shear
   vorticity `omega = du/dz` (spanwise) from the velocity profile instead of
   isotropic random phase. Convergence study (`run_convergence_study.py`): the
   ensemble-mean induced streamwise velocity at a fixed probe settles on a
   stable non-zero limit (~-0.37 ft/s; drift < 2 % past N=2000) with SNR growing
   1.2 -> 5.3 across N=500..8000, whereas random-phase seeding averages to ~0
   (SNR ~0). Pier/coherent structures continue to enter via the existing pier
   shedding.
5. **Unify kernels (done) + recalibrate (spec'd, not yet fitted).** The
   duplicate `swmm_node.py` induction kernel now delegates to the corrected core
   kernel (single source of truth); this is output-neutral because the `1.2x`
   friction-velocity floor dominates the bundled scenarios. The empirical
   constants that must be refit before the field changes underneath them
   (`1.2x` floor, per-sediment `scour_steepness`/`scour_midpoint`, Tier 2
   vorticity proxy) and the reference data each needs are catalogued in
   `docs/RECALIBRATION_SPEC.md`. Fitting them requires scour reference/flume data
   and is intentionally left to a data-in-hand calibration pass.
6. **Defaults (done, partial).** Measured the effect of flipping the defaults:
   - **Coherent seeding is now the default** (`seeding="coherent"`). It gives the
     field a converged limit, has no per-step cost, and all 24 ALR checks pass;
     only a default-asserting test needed updating. `seeding="random"` remains
     available for reproducing earlier results.
   - **Refinement stays opt-in** (`enable_refinement=False`) on purpose. Turning
     it on by default made the ALR study ~4x slower (refinement runs every step
     and grows the particle count) and regressed the vorticity-convergence check
     (`relative_diff` 0.0 -> 0.37). It is a tool to invoke when high in-zone
     resolution is needed, not an always-on cost.
   - Note: this only affects the research/ALR path -- `VortexParticleField` is
     not used by the engineering scour reports (`swmm_node`/`swmm_2d` build their
     own particle clouds), so no scour deliverable changes.

### Not done: fitting the engineering scour constants (Phase 5 refit)

Attempted and found not well-posed with the current outputs: the design curves
predict *equilibrium local pier-scour depth* (CSU/Froehlich, ft), but the tool
outputs a 0-1 shear-based risk index and a transport-based *general* degradation
rate (ft/yr) -- different physical quantities. Calibrating the constants to
"reproduce CSU" would require first adding a local-pier-scour depth model to the
tool, then real flume data to validate. Documented in RECALIBRATION_SPEC.md;
left for a data-in-hand calibration pass rather than fabricated.

### Status note (Phases 3-4)

Refinement now restores the overlap condition by adding degrees of freedom
(conservatively), and coherent seeding gives the field a converged limit to
approach -- the two pieces the "adaptive Lagrangian refinement" claim needs.
Both are **off by default** (`enable_refinement=False`, `seeding="random"`) and
change no reported result until enabled. What remains before enabling by default
is Phase 5: recalibrating the empirically-tuned 2D scour model (and unifying the
duplicate induction kernel) against the new field, since its `1.2x` floor and
logistic parameters were fitted to the old random-phase magnitudes.

Note on the convergence metric: pointwise induced velocity from a *regularized*
vortex field is resolution-sensitive when `sigma` shrinks with `N` (the kernel
itself changes), so the convergence study holds `sigma` fixed and refines only
the quadrature -- the appropriate way to isolate the seeding effect. Full
pointwise convergence under simultaneous `sigma -> 0` is a weak/integral
statement and is out of scope for the screening tool.

## 7. Validation criteria

- Moment conservation: `|sum(alpha_after) - sum(alpha_before)| / |sum(alpha_before)| < 1e-10`.
- Overlap: in-zone `overlap_ratio()["mean"]` within ~30 % of exterior after
  refinement.
- Convergence: in-zone induced velocity at a fixed probe changes by `< X %`
  when the particle budget doubles (target set once coherent seeding is in).
- No regression: full suite green with refinement disabled.
