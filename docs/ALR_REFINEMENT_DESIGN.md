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
2. **Per-particle volumes.** Swap scalar `particle_volume` for `_volumes`;
   induction uses per-particle strength. Verify no change vs. today when volumes
   are uniform (bit-for-bit or within fp tolerance).
3. **Conservative split/merge** behind `enable_refinement=False`. Unit tests:
   zeroth/first-moment conservation; `overlap_ratio` in-zone drops toward 1.
4. **Coherent seeding** option; convergence study across particle budgets.
5. **Recalibrate** the 2D scour model (`swmm_node.py` kernel + `1.2x` floor +
   logistic params) against the refined field, and unify the two induction
   kernels.
6. Flip refinement on by default only after 3-5 validate; update the paper's
   claims to match the measured convergence.

## 7. Validation criteria

- Moment conservation: `|sum(alpha_after) - sum(alpha_before)| / |sum(alpha_before)| < 1e-10`.
- Overlap: in-zone `overlap_ratio()["mean"]` within ~30 % of exterior after
  refinement.
- Convergence: in-zone induced velocity at a fixed probe changes by `< X %`
  when the particle budget doubles (target set once coherent seeding is in).
- No regression: full suite green with refinement disabled.
