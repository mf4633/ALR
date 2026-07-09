# Revised Manuscript Sections — ALR / JWMM

This document contains replacement text for every passage affected by the
code audit. Sections not listed here are unchanged. All numbers are reproducible
from the tagged open-source release. Struck/old values are shown in *[brackets]*
for the editor's convenience and should be removed in the clean copy.

---

## Abstract (replacement)

Hydraulic engineers in practice face a pronounced modeling gap. The Manning
equation is fast and inexpensive but collapses the turbulent boundary layer into
a single empirical roughness coefficient. Three-dimensional computational fluid
dynamics resolves that boundary layer but typically requires weeks of setup,
specialized software, and a dedicated consultant, placing it out of reach for
preliminary scour screening, design-alternative comparison, and routine
stormwater, transportation, and dam-safety work. This paper presents a
vortex-particle post-processor for the Storm Water Management Model and its
Personal Computer variant that provides physics-based turbulence information at
user-selected critical locations without mesh generation or outside expertise.
The method, termed Adaptive Lagrangian Refinement, uses an observation-dependent
resolution rule: particle core sizes contract near locations the engineer
designates as critical and expand elsewhere, concentrating computation only
where it affects the answer. The underlying symmetrized variable-blob
Biot–Savart kernel preserves circulation to 0.03%. Because vorticity error
scales as 1/√N, observation-concentrated placement reaches a given in-zone
accuracy with approximately **4.7× fewer particles** than uniform placement, at
correspondingly lower wall time. *[replaces: "a 500-particle run recovers the
vorticity field of a 6,000-particle uniform baseline to 0.18% error at roughly
19 times the wall-clock speed"]* Benchmark validation cross-checks the Tier-1
bed-shear field against six empirical design families for bridge-pier and
contraction scour and against published computational-fluid-dynamics results
from the bridge-pier literature. The reference implementation is open-source.

---

## Section 2.4 (revise closing sentence)

The critical shear stress governing the Shields parameter and Meyer-Peter-Müller
transport is derived from the Shields curve using the Soulsby & Whitehouse (1997)
explicit fit, θ_c = 0.30/(1 + 1.2 D*) + 0.055[1 − exp(−0.020 D*)], with D* the
dimensionless grain size. This yields critical Shields parameters in the standard
0.03–0.06 band for sand through gravel. A separate, empirically calibrated design
shear centres the logistic scour-severity classifier; incipient motion and
scour-design onset are treated as distinct thresholds.

---

## Section 3.2 (Shields discussion — revise trend sentence)

The computed critical Shields parameters follow the Shields curve, which is
**non-monotonic** in grain size: θ_c passes through a minimum near D* ≈ 10
(medium-to-coarse sand) and rises both toward finer (viscous) grains and toward
the fully-rough plateau (≈0.05–0.06) for gravel. *[replaces any statement that
θ_c decreases monotonically with grain size.]* The HEC-18 CSU correlation
(r = 0.605) is unchanged.

---

## Section 3.4 (Melville comparison — revise the amplification discussion)

For this configuration (3-ft / 0.9-m pier in a 40-ft / 12-m channel, blockage
ratio 0.075), the Tier-1 Colebrook–White analysis provides a geometric-blockage
shear amplification of ≈1.11×, independent of flow intensity (Table 3). This
geometric factor is complementary to Melville's flow-intensity parameter K_I:
Tier-1 captures the blockage (pier width / channel width), while Melville's K_I
captures the flow-intensity dependence. A practitioner would use both:

  d_s,combined = d_s,Melville × (Tier-1 blockage factor)

*[Delete the previous sentence attributing a higher 1.44× amplification to Tier-2
Biot–Savart Reynolds stresses. See Section 4.3 and the revised Section 3.5.]*

---

## Section 3.5 (Comparison with published CFD — replacement)

The empirical benchmarks of Sections 3.1–3.4 situate ALR within established
engineering design practice. A complementary comparison against a published CFD
result situates it within the physics-based modeling hierarchy.

Roulund et al. (2005) reported combined laboratory-flume and 3-D RANS simulations
of the flow around a circular pile (D = 0.1 m, approach velocity 0.46 m/s,
Re_D ≈ 46,000). Their key result is the bed-shear amplification τ/τ∞ around the
pile: peak amplification of order 10–11 at the boundary-layer separation line on
an undeformed bed, decreasing to order 3–4 as a scour hole develops. These values
are the accepted physics-based reference for near-pier bed shear.

ALR does not attempt to reproduce this near-pier peak. Its Tier-1 analysis
provides a zone-scale geometric-blockage amplification (≈1.09× at the synthetic
3-ft pier of Section 4.3), and its Tier-2 vortex-particle step, seeded from the
mean shear over an engineering-scale observation zone, correctly adds no further
bed-shear amplification (Section 4.3): a mean-shear-only vortex field satisfies
the constant-stress-layer limit −⟨u′w′⟩ → u*² and therefore returns unit
amplification. Resolving the CFD-scale peak would require injecting the coherent
near-pier structures (horseshoe and shedding vortices) into the control volume
and calibrating their magnitude against flume or CFD data; this is identified as
future work (Section 5), not a result claimed here.

*[This replaces the prior text, which reported a 1.44× Tier-2 amplification and
argued it was "quantitatively consistent with the Roulund peak of ≈11 averaged
over the zone." That 1.44× was an artifact of a floored friction velocity in the
Tier-2 routine and has been corrected to 1.00×.]*

The design implication is unchanged and is the central argument of this paper.
ALR is not a substitute for CFD when boundary-layer-resolved peak shear is
required. It is an upgrade over Manning's equation for the much larger class of
routine work — screening, alternative comparison, model forensics — where the
current alternative to CFD is Manning's with empirical correction factors, and
where a physics-based Tier-1 bed-shear field plus a resolved turbulence-intensity
diagnostic materially improves engineering judgement.

---

## Section 4.2 (Computational Performance — replacement)

A principal advantage of the observation-dependent resolution rule is that it
allocates particles where they affect the in-zone answer. Because vorticity error
scales as 1/√N, the natural cost–benefit metric is the particle count required to
reach a given in-zone accuracy. Across particle budgets from 500 to 16,000,
observation-concentrated placement reaches the same in-zone vorticity accuracy as
uniform placement with **≈4.7× fewer particles** (Table 6), at correspondingly
lower wall time under the sub-quadratic 6σ-cutoff scaling (Section 5.4). This
in-zone benefit is accompanied, as expected, by higher error outside the
observation zone — the resolution the method deliberately removes from regions
the engineer has not designated as critical.

*[replaces: "At the optimal operating point of 500 ALR particles, the method
achieves a 19× wall-time speedup with only 0.18% vorticity error relative to the
6,000-particle baseline."]*

---

## Section 4.3 (Engineering Scour — replacement)

At the synthetic 3-ft (0.9-m) bridge pier, Tier-1 (vectorized Colebrook–White)
reports a geometric-blockage bed-shear amplification of ≈1.09× (pier-zone
0.138 psf vs approach 0.126 psf). The Tier-2 vortex-particle step, seeded from
the mean-shear vorticity over the observation zone, returns an amplification of
**1.00×** — i.e., it adds no bed-shear amplification beyond the Tier-1 geometric
estimate, consistent with the constant-stress-layer limit. The reported Shields
parameter at the pier is 0.86 and the logistic scour-severity index is 0.88.
*[replaces: "Tier 2 amplifies bed shear by 1.44×. Shields parameter = 1.24,
scour severity index = 0.98."]* Tier-2's physical content at this configuration
is the resolved turbulence-intensity / TKE field, not a shear multiplier.

---

## Section 4.4 (Extensibility — revise results paragraph)

The surface coarsened from 0.80 mm (medium sand) to 5.0 mm (fine gravel) — a
6.3× increase — forming a gravel armor that limits further transport, with the
bed degrading by 0.31 ft. *[replaces: "from 0.80 mm to 7.2 mm — a 9× increase …
10.6 ft of clear-water scour."]* Mass is conserved to ~10⁻¹⁵ over the hydrograph,
and the erosion is transport-limited (the per-step Courant limiter never binds).
This self-limiting armoring is the primary physical mechanism controlling
long-term degradation below dams and is consistent with field observations
(Williams and Wolman, 1984). The demonstration is included as evidence of
extensibility; full pier-scale coupling between the ALR vortex field and the
morphodynamic engine is future work (Section 5).

---

## Section 5.1 (revise the positioning sentence)

The benchmarks of Section 3 show the two faces of that positioning: good
correlation with empirical scour formulas (Laursen r = 0.998, HEC-18 r = 0.605;
Sections 3.2–3.3) via the Tier-1 geometric bed-shear field, and a resolved
turbulence-intensity diagnostic at designated locations. *[Delete the clause
claiming a zone-averaged Tier-2 shear amplification "well above the Manning +
geometric-blockage baseline."]* The computational cost is of the same order as
Manning's (≈0.02 s per pier scenario), rather than the hours-to-days of CFD.

---

## Section 5.3 (Limitations — add one item)

Add: The Tier-2 vortex-particle step, as currently seeded from the mean shear,
resolves turbulence intensity but does not by itself amplify bed shear above the
Tier-1 geometric estimate; capturing near-structure amplification (e.g., the
pier horseshoe vortex) requires injecting the corresponding coherent structures
and calibrating their strength against flume or CFD data. This is the principal
extension required before ALR can report a physics-based near-pier peak shear.

---

## Conclusions (revise items 2, 4, 5)

2. The core algorithmic contribution is an observation-dependent resolution rule
   implemented on a symmetrized Biot–Savart kernel (Barba and Rossi, 2010).
   Observation-concentrated placement reaches a given in-zone vorticity accuracy
   with ≈4.7× fewer particles than uniform placement, with circulation conserved
   to 0.03%. *[replaces the "500-particle … 0.18% … 19×" sentence.]*

4. Comparison against the published CFD dataset of Roulund et al. (2005) situates
   ALR between Manning's + HEC-18 screening and boundary-layer-resolved 3-D RANS.
   ALR provides a Tier-1 geometric bed-shear field and a resolved
   turbulence-intensity diagnostic; it does not reproduce the near-pier CFD peak,
   which would require injected coherent structures with calibrated magnitude
   (future work). *[replaces text asserting the 1.44× Tier-2 amplification.]*

5. Extensibility is demonstrated by coupling the ALR particle field to a
   mass-conserving quasi-unsteady fractional sediment-transport engine with
   Hirano active-layer armoring, producing self-limiting surface coarsening
   (~6×) and sub-foot to ~1 ft degradation for a dam-release scenario consistent
   with field observations (Williams and Wolman, 1984). *[replaces "10.6 ft … 9×".]*

---

## Regenerated Tables

### Table 4 — Positioning of ALR (bed-shear amplification)

| Method | Bed-shear amplification τ/τ∞ | Resolution | Cost |
|---|---|---|---|
| Manning's + HEC-18 K-factors | empirical, no field | reach-average | seconds |
| **ALR Tier-1 (geometric blockage)** | **≈1.09×** | observation-zone | ≈0.02 s |
| **ALR Tier-2 (mean-shear vortex)** | **1.00× (turbulence-intensity diagnostic)** | observation-zone | ≈0.02 s |
| 3-D RANS (Roulund et al., 2005) | 10–11 peak (undeformed), 3–4 (scoured) | y⁺ ≲ 1 | hours–days |

*[Old Table 4 reported ALR Tier-2 at 1.44×.]*

### Table 5 — Convergence of ALR metrics with observation radius (unchanged, re-verified)

| obs radius (ft) | mean σ | mean vorticity | mean enstrophy |
|---|---|---|---|
| 5 | 0.693 | 0.34792 | 0.18532 |
| 10 | 0.490 | 0.34792 | 0.18532 |
| 15 | 0.333 | 0.34792 | 0.18532 |
| 25 | 0.215 | 0.34769 | 0.18523 |
| 50 | 0.172 | 0.34543 | 0.18218 |
| 100 | 0.163 | 0.34543 | 0.18218 |

Vorticity converges to <0.4% between the last two radii; circulation conserved to 0.03%.

### Table 6 — Cost–benefit of observation-concentrated placement (regenerated)

| N particles | in-zone err, uniform | in-zone err, concentrated | wall time (s) |
|---|---|---|---|
| 500 | 1.162 | 0.626 | 0.027 |
| 1,000 | 1.090 | 0.432 | 0.040 |
| 2,000 | 0.659 | 0.369 | 0.058 |
| 4,000 | 0.505 | 0.226 | 0.094 |
| 8,000 | 0.358 | 0.145 | 0.161 |
| 16,000 | 0.202 | 0.118 | 0.389 |

Particle-count reduction to equal in-zone accuracy (error ∝ 1/√N): **≈4.7×**.
*[Old table framed this as a 12× particle reduction / 0.18% error / 19× speedup.]*

### Table 7 — Sediment extensibility (regenerated, paper's exact scenario)

Scenario: 500 × 40 ft channel, slope 0.002, 6-fraction sand-gravel bed, zero
upstream feed, 5-step hydrograph (100/300/600/900/300 cfs over 2000/600/100/20/200 h;
2920 h total, 900 cfs peak).

| Quantity | Corrected engine | *[Old / submitted]* |
|---|---|---|
| Initial surface d50 | 0.80 mm | 0.80 mm |
| Final surface d50 | 5.0 mm | *[7.2 mm]* |
| Coarsening ratio | 6.3× | *[9×]* |
| Bed degradation | 0.31 ft | *[10.6 ft]* |
| Armor forms | yes | yes |
| Mass residual | ~10⁻¹⁵ | *[not conserved]* |

---

## Figures

- **Figure 3** (Melville K_I + scour with/without amplification): the "with
  amplification" curve already used the Tier-1 geometric factor (≈1.11×), not the
  1.44× Tier-2 value, so the plotted curves are **unchanged**. Only the caption /
  §3.4 text referencing a separate 1.44× Tier-2 amplification is revised. A clean
  regenerated version (`Figure3_revised.png`) is provided.
- **Figure 5** (surface d50 coarsening): regenerated (`Figure5_revised.png`) —
  0.80 → 5.0 mm (6.3×), self-limiting armor, 0.31 ft degradation.
