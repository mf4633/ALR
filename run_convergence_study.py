#!/usr/bin/env python
"""
Convergence study: coherent mean-shear seeding vs random-phase seeding.

Demonstrates the central premise behind adaptive Lagrangian refinement
(docs/ALR_REFINEMENT_DESIGN.md): a vortex particle field only has a converged
limit if the seeded vorticity is a coherent (resolved) field, not a random-phase
turbulence proxy.

For each seeding mode and particle budget N, we form an ensemble over random
placements and report the ensemble mean and standard deviation of the induced
streamwise velocity (vx) at a fixed interior probe.

Expected outcome:
  - COHERENT: ensemble-mean vx settles on a stable, non-zero value while the
    spread shrinks ~1/sqrt(N); signal-to-noise |mean|/std grows with N.
  - RANDOM-PHASE: ensemble-mean vx -> 0 (no coherent signal); SNR stays ~0.

Run:  python run_convergence_study.py [--seeds 60]
"""

import argparse
import numpy as np

from quantum_hydraulics.core.hydraulics import HydraulicsEngine
from quantum_hydraulics.core.vortex_field import VortexParticleField

PROBE = np.array([50.0, 15.0, 2.5])
BUDGETS = [500, 1000, 2000, 4000, 8000]
SIGMA = 0.5  # fixed core size, so only the quadrature (not the kernel) refines


def _induced_vx_at(field: VortexParticleField, x0: np.ndarray) -> float:
    """Streamwise induced velocity at x0 from the whole field (direct sum)."""
    r = field._positions - x0
    r2 = np.sum(r ** 2, axis=1)
    denom = (r2 + SIGMA ** 2) ** 1.5 + 1e-12
    K = 1.0 / (4.0 * np.pi * denom)
    strength = field._vorticities * field._volumes[:, None]
    v = (K[:, None] * np.cross(strength, r)).sum(axis=0)
    return float(v[0])


def _ensemble(engine, seeding, n, seeds):
    vals = []
    for s in range(seeds):
        field = VortexParticleField(engine, length=100, n_particles=n,
                                    seeding=seeding)
        # Re-seed with a per-ensemble seed and a fixed core size so only the
        # particle placement varies across the ensemble.
        field.seeding = seeding
        field._seed_field(seed=s)
        field._sigmas = np.full(len(field._positions), SIGMA)
        field._volumes = np.full(len(field._positions),
                                 (field.L * field.W * field.H) / len(field._positions))
        vals.append(_induced_vx_at(field, PROBE))
    return np.array(vals)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=60,
                    help="ensemble size per budget (default 60)")
    args = ap.parse_args()

    engine = HydraulicsEngine(Q=600, width=30, depth=5, slope=0.002,
                              roughness_ks=0.15)

    print("=" * 70)
    print("Convergence study: induced vx at probe", PROBE, f"(sigma={SIGMA})")
    print(f"ensemble size = {args.seeds} placements per budget")
    print("=" * 70)

    results = {}
    for seeding in ("coherent", "random"):
        print(f"\n{seeding.upper()} seeding:")
        print(f"  {'N':>7}  {'mean(vx)':>10}  {'std':>8}  {'SNR':>6}")
        prev = None
        rows = []
        for n in BUDGETS:
            vs = _ensemble(engine, seeding, n, args.seeds)
            mean, std = vs.mean(), vs.std()
            snr = abs(mean) / max(std, 1e-12)
            rows.append((n, mean, std, snr))
            drift = "" if prev is None else f"  dmean={100*abs(mean-prev)/max(abs(prev),1e-9):5.1f}%"
            print(f"  {n:>7}  {mean:>+10.4f}  {std:>8.4f}  {snr:>6.2f}{drift}")
            prev = mean
        results[seeding] = rows

    # Verdict
    coh = results["coherent"]
    ran = results["random"]
    coh_snr_grows = coh[-1][3] > coh[0][3] * 1.5
    coh_stable = abs(coh[-1][1] - coh[-2][1]) / max(abs(coh[-2][1]), 1e-9) < 0.05
    ran_to_zero = abs(ran[-1][1]) < 0.25 * abs(coh[-1][1])

    print("\n" + "-" * 70)
    print(f"  coherent SNR grows with N .............. {'PASS' if coh_snr_grows else 'FAIL'}"
          f"  ({coh[0][3]:.2f} -> {coh[-1][3]:.2f})")
    print(f"  coherent mean(vx) stabilizes .......... {'PASS' if coh_stable else 'FAIL'}"
          f"  (limit ~ {coh[-1][1]:+.3f})")
    print(f"  random-phase mean(vx) -> 0 ............ {'PASS' if ran_to_zero else 'FAIL'}"
          f"  ({ran[-1][1]:+.3f} vs coherent {coh[-1][1]:+.3f})")
    ok = coh_snr_grows and coh_stable and ran_to_zero
    print("-" * 70)
    print("  RESULT:", "convergence demonstrated" if ok else "inconclusive")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
