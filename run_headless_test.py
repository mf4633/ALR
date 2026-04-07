"""
Quantum Hydraulics — Headless Validation Scenario
==================================================

Simple trapezoidal channel test that exercises the full physics stack
without any GUI, visualization, or SWMM dependency.

Scenario:
  - 30-ft bottom width, 2:1 side slopes
  - 0.2% bed slope
  - 5 ft water depth
  - Q = 600 cfs
  - ks = 0.1 ft (equivalent sand roughness)
  - Sand bed (tau_c = 0.10 psf)

Validates:
  1. Geometry (A, P, R)          — exact closed-form
  2. Continuity (Q = V * A)      — conservation law
  3. Colebrook-White convergence  — friction factor within Moody bounds
  4. Log-law velocity profile     — monotonic increase, bounded by theory
  5. Kolmogorov cascade           — eta < lambda_T < R
  6. Vortex field evolution       — energy bounded, particles contained
  7. QuantumNode scour metrics    — physically plausible ranges

Exit code 0 = all checks pass, 1 = failure.

Usage:
  python run_headless_test.py
  python run_headless_test.py --verbose
  python run_headless_test.py --json
"""

import sys
import json
import time
import argparse
import numpy as np

# ── Ensure package is importable ────────────────────────────────────────────
sys.path.insert(0, ".")

from quantum_hydraulics.core.hydraulics import HydraulicsEngine
from quantum_hydraulics.core.vortex_field import VortexParticleField
from quantum_hydraulics.integration.swmm_node import QuantumNode, SedimentProperties
from quantum_hydraulics.validation.analytical import (
    colebrook_white,
    log_law_velocity,
    kolmogorov_scales,
)


# ── Scenario parameters ────────────────────────────────────────────────────
Q = 600.0       # cfs
WIDTH = 30.0    # ft (bottom width)
DEPTH = 5.0     # ft
SLOPE = 0.002   # ft/ft
KS = 0.1        # ft (equivalent sand roughness)
SIDE_SLOPE = 2.0  # H:V
NU = 1.1e-5     # ft²/s
RHO = 1.94      # slug/ft³
G = 32.2        # ft/s²


class CheckResult:
    """Single pass/fail check with diagnostics."""

    def __init__(self, name: str, passed: bool, detail: str, values: dict = None):
        self.name = name
        self.passed = passed
        self.detail = detail
        self.values = values or {}

    def __str__(self):
        tag = "PASS" if self.passed else "FAIL"
        return f"  [{tag}] {self.name}: {self.detail}"


def run_checks(verbose: bool = False) -> list[CheckResult]:
    """Run all validation checks, return list of results."""
    results = []
    t0 = time.perf_counter()

    # ════════════════════════════════════════════════════════════════════════
    # 1. HYDRAULICS ENGINE
    # ════════════════════════════════════════════════════════════════════════
    engine = HydraulicsEngine(
        Q=Q, width=WIDTH, depth=DEPTH, slope=SLOPE,
        roughness_ks=KS, side_slope=SIDE_SLOPE,
    )

    # ── 1a. Geometry (exact) ──
    A_exact = WIDTH * DEPTH + SIDE_SLOPE * DEPTH ** 2  # 30*5 + 2*25 = 200 ft²
    P_exact = WIDTH + 2 * DEPTH * np.sqrt(1 + SIDE_SLOPE ** 2)  # 30 + 10*sqrt(5)
    R_exact = A_exact / P_exact

    results.append(CheckResult(
        "Geometry — Area",
        abs(engine.A - A_exact) < 1e-10,
        f"computed={engine.A:.4f}, exact={A_exact:.4f}",
        {"A_computed": engine.A, "A_exact": A_exact},
    ))
    results.append(CheckResult(
        "Geometry — Hydraulic radius",
        abs(engine.R - R_exact) < 1e-10,
        f"computed={engine.R:.6f}, exact={R_exact:.6f}",
        {"R_computed": engine.R, "R_exact": R_exact},
    ))

    # ── 1b. Continuity Q = V * A ──
    Q_check = engine.V_mean * engine.A
    results.append(CheckResult(
        "Continuity — Q = V·A",
        abs(Q_check - Q) < 1e-8,
        f"V·A={Q_check:.6f}, Q={Q:.1f}",
        {"Q_VA": Q_check, "Q_input": Q},
    ))

    # ── 1c. Colebrook-White convergence ──
    # Independent solve using analytical module
    epsilon_D = KS / (4 * R_exact)
    Re = engine.V_mean * R_exact / NU
    f_independent = colebrook_white(Re, epsilon_D)
    f_engine = engine.friction_factor

    results.append(CheckResult(
        "Colebrook-White — convergence",
        abs(f_engine - f_independent) / f_independent < 0.01,
        f"engine f={f_engine:.6f}, independent f={f_independent:.6f}",
        {"f_engine": f_engine, "f_independent": f_independent},
    ))

    # Moody diagram bounds: f should be between 0.005 and 0.08 for turbulent flow
    results.append(CheckResult(
        "Colebrook-White — Moody bounds",
        0.005 < f_engine < 0.08,
        f"f={f_engine:.6f} (expect 0.005–0.08)",
        {"f": f_engine},
    ))

    # ── 1d. Froude number ──
    D_h = A_exact / (WIDTH + 2 * SIDE_SLOPE * DEPTH)
    Fr_check = engine.V_mean / np.sqrt(G * D_h)
    results.append(CheckResult(
        "Froude number — subcritical",
        engine.Fr < 1.0,
        f"Fr={engine.Fr:.4f} (subcritical for this channel)",
        {"Fr": engine.Fr},
    ))

    # ── 1e. Reynolds number — turbulent ──
    results.append(CheckResult(
        "Reynolds number — turbulent",
        engine.Re > 4000,
        f"Re={engine.Re:,.0f}",
        {"Re": engine.Re},
    ))

    # ════════════════════════════════════════════════════════════════════════
    # 2. VELOCITY PROFILE
    # ════════════════════════════════════════════════════════════════════════
    # Inner layer (log-law): 0 < z < 0.2*depth
    z_inner_pts = np.linspace(0.01, 0.18 * DEPTH, 25)
    v_inner = np.array([engine.velocity_profile(z) for z in z_inner_pts])
    diffs_inner = np.diff(v_inner)
    inner_monotonic = np.all(diffs_inner >= -1e-6)

    # Outer layer (power law): z > 0.2*depth
    z_outer_pts = np.linspace(0.22 * DEPTH, 0.95 * DEPTH, 25)
    v_outer = np.array([engine.velocity_profile(z) for z in z_outer_pts])
    diffs_outer = np.diff(v_outer)
    outer_monotonic = np.all(diffs_outer >= -1e-6)

    results.append(CheckResult(
        "Velocity profile — monotonic (per regime)",
        inner_monotonic and outer_monotonic,
        f"inner_min_diff={diffs_inner.min():.6f}, outer_min_diff={diffs_outer.min():.6f}",
        {"inner_min_diff": float(diffs_inner.min()), "outer_min_diff": float(diffs_outer.min())},
    ))

    # Bed velocity should be small, surface velocity near V_mean
    results.append(CheckResult(
        "Velocity profile — near-bed < surface",
        v_inner[0] < v_outer[-1],
        f"v_bed={v_inner[0]:.3f}, v_surface={v_outer[-1]:.3f}",
        {"v_bed": float(v_inner[0]), "v_surface": float(v_outer[-1])},
    ))

    # Cross-check against analytical log_law_velocity at inner layer
    z_inner = 0.05  # well within inner layer (< 0.2 * depth)
    z0 = KS / 30.0
    v_analytical = float(log_law_velocity(np.array([z_inner]), engine.u_star, z0).item())
    v_engine = engine.velocity_profile(z_inner)
    results.append(CheckResult(
        "Velocity profile — log-law match (inner layer)",
        abs(v_engine - v_analytical) / max(v_analytical, 1e-6) < 0.01,
        f"engine={v_engine:.4f}, analytical={v_analytical:.4f}",
        {"v_engine": v_engine, "v_analytical": v_analytical},
    ))

    # ════════════════════════════════════════════════════════════════════════
    # 3. TURBULENCE SCALES (Kolmogorov cascade)
    # ════════════════════════════════════════════════════════════════════════
    eta_ref, tau_ref, v_ref = kolmogorov_scales(engine.epsilon, NU)

    results.append(CheckResult(
        "Kolmogorov scale — eta matches analytical",
        abs(engine.eta_kolmogorov - eta_ref) / eta_ref < 0.01,
        f"engine={engine.eta_kolmogorov:.8f}, ref={eta_ref:.8f}",
        {"eta_engine": engine.eta_kolmogorov, "eta_ref": eta_ref},
    ))

    # Scale hierarchy: eta < lambda_taylor < R
    results.append(CheckResult(
        "Scale hierarchy — eta < lambda_T < R",
        engine.eta_kolmogorov < engine.lambda_taylor < engine.R,
        f"eta={engine.eta_kolmogorov:.6f}, lambda={engine.lambda_taylor:.6f}, R={engine.R:.4f}",
        {"eta": engine.eta_kolmogorov, "lambda_T": engine.lambda_taylor, "R": engine.R},
    ))

    # ════════════════════════════════════════════════════════════════════════
    # 4. VORTEX PARTICLE FIELD (small, fast)
    # ════════════════════════════════════════════════════════════════════════
    n_test = 200  # small for speed
    vfield = VortexParticleField(engine, length=100.0, n_particles=n_test)

    results.append(CheckResult(
        "Vortex field — particle count",
        vfield._positions.shape[0] == n_test,
        f"n={vfield._positions.shape[0]}",
        {"n_particles": vfield._positions.shape[0]},
    ))

    # Evolve 10 steps, check particles stay in domain
    for _ in range(10):
        vfield.step(dt=0.02)

    in_x = np.all((vfield._positions[:, 0] >= 0) & (vfield._positions[:, 0] <= 100.0))
    in_z = np.all((vfield._positions[:, 2] >= 0) & (vfield._positions[:, 2] <= DEPTH))
    results.append(CheckResult(
        "Vortex field — particles bounded after 10 steps",
        in_x and in_z,
        f"x_range=[{vfield._positions[:,0].min():.2f}, {vfield._positions[:,0].max():.2f}], "
        f"z_range=[{vfield._positions[:,2].min():.4f}, {vfield._positions[:,2].max():.4f}]",
    ))

    # Energy should be finite and positive
    state = vfield.get_state()
    e_max = state.energies.max()
    e_finite = np.all(np.isfinite(state.energies))
    results.append(CheckResult(
        "Vortex field — energy finite & positive",
        e_finite and e_max > 0,
        f"max_energy={e_max:.4f}, all_finite={e_finite}",
        {"max_energy": float(e_max)},
    ))

    # ════════════════════════════════════════════════════════════════════════
    # 5. QUANTUM NODE — synthetic SWMM-like scenario
    # ════════════════════════════════════════════════════════════════════════
    node = QuantumNode(
        node_id="HEADLESS_TEST",
        width=WIDTH,
        length=30.0,
        roughness_ks=KS,
        sediment=SedimentProperties.sand(),
    )

    # Feed a simple rising-then-falling hydrograph (10 steps)
    depths =  [1.0, 2.0, 3.5, 5.0, 5.0, 4.5, 3.0, 2.0, 1.0, 0.5]
    inflows = [50, 150, 350, 600, 600, 500, 250, 100, 30, 10]

    for d, q in zip(depths, inflows):
        node.update_and_evolve(depth=d, inflow=q, dt=0.5)

    # After peak (600 cfs at 5 ft), compute turbulence
    node.update_from_swmm(depth=5.0, inflow=600.0)
    node.compute_turbulence(n_particles=300)

    m = node.metrics

    results.append(CheckResult(
        "QuantumNode — has particles",
        node.n_particles > 0,
        f"n_particles={node.n_particles}",
        {"n_particles": node.n_particles},
    ))

    results.append(CheckResult(
        "QuantumNode — max velocity > 0",
        m.max_velocity > 0,
        f"v_max={m.max_velocity:.3f} ft/s",
        {"max_velocity": m.max_velocity},
    ))

    results.append(CheckResult(
        "QuantumNode — bed shear > 0",
        m.bed_shear_stress > 0,
        f"tau={m.bed_shear_stress:.4f} psf",
        {"bed_shear_stress": m.bed_shear_stress},
    ))

    results.append(CheckResult(
        "QuantumNode — scour risk in [0, 1]",
        0.0 <= m.scour_risk_index <= 1.0,
        f"risk={m.scour_risk_index:.4f}",
        {"scour_risk_index": m.scour_risk_index},
    ))

    # At 600 cfs / 200 ft² = 3 ft/s mean → expect non-trivial scour on sand
    results.append(CheckResult(
        "QuantumNode — scour detected on sand bed",
        m.scour_risk_index > 0.1,
        f"risk={m.scour_risk_index:.4f} (expect > 0.1 for sand at 600 cfs)",
        {"scour_risk_index": m.scour_risk_index},
    ))

    # Shields parameter should be positive
    results.append(CheckResult(
        "QuantumNode — Shields parameter > 0",
        m.shields_parameter > 0,
        f"theta={m.shields_parameter:.4f}",
        {"shields_parameter": m.shields_parameter},
    ))

    # Assessment strings should be non-empty
    results.append(CheckResult(
        "QuantumNode — assessments populated",
        len(node.get_scour_assessment()) > 0 and len(node.get_velocity_assessment()) > 0,
        f"scour: {node.get_scour_assessment()[:60]}",
    ))

    elapsed = time.perf_counter() - t0

    results.append(CheckResult(
        "Runtime",
        elapsed < 120.0,
        f"{elapsed:.2f}s",
        {"elapsed_s": elapsed},
    ))

    return results


def main():
    parser = argparse.ArgumentParser(description="Quantum Hydraulics headless validation")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show all check details")
    parser.add_argument("--json", action="store_true", help="Output results as JSON")
    args = parser.parse_args()

    results = run_checks(verbose=args.verbose)

    n_pass = sum(1 for r in results if r.passed)
    n_fail = sum(1 for r in results if not r.passed)
    all_pass = n_fail == 0

    if args.json:
        output = {
            "scenario": "30-ft trapezoidal channel, Q=600 cfs, depth=5 ft, slope=0.2%",
            "passed": n_pass,
            "failed": n_fail,
            "all_pass": all_pass,
            "checks": [
                {
                    "name": r.name,
                    "passed": r.passed,
                    "detail": r.detail,
                    **r.values,
                }
                for r in results
            ],
        }
        print(json.dumps(output, indent=2, default=lambda o: o.item() if hasattr(o, 'item') else str(o)))
    else:
        print()
        print("=" * 72)
        print("QUANTUM HYDRAULICS — HEADLESS VALIDATION")
        print("=" * 72)
        print(f"Scenario: 30-ft trapezoidal channel, Q=600 cfs, y=5 ft, S=0.002")
        print(f"          ks=0.1 ft, side slope 2:1, sand bed")
        print("-" * 72)

        for r in results:
            if args.verbose or not r.passed:
                print(r)
            elif r.passed:
                print(f"  [PASS] {r.name}")

        print("-" * 72)
        print(f"  {n_pass}/{n_pass + n_fail} checks passed", end="")
        if all_pass:
            print(" — ALL PASS")
        else:
            print(f" — {n_fail} FAILED")
            for r in results:
                if not r.passed:
                    print(f"         X {r.name}: {r.detail}")

        print("=" * 72)
        print()

    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
