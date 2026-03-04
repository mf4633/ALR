#!/usr/bin/env python
"""
Quick verification script for quantum_hydraulics package.

Run from the quantum directory:
    python verify_package.py
"""

import sys
import os

# Add package to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that all imports work."""
    print("Testing imports...")

    # Core modules
    print("  - Importing core modules...", end=" ")
    from quantum_hydraulics.core.particle import VortexParticle
    from quantum_hydraulics.core.hydraulics import HydraulicsEngine
    from quantum_hydraulics.core.vortex_field import VortexParticleField, FieldState
    print("OK")

    # Visualization modules
    print("  - Importing visualization modules...", end=" ")
    from quantum_hydraulics.visualization.theme import Theme, THEMES, get_theme
    from quantum_hydraulics.visualization.renderers import plot_plan_view
    print("OK")

    # Validation modules
    print("  - Importing validation modules...", end=" ")
    from quantum_hydraulics.validation.analytical import lamb_oseen_vortex, kolmogorov_spectrum
    print("OK")

    # Integration modules
    print("  - Importing integration modules...", end=" ")
    from quantum_hydraulics.integration.swmm_node import QuantumNode
    print("OK")

    # Top-level imports
    print("  - Importing from package root...", end=" ")
    import quantum_hydraulics
    print(f"OK (version {quantum_hydraulics.__version__})")

    return True


def test_hydraulics():
    """Test basic hydraulics computation."""
    print("\nTesting HydraulicsEngine...")

    from quantum_hydraulics.core.hydraulics import HydraulicsEngine

    h = HydraulicsEngine(
        Q=600,
        width=30,
        depth=5,
        slope=0.002,
        roughness_ks=0.15
    )

    print(f"  Q = {h.Q} cfs")
    print(f"  V = {h.V_mean:.2f} ft/s")
    print(f"  A = {h.A:.1f} ft2")
    print(f"  Re = {h.Re:.0f}")
    print(f"  Fr = {h.Fr:.3f}")
    print(f"  f = {h.friction_factor:.5f}")
    print(f"  eta_K = {h.eta_kolmogorov:.6f} ft")

    # Verify continuity
    Q_check = h.V_mean * h.A
    assert abs(Q_check - h.Q) < 1e-10, "Continuity check failed!"
    print("  Continuity check: PASS")

    return True


def test_particle():
    """Test particle creation and manipulation."""
    print("\nTesting VortexParticle...")

    import numpy as np
    from quantum_hydraulics.core.particle import VortexParticle

    p = VortexParticle.create(
        position=[10, 15, 2.5],
        vorticity=[0.1, 0.2, 0.3],
        core_size=0.5
    )

    print(f"  Position: {p.pos}")
    print(f"  Vorticity: {p.omega}")
    print(f"  Sigma: {p.sigma}")
    print(f"  Energy: {p.energy:.4f}")
    print(f"  Circulation: {p.circulation:.4f}")

    # Test advection
    p.advect(np.array([1.0, 0.5, 0.0]), dt=0.1)
    print(f"  After advection: pos = {p.pos}")

    return True


def test_vortex_field():
    """Test vortex field creation."""
    print("\nTesting VortexParticleField...")

    from quantum_hydraulics.core.hydraulics import HydraulicsEngine
    from quantum_hydraulics.core.vortex_field import VortexParticleField

    h = HydraulicsEngine(Q=600, width=30, depth=5, slope=0.002, roughness_ks=0.15)
    field = VortexParticleField(h, length=100, n_particles=500)

    print(f"  Domain: {field.L} x {field.W} x {field.H}")
    print(f"  Particles: {len(field.particles)}")
    print(f"  Base sigma: {field.base_sigma:.4f}")
    print(f"  Min sigma: {field.min_sigma:.6f}")
    print(f"  Observation active: {field.observation_active}")

    # Test stepping
    print("  Running 5 timesteps...", end=" ")
    for _ in range(5):
        field.step(dt=0.05)
    print("OK")

    # Get state
    state = field.get_state()
    print(f"  State has {state.n_particles} particles")

    return True


def test_themes():
    """Test theme system."""
    print("\nTesting Theme system...")

    from quantum_hydraulics.visualization.theme import THEMES, get_theme, list_themes

    print(f"  Available themes: {list_themes()}")

    for name in list_themes():
        theme = get_theme(name)
        print(f"  - {name}: bg={theme.background}, accent={theme.accent_primary}")

    return True


def test_analytical():
    """Test analytical solutions."""
    print("\nTesting analytical solutions...")

    import numpy as np
    from quantum_hydraulics.validation.analytical import (
        lamb_oseen_vortex,
        kolmogorov_spectrum,
        colebrook_white
    )

    # Lamb-Oseen
    r = np.array([0.1, 0.5, 1.0])
    v = lamb_oseen_vortex(r, t=1.0, gamma=1.0)
    print(f"  Lamb-Oseen v(r): {v}")

    # Kolmogorov spectrum
    k = np.array([1, 10, 100])
    E = kolmogorov_spectrum(k, epsilon=0.1)
    print(f"  Kolmogorov E(k): {E}")

    # Colebrook-White
    f = colebrook_white(Re=100000, epsilon_D=0.01)
    print(f"  Colebrook-White f: {f:.5f}")

    return True


def main():
    """Run all verification tests."""
    print("=" * 60)
    print("QUANTUM HYDRAULICS PACKAGE VERIFICATION")
    print("=" * 60)

    tests = [
        ("Imports", test_imports),
        ("Hydraulics", test_hydraulics),
        ("Particle", test_particle),
        ("Vortex Field", test_vortex_field),
        ("Themes", test_themes),
        ("Analytical", test_analytical),
    ]

    results = []
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, "PASS" if success else "FAIL"))
        except Exception as e:
            print(f"\n  ERROR: {e}")
            results.append((name, "FAIL"))

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    for name, status in results:
        symbol = "OK" if status == "PASS" else "X"
        print(f"  [{symbol}] {name}")

    passed = sum(1 for _, s in results if s == "PASS")
    total = len(results)
    print(f"\nPassed: {passed}/{total}")

    if passed == total:
        print("\nAll tests passed! Package is ready to use.")
        print("\nTry running:")
        print("  python -m quantum_hydraulics --quick")
        print("  python -m quantum_hydraulics --validate")

    print("=" * 60)

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
