"""
Main entry point for running quantum_hydraulics as a module.

Usage:
    python -m quantum_hydraulics           # Interactive demo
    python -m quantum_hydraulics --help    # Show options
    python -m quantum_hydraulics --quick   # Quick demo
    python -m quantum_hydraulics --validate  # Run validation
"""

import sys
import argparse


def main():
    parser = argparse.ArgumentParser(
        prog="quantum_hydraulics",
        description="Quantum-inspired vortex particle hydraulics simulator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m quantum_hydraulics              # Launch interactive demo
  python -m quantum_hydraulics --quick      # Quick demo (fewer particles)
  python -m quantum_hydraulics --validate   # Run validation tests
  python -m quantum_hydraulics --conceptual # Conceptual demo
  python -m quantum_hydraulics --info       # Show package info
        """,
    )

    parser.add_argument(
        "--quick", action="store_true",
        help="Run quick demo with fewer particles"
    )
    parser.add_argument(
        "--validate", action="store_true",
        help="Run validation tests"
    )
    parser.add_argument(
        "--conceptual", action="store_true",
        help="Run conceptual demo"
    )
    parser.add_argument(
        "--info", action="store_true",
        help="Show package information"
    )
    parser.add_argument(
        "--theme", type=str, default="dark_professional",
        choices=["dark_professional", "light_publication", "hec_ras_style", "ocean_blue"],
        help="Visual theme"
    )
    parser.add_argument(
        "--particles", type=int, default=6000,
        help="Number of particles"
    )
    parser.add_argument(
        "--export", type=str,
        help="Export animation to file"
    )

    args = parser.parse_args()

    if args.info:
        from quantum_hydraulics import __version__
        print(f"""
Quantum Hydraulics v{__version__}
==============================

A physics-based hydraulic simulation package using:
- Vortex particle methods (Biot-Savart law)
- First-principles physics (Colebrook-White, Kolmogorov)
- Observation-dependent adaptive resolution

Package location: {__file__}

Run 'python -m quantum_hydraulics --help' for options.
        """)
        return

    if args.validate:
        print("Running validation tests...")
        from quantum_hydraulics.validation.benchmarks import run_validation_report
        success = run_validation_report()
        sys.exit(0 if success else 1)

    if args.conceptual:
        from quantum_hydraulics.demos.conceptual_demo import run_conceptual_demo
        run_conceptual_demo()
        return

    # Default: run engineering demo
    from quantum_hydraulics.demos.engineering_demo import run_engineering_demo

    n_particles = 2000 if args.quick else args.particles

    if args.export:
        from quantum_hydraulics.demos.engineering_demo import export_demo_animation
        export_demo_animation(args.export)
    else:
        run_engineering_demo(
            theme=args.theme,
            n_particles=n_particles,
        )


if __name__ == "__main__":
    main()
