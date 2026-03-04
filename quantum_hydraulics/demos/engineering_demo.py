"""
Engineering Demo - Full interactive quantum hydraulics simulator.

This is the main demonstration showing all features:
- First-principles physics (Colebrook-White, Kolmogorov cascade)
- Vortex particle method (Biot-Savart law)
- Observation-dependent resolution (adaptive sigma)
- Professional visualization with themes
"""

import os
import sys


def run_engineering_demo(
    Q: float = 600.0,
    width: float = 30.0,
    depth: float = 5.0,
    slope: float = 0.002,
    roughness: float = 0.15,
    theme: str = "dark_professional",
    n_particles: int = 6000,
):
    """
    Launch the interactive engineering simulator.

    Parameters
    ----------
    Q : float
        Discharge in cfs (default 600)
    width : float
        Channel width in feet (default 30)
    depth : float
        Water depth in feet (default 5)
    slope : float
        Bed slope (default 0.002)
    roughness : float
        Roughness ks in feet (default 0.15)
    theme : str
        Visual theme (default 'dark_professional')
    n_particles : int
        Number of particles (default 6000)
    """
    from quantum_hydraulics.visualization.interactive import InteractiveSimulator

    print("\n" + "=" * 80)
    print(" " * 15 + "QUANTUM-INSPIRED VORTEX PARTICLE SIMULATOR")
    print("=" * 80)
    print("\nCombining:")
    print("  - First-principles physics (Colebrook-White, Kolmogorov cascade)")
    print("  - True 3D vortex particle method (Biot-Savart law)")
    print("  - Observation-dependent resolution (adaptive core size sigma)")
    print("  - Professional visualization with multiple themes")
    print("\nThis is physics-based hydraulic simulation, not empirical formulas.")
    print("=" * 80 + "\n")

    print("Initializing simulator...")
    print("  - Computing hydraulics from conservation laws...")
    print("  - Generating vortex particles across Kolmogorov cascade...")
    print("  - Building spatial acceleration structures...")
    print("  - Creating interactive UI...\n")

    sim = InteractiveSimulator(
        Q=Q,
        width=width,
        depth=depth,
        slope=slope,
        roughness=roughness,
        theme=theme,
        n_particles=n_particles,
    )

    sim.run()

    print("\n" + "=" * 80)
    print("Simulator closed.")
    print("=" * 80)


def run_quick_demo():
    """Run a quick demo with fewer particles for faster startup."""
    run_engineering_demo(n_particles=2000)


def run_high_resolution_demo():
    """Run a high-resolution demo with more particles."""
    run_engineering_demo(n_particles=10000)


def export_demo_animation(output_path: str = None, n_frames: int = 100):
    """
    Export a demo animation to video file.

    Parameters
    ----------
    output_path : str, optional
        Output file path (default: quantum_demo.mp4 in current directory)
    n_frames : int
        Number of frames to render
    """
    from quantum_hydraulics.core.hydraulics import HydraulicsEngine
    from quantum_hydraulics.core.vortex_field import VortexParticleField
    from quantum_hydraulics.visualization.export import export_animation
    from quantum_hydraulics.visualization.theme import get_theme

    if output_path is None:
        output_path = os.path.join(os.getcwd(), "quantum_demo.mp4")

    print("Creating demo animation...")

    hydraulics = HydraulicsEngine(
        Q=600, width=30, depth=5, slope=0.002, roughness_ks=0.15
    )
    field = VortexParticleField(hydraulics, length=200, n_particles=4000)

    def progress(current, total):
        pct = current / total * 100
        print(f"\r  Rendering: {current}/{total} ({pct:.0f}%)", end="", flush=True)

    export_animation(
        output_path,
        field,
        hydraulics,
        n_frames=n_frames,
        fps=20,
        dpi=150,
        progress_callback=progress,
    )

    print(f"\n\nAnimation saved to: {output_path}")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Quantum Hydraulics Engineering Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m quantum_hydraulics.demos.engineering_demo
  python -m quantum_hydraulics.demos.engineering_demo --quick
  python -m quantum_hydraulics.demos.engineering_demo --theme light_publication
  python -m quantum_hydraulics.demos.engineering_demo --export demo.mp4
        """,
    )

    parser.add_argument("--quick", action="store_true", help="Quick demo with fewer particles")
    parser.add_argument("--hires", action="store_true", help="High resolution with more particles")
    parser.add_argument("--theme", type=str, default="dark_professional",
                        choices=["dark_professional", "light_publication", "hec_ras_style", "ocean_blue"],
                        help="Visual theme")
    parser.add_argument("--particles", type=int, default=6000, help="Number of particles")
    parser.add_argument("--export", type=str, help="Export animation to file instead of interactive")
    parser.add_argument("--frames", type=int, default=100, help="Frames for animation export")

    parser.add_argument("--Q", type=float, default=600.0, help="Discharge (cfs)")
    parser.add_argument("--width", type=float, default=30.0, help="Channel width (ft)")
    parser.add_argument("--depth", type=float, default=5.0, help="Water depth (ft)")
    parser.add_argument("--slope", type=float, default=0.002, help="Bed slope")
    parser.add_argument("--roughness", type=float, default=0.15, help="Roughness ks (ft)")

    args = parser.parse_args()

    if args.export:
        export_demo_animation(args.export, args.frames)
    elif args.quick:
        run_quick_demo()
    elif args.hires:
        run_high_resolution_demo()
    else:
        run_engineering_demo(
            Q=args.Q,
            width=args.width,
            depth=args.depth,
            slope=args.slope,
            roughness=args.roughness,
            theme=args.theme,
            n_particles=args.particles,
        )


if __name__ == "__main__":
    main()
