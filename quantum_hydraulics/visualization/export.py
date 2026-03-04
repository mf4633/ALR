"""
Export utilities for animations and frame sequences.

Supports:
- MP4 video export (via FFmpeg)
- GIF export (via Pillow)
- PNG sequence export
"""

import os
from typing import Optional, Callable, Any
import numpy as np

try:
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
    _HAS_MATPLOTLIB = True
except ImportError:
    _HAS_MATPLOTLIB = False

from quantum_hydraulics.core.hydraulics import HydraulicsEngine
from quantum_hydraulics.core.vortex_field import VortexParticleField
from quantum_hydraulics.visualization.theme import Theme, get_theme
from quantum_hydraulics.visualization.renderers import (
    create_figure_layout,
    plot_plan_view,
    plot_profile_view,
    plot_velocity_profile,
    plot_energy_spectrum,
    plot_detail_map,
    plot_theory_panel,
)


def export_animation(
    output_path: str,
    vortex_field: VortexParticleField,
    hydraulics: HydraulicsEngine,
    n_frames: int = 100,
    dt: float = 0.05,
    steps_per_frame: int = 3,
    fps: int = 20,
    dpi: int = 150,
    theme: Optional[Theme] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    format: Optional[str] = None,
) -> str:
    """
    Export simulation animation to video file.

    Parameters
    ----------
    output_path : str
        Output file path (extension determines format: .mp4, .gif)
    vortex_field : VortexParticleField
        Vortex particle field to animate
    hydraulics : HydraulicsEngine
        Hydraulics engine for parameters
    n_frames : int
        Number of frames to render
    dt : float
        Timestep per physics step
    steps_per_frame : int
        Physics steps per animation frame
    fps : int
        Frames per second in output
    dpi : int
        Resolution in dots per inch
    theme : Theme, optional
        Visual theme
    progress_callback : callable, optional
        Callback(current_frame, total_frames) for progress updates
    format : str, optional
        Force output format ('mp4' or 'gif'). Auto-detected from extension if None.

    Returns
    -------
    str
        Path to exported file

    Raises
    ------
    ImportError
        If matplotlib is not available
    ValueError
        If format is not supported
    """
    if not _HAS_MATPLOTLIB:
        raise ImportError("Matplotlib required for animation export")

    if theme is None:
        theme = get_theme()

    # Determine format
    if format is None:
        ext = os.path.splitext(output_path)[1].lower()
        if ext == ".mp4":
            format = "mp4"
        elif ext == ".gif":
            format = "gif"
        else:
            format = "mp4"
            output_path = output_path + ".mp4"

    # Create figure
    fig, axes = create_figure_layout(theme)

    def update(frame):
        # Step physics
        for _ in range(steps_per_frame):
            vortex_field.step(dt=dt)

        # Get state
        state = vortex_field.get_state()

        # Update plots
        plot_plan_view(axes["plan"], state, hydraulics, theme)
        plot_theory_panel(axes["theory"], hydraulics, vortex_field, theme)
        plot_profile_view(axes["profile"], state, hydraulics, theme)
        plot_detail_map(axes["detail"], state, vortex_field, theme)
        plot_velocity_profile(axes["velocity"], hydraulics, theme)
        plot_energy_spectrum(axes["spectrum"], state, hydraulics, theme)

        if progress_callback:
            progress_callback(frame + 1, n_frames)

        return []

    anim = FuncAnimation(fig, update, frames=n_frames, interval=1000 // fps, blit=False)

    # Export
    if format == "mp4":
        try:
            writer = FFMpegWriter(fps=fps, metadata={"title": "Quantum Hydraulics Simulation"})
            anim.save(output_path, writer=writer, dpi=dpi)
        except (RuntimeError, FileNotFoundError) as e:
            print(f"FFmpeg not available ({e}), falling back to GIF")
            output_path = output_path.replace(".mp4", ".gif")
            format = "gif"

    if format == "gif":
        writer = PillowWriter(fps=fps)
        anim.save(output_path, writer=writer, dpi=dpi // 2)  # Lower DPI for GIF

    plt.close(fig)

    print(f"Animation exported to: {output_path}")
    return output_path


def export_frames(
    output_dir: str,
    vortex_field: VortexParticleField,
    hydraulics: HydraulicsEngine,
    n_frames: int = 50,
    dt: float = 0.05,
    steps_per_frame: int = 3,
    dpi: int = 150,
    theme: Optional[Theme] = None,
    prefix: str = "frame",
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> list:
    """
    Export simulation as PNG frame sequence.

    Parameters
    ----------
    output_dir : str
        Directory to save frames
    vortex_field : VortexParticleField
        Vortex particle field to animate
    hydraulics : HydraulicsEngine
        Hydraulics engine
    n_frames : int
        Number of frames to export
    dt : float
        Timestep per physics step
    steps_per_frame : int
        Physics steps per frame
    dpi : int
        Resolution
    theme : Theme, optional
        Visual theme
    prefix : str
        Filename prefix
    progress_callback : callable, optional
        Progress callback

    Returns
    -------
    list
        List of exported file paths
    """
    if not _HAS_MATPLOTLIB:
        raise ImportError("Matplotlib required for frame export")

    if theme is None:
        theme = get_theme()

    os.makedirs(output_dir, exist_ok=True)

    exported_files = []

    for frame in range(n_frames):
        # Step physics
        for _ in range(steps_per_frame):
            vortex_field.step(dt=dt)

        # Get state
        state = vortex_field.get_state()

        # Create figure
        fig, axes = create_figure_layout(theme)

        # Render plots
        plot_plan_view(axes["plan"], state, hydraulics, theme)
        plot_theory_panel(axes["theory"], hydraulics, vortex_field, theme)
        plot_profile_view(axes["profile"], state, hydraulics, theme)
        plot_detail_map(axes["detail"], state, vortex_field, theme)
        plot_velocity_profile(axes["velocity"], hydraulics, theme)
        plot_energy_spectrum(axes["spectrum"], state, hydraulics, theme)

        # Save
        filename = f"{prefix}_{frame:04d}.png"
        filepath = os.path.join(output_dir, filename)
        fig.savefig(filepath, dpi=dpi, facecolor=theme.background, bbox_inches="tight")
        plt.close(fig)

        exported_files.append(filepath)

        if progress_callback:
            progress_callback(frame + 1, n_frames)

    print(f"Exported {n_frames} frames to: {output_dir}")
    return exported_files


def export_single_frame(
    output_path: str,
    vortex_field: VortexParticleField,
    hydraulics: HydraulicsEngine,
    dpi: int = 200,
    theme: Optional[Theme] = None,
) -> str:
    """
    Export single frame snapshot.

    Parameters
    ----------
    output_path : str
        Output file path
    vortex_field : VortexParticleField
        Vortex field
    hydraulics : HydraulicsEngine
        Hydraulics engine
    dpi : int
        Resolution
    theme : Theme, optional
        Visual theme

    Returns
    -------
    str
        Path to exported file
    """
    if not _HAS_MATPLOTLIB:
        raise ImportError("Matplotlib required for frame export")

    if theme is None:
        theme = get_theme()

    state = vortex_field.get_state()

    fig, axes = create_figure_layout(theme)

    plot_plan_view(axes["plan"], state, hydraulics, theme)
    plot_theory_panel(axes["theory"], hydraulics, vortex_field, theme)
    plot_profile_view(axes["profile"], state, hydraulics, theme)
    plot_detail_map(axes["detail"], state, vortex_field, theme)
    plot_velocity_profile(axes["velocity"], hydraulics, theme)
    plot_energy_spectrum(axes["spectrum"], state, hydraulics, theme)

    fig.savefig(output_path, dpi=dpi, facecolor=theme.background, bbox_inches="tight")
    plt.close(fig)

    print(f"Frame exported to: {output_path}")
    return output_path


def export_plan_view_only(
    output_path: str,
    vortex_field: VortexParticleField,
    hydraulics: HydraulicsEngine,
    figsize: tuple = (12, 8),
    dpi: int = 200,
    theme: Optional[Theme] = None,
) -> str:
    """
    Export just the plan view (useful for presentations).

    Parameters
    ----------
    output_path : str
        Output file path
    vortex_field : VortexParticleField
        Vortex field
    hydraulics : HydraulicsEngine
        Hydraulics engine
    figsize : tuple
        Figure size
    dpi : int
        Resolution
    theme : Theme, optional
        Visual theme

    Returns
    -------
    str
        Path to exported file
    """
    if not _HAS_MATPLOTLIB:
        raise ImportError("Matplotlib required for export")

    if theme is None:
        theme = get_theme()

    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(theme.background)

    state = vortex_field.get_state()
    plot_plan_view(ax, state, hydraulics, theme)

    fig.savefig(output_path, dpi=dpi, facecolor=theme.background, bbox_inches="tight")
    plt.close(fig)

    return output_path
