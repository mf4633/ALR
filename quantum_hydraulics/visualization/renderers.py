"""
Rendering functions for quantum hydraulics visualization.

Provides clean, separated plot functions for different views.
All functions accept FieldState and Theme for consistent rendering.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.lines import Line2D
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.colorbar import Colorbar
from typing import Dict, Optional, Tuple, Any

from quantum_hydraulics.core.vortex_field import FieldState
from quantum_hydraulics.core.hydraulics import HydraulicsEngine
from quantum_hydraulics.visualization.theme import Theme, get_theme


def _cleanup_colorbar(ax: Axes):
    """Remove existing colorbar to prevent accumulation."""
    if hasattr(ax, "_qh_colorbar") and ax._qh_colorbar is not None:
        try:
            ax._qh_colorbar.remove()
        except (ValueError, AttributeError):
            pass
        ax._qh_colorbar = None


def _store_colorbar(ax: Axes, cbar: Colorbar):
    """Store colorbar reference for later cleanup."""
    ax._qh_colorbar = cbar


def create_figure_layout(
    theme: Optional[Theme] = None, figsize: Tuple[int, int] = (20, 11)
) -> Tuple[Figure, Dict[str, Axes]]:
    """
    Create standard figure layout with all plot areas.

    Parameters
    ----------
    theme : Theme, optional
        Visual theme to apply
    figsize : tuple, optional
        Figure size in inches

    Returns
    -------
    fig : Figure
        Matplotlib figure
    axes : dict
        Dictionary of axes: 'plan', 'theory', 'profile', 'detail', 'velocity', 'spectrum'
    """
    if theme is None:
        theme = get_theme()

    fig = plt.figure(figsize=figsize)
    fig.patch.set_facecolor(theme.background)

    gs = fig.add_gridspec(
        3, 4, hspace=0.3, wspace=0.3, left=0.08, right=0.98, top=0.94, bottom=0.12
    )

    axes = {
        "plan": fig.add_subplot(gs[0:2, 0:3]),
        "theory": fig.add_subplot(gs[0, 3]),
        "profile": fig.add_subplot(gs[1, 3]),
        "detail": fig.add_subplot(gs[2, 0:2]),
        "velocity": fig.add_subplot(gs[2, 2]),
        "spectrum": fig.add_subplot(gs[2, 3]),
    }

    # Apply theme to all axes
    for ax in axes.values():
        ax.set_facecolor(theme.background)
        ax.tick_params(colors=theme.foreground, labelsize=8)

    return fig, axes


def plot_plan_view(
    ax: Axes,
    state: FieldState,
    hydraulics: HydraulicsEngine,
    theme: Optional[Theme] = None,
    show_trails: bool = True,
    show_vectors: bool = False,
) -> Axes:
    """
    Plot plan view with vortex particles.

    Parameters
    ----------
    ax : Axes
        Matplotlib axes to plot on
    state : FieldState
        Current field state
    hydraulics : HydraulicsEngine
        Hydraulics engine for summary info
    theme : Theme, optional
        Visual theme
    show_trails : bool
        Whether to show particle trails
    show_vectors : bool
        Whether to show velocity vectors

    Returns
    -------
    Axes
        The axes with the plot
    """
    if theme is None:
        theme = get_theme()

    # Store original position before clearing
    if not hasattr(ax, "_original_position"):
        ax._original_position = ax.get_position()

    _cleanup_colorbar(ax)
    ax.clear()
    ax.set_facecolor(theme.background)
    ax.set_position(ax._original_position)

    if state.n_particles == 0:
        ax.text(0.5, 0.5, "No particles", ha="center", va="center", color=theme.foreground)
        return ax

    positions = state.positions
    energies = state.energies
    sigmas = state.sigmas

    # Particle size reflects resolution
    sizes = 1200 / (sigmas ** 2)
    sizes = np.clip(sizes, 4, 120)

    # Plot particles
    scatter = ax.scatter(
        positions[:, 0],
        positions[:, 1],
        c=energies,
        cmap=theme.energy_cmap,
        s=sizes,
        alpha=0.8,
        vmin=0,
        vmax=np.percentile(energies, 95) if len(energies) > 0 else 1,
        linewidths=0,
    )

    # Plot trails
    if show_trails:
        for trail in state.trails:
            if len(trail) > 1:
                ax.plot(
                    trail[:, 0],
                    trail[:, 1],
                    color=theme.foreground,
                    alpha=0.08,
                    linewidth=0.8,
                )

    # Observation zone
    if state.observation_active:
        obs_circle = Circle(
            (state.obs_center[0], state.obs_center[1]),
            state.obs_radius,
            fill=False,
            edgecolor=theme.observation_color,
            linewidth=3,
            linestyle="--",
            alpha=0.9,
        )
        ax.add_patch(obs_circle)
        ax.text(
            state.obs_center[0],
            state.obs_center[1] + state.obs_radius + 3,
            "OBSERVATION\nZONE",
            color=theme.observation_color,
            fontsize=10,
            ha="center",
            weight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor=theme.background, alpha=0.7),
        )

    # Channel boundaries
    ax.plot(
        [0, state.domain_length],
        [0, 0],
        color=theme.channel_color,
        linewidth=3,
        alpha=0.7,
    )
    ax.plot(
        [0, state.domain_length],
        [state.domain_width, state.domain_width],
        color=theme.channel_color,
        linewidth=3,
        alpha=0.7,
    )

    ax.set_xlim(0, state.domain_length)
    ax.set_ylim(-2, state.domain_width + 2)
    ax.set_aspect("equal")
    ax.set_xlabel("Streamwise Distance (ft)", color=theme.foreground, fontsize=10)
    ax.set_ylabel("Cross-Stream Distance (ft)", color=theme.foreground, fontsize=10)

    # Title with flow regime
    summary = hydraulics.get_summary()
    title_color = theme.get_title_color(summary["Fr"])
    ax.set_title(
        f"3D VORTEX PARTICLE FIELD - {summary['flow_regime']} - "
        f"{state.n_particles} Particles",
        color=title_color,
        fontsize=13,
        weight="bold",
        pad=10,
    )

    # Info box
    info_text = (
        "FIRST-PRINCIPLES PHYSICS + VORTEX PARTICLES\n"
        "Biot-Savart Law - Kolmogorov Cascade - Adaptive Resolution"
    )
    ax.text(
        0.01,
        0.97,
        info_text,
        transform=ax.transAxes,
        fontsize=9,
        color=theme.accent_primary,
        va="top",
        weight="bold",
        bbox=dict(boxstyle="round,pad=0.5", facecolor=theme.background, alpha=0.9),
    )

    # Legend
    old_legend = ax.get_legend()
    if old_legend is not None:
        old_legend.remove()

    legend_elements = [
        Line2D(
            [0], [0], marker="o", color="w",
            markerfacecolor=theme.accent_primary,
            markersize=4, linestyle="", label="High Resolution (small sigma)"
        ),
        Line2D(
            [0], [0], marker="o", color="w",
            markerfacecolor=theme.accent_primary,
            markersize=10, linestyle="", label="Low Resolution (large sigma)"
        ),
    ]
    ax.legend(
        handles=legend_elements,
        loc="lower left",
        fontsize=8,
        facecolor=theme.background,
        edgecolor=theme.foreground,
        labelcolor=theme.foreground,
        framealpha=0.9,
    )

    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax, pad=0.01, fraction=0.03)
    cbar.set_label("Vortex Energy", color=theme.foreground, fontsize=9)
    cbar.ax.tick_params(colors=theme.foreground, labelsize=8)
    _store_colorbar(ax, cbar)

    return ax


def plot_profile_view(
    ax: Axes,
    state: FieldState,
    hydraulics: HydraulicsEngine,
    theme: Optional[Theme] = None,
) -> Axes:
    """
    Plot longitudinal profile view.

    Parameters
    ----------
    ax : Axes
        Matplotlib axes
    state : FieldState
        Current field state
    hydraulics : HydraulicsEngine
        Hydraulics engine
    theme : Theme, optional
        Visual theme

    Returns
    -------
    Axes
    """
    if theme is None:
        theme = get_theme()

    if not hasattr(ax, "_original_position"):
        ax._original_position = ax.get_position()

    ax.clear()
    ax.set_facecolor(theme.background)
    ax.set_position(ax._original_position)

    length = state.domain_length
    depth = state.domain_depth
    slope = hydraulics.slope

    x_profile = np.array([0, length])
    bed_elev = np.array([10, 10 - slope * length])
    wse = bed_elev + depth

    ax.fill_between(x_profile, bed_elev, wse, color=theme.water_color, alpha=0.4)
    ax.plot(x_profile, wse, color=theme.water_color, linewidth=2.5, label="Water Surface")
    ax.plot(x_profile, bed_elev, color=theme.bed_color, linewidth=3, label="Channel Bed")

    if state.observation_active:
        ax.axvline(
            state.obs_center[0],
            color=theme.observation_color,
            linestyle="--",
            linewidth=2,
            alpha=0.8,
            label="Observation",
        )

    ax.set_xlim(0, length)
    ax.set_ylim(bed_elev[-1] - 1, wse[0] + 1)
    ax.set_xlabel("Distance (ft)", color=theme.foreground, fontsize=9)
    ax.set_ylabel("Elevation (ft)", color=theme.foreground, fontsize=9)
    ax.set_title("LONGITUDINAL PROFILE", color=theme.foreground, fontsize=10, weight="bold")
    ax.legend(fontsize=7, facecolor=theme.background, edgecolor=theme.foreground, labelcolor=theme.foreground)
    ax.grid(True, alpha=theme.grid_alpha, color=theme.grid_color, linestyle=":")
    ax.tick_params(colors=theme.foreground)

    return ax


def plot_velocity_profile(
    ax: Axes,
    hydraulics: HydraulicsEngine,
    theme: Optional[Theme] = None,
) -> Axes:
    """
    Plot vertical velocity profile.

    Parameters
    ----------
    ax : Axes
        Matplotlib axes
    hydraulics : HydraulicsEngine
        Hydraulics engine
    theme : Theme, optional
        Visual theme

    Returns
    -------
    Axes
    """
    if theme is None:
        theme = get_theme()

    if not hasattr(ax, "_original_position"):
        ax._original_position = ax.get_position()

    ax.clear()
    ax.set_facecolor(theme.background)
    ax.set_position(ax._original_position)

    z_values = np.linspace(0.001, hydraulics.depth, 100)
    u_values = [hydraulics.velocity_profile(z) for z in z_values]

    ax.plot(u_values, z_values, color=theme.accent_primary, linewidth=2.5, label="Theory")
    ax.axhline(
        hydraulics.depth * 0.2,
        color=theme.accent_secondary,
        linestyle=":",
        linewidth=1,
        alpha=0.7,
        label="Log-Law Limit",
    )
    ax.axvline(
        hydraulics.V_mean,
        color=theme.subcritical_color,
        linestyle="--",
        linewidth=1.5,
        alpha=0.7,
        label="Mean Velocity",
    )

    ax.set_xlabel("Velocity (ft/s)", color=theme.foreground, fontsize=9)
    ax.set_ylabel("Height Above Bed (ft)", color=theme.foreground, fontsize=9)
    ax.set_title("VELOCITY PROFILE\nLog Law + Power Law", color=theme.foreground, fontsize=10, weight="bold")
    ax.legend(fontsize=7, facecolor=theme.background, labelcolor=theme.foreground)
    ax.grid(True, alpha=theme.grid_alpha, color=theme.grid_color, linestyle=":")
    ax.set_xlim(0, max(u_values) * 1.1 if u_values else 1)
    ax.set_ylim(0, hydraulics.depth)
    ax.tick_params(colors=theme.foreground)

    return ax


def plot_energy_spectrum(
    ax: Axes,
    state: FieldState,
    hydraulics: HydraulicsEngine,
    theme: Optional[Theme] = None,
) -> Axes:
    """
    Plot turbulent energy spectrum with Kolmogorov -5/3 law.

    Parameters
    ----------
    ax : Axes
        Matplotlib axes
    state : FieldState
        Current field state
    hydraulics : HydraulicsEngine
        Hydraulics engine
    theme : Theme, optional
        Visual theme

    Returns
    -------
    Axes
    """
    if theme is None:
        theme = get_theme()

    if not hasattr(ax, "_original_position"):
        ax._original_position = ax.get_position()

    ax.clear()
    ax.set_facecolor(theme.background)
    ax.set_position(ax._original_position)

    if state.n_particles == 0:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", color=theme.foreground)
        return ax

    energies = state.energies
    scales = state.sigmas

    # Bin by scale
    scale_bins = np.logspace(np.log10(scales.min()), np.log10(scales.max()), 15)
    energy_binned = []
    scale_centers = []

    for i in range(len(scale_bins) - 1):
        mask = (scales >= scale_bins[i]) & (scales < scale_bins[i + 1])
        if np.any(mask):
            energy_binned.append(energies[mask].sum())
            scale_centers.append(np.sqrt(scale_bins[i] * scale_bins[i + 1]))

    if len(scale_centers) > 0:
        ax.loglog(
            scale_centers,
            energy_binned,
            "o-",
            color=theme.accent_primary,
            linewidth=2,
            markersize=6,
            label="Vortex Energy",
        )

        # Theoretical -5/3 slope
        k = 1.0 / np.array(scale_centers)
        E_theory = hydraulics.epsilon ** (2 / 3) * k ** (-5 / 3)
        mid_idx = len(energy_binned) // 2
        E_theory *= energy_binned[mid_idx] / E_theory[mid_idx]
        ax.loglog(
            scale_centers,
            E_theory,
            "--",
            color=theme.observation_color,
            linewidth=2,
            alpha=0.7,
            label="Kolmogorov -5/3",
        )

    ax.set_xlabel("Length Scale (ft)", color=theme.foreground, fontsize=9)
    ax.set_ylabel("Energy", color=theme.foreground, fontsize=9)
    ax.set_title("ENERGY SPECTRUM\nKolmogorov Cascade", color=theme.foreground, fontsize=10, weight="bold")
    ax.legend(fontsize=7, facecolor=theme.background, labelcolor=theme.foreground)
    ax.grid(True, alpha=theme.grid_alpha, color=theme.grid_color, linestyle=":")
    ax.tick_params(colors=theme.foreground)

    return ax


def plot_detail_map(
    ax: Axes,
    state: FieldState,
    vortex_field: Any,
    theme: Optional[Theme] = None,
) -> Axes:
    """
    Plot computational detail map showing observation-dependent resolution.

    Parameters
    ----------
    ax : Axes
        Matplotlib axes
    state : FieldState
        Current field state
    vortex_field : VortexParticleField
        Vortex field for computing adaptive core sizes
    theme : Theme, optional
        Visual theme

    Returns
    -------
    Axes
    """
    if theme is None:
        theme = get_theme()

    if not hasattr(ax, "_original_position"):
        ax._original_position = ax.get_position()

    _cleanup_colorbar(ax)
    ax.clear()
    ax.set_facecolor(theme.background)
    ax.set_position(ax._original_position)

    # Create grid
    x_grid = np.linspace(0, state.domain_length, 80)
    y_grid = np.linspace(0, state.domain_width, 40)
    X, Y = np.meshgrid(x_grid, y_grid)

    # Compute core size at each point
    Z = np.zeros_like(X)
    for i in range(len(x_grid)):
        for j in range(len(y_grid)):
            pos = np.array([x_grid[i], y_grid[j], state.domain_depth / 2])
            Z[j, i] = vortex_field.get_adaptive_core_size(pos)

    # Plot as heatmap (inverted: small sigma = bright = expensive)
    im = ax.contourf(X, Y, 1.0 / Z, levels=25, cmap=theme.detail_cmap)

    # Observation zone
    if state.observation_active:
        obs_circle = Circle(
            (state.obs_center[0], state.obs_center[1]),
            state.obs_radius,
            fill=False,
            edgecolor=theme.observation_color,
            linewidth=3,
            linestyle="--",
        )
        ax.add_patch(obs_circle)

    ax.set_xlim(0, state.domain_length)
    ax.set_ylim(0, state.domain_width)
    ax.set_xlabel("Distance (ft)", color=theme.foreground, fontsize=9)
    ax.set_ylabel("Width (ft)", color=theme.foreground, fontsize=9)
    ax.set_title(
        "COMPUTATIONAL RESOLUTION MAP\n(Bright = High Resolution = Expensive)",
        color=theme.foreground,
        fontsize=10,
        weight="bold",
    )

    cbar = plt.colorbar(im, ax=ax, label="1/sigma (Resolution)")
    cbar.ax.tick_params(colors=theme.foreground, labelsize=8)
    cbar.set_label("Resolution (1/sigma)", color=theme.foreground, fontsize=8)
    _store_colorbar(ax, cbar)
    ax.tick_params(colors=theme.foreground)

    return ax


def plot_theory_panel(
    ax: Axes,
    hydraulics: HydraulicsEngine,
    vortex_field: Any,
    theme: Optional[Theme] = None,
) -> Axes:
    """
    Plot physics theory information panel.

    Parameters
    ----------
    ax : Axes
        Matplotlib axes
    hydraulics : HydraulicsEngine
        Hydraulics engine
    vortex_field : VortexParticleField
        Vortex field
    theme : Theme, optional
        Visual theme

    Returns
    -------
    Axes
    """
    if theme is None:
        theme = get_theme()

    if not hasattr(ax, "_original_position"):
        ax._original_position = ax.get_position()

    ax.clear()
    ax.set_facecolor(theme.background)
    ax.set_position(ax._original_position)
    ax.axis("off")

    summary = hydraulics.get_summary()

    theory_text = (
        "===========================\n"
        "   FIRST-PRINCIPLES PHYSICS\n"
        "===========================\n\n"
        "Conservation Laws:\n"
        f"|- Q = {summary['Q']:.0f} cfs\n"
        f"|- V = {summary['V']:.2f} ft/s\n"
        f"|- A = {summary['A']:.1f} ft2\n"
        f"+- R = {summary['R']:.2f} ft\n\n"
        "Dimensionless Groups:\n"
        f"|- Re = {summary['Re']:.0f}\n"
        f"|- Fr = {summary['Fr']:.3f}\n"
        f"+- Re_t = {summary['Re_t']:.0f}\n\n"
        "Friction (Colebrook):\n"
        f"|- f = {summary['f']:.5f}\n"
        f"|- u* = {summary['u_star']:.3f} ft/s\n"
        f"+- Sf = {summary['Sf']:.5f}\n\n"
        "Turbulence Scales:\n"
        f"|- eps = {summary['epsilon']:.4f} ft2/s3\n"
        f"|- eta_K = {summary['eta']:.5f} ft\n"
        f"+- k = {summary['TKE']:.3f} ft2/s2\n\n"
        "Vortex Particles:\n"
        f"|- N = {len(vortex_field.particles)}\n"
        f"|- sigma_base = {vortex_field.base_sigma:.4f} ft\n"
        f"+- sigma_min = {vortex_field.min_sigma:.5f} ft\n\n"
        f"Flow: {summary['flow_regime']}\n"
        f"Type: {summary['uniformity']}"
    )

    ax.text(
        0.05,
        0.95,
        theory_text,
        transform=ax.transAxes,
        fontsize=8,
        color=theme.subcritical_color,
        family="monospace",
        va="top",
        bbox=dict(boxstyle="round,pad=0.8", facecolor=theme.background, alpha=0.95),
    )

    return ax
