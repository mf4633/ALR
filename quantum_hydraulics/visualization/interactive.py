"""
Interactive simulator with controls and animation.

Provides the main interactive visualization with sliders, buttons,
and animation controls.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, CheckButtons, Button
from matplotlib.animation import FuncAnimation
from typing import Optional, Dict, Any, Callable

from quantum_hydraulics.core.hydraulics import HydraulicsEngine
from quantum_hydraulics.core.vortex_field import VortexParticleField
from quantum_hydraulics.visualization.theme import Theme, get_theme
from quantum_hydraulics.visualization.renderers import (
    plot_plan_view,
    plot_profile_view,
    plot_velocity_profile,
    plot_energy_spectrum,
    plot_detail_map,
    plot_theory_panel,
)


class InteractiveSimulator:
    """
    Interactive quantum hydraulics simulator with full UI.

    Parameters
    ----------
    Q : float, optional
        Initial discharge in cfs (default 600)
    width : float, optional
        Initial channel width in feet (default 30)
    depth : float, optional
        Initial water depth in feet (default 5)
    slope : float, optional
        Initial bed slope (default 0.002)
    roughness : float, optional
        Initial roughness ks in feet (default 0.15)
    length : float, optional
        Channel length in feet (default 200)
    n_particles : int, optional
        Number of particles (default 6000)
    theme : str or Theme, optional
        Visual theme name or Theme object (default 'dark_professional')
    """

    def __init__(
        self,
        Q: float = 600.0,
        width: float = 30.0,
        depth: float = 5.0,
        slope: float = 0.002,
        roughness: float = 0.15,
        length: float = 200.0,
        n_particles: int = 6000,
        theme: Optional[str] = None,
    ):
        # Parameters
        self.Q = Q
        self.width = width
        self.depth = depth
        self.slope = slope
        self.roughness = roughness
        self.length = length
        self.n_particles = n_particles

        # Theme
        if isinstance(theme, str):
            self.theme = get_theme(theme)
        elif isinstance(theme, Theme):
            self.theme = theme
        else:
            self.theme = get_theme()

        # Initialize physics
        self.hydraulics = HydraulicsEngine(
            self.Q, self.width, self.depth, self.slope, self.roughness
        )
        self.vortex_field = VortexParticleField(
            self.hydraulics, length=self.length, n_particles=n_particles
        )

        # Animation control
        self.running = False
        self.steps_per_frame = 3
        self.anim: Optional[FuncAnimation] = None

        # UI elements
        self.fig = None
        self.axes: Dict[str, Any] = {}
        self.sliders: Dict[str, Slider] = {}
        self.checkbox: Optional[CheckButtons] = None
        self.button: Optional[Button] = None

        # Callbacks
        self._on_parameter_change_callback: Optional[Callable] = None

    def _create_ui(self):
        """Create the interactive UI."""
        plt.style.use("dark_background")
        self.fig = plt.figure(figsize=(20, 11))
        self.fig.patch.set_facecolor(self.theme.background)

        # Main plot areas
        gs = self.fig.add_gridspec(
            3, 4, hspace=0.3, wspace=0.3, left=0.08, right=0.98, top=0.94, bottom=0.12
        )

        self.axes = {
            "plan": self.fig.add_subplot(gs[0:2, 0:3]),
            "theory": self.fig.add_subplot(gs[0, 3]),
            "profile": self.fig.add_subplot(gs[1, 3]),
            "detail": self.fig.add_subplot(gs[2, 0:2]),
            "velocity": self.fig.add_subplot(gs[2, 2]),
            "spectrum": self.fig.add_subplot(gs[2, 3]),
        }

        self._create_controls()
        self._update_visualization()

    def _create_controls(self):
        """Create sliders and buttons."""
        slider_specs = [
            ("Q", "Discharge Q (cfs)", 100, 2000, self.Q, "%1.0f"),
            ("width", "Channel Width (ft)", 15, 60, self.width, "%1.0f"),
            ("depth", "Water Depth (ft)", 2, 12, self.depth, "%1.1f"),
            ("slope", "Bed Slope", 0.0001, 0.01, self.slope, "%1.4f"),
            ("roughness", "Roughness ks (ft)", 0.01, 0.5, self.roughness, "%1.3f"),
        ]

        slider_x = 0.12
        slider_width = 0.25
        slider_height = 0.015
        y_positions = [0.08, 0.06, 0.04, 0.02, 0.00]

        for (key, label, vmin, vmax, vinit, fmt), y_pos in zip(slider_specs, y_positions):
            ax_slider = plt.axes(
                [slider_x, y_pos, slider_width, slider_height],
                facecolor=self.theme.background,
            )
            slider = Slider(
                ax_slider, label, vmin, vmax, valinit=vinit, valfmt=fmt,
                color=self.theme.accent_primary
            )
            slider.label.set_color(self.theme.foreground)
            slider.label.set_fontsize(9)
            slider.valtext.set_color(self.theme.foreground)
            slider.on_changed(lambda val, k=key: self._on_parameter_change(k, val))
            self.sliders[key] = slider

        # Checkbox for observation
        ax_check = plt.axes(
            [slider_x + slider_width + 0.02, 0.01, 0.12, 0.08],
            facecolor=self.theme.background,
        )
        self.checkbox = CheckButtons(ax_check, ["Quantum\nObservation"], [True])
        self.checkbox.labels[0].set_color(self.theme.accent_primary)
        self.checkbox.labels[0].set_fontsize(10)
        self.checkbox.on_clicked(self._toggle_observation)

        # Start/Stop button
        ax_button = plt.axes([slider_x + slider_width + 0.15, 0.01, 0.08, 0.04])
        self.button = Button(
            ax_button, "Pause",
            color=self.theme.background, hovercolor="#4a4a4a"
        )
        self.button.label.set_color(self.theme.foreground)
        self.button.on_clicked(self._toggle_animation)
        self.running = True

    def _on_parameter_change(self, param: str, value: float):
        """Handle parameter slider changes."""
        setattr(self, param, value)

        self.hydraulics = HydraulicsEngine(
            self.Q, self.width, self.depth, self.slope, self.roughness
        )
        self.vortex_field.update_hydraulics(self.hydraulics)

        if self._on_parameter_change_callback:
            self._on_parameter_change_callback(param, value)

    def _toggle_observation(self, label: str):
        """Toggle observation zone."""
        self.vortex_field.toggle_observation()

    def _toggle_animation(self, event):
        """Toggle animation."""
        self.running = not self.running
        if self.button:
            self.button.label.set_text("Resume" if not self.running else "Pause")

    def _animate_frame(self, frame: int):
        """Animation update function."""
        if self.running:
            for _ in range(self.steps_per_frame):
                self.vortex_field.step(dt=0.05)
            self._update_visualization()

    def _update_visualization(self):
        """Update all plots."""
        state = self.vortex_field.get_state()

        plot_plan_view(self.axes["plan"], state, self.hydraulics, self.theme)
        plot_theory_panel(self.axes["theory"], self.hydraulics, self.vortex_field, self.theme)
        plot_profile_view(self.axes["profile"], state, self.hydraulics, self.theme)
        plot_detail_map(self.axes["detail"], state, self.vortex_field, self.theme)
        plot_velocity_profile(self.axes["velocity"], self.hydraulics, self.theme)
        plot_energy_spectrum(self.axes["spectrum"], state, self.hydraulics, self.theme)

        plt.draw()

    def run(self, interval: int = 50):
        """
        Launch the interactive simulator.

        Parameters
        ----------
        interval : int
            Animation interval in milliseconds
        """
        self._create_ui()

        self.anim = FuncAnimation(
            self.fig, self._animate_frame,
            interval=interval, blit=False, cache_frame_data=False
        )

        plt.show()

    def set_observation(self, x: float, y: float, z: float, radius: float):
        """
        Set observation zone location.

        Parameters
        ----------
        x, y, z : float
            Observation center coordinates
        radius : float
            Observation radius
        """
        self.vortex_field.set_observation(np.array([x, y, z]), radius)

    def set_theme(self, theme_name: str):
        """
        Change visual theme.

        Parameters
        ----------
        theme_name : str
            Name of theme to apply
        """
        self.theme = get_theme(theme_name)

    def on_parameter_change(self, callback: Callable):
        """
        Register callback for parameter changes.

        Parameters
        ----------
        callback : Callable
            Function(param_name, new_value) called when parameters change
        """
        self._on_parameter_change_callback = callback

    def get_state(self):
        """Get current field state."""
        return self.vortex_field.get_state()

    def get_hydraulics_summary(self) -> Dict:
        """Get current hydraulics summary."""
        return self.hydraulics.get_summary()


def run_interactive(
    Q: float = 600.0,
    width: float = 30.0,
    depth: float = 5.0,
    slope: float = 0.002,
    roughness: float = 0.15,
    theme: str = "dark_professional",
    n_particles: int = 6000,
):
    """
    Convenience function to launch the interactive simulator.

    Parameters
    ----------
    Q : float
        Discharge in cfs
    width : float
        Channel width in feet
    depth : float
        Water depth in feet
    slope : float
        Bed slope
    roughness : float
        Roughness ks in feet
    theme : str
        Theme name
    n_particles : int
        Number of particles
    """
    print("\n" + "=" * 80)
    print(" " * 15 + "QUANTUM-INSPIRED VORTEX PARTICLE SIMULATOR")
    print("=" * 80)
    print("\nCombining:")
    print("  - First-principles physics (Colebrook-White, Kolmogorov cascade)")
    print("  - True 3D vortex particle method (Biot-Savart law)")
    print("  - Observation-dependent resolution (adaptive core size sigma)")
    print("  - Professional visualization with theme system")
    print("=" * 80 + "\n")

    sim = InteractiveSimulator(
        Q=Q, width=width, depth=depth, slope=slope,
        roughness=roughness, theme=theme, n_particles=n_particles
    )
    sim.run()

    return sim
