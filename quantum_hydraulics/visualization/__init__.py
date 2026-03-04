"""
Visualization modules for quantum hydraulics.

Contains:
- Theme: Professional color schemes
- Renderers: Plot functions for different views
- Interactive: Interactive simulator with controls
- Export: Animation and frame export utilities
"""

from quantum_hydraulics.visualization.theme import Theme, THEMES, get_theme
from quantum_hydraulics.visualization.renderers import (
    plot_plan_view,
    plot_profile_view,
    plot_velocity_profile,
    plot_energy_spectrum,
    plot_detail_map,
    create_figure_layout,
)
from quantum_hydraulics.visualization.interactive import InteractiveSimulator
from quantum_hydraulics.visualization.export import export_animation, export_frames

__all__ = [
    "Theme",
    "THEMES",
    "get_theme",
    "plot_plan_view",
    "plot_profile_view",
    "plot_velocity_profile",
    "plot_energy_spectrum",
    "plot_detail_map",
    "create_figure_layout",
    "InteractiveSimulator",
    "export_animation",
    "export_frames",
]
