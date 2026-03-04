"""
Theme system for quantum hydraulics visualization.

Provides professional color schemes for different use cases:
- dark_professional: Dark theme with cyan accents (default)
- light_publication: Light theme for papers and reports
- hec_ras_style: Familiar colors for HEC-RAS users
"""

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class Theme:
    """
    Visual theme configuration for plots.

    Attributes
    ----------
    name : str
        Theme name
    background : str
        Background color (hex)
    foreground : str
        Foreground/text color (hex)
    accent_primary : str
        Primary accent color (hex)
    accent_secondary : str
        Secondary accent color (hex)
    velocity_cmap : str
        Colormap for velocity fields
    energy_cmap : str
        Colormap for energy fields
    detail_cmap : str
        Colormap for computational detail
    grid_color : str
        Grid line color (hex)
    grid_alpha : float
        Grid transparency
    channel_color : str
        Channel boundary color (hex)
    water_color : str
        Water surface color (hex)
    bed_color : str
        Channel bed color (hex)
    observation_color : str
        Observation zone marker color (hex)
    supercritical_color : str
        Title color for supercritical flow
    subcritical_color : str
        Title color for subcritical flow
    vortex_positive_color : str
        Color for positive vorticity
    vortex_negative_color : str
        Color for negative vorticity
    """

    name: str
    background: str
    foreground: str
    accent_primary: str
    accent_secondary: str
    velocity_cmap: str
    energy_cmap: str
    detail_cmap: str
    grid_color: str
    grid_alpha: float
    channel_color: str
    water_color: str
    bed_color: str
    observation_color: str
    supercritical_color: str
    subcritical_color: str
    vortex_positive_color: str
    vortex_negative_color: str

    def get_title_color(self, froude_number: float) -> str:
        """Get title color based on flow regime."""
        return self.supercritical_color if froude_number > 1 else self.subcritical_color


# Define available themes
THEMES: Dict[str, Theme] = {
    "dark_professional": Theme(
        name="dark_professional",
        background="#0a0a0a",
        foreground="#ffffff",
        accent_primary="#00d4ff",
        accent_secondary="#ff6b6b",
        velocity_cmap="turbo",
        energy_cmap="plasma",
        detail_cmap="viridis",
        grid_color="#ffffff",
        grid_alpha=0.2,
        channel_color="#8B4513",
        water_color="#00d4ff",
        bed_color="#8B4513",
        observation_color="#ffff00",
        supercritical_color="#ff4444",
        subcritical_color="#44ff44",
        vortex_positive_color="#00ffff",
        vortex_negative_color="#ff4444",
    ),
    "light_publication": Theme(
        name="light_publication",
        background="#ffffff",
        foreground="#000000",
        accent_primary="#0066cc",
        accent_secondary="#cc3300",
        velocity_cmap="coolwarm",
        energy_cmap="YlOrRd",
        detail_cmap="Blues",
        grid_color="#888888",
        grid_alpha=0.3,
        channel_color="#654321",
        water_color="#4488cc",
        bed_color="#8B7355",
        observation_color="#ff8800",
        supercritical_color="#cc0000",
        subcritical_color="#006600",
        vortex_positive_color="#0066cc",
        vortex_negative_color="#cc3300",
    ),
    "hec_ras_style": Theme(
        name="hec_ras_style",
        background="#f0f0f0",
        foreground="#000000",
        accent_primary="#0000ff",
        accent_secondary="#ff0000",
        velocity_cmap="jet",
        energy_cmap="jet",
        detail_cmap="jet",
        grid_color="#808080",
        grid_alpha=0.5,
        channel_color="#000000",
        water_color="#0000ff",
        bed_color="#808000",
        observation_color="#ff00ff",
        supercritical_color="#ff0000",
        subcritical_color="#0000ff",
        vortex_positive_color="#0000ff",
        vortex_negative_color="#ff0000",
    ),
    "ocean_blue": Theme(
        name="ocean_blue",
        background="#001a33",
        foreground="#ffffff",
        accent_primary="#00ccff",
        accent_secondary="#ff9933",
        velocity_cmap="ocean",
        energy_cmap="hot",
        detail_cmap="cubehelix",
        grid_color="#336699",
        grid_alpha=0.3,
        channel_color="#996633",
        water_color="#0099cc",
        bed_color="#663300",
        observation_color="#ffcc00",
        supercritical_color="#ff6600",
        subcritical_color="#00ff99",
        vortex_positive_color="#00ccff",
        vortex_negative_color="#ff6600",
    ),
}

# Default theme
_current_theme: Theme = THEMES["dark_professional"]


def get_theme(name: Optional[str] = None) -> Theme:
    """
    Get a theme by name.

    Parameters
    ----------
    name : str, optional
        Theme name. If None, returns current theme.
        Available: 'dark_professional', 'light_publication', 'hec_ras_style', 'ocean_blue'

    Returns
    -------
    Theme
        The requested theme

    Raises
    ------
    ValueError
        If theme name not found
    """
    if name is None:
        return _current_theme
    if name not in THEMES:
        available = ", ".join(THEMES.keys())
        raise ValueError(f"Theme '{name}' not found. Available: {available}")
    return THEMES[name]


def set_default_theme(name: str):
    """
    Set the default theme.

    Parameters
    ----------
    name : str
        Theme name to set as default
    """
    global _current_theme
    _current_theme = get_theme(name)


def list_themes() -> list:
    """Return list of available theme names."""
    return list(THEMES.keys())


def apply_theme_to_axes(ax, theme: Optional[Theme] = None):
    """
    Apply theme colors to matplotlib axes.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to style
    theme : Theme, optional
        Theme to apply. Uses current if None.
    """
    if theme is None:
        theme = _current_theme

    ax.set_facecolor(theme.background)
    ax.tick_params(colors=theme.foreground, labelsize=8)
    for spine in ax.spines.values():
        spine.set_color(theme.foreground)
    ax.xaxis.label.set_color(theme.foreground)
    ax.yaxis.label.set_color(theme.foreground)
    ax.title.set_color(theme.foreground)
