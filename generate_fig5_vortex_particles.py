"""
Generate Figure 5: Vortex Particle Distribution at Bridge Pier.

Publication-quality two-panel visualization showing horseshoe vortex
particles shed around a circular bridge pier, advected downstream by
the approach flow with turbulent diffusion.

  Left panel  -- Plan view (x-y): pier as filled gray circle, particles
                colored by vorticity magnitude, flow direction arrow.
  Right panel -- Cross-section (x-z): pier as gray rectangle, particles
                colored by vorticity magnitude, horseshoe vortex
                concentration near the bed vs surface separation higher up.

Usage:
    python generate_fig5_vortex_particles.py
"""

import sys
import os

# Ensure the quantum_hydraulics package is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

# Force non-interactive backend before any other matplotlib import
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
import matplotlib.colors as mcolors

from quantum_hydraulics.core.pier_shedding import PierBody
from quantum_hydraulics.core.hydraulics import HydraulicsEngine

# ── Theme ─────────────────────────────────────────────────────────────
try:
    from quantum_hydraulics.visualization.theme import THEMES
    theme = THEMES["light_publication"]
except Exception:
    theme = None

bg = theme.background if theme else "#ffffff"
fg = theme.foreground if theme else "#000000"
c1 = theme.accent_primary if theme else "#0066cc"       # navy
c2 = theme.accent_secondary if theme else "#cc3300"      # red
gc = theme.grid_color if theme else "#888888"
ga = theme.grid_alpha if theme else 0.3

# ── Simulation parameters ─────────────────────────────────────────────
PIER_X, PIER_Y = 25.0, 5.0
PIER_D = 5.0           # ft diameter
CHANNEL_W = 40.0       # ft
DEPTH = 10.0           # ft
V_APPROACH = 5.0       # ft/s
DT = 0.25              # s  (finer timestep for denser shedding)
N_STEPS = 200

# Turbulent diffusion coefficient (ft^2/s) -- representative for open-channel
DIFF_XY = 0.15
DIFF_Z = 0.05

# ── Run pier shedding simulation with advection ──────────────────────
# Each timestep: (1) advect all existing particles downstream,
#                (2) shed new particles at the pier,
#                (3) reflect any that drifted inside the pier.
pier = PierBody(x=PIER_X, y=PIER_Y, diameter=PIER_D)
rng = np.random.default_rng(42)

live_pos = np.empty((0, 3))
live_omega = np.empty((0, 3))
live_sigma = np.empty((0,))

for step in range(N_STEPS):
    n = len(live_pos)

    # 1. Advect existing particles
    if n > 0:
        # Mean-flow advection (downstream in +x)
        live_pos[:, 0] += V_APPROACH * DT

        # Lateral deflection around pier (simple potential-flow dipole push)
        dx = live_pos[:, 0] - PIER_X
        dy = live_pos[:, 1] - PIER_Y
        r2 = dx ** 2 + dy ** 2
        r2 = np.maximum(r2, (PIER_D * 0.6) ** 2)  # soften near pier
        # Dipole lateral velocity ~ U * R^2 * sin(2*theta) / r^2
        v_lat = V_APPROACH * (PIER_D / 2) ** 2 * 2 * dx * dy / (r2 ** 2) * DT
        live_pos[:, 1] += v_lat * 0.5

        # Turbulent diffusion (Brownian increments)
        live_pos[:, 0] += rng.normal(0, np.sqrt(2 * DIFF_XY * DT), n)
        live_pos[:, 1] += rng.normal(0, np.sqrt(2 * DIFF_XY * DT), n)
        live_pos[:, 2] += rng.normal(0, np.sqrt(2 * DIFF_Z * DT), n)

        # Clamp z to [0, depth]
        live_pos[:, 2] = np.clip(live_pos[:, 2], 0.0, DEPTH)

        # Reflect particles out of pier
        live_pos = pier.reflect_particles(live_pos)

        # Vortex stretching: slight decay of vorticity over time
        live_omega *= 0.98

    # 2. Shed new particles
    result = pier.shed_particles(V_approach=V_APPROACH, depth=DEPTH, dt=DT)
    if result is not None:
        new_pos, new_omega, new_sigma = result
        live_pos = np.vstack([live_pos, new_pos]) if n > 0 else new_pos.copy()
        live_omega = np.vstack([live_omega, new_omega]) if n > 0 else new_omega.copy()
        live_sigma = np.concatenate([live_sigma, new_sigma]) if n > 0 else new_sigma.copy()

positions = live_pos
vorticities = live_omega
sigmas = live_sigma

# Vorticity magnitude for coloring
omega_mag = np.linalg.norm(vorticities, axis=1)

print(f"Total particles shed: {len(positions)}")
print(f"  x range: [{positions[:,0].min():.1f}, {positions[:,0].max():.1f}] ft")
print(f"  y range: [{positions[:,1].min():.1f}, {positions[:,1].max():.1f}] ft")
print(f"  z range: [{positions[:,2].min():.1f}, {positions[:,2].max():.1f}] ft")
print(f"  |omega| range: [{omega_mag.min():.2f}, {omega_mag.max():.2f}]")

# ── Classify particles for annotation ────────────────────────────────
# Horseshoe particles sit near the bed (z < 0.2 * depth)
bed_mask = positions[:, 2] < 0.2 * DEPTH
surface_mask = ~bed_mask

print(f"  Horseshoe (near-bed) particles: {bed_mask.sum()}")
print(f"  Surface separation particles:   {surface_mask.sum()}")

# ── View window — focus on near-pier region (8D downstream) ──────────
VIEW_D = 12  # diameters downstream to show
x_lo = PIER_X - PIER_D * 3.5
x_hi = PIER_X + PIER_D * VIEW_D
y_lo = PIER_Y - PIER_D * 2.5
y_hi = PIER_Y + PIER_D * 2.5

# Filter particles to view window (for cleaner plotting)
in_view = (
    (positions[:, 0] >= x_lo) & (positions[:, 0] <= x_hi) &
    (positions[:, 1] >= y_lo) & (positions[:, 1] <= y_hi)
)
vp = positions[in_view]
vo = vorticities[in_view]
vs = sigmas[in_view]
vm = omega_mag[in_view]
bm = bed_mask[in_view]
sm = surface_mask[in_view]

print(f"  Particles in view window: {in_view.sum()} / {len(positions)}")

# ── Figure — stacked vertically for equal-aspect plan + cross-section ─
fig, (ax_plan, ax_xsec) = plt.subplots(
    2, 1, figsize=(11, 9.5), gridspec_kw={"hspace": 0.32}
)
fig.patch.set_facecolor(bg)

# Color normalization shared by both panels
vmin, vmax = 0.0, omega_mag.max()
norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
cmap = plt.cm.coolwarm

# Particle sizes: scale by sigma (larger core = larger dot)
s_base = 35
s_arr = s_base * (vs / vs.min()) ** 1.3 if len(vs) > 0 else np.array([])

# ── LEFT PANEL: Plan View (x-y) ──────────────────────────────────────
ax = ax_plan
ax.set_facecolor(bg)

# Very faint blue channel tint
ax.axhspan(y_lo, y_hi, color="#f2f6fc", zorder=0)

# Scatter particles
sc_plan = ax.scatter(
    vp[:, 0], vp[:, 1],
    c=vm, cmap=cmap, norm=norm,
    s=s_arr, alpha=0.80, edgecolors="#444444", linewidths=0.3, zorder=3,
)

# Pier circle
pier_circle = Circle(
    (PIER_X, PIER_Y), PIER_D / 2,
    facecolor="#b0b0b0", edgecolor="#333333", linewidth=1.8, zorder=4,
)
ax.add_patch(pier_circle)
ax.text(PIER_X, PIER_Y, f"D={PIER_D:.0f} ft", ha="center", va="center",
        fontsize=7.5, color="#222222", weight="bold", zorder=5)

# Flow direction arrow (upstream of pier)
arr_x0 = PIER_X - PIER_D * 2.8
arr_x1 = PIER_X - PIER_D * 1.3
ax.annotate(
    "", xy=(arr_x1, PIER_Y), xytext=(arr_x0, PIER_Y),
    arrowprops=dict(arrowstyle="-|>", color=c1, lw=2.5),
    zorder=5,
)
ax.text(
    (arr_x0 + arr_x1) / 2, PIER_Y + 1.0,
    f"V = {V_APPROACH:.0f} ft/s", ha="center", va="bottom",
    fontsize=9, color=c1, weight="bold",
)

ax.set_xlim(x_lo, x_hi)
ax.set_ylim(y_lo, y_hi)
ax.set_xlabel("x (ft)", color=fg, fontsize=10)
ax.set_ylabel("y (ft)", color=fg, fontsize=10)
ax.set_title("(a) Plan View", color=fg, fontsize=11, weight="bold")
ax.set_aspect("equal")
ax.tick_params(colors=fg, labelsize=8)
ax.grid(True, color=gc, alpha=ga, linewidth=0.5)
for spine in ax.spines.values():
    spine.set_color(fg)

# Annotate downstream wake
wake_in_view = vp[:, 0] > PIER_X + PIER_D * 2
if wake_in_view.sum() > 3:
    wcx = np.median(vp[wake_in_view, 0])
    wcy = np.median(vp[wake_in_view, 1])
    ax.annotate(
        "Downstream wake",
        xy=(wcx, wcy),
        xytext=(wcx, wcy + PIER_D * 1.8),
        fontsize=8, color=fg, ha="center",
        arrowprops=dict(arrowstyle="->", color=fg, lw=0.8),
        zorder=5,
    )

# ── RIGHT PANEL: Cross-section (x-z) ─────────────────────────────────
ax = ax_xsec
ax.set_facecolor(bg)

# Very faint blue water fill
ax.axhspan(0, DEPTH, color="#f2f6fc", zorder=0)

# Scatter particles colored by vorticity magnitude
sc_xsec = ax.scatter(
    vp[:, 0], vp[:, 2],
    c=vm, cmap=cmap, norm=norm,
    s=s_arr, alpha=0.80, edgecolors="#444444", linewidths=0.3, zorder=3,
)

# Pier as gray rectangle from bed to surface
pier_rect = Rectangle(
    (PIER_X - PIER_D / 2, 0), PIER_D, DEPTH,
    facecolor="#b0b0b0", edgecolor="#333333", linewidth=1.8, zorder=4,
)
ax.add_patch(pier_rect)

# Bed line with fill
ax.axhline(0, color="#8B7355", linewidth=2.0, zorder=2)
ax.fill_between(
    [x_lo, x_hi], -1.0, 0,
    color="#c4a87a", alpha=0.45, zorder=1,
)

# Water surface line
ax.axhline(DEPTH, color="#4488cc", linewidth=1.5, linestyle="--", zorder=2)
ax.text(x_hi - 0.5, DEPTH + 0.2, "W.S.", fontsize=8, color="#4488cc",
        ha="right", va="bottom")

ax.set_xlim(x_lo, x_hi)
ax.set_ylim(-1.0, DEPTH + 1.2)
ax.set_xlabel("x (ft)", color=fg, fontsize=10)
ax.set_ylabel("z (ft)", color=fg, fontsize=10)
ax.set_title("(b) Cross-Section (x-z)", color=fg, fontsize=11, weight="bold")
ax.tick_params(colors=fg, labelsize=8)
ax.grid(True, color=gc, alpha=ga, linewidth=0.5)
for spine in ax.spines.values():
    spine.set_color(fg)

# Annotate horseshoe vortex region near bed
bed_particles_v = vp[bm]
if len(bed_particles_v) > 0:
    bcx = np.median(bed_particles_v[:, 0])
    bcz = np.median(bed_particles_v[:, 2])
    # Place annotation text above and to the right
    tx = min(bcx + PIER_D * 2.0, x_hi - 4)
    tz = bcz + 3.0
    ax.annotate(
        "Horseshoe vortex\n(scour driver)",
        xy=(bcx, bcz),
        xytext=(tx, tz),
        fontsize=8.5, color=c2, weight="bold", ha="center",
        arrowprops=dict(arrowstyle="->", color=c2, lw=1.2),
        zorder=5,
    )

# Annotate surface separation particles
surf_particles_v = vp[sm]
if len(surf_particles_v) > 0:
    scx = np.median(surf_particles_v[:, 0])
    scz = np.median(surf_particles_v[:, 2])
    tx2 = max(scx - PIER_D * 2.0, x_lo + 4)
    tz2 = min(scz + 2.0, DEPTH - 0.5)
    ax.annotate(
        "Surface separation\nparticles",
        xy=(scx, scz),
        xytext=(tx2, tz2),
        fontsize=8.5, color=c1, weight="bold", ha="center",
        arrowprops=dict(arrowstyle="->", color=c1, lw=1.2),
        zorder=5,
    )

# Bed label
ax.text(x_lo + 0.8, -0.55, "Channel Bed", fontsize=8, color="#654321",
        va="center", ha="left")

# ── Shared colorbar ──────────────────────────────────────────────────
cbar = fig.colorbar(sc_xsec, ax=[ax_plan, ax_xsec], shrink=0.45, pad=0.03,
                     aspect=20, location="right")
cbar.set_label("Vorticity Magnitude (1/s)", color=fg, fontsize=9)
cbar.ax.tick_params(colors=fg, labelsize=8)

# ── Suptitle ─────────────────────────────────────────────────────────
fig.suptitle(
    "Vortex Particle Distribution at Bridge Pier",
    color=fg, fontsize=14, weight="bold", y=0.97,
)

# ── Save ─────────────────────────────────────────────────────────────
output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "Scour_Benchmark_figures")
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "fig5_vortex_particles.png")

fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor=bg)
plt.close(fig)

print(f"\nFigure saved: {output_path}")
