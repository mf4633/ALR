"""
Conceptual Demo - Quantum-inspired lazy evaluation demonstration.

This demo illustrates the philosophical concepts:
1. Probabilistic representation (like quantum superposition)
2. Lazy evaluation (only compute details where "observed")
3. Emergent complexity from simple statistical rules
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle
from typing import List, Tuple, Optional


class ConceptualVortex:
    """
    A vortex represented probabilistically.

    Exists in 'superposition' until observed (rendered at high resolution).
    """

    def __init__(self, x: float, y: float, strength: float, size: float):
        self.x = x
        self.y = y
        self.strength = strength
        self.size = size
        self.phase = np.random.uniform(0, 2 * np.pi)

    def influence(self, xi: float, yi: float, observation_level: float = 1.0) -> Tuple[float, float]:
        """
        Calculate velocity influence at a point.

        Higher observation_level = more detailed computation.
        """
        dx = xi - self.x
        dy = yi - self.y
        r = np.sqrt(dx ** 2 + dy ** 2)

        if r < 0.01:
            return 0.0, 0.0

        base_strength = self.strength / (r + self.size)

        # Add detail based on observation level
        if observation_level > 1:
            turbulent_factor = 1 + 0.3 * np.sin(5 * r + self.phase) * observation_level
            base_strength *= turbulent_factor

        if observation_level > 2:
            sub_vortex = 0.1 * np.sin(20 * r + 2 * self.phase) * np.cos(10 * np.arctan2(dy, dx))
            base_strength *= (1 + sub_vortex)

        vx = -base_strength * dy / r
        vy = base_strength * dx / r

        return vx, vy

    def evolve(self, dt: float, all_vortices: List["ConceptualVortex"]):
        """Evolve based on interactions."""
        vx_total, vy_total = 0.0, 0.0

        for other in all_vortices:
            if other is not self:
                vx, vy = other.influence(self.x, self.y, observation_level=1)
                vx_total += vx
                vy_total += vy

        vx_total += np.random.normal(0, 0.02 * dt)
        vy_total += np.random.normal(0, 0.02 * dt)

        self.x = (self.x + vx_total * dt) % 10
        self.y = (self.y + vy_total * dt) % 10
        self.phase += 0.5 * dt


class ProbabilisticField:
    """
    Flow field represented probabilistically.

    Demonstrates lazy evaluation - detail computed only where observed.
    """

    def __init__(self, domain_size: float = 10.0, n_vortices: int = 12):
        self.domain_size = domain_size
        self.vortices: List[ConceptualVortex] = []

        for _ in range(n_vortices):
            x = np.random.uniform(0, domain_size)
            y = np.random.uniform(0, domain_size)
            strength = np.random.uniform(-2, 2)
            size = np.random.uniform(0.3, 0.8)
            self.vortices.append(ConceptualVortex(x, y, strength, size))

        self.observation_zones: List[Tuple[float, float, float]] = []

    def add_observation(self, x: float, y: float, radius: float = 2.0):
        """Add observation zone - forces higher detail computation."""
        self.observation_zones.append((x, y, radius))

    def get_observation_level(self, x: float, y: float) -> float:
        """Determine detail level based on observation proximity."""
        base_level = 1.0

        for ox, oy, radius in self.observation_zones:
            dist = np.sqrt((x - ox) ** 2 + (y - oy) ** 2)
            if dist < radius:
                proximity = 1 - (dist / radius)
                level = 1 + 2 * proximity
                base_level = max(base_level, level)

        return base_level

    def render(self, resolution: int = 40) -> Tuple[np.ndarray, ...]:
        """Render velocity field - 'collapse' the probabilistic state."""
        x = np.linspace(0, self.domain_size, resolution)
        y = np.linspace(0, self.domain_size, resolution)
        X, Y = np.meshgrid(x, y)

        U = np.zeros_like(X)
        V = np.zeros_like(Y)
        detail_map = np.zeros_like(X)

        for i in range(resolution):
            for j in range(resolution):
                xi, yi = X[i, j], Y[i, j]
                obs_level = self.get_observation_level(xi, yi)
                detail_map[i, j] = obs_level

                for vortex in self.vortices:
                    vx, vy = vortex.influence(xi, yi, observation_level=obs_level)
                    U[i, j] += vx
                    V[i, j] += vy

        return X, Y, U, V, detail_map

    def evolve(self, dt: float = 0.1):
        """Time evolution."""
        for vortex in self.vortices:
            vortex.evolve(dt, self.vortices)


def run_conceptual_demo():
    """Main demonstration of quantum-inspired concepts."""
    print("=" * 70)
    print("QUANTUM-INSPIRED FLOW DEMONSTRATION")
    print("=" * 70)
    print("\nKey Concepts:")
    print("1. PROBABILISTIC REPRESENTATION: Flow as vortex 'entities'")
    print("2. LAZY EVALUATION: Detail computed only where observed")
    print("3. EMERGENCE: Complex behavior from simple rules")
    print("4. OBSERVATION AFFECTS REALITY: Adding observation zones")
    print("   forces the system to compute more detail")
    print("=" * 70)

    # Scenario 1: Unobserved
    print("\n[1/3] Unobserved flow...")
    flow1 = ProbabilisticField()
    for _ in range(5):
        flow1.evolve(0.1)

    # Scenario 2: Single observation
    print("[2/3] Single observation zone...")
    flow2 = ProbabilisticField()
    flow2.add_observation(5, 5, radius=2.5)
    for _ in range(5):
        flow2.evolve(0.1)

    # Scenario 3: Multiple observations
    print("[3/3] Multiple observation zones...")
    flow3 = ProbabilisticField()
    flow3.add_observation(3, 3, radius=2.0)
    flow3.add_observation(7, 7, radius=2.0)
    for _ in range(5):
        flow3.evolve(0.1)

    # Create visualization
    fig, axes = plt.subplots(3, 2, figsize=(14, 15))
    fig.patch.set_facecolor("#0a0a0a")

    for idx, (flow, title) in enumerate([
        (flow1, "Unobserved (Superposition)"),
        (flow2, "Single Observation"),
        (flow3, "Multiple Observations"),
    ]):
        X, Y, U, V, detail_map = flow.render()

        # Flow plot
        ax1 = axes[idx, 0]
        ax1.set_facecolor("#0a0a0a")
        speed = np.sqrt(U ** 2 + V ** 2)
        ax1.streamplot(X, Y, U, V, color=speed, cmap="plasma", linewidth=1.5, density=1.5)

        for vortex in flow.vortices:
            color = "cyan" if vortex.strength > 0 else "red"
            circle = Circle((vortex.x, vortex.y), vortex.size, fill=False, edgecolor=color, linewidth=2, alpha=0.6)
            ax1.add_patch(circle)

        for ox, oy, radius in flow.observation_zones:
            obs = Circle((ox, oy), radius, fill=False, edgecolor="yellow", linewidth=3, linestyle="--")
            ax1.add_patch(obs)

        ax1.set_xlim(0, 10)
        ax1.set_ylim(0, 10)
        ax1.set_aspect("equal")
        ax1.set_title(f"Flow Field: {title}", color="white", fontsize=12)
        ax1.tick_params(colors="white")

        # Detail map
        ax2 = axes[idx, 1]
        ax2.set_facecolor("#0a0a0a")
        im = ax2.contourf(X, Y, detail_map, levels=20, cmap="viridis")
        plt.colorbar(im, ax=ax2, label="Computation Detail")

        for ox, oy, radius in flow.observation_zones:
            obs = Circle((ox, oy), radius, fill=False, edgecolor="yellow", linewidth=3, linestyle="--")
            ax2.add_patch(obs)

        ax2.set_xlim(0, 10)
        ax2.set_ylim(0, 10)
        ax2.set_aspect("equal")
        ax2.set_title("Computational Effort Map", color="white", fontsize=12)
        ax2.tick_params(colors="white")

    fig.suptitle(
        "Quantum-Inspired Flow: Observation Determines Computation\n"
        "(Bright = High Detail/Observed, Dark = Low Detail/Superposition)",
        color="white",
        fontsize=14,
        y=0.98,
    )

    plt.tight_layout()
    plt.subplots_adjust(top=0.92)

    print("\n" + "=" * 70)
    print("INTERPRETATION:")
    print("=" * 70)
    print("Notice how the SAME physical flow requires DIFFERENT amounts")
    print("of computation depending on where we 'observe' it.")
    print("\nThis demonstrates the key insight:")
    print("- Reality might not compute full detail everywhere")
    print("- Observation forces specifics to be 'rendered'")
    print("- This is legitimate physics applied to turbulence closure")
    print("=" * 70)

    plt.show()


def main():
    """Main entry point."""
    run_conceptual_demo()


if __name__ == "__main__":
    main()
