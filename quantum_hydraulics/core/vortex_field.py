"""
VortexParticleField - 3D vortex particle system with adaptive resolution.

Implements the regularized Biot-Savart law for velocity induction and a
relaxation-based approximation of viscous diffusion (see apply_diffusion).

OPTIMIZED VERSION: Uses Structure-of-Arrays for 5-10x speedup.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from collections import deque

try:
    from scipy.spatial import cKDTree
except ImportError:
    cKDTree = None

try:
    from numba import njit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False

from quantum_hydraulics.core.particle import VortexParticle
from quantum_hydraulics.core.hydraulics import HydraulicsEngine


@dataclass
class FieldState:
    """
    Dataclass for passing field state between physics and rendering.

    This provides a clean interface for visualization without coupling
    to the internal particle representation.
    """

    positions: np.ndarray  # Shape (N, 3)
    vorticities: np.ndarray  # Shape (N, 3)
    energies: np.ndarray  # Shape (N,)
    sigmas: np.ndarray  # Shape (N,)
    ages: np.ndarray  # Shape (N,)
    obs_center: np.ndarray  # Shape (3,)
    obs_radius: float
    observation_active: bool
    domain_length: float
    domain_width: float
    domain_depth: float
    trails: List[np.ndarray] = field(default_factory=list)

    @property
    def n_particles(self) -> int:
        return len(self.positions)


# Numba-accelerated kernels (if available)
if HAS_NUMBA:
    @njit(parallel=True, fastmath=True)
    def _compute_velocity_induction_numba(
        positions: np.ndarray,
        vorticities: np.ndarray,
        sigmas: np.ndarray,
        cutoff_multiplier: float = 6.0
    ) -> np.ndarray:
        """
        Vectorized Biot-Savart velocity induction with Numba acceleration.

        Parameters
        ----------
        positions : np.ndarray
            Particle positions, shape (N, 3)
        vorticities : np.ndarray
            Particle vorticities, shape (N, 3)
        sigmas : np.ndarray
            Particle core sizes, shape (N,)
        cutoff_multiplier : float
            Cutoff distance as multiple of sigma

        Returns
        -------
        np.ndarray
            Induced velocities, shape (N, 3)
        """
        n = positions.shape[0]
        velocities = np.zeros((n, 3), dtype=np.float64)

        for i in prange(n):
            sigma_i_sq = sigmas[i] ** 2

            for j in range(n):
                if i == j:
                    continue

                # Displacement vector
                rx = positions[j, 0] - positions[i, 0]
                ry = positions[j, 1] - positions[i, 1]
                rz = positions[j, 2] - positions[i, 2]
                r_sq = rx * rx + ry * ry + rz * rz

                sigma_j_sq = sigmas[j] ** 2

                # Symmetrized core size: sigma_ij^2 = sigma_i^2 + sigma_j^2
                # (Barba et al. 2005, variable-blob Biot-Savart)
                sigma_ij_sq = sigma_i_sq + sigma_j_sq

                # Cutoff at cutoff_multiplier * symmetrized core (consistent with
                # the kernel width used below)
                cutoff_sq = cutoff_multiplier ** 2 * sigma_ij_sq
                if r_sq > cutoff_sq:
                    continue

                # Low-order algebraic regularized Biot-Savart kernel:
                #   K = 1 / (4 pi (r^2 + sigma_ij^2)^{3/2})
                # This is the standard second-order algebraic (Rosenhead-Moore)
                # smoothing; it is finite at r -> 0 and needs no extra cutoff
                # function.  (The previous code multiplied this by a Gaussian
                # 1 - exp(-r^2/sigma^2) factor, which does not correspond to any
                # published blob and double-regularized the kernel.)
                denom = (r_sq + sigma_ij_sq) ** 1.5 + 1e-12
                K = 1.0 / (4.0 * np.pi * denom)

                # Cross product: omega x r
                ox, oy, oz = vorticities[j, 0], vorticities[j, 1], vorticities[j, 2]
                cross_x = oy * rz - oz * ry
                cross_y = oz * rx - ox * rz
                cross_z = ox * ry - oy * rx

                velocities[i, 0] += K * cross_x
                velocities[i, 1] += K * cross_y
                velocities[i, 2] += K * cross_z

        return velocities

    @njit(parallel=True, fastmath=True)
    def _apply_diffusion_numba(
        positions: np.ndarray,
        vorticities: np.ndarray,
        sigmas: np.ndarray,
        nu: float,
        dt: float,
        search_multiplier: float = 4.0
    ) -> np.ndarray:
        """
        Relaxation-based viscous diffusion with Numba acceleration.

        Approximates d(omega)/dt = nu * laplacian(omega) by relaxing each
        particle's vorticity toward the Gaussian-weighted local mean over a
        timestep dt.  This is a smoothing approximation (it is not the strictly
        conservative Particle Strength Exchange scheme); with molecular nu the
        per-step change is negligible, as intended for the resolved scales here.

        Parameters
        ----------
        positions : np.ndarray
            Particle positions, shape (N, 3)
        vorticities : np.ndarray
            Particle vorticities (will be modified in place), shape (N, 3)
        sigmas : np.ndarray
            Particle core sizes, shape (N,)
        nu : float
            Kinematic viscosity
        dt : float
            Timestep in seconds
        search_multiplier : float
            Search radius as multiple of sigma

        Returns
        -------
        np.ndarray
            Updated vorticities, shape (N, 3)
        """
        n = positions.shape[0]
        new_vorticities = np.zeros((n, 3), dtype=np.float64)

        for i in prange(n):
            sigma_i_sq = sigmas[i] ** 2
            search_sq = (search_multiplier * sigmas[i]) ** 2

            # Accumulate weighted vorticity
            weight_sum = 0.0
            omega_avg = np.zeros(3, dtype=np.float64)

            for j in range(n):
                rx = positions[j, 0] - positions[i, 0]
                ry = positions[j, 1] - positions[i, 1]
                rz = positions[j, 2] - positions[i, 2]
                r_sq = rx * rx + ry * ry + rz * rz

                if r_sq > search_sq:
                    continue

                # Symmetrized PSE kernel: use average sigma
                sigma_avg_sq = 0.5 * (sigma_i_sq + sigmas[j] ** 2)
                weight = np.exp(-r_sq / (2.0 * sigma_avg_sq))
                weight_sum += weight
                omega_avg[0] += weight * vorticities[j, 0]
                omega_avg[1] += weight * vorticities[j, 1]
                omega_avg[2] += weight * vorticities[j, 2]

            if weight_sum > 1e-10:
                omega_avg[0] /= weight_sum
                omega_avg[1] /= weight_sum
                omega_avg[2] /= weight_sum

                # Time-consistent relaxation fraction (dimensionless), clamped
                # to <= 1 for stability.  Units: (1/s) * s = dimensionless.
                diffusion_rate = 2.0 * nu * dt / (sigma_i_sq + 1e-12)
                if diffusion_rate > 1.0:
                    diffusion_rate = 1.0

                new_vorticities[i, 0] = vorticities[i, 0] + diffusion_rate * (omega_avg[0] - vorticities[i, 0])
                new_vorticities[i, 1] = vorticities[i, 1] + diffusion_rate * (omega_avg[1] - vorticities[i, 1])
                new_vorticities[i, 2] = vorticities[i, 2] + diffusion_rate * (omega_avg[2] - vorticities[i, 2])
            else:
                new_vorticities[i] = vorticities[i]

        return new_vorticities


def _compute_velocity_induction_numpy(
    positions: np.ndarray,
    vorticities: np.ndarray,
    sigmas: np.ndarray,
    spatial_tree: Optional["cKDTree"] = None,
    cutoff_multiplier: float = 6.0
) -> np.ndarray:
    """
    Vectorized Biot-Savart velocity induction using NumPy.

    Falls back to this when Numba is not available.
    """
    n = positions.shape[0]
    if n == 0:
        return np.zeros((0, 3))

    velocities = np.zeros((n, 3), dtype=np.float64)

    # Build tree if not provided and we have enough particles
    if spatial_tree is None and cKDTree is not None and n > 50:
        spatial_tree = cKDTree(positions)

    # Global cutoff radius (conservative: use max sigma)
    max_cutoff = cutoff_multiplier * sigmas.max() if len(sigmas) > 0 else 0

    for i in range(n):
        # Use spatial tree for neighbor search if available
        if spatial_tree is not None:
            cutoff = cutoff_multiplier * sigmas[i]
            neighbor_indices = spatial_tree.query_ball_point(positions[i], cutoff)
        else:
            neighbor_indices = list(range(n))

        if len(neighbor_indices) < 2:
            continue

        # Vectorized computation over neighbors
        neighbor_indices = np.array(neighbor_indices, dtype=np.int64)
        pos_neighbors = positions[neighbor_indices]
        omega_neighbors = vorticities[neighbor_indices]
        sigma_neighbors = sigmas[neighbor_indices]

        r_vecs = pos_neighbors - positions[i]
        r_squared = np.sum(r_vecs ** 2, axis=1, keepdims=True)

        # Symmetrized sigma: sigma_ij^2 = sigma_i^2 + sigma_j^2
        sigma_ij_sq = sigmas[i] ** 2 + sigma_neighbors ** 2

        # Low-order algebraic regularized Biot-Savart kernel (Rosenhead-Moore):
        #   K = 1 / (4 pi (r^2 + sigma_ij^2)^{3/2})
        denominator = (r_squared + sigma_ij_sq[:, np.newaxis]) ** 1.5 + 1e-12
        K = 1.0 / (4 * np.pi * denominator)

        # Cross product: omega x r
        cross_products = np.cross(omega_neighbors, r_vecs)

        velocities[i] = np.sum(K * cross_products, axis=0)

    return velocities


def _apply_diffusion_numpy(
    positions: np.ndarray,
    vorticities: np.ndarray,
    sigmas: np.ndarray,
    nu: float,
    dt: float,
    spatial_tree: Optional["cKDTree"] = None,
    search_multiplier: float = 4.0
) -> np.ndarray:
    """
    Relaxation-based viscous diffusion using NumPy.

    Falls back to this when Numba is not available.  See the Numba variant for
    the approximation used (relaxation toward the local Gaussian-weighted mean
    over a timestep dt; not strictly conservative PSE).
    """
    n = positions.shape[0]
    if n == 0:
        return vorticities.copy()

    new_vorticities = vorticities.copy()

    for i in range(n):
        search_radius = search_multiplier * sigmas[i]

        if spatial_tree is not None:
            neighbor_indices = spatial_tree.query_ball_point(positions[i], search_radius)
        else:
            neighbor_indices = list(range(n))

        if len(neighbor_indices) < 2:
            continue

        neighbor_indices = np.array(neighbor_indices, dtype=np.int64)
        neighbor_pos = positions[neighbor_indices]
        neighbor_omega = vorticities[neighbor_indices]

        dx = neighbor_pos - positions[i]
        r_squared = np.sum(dx ** 2, axis=1)

        # Symmetrized PSE diffusion kernel
        sigma_avg_sq = 0.5 * (sigmas[i] ** 2 + sigmas[neighbor_indices] ** 2)
        weights = np.exp(-r_squared / (2 * sigma_avg_sq))
        weight_sum = weights.sum()

        if weight_sum < 1e-10:
            continue

        weights /= weight_sum
        omega_avg = np.average(neighbor_omega, axis=0, weights=weights)

        # Time-consistent relaxation fraction (dimensionless), clamped to <= 1.
        diffusion_rate = min(1.0, 2.0 * nu * dt / (sigmas[i] ** 2 + 1e-12))
        new_vorticities[i] += diffusion_rate * (omega_avg - vorticities[i])

    return new_vorticities


class VortexParticleField:
    """
    3D vortex particle system with observation-dependent resolution.

    Implements Biot-Savart law for velocity induction and
    particle strength exchange for viscous diffusion.

    OPTIMIZED: Uses Structure-of-Arrays for 5-10x faster computation.

    Parameters
    ----------
    hydraulics : HydraulicsEngine
        Hydraulics engine providing flow parameters
    length : float, optional
        Domain length in feet (default 200.0)
    n_particles : int, optional
        Target number of particles (default 6000)

    Attributes
    ----------
    _positions : np.ndarray
        Particle positions, shape (N, 3)
    _vorticities : np.ndarray
        Particle vorticities, shape (N, 3)
    _sigmas : np.ndarray
        Particle core sizes, shape (N,)
    _ages : np.ndarray
        Particle ages, shape (N,)
    obs_center : np.ndarray
        Center of observation zone [x, y, z]
    obs_radius : float
        Radius of observation zone in feet
    observation_active : bool
        Whether observation-dependent resolution is active
    """

    def __init__(
        self,
        hydraulics: HydraulicsEngine,
        length: float = 200.0,
        n_particles: int = 6000,
        seeding: str = "random",
    ):
        self.hydraulics = hydraulics
        # Vorticity seeding mode: "random" (isotropic turbulence proxy, the
        # historical default) or "coherent" (deterministic mean-shear vorticity
        # omega = du/dz, which has a converged limit -- see _seed_coherent and
        # docs/ALR_REFINEMENT_DESIGN.md).
        self.seeding = seeding
        self.L = length
        self.W = hydraulics.width
        self.H = hydraulics.depth

        # Observation zone (location of interest for resolution concentration)
        self.obs_center = np.array([length / 2, hydraulics.width / 2, hydraulics.depth / 2])
        self.obs_radius = 25.0
        self.observation_active = True
        # Multi-zone support: list of (center_array, radius) tuples
        self.obs_zones: Optional[list] = None
        # Optional pier bodies for vortex shedding
        self.pier_bodies: Optional[list] = None

        # Adaptive Lagrangian refinement (flaw #2). When enabled, particles in
        # observation zones whose overlap ratio h/sigma exceeds ``refine_split_ratio``
        # are split conservatively (adding degrees of freedom), and over-dense
        # particles below ``refine_merge_ratio`` are merged, capped at
        # ``refine_n_max``. OFF by default -- see docs/ALR_REFINEMENT_DESIGN.md.
        self.enable_refinement = False
        self.refine_split_ratio = 1.4
        self.refine_merge_ratio = 0.4
        self.refine_stencil_frac = 0.7
        self.refine_n_max: Optional[int] = None

        # Structure-of-Arrays particle storage (OPTIMIZED)
        self.n_particles = n_particles
        self._positions: np.ndarray = np.zeros((0, 3), dtype=np.float64)
        self._vorticities: np.ndarray = np.zeros((0, 3), dtype=np.float64)
        self._sigmas: np.ndarray = np.zeros(0, dtype=np.float64)
        self._ages: np.ndarray = np.zeros(0, dtype=np.float64)
        # Per-particle volume element (ft^3); strength_j = omega_j * Vol_j.
        self._volumes: np.ndarray = np.zeros(0, dtype=np.float64)

        # Core size parameters
        self.base_sigma = self.H / 5.0
        self.min_sigma = self.hydraulics.eta_kolmogorov * 3
        self.max_sigma = self.base_sigma * 2

        # Visualization trails
        self.trails: deque = deque(maxlen=40)
        self._trail_frequency = 0

        # Spatial acceleration
        self._spatial_tree: Optional[cKDTree] = None
        self._tree_build_counter = 0

        # Pre-computed velocity profile lookup table
        self._velocity_lut_z: Optional[np.ndarray] = None
        self._velocity_lut_v: Optional[np.ndarray] = None
        self._build_velocity_lut()

        # Initialize particles
        self._seed_field()

        # Volume element carried by each particle.  A vortex particle represents
        # the vorticity integrated over a finite region: strength = omega * Vol
        # (units of circulation * length), which is the ``dV`` in the Biot-Savart
        # integral.  Without it the discretized sum carries the wrong units and
        # its magnitude grows like sqrt(N) with particle count.
        #
        # Volume is stored *per particle* so that refinement (splitting particles
        # in observation zones) can subdivide a parent's volume among its
        # children while conserving total strength.  At seeding the distribution
        # is uniform, so every particle gets the mean domain volume per particle;
        # this reproduces the earlier scalar-volume behaviour exactly (the kernel
        # is linear in the source strength, so a uniform factor pulls out).
        #
        # NOTE: the volume weight corrects units/discretization but does NOT by
        # itself make the induced field converge with N.  The seeded vorticity is
        # a random-phase turbulence proxy, so the net induced velocity is a
        # random walk (~1/sqrt(N) once volume-weighted).  A discretization-
        # independent field also needs a coherent (resolved) vorticity
        # distribution and true refinement -- see docs/ALR_REFINEMENT_DESIGN.md.
        self._ref_volume = self._reference_particle_volume()
        self._volumes = np.full(len(self._positions), self._ref_volume,
                                dtype=np.float64)

    def _reference_particle_volume(self) -> float:
        """Mean domain volume per particle (ft^3); the seed-time volume weight."""
        n = max(len(self._positions), 1)
        return (self.L * self.W * self.H) / n

    @property
    def particle_volume(self) -> float:
        """Mean per-particle volume (ft^3).

        Backwards-compatible scalar view of ``_volumes``; equals the uniform
        seed volume until refinement makes volumes non-uniform.
        """
        if len(self._volumes) == 0:
            return self._reference_particle_volume()
        return float(self._volumes.mean())

    def _build_velocity_lut(self, n_points: int = 200):
        """Pre-compute velocity profile lookup table for fast interpolation."""
        self._velocity_lut_z = np.linspace(0, self.H, n_points)
        self._velocity_lut_v = np.array([
            self.hydraulics.velocity_profile(z) for z in self._velocity_lut_z
        ])

    def _velocity_profile_fast(self, z: np.ndarray) -> np.ndarray:
        """Fast velocity profile lookup using pre-computed table."""
        if self._velocity_lut_z is None:
            self._build_velocity_lut()
        return np.interp(z, self._velocity_lut_z, self._velocity_lut_v)

    def _seed_field(self, seed: int = 42):
        """Seed the vorticity field according to ``self.seeding``."""
        if self.seeding == "coherent":
            self._seed_coherent(seed)
        else:
            self._seed_particles(seed)

    def _seed_coherent(self, seed: int = 42):
        """
        Seed a coherent (resolved) mean-shear vorticity field.

        The mean streamwise flow ``u(z)`` has vorticity ``omega = curl(u) =
        (0, du/dz, 0)`` -- a deterministic, spanwise vorticity sheet rather than
        isotropic random turbulence.  Because the vorticity is a fixed function
        of position (not zero-mean noise), the discrete Biot-Savart sum has a
        non-trivial converged limit: the ensemble-mean induced velocity settles
        on a stable value with signal-to-noise growing like sqrt(N), whereas the
        random-phase seed averages to zero.  This is the seeding required before
        refinement can be validated as convergent (see
        docs/ALR_REFINEMENT_DESIGN.md).

        Positions are drawn uniformly in the interior; only the vorticity differs
        from the random seed.
        """
        rng = np.random.default_rng(seed)
        n = self.n_particles

        positions = np.column_stack([
            rng.uniform(0.1 * self.L, 0.9 * self.L, n),
            rng.uniform(0.1 * self.W, 0.9 * self.W, n),
            rng.uniform(0.1 * self.H, 0.9 * self.H, n),
        ])

        # Mean-shear vorticity omega_y = du/dz via central difference on the
        # (log-law + power-law) velocity profile.
        dz = 0.01 * self.H
        zc = positions[:, 2]
        u_plus = self.hydraulics.velocity_profile_vectorized(zc + dz)
        u_minus = self.hydraulics.velocity_profile_vectorized(
            np.maximum(zc - dz, 1e-6))
        dudz = (u_plus - u_minus) / (2.0 * dz)

        vorticities = np.zeros((n, 3), dtype=np.float64)
        vorticities[:, 1] = dudz

        self._positions = positions
        self._vorticities = vorticities
        self._sigmas = self._get_adaptive_core_sizes_batch(positions)
        self._ages = np.zeros(n, dtype=np.float64)

    def _seed_particles(self, seed: int = 42):
        """
        Seed particles representing turbulent vorticity field.

        Uses Kolmogorov cascade theory to distribute across scales.
        """
        rng = np.random.default_rng(seed)

        # Scale hierarchy: energy cascade
        scales = [self.H, self.H / 2, self.H / 4, self.H / 8]
        weights = [0.15, 0.25, 0.35, 0.25]

        # Pre-allocate arrays
        total_particles = self.n_particles
        positions = np.zeros((total_particles, 3), dtype=np.float64)
        vorticities = np.zeros((total_particles, 3), dtype=np.float64)
        sigmas = np.zeros(total_particles, dtype=np.float64)

        idx = 0
        for scale, weight in zip(scales, weights):
            n = int(total_particles * weight)
            end_idx = min(idx + n, total_particles)
            actual_n = end_idx - idx

            # Random positions in channel (avoid boundaries)
            positions[idx:end_idx, 0] = rng.uniform(0.1 * self.L, 0.9 * self.L, actual_n)
            positions[idx:end_idx, 1] = rng.uniform(0.1 * self.W, 0.9 * self.W, actual_n)
            positions[idx:end_idx, 2] = rng.uniform(0.1 * self.H, 0.9 * self.H, actual_n)

            # Vorticity magnitude from Kolmogorov scaling
            if scale > 10 * self.hydraulics.eta_kolmogorov:
                omega_mag = self.hydraulics.V_mean / scale
            else:
                omega_mag = np.sqrt(self.hydraulics.epsilon / self.hydraulics.nu)

            # Random orientations (anisotropic for channel flow)
            theta = rng.uniform(0, 2 * np.pi, actual_n)
            phi = rng.uniform(0, np.pi, actual_n)

            vorticities[idx:end_idx, 0] = 0.3 * omega_mag * rng.normal(size=actual_n)
            vorticities[idx:end_idx, 1] = omega_mag * np.sin(phi) * np.sin(theta)
            vorticities[idx:end_idx, 2] = 0.5 * omega_mag * rng.normal(size=actual_n)

            idx = end_idx

        # Compute adaptive sigmas vectorized
        self._positions = positions[:idx]
        self._vorticities = vorticities[:idx]
        self._sigmas = self._get_adaptive_core_sizes_batch(positions[:idx])
        self._ages = np.zeros(idx, dtype=np.float64)

    def _get_adaptive_core_sizes_batch(self, positions: np.ndarray) -> np.ndarray:
        """
        Compute observation-dependent core sizes for batch of positions.

        Supports multiple observation zones via ``self.obs_zones``.  When set,
        the enhancement factor is the element-wise *maximum* across all zones
        (giving the smallest sigma — highest resolution — where any zone applies).

        Parameters
        ----------
        positions : np.ndarray
            Positions, shape (N, 3)

        Returns
        -------
        np.ndarray
            Adaptive core sizes, shape (N,)
        """
        if not self.observation_active:
            return np.full(len(positions), self.base_sigma, dtype=np.float64)

        # Build zone list: use multi-zone if set, else fall back to single zone
        zones = self.obs_zones if self.obs_zones is not None else [
            (self.obs_center, self.obs_radius)
        ]

        enhancement_factor = np.ones(len(positions), dtype=np.float64)
        for center, radius in zones:
            center = np.asarray(center, dtype=np.float64)
            dx = positions[:, 0] - center[0]
            dy = positions[:, 1] - center[1]
            dz = (positions[:, 2] - center[2]) * 0.5  # Weight vertical less
            dist = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
            zone_enhancement = 1.0 + 4.0 * np.exp(-(dist / radius) ** 2)
            np.maximum(enhancement_factor, zone_enhancement, out=enhancement_factor)

        # Smaller sigma = higher resolution
        sigma_adaptive = self.base_sigma / enhancement_factor

        return np.clip(sigma_adaptive, self.min_sigma, self.max_sigma)

    def get_adaptive_core_size(self, position: np.ndarray) -> float:
        """
        Compute observation-dependent core size.

        Near observation zone: Small sigma -> high resolution
        Far from observation: Large sigma -> coarse approximation

        Parameters
        ----------
        position : np.ndarray
            3D position [x, y, z]

        Returns
        -------
        float
            Adaptive core size sigma
        """
        if not self.observation_active:
            return self.base_sigma

        zones = self.obs_zones if self.obs_zones is not None else [
            (self.obs_center, self.obs_radius)
        ]

        best_enhancement = 1.0
        for center, radius in zones:
            center = np.asarray(center, dtype=np.float64)
            dx = position[0] - center[0]
            dy = position[1] - center[1]
            dz = (position[2] - center[2]) * 0.5
            dist = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
            enhancement = 1.0 + 4.0 * np.exp(-(dist / radius) ** 2)
            if enhancement > best_enhancement:
                best_enhancement = enhancement

        sigma_adaptive = self.base_sigma / best_enhancement

        return np.clip(sigma_adaptive, self.min_sigma, self.max_sigma)

    def compute_velocity_induction(self) -> np.ndarray:
        """
        Compute velocity induced on each particle by all others.

        Uses Biot-Savart law with regularized kernel:
        v = (1/4pi) * integral( (omega x r) / |r|^3 dV )

        The integral is discretized as a sum over particles, each contributing
        its strength omega_j * Vol_j.  The volume weight is folded into the source
        strength per particle, so non-uniform volumes (from refinement) are
        handled directly and, for uniform volumes, the result is identical to
        scaling by a single scalar.

        OPTIMIZED: Uses Numba JIT compilation when available.

        Returns
        -------
        np.ndarray
            Array of velocities, shape (N, 3)
        """
        n = len(self._positions)
        if n == 0:
            return np.zeros((0, 3))

        # Build spatial tree periodically for efficient neighbor search
        if cKDTree is not None and self._tree_build_counter % 10 == 0:
            self._spatial_tree = cKDTree(self._positions)
        self._tree_build_counter += 1

        # Source strength = vorticity * per-particle volume element.  The kernel
        # is linear in the source term, so weighting here applies Vol_j to each
        # source particle j (the target's own volume does not affect its induced
        # velocity).  Guard against hand-built state whose _volumes was not kept
        # in sync with the particle arrays by falling back to a uniform volume.
        vols = self._volumes
        if len(vols) != n:
            vols = np.full(n, self._ref_volume, dtype=np.float64)
        strengths = self._vorticities * vols[:, np.newaxis]

        # Use Numba-accelerated version if available
        if HAS_NUMBA and n > 100:
            velocities = _compute_velocity_induction_numba(
                self._positions, strengths, self._sigmas
            )
        else:
            velocities = _compute_velocity_induction_numpy(
                self._positions, strengths, self._sigmas, self._spatial_tree
            )

        return velocities

    def apply_diffusion(self, dt: float = 0.05):
        """
        Apply viscous diffusion over a timestep dt.

        Approximates d(omega)/dt = nu * nabla^2(omega) by relaxation toward the
        local Gaussian-weighted mean vorticity (see the kernel docstrings for the
        approximation and its limitations).

        OPTIMIZED: Uses Numba JIT compilation when available.

        Parameters
        ----------
        dt : float
            Timestep in seconds
        """
        n = len(self._positions)
        if n == 0:
            return

        if self._spatial_tree is None and cKDTree is not None:
            self._spatial_tree = cKDTree(self._positions)

        # Use Numba-accelerated version if available
        if HAS_NUMBA and n > 100:
            self._vorticities = _apply_diffusion_numba(
                self._positions, self._vorticities, self._sigmas,
                self.hydraulics.nu, dt
            )
        else:
            self._vorticities = _apply_diffusion_numpy(
                self._positions, self._vorticities, self._sigmas,
                self.hydraulics.nu, dt, self._spatial_tree
            )

    def step(self, dt: float = 0.05):
        """
        Advance particle system one timestep.

        OPTIMIZED: Single-pass update using vectorized operations.

        Parameters
        ----------
        dt : float
            Timestep in seconds
        """
        n = len(self._positions)
        if n == 0:
            return

        # Compute induced velocities
        velocities = self.compute_velocity_induction()

        # Add mean flow using fast lookup (vectorized)
        u_mean = self._velocity_profile_fast(self._positions[:, 2])
        velocities[:, 0] += u_mean

        # Advect particles (vectorized)
        self._positions += velocities * dt
        self._ages += dt

        # Apply boundary conditions (vectorized)
        self._positions[:, 0] = self._positions[:, 0] % self.L
        self._positions[:, 1] = np.clip(self._positions[:, 1], 0.5, self.W - 0.5)
        self._positions[:, 2] = np.clip(self._positions[:, 2], 0.1, self.H - 0.1)

        # Apply viscous diffusion
        self.apply_diffusion(dt)

        # Update core sizes based on observation (vectorized)
        self._sigmas = self._get_adaptive_core_sizes_batch(self._positions)

        # Adaptive Lagrangian refinement (opt-in; no-op when disabled)
        if self.enable_refinement:
            self.refine()

        # Pier vortex shedding (optional)
        if self.pier_bodies:
            V_approach = self.hydraulics.V_mean
            for pier in self.pier_bodies:
                # Reflect particles out of pier
                self._positions = pier.reflect_particles(self._positions)
                # Shed new particles
                result = pier.shed_particles(V_approach, self.H, dt)
                if result is not None:
                    new_pos, new_omega, new_sig = result
                    self._positions = np.vstack([self._positions, new_pos])
                    self._vorticities = np.vstack([self._vorticities, new_omega])
                    self._sigmas = np.concatenate([self._sigmas, new_sig])
                    self._ages = np.concatenate([
                        self._ages, np.zeros(len(new_pos))
                    ])
                    # Shed particles carry the reference (seed-time) volume.
                    self._volumes = np.concatenate([
                        self._volumes,
                        np.full(len(new_pos), self._ref_volume),
                    ])

        # Store trail for visualization
        self._trail_frequency += 1
        if self._trail_frequency % 3 == 0:
            # Get energies and find top particles
            energies = self._compute_energies()
            top_indices = np.argsort(energies)[-500:]
            trail_positions = self._positions[top_indices].copy()
            self.trails.append(trail_positions)

    def _compute_energies(self) -> np.ndarray:
        """Compute particle energies: |omega|^2 * sigma^3"""
        omega_mag_sq = np.sum(self._vorticities ** 2, axis=1)
        return omega_mag_sq * self._sigmas ** 3

    def overlap_ratio(self, mask: Optional[np.ndarray] = None) -> dict:
        """
        Diagnose the vortex-blob overlap condition h / sigma.

        Vortex particle methods are only accurate when neighbouring blobs
        overlap, i.e. the local inter-particle spacing ``h`` is at most of order
        the core size ``sigma`` (h / sigma <~ 1).  Reducing sigma in an
        observation zone *without* adding particles leaves ``h`` unchanged, so
        h / sigma rises and the field becomes under-resolved exactly where the
        highest resolution is advertised.  This method quantifies that: values
        well above 1 indicate the overlap condition is violated.

        Parameters
        ----------
        mask : np.ndarray, optional
            Boolean mask selecting a subset of particles (e.g. an observation
            zone).  If omitted, all particles are used.

        Returns
        -------
        dict
            ``{"mean", "median", "frac_gt_1", "n"}`` for h / sigma over the
            selected particles.  Empty-safe (returns NaNs / 0 for < 2 points).
        """
        pos = self._positions
        sig = self._sigmas
        if mask is not None:
            pos = pos[mask]
            sig = sig[mask]
        n = len(pos)
        if n < 2:
            return {"mean": float("nan"), "median": float("nan"),
                    "frac_gt_1": float("nan"), "n": n}

        if cKDTree is not None:
            tree = cKDTree(pos)
            # k=2: nearest neighbour excluding self
            dist, _ = tree.query(pos, k=2)
            h = dist[:, 1]
        else:
            # O(N^2) fallback
            h = np.empty(n)
            for i in range(n):
                d = np.linalg.norm(pos - pos[i], axis=1)
                d[i] = np.inf
                h[i] = d.min()

        ratio = h / np.maximum(sig, 1e-12)
        return {
            "mean": float(ratio.mean()),
            "median": float(np.median(ratio)),
            "frac_gt_1": float(np.mean(ratio > 1.0)),
            "n": int(n),
        }

    def observation_zone_mask(self, factor: float = 1.0) -> np.ndarray:
        """
        Boolean mask of particles inside the (single- or multi-zone) observation
        region, each zone taken as a sphere of ``factor * radius``.

        Returns an all-False mask when observation is inactive.
        """
        n = len(self._positions)
        if not self.observation_active or n == 0:
            return np.zeros(n, dtype=bool)

        zones = self.obs_zones if self.obs_zones is not None else [
            (self.obs_center, self.obs_radius)
        ]
        mask = np.zeros(n, dtype=bool)
        for center, radius in zones:
            center = np.asarray(center, dtype=np.float64)
            dist = np.linalg.norm(self._positions - center, axis=1)
            mask |= dist < (factor * radius)
        return mask

    # ------------------------------------------------------------------
    # Adaptive Lagrangian refinement (flaw #2) -- opt-in, default off
    # ------------------------------------------------------------------

    def _nearest_neighbor_spacing(self) -> np.ndarray:
        """Distance from each particle to its nearest neighbour, shape (N,)."""
        pos = self._positions
        n = len(pos)
        if n < 2:
            return np.full(n, np.inf)
        if cKDTree is not None:
            tree = cKDTree(pos)
            dist, _ = tree.query(pos, k=2)
            return dist[:, 1]
        h = np.empty(n)
        for i in range(n):
            d = np.linalg.norm(pos - pos[i], axis=1)
            d[i] = np.inf
            h[i] = d.min()
        return h

    @staticmethod
    def _octahedral_stencil() -> np.ndarray:
        """Unit octahedral split stencil (centre + 6 axis points), sum = 0."""
        return np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0], [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0], [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0], [0.0, 0.0, -1.0],
        ])

    def _split_particles(self, idx: np.ndarray) -> int:
        """
        Split the particles at ``idx`` into 7 children each (octahedral stencil).

        Conservation (exact, independent of stencil radius, because the stencil
        is centrally symmetric):
          - total volume: children share the parent volume (Vol/7 each);
          - zeroth strength moment: sum(omega*Vol) unchanged (omega inherited,
            volume split);
          - first strength moment: strength-weighted centroid = parent position.

        Returns the net change in particle count.
        """
        idx = np.asarray(idx, dtype=np.int64)
        if idx.size == 0:
            return 0

        M = 7
        stencil = self._octahedral_stencil()             # (7, 3)
        keep = np.ones(len(self._positions), dtype=bool)
        keep[idx] = False

        P = self._positions[idx]                          # (k, 3)
        O = self._vorticities[idx]
        S = self._sigmas[idx]
        V = self._volumes[idx]
        A = self._ages[idx]

        radii = self.refine_stencil_frac * S              # (k,)
        # children positions: P_i + radius_i * stencil_k
        child_pos = (P[:, None, :]
                     + radii[:, None, None] * stencil[None, :, :]).reshape(-1, 3)
        child_omega = np.repeat(O, M, axis=0)             # vorticity inherited
        child_sigma = np.repeat(S, M)                     # core size inherited
        child_vol = np.repeat(V / M, M)                   # volume split
        child_age = np.repeat(A, M)

        self._positions = np.vstack([self._positions[keep], child_pos])
        self._vorticities = np.vstack([self._vorticities[keep], child_omega])
        self._sigmas = np.concatenate([self._sigmas[keep], child_sigma])
        self._volumes = np.concatenate([self._volumes[keep], child_vol])
        self._ages = np.concatenate([self._ages[keep], child_age])

        return idx.size * (M - 1)

    def _merge_particles(self, pairs: np.ndarray) -> int:
        """
        Merge disjoint particle pairs, conserving total strength and volume.

        Each merged particle carries the summed volume and the summed vector
        strength (omega*Vol), so the total zeroth moment sum(omega*Vol) and the
        total volume are exactly preserved; it is placed at the
        strength-magnitude-weighted centroid (first moment approximate).

        ``pairs`` is an (m, 2) int array of disjoint indices. Returns the net
        change in particle count (negative).
        """
        pairs = np.asarray(pairs, dtype=np.int64)
        if pairs.size == 0:
            return 0

        i, j = pairs[:, 0], pairs[:, 1]
        Vi, Vj = self._volumes[i], self._volumes[j]
        Oi, Oj = self._vorticities[i], self._vorticities[j]
        # Vector strengths alpha = omega * Vol
        Ai = Oi * Vi[:, None]
        Aj = Oj * Vj[:, None]
        V_new = Vi + Vj
        A_new = Ai + Aj
        O_new = A_new / V_new[:, None]                    # conserves sum(omega*Vol)
        # strength-magnitude-weighted centroid
        wi = np.linalg.norm(Ai, axis=1)
        wj = np.linalg.norm(Aj, axis=1)
        wsum = np.maximum(wi + wj, 1e-30)
        P_new = (self._positions[i] * wi[:, None]
                 + self._positions[j] * wj[:, None]) / wsum[:, None]
        S_new = (self._sigmas[i] * Vi + self._sigmas[j] * Vj) / V_new
        Age_new = np.maximum(self._ages[i], self._ages[j])

        drop = np.zeros(len(self._positions), dtype=bool)
        drop[i] = True
        drop[j] = True
        self._positions = np.vstack([self._positions[~drop], P_new])
        self._vorticities = np.vstack([self._vorticities[~drop], O_new])
        self._sigmas = np.concatenate([self._sigmas[~drop], S_new])
        self._volumes = np.concatenate([self._volumes[~drop], V_new])
        self._ages = np.concatenate([self._ages[~drop], Age_new])

        return -pairs.shape[0]

    def refine(self) -> dict:
        """
        Run one adaptive-refinement pass (split under-resolved in-zone particles,
        merge over-dense ones), respecting ``refine_n_max``.

        No-op unless ``enable_refinement`` is True. Returns a summary dict with
        the number split/merged and the resulting particle count.

        Total vortex strength sum(omega*Vol) and total volume are conserved by
        the split step exactly and by the merge step to the zeroth moment.
        """
        summary = {"split": 0, "merged": 0, "n": len(self._positions)}
        if not self.enable_refinement or len(self._positions) < 2:
            return summary

        n_max = self.refine_n_max or (5 * self.n_particles)
        h = self._nearest_neighbor_spacing()
        ratio = h / np.maximum(self._sigmas, 1e-12)

        # --- split: in-zone particles that violate overlap ---
        in_zone = self.observation_zone_mask()
        split_candidates = np.where(in_zone & (ratio > self.refine_split_ratio))[0]
        if split_candidates.size > 0:
            # respect the population cap (each split adds 6 particles)
            budget = max(0, (n_max - len(self._positions)) // 6)
            if split_candidates.size > budget:
                # split the worst-overlap particles first
                order = np.argsort(ratio[split_candidates])[::-1]
                split_candidates = split_candidates[order[:budget]]
            if split_candidates.size > 0:
                added = self._split_particles(split_candidates)
                summary["split"] = split_candidates.size
                # keep particles inside the domain after placing children
                self._apply_domain_bounds()

        # --- merge: over-dense particles (outside observation zones) ---
        h = self._nearest_neighbor_spacing()
        ratio = h / np.maximum(self._sigmas, 1e-12)
        dense = np.where((~self.observation_zone_mask())
                         & (ratio < self.refine_merge_ratio))[0]
        if dense.size >= 2 and cKDTree is not None:
            pairs = self._disjoint_nearest_pairs(dense)
            if pairs.size > 0:
                self._merge_particles(pairs)
                summary["merged"] = pairs.shape[0]

        summary["n"] = len(self._positions)
        return summary

    def _disjoint_nearest_pairs(self, idx: np.ndarray) -> np.ndarray:
        """Greedily pair each candidate with its nearest candidate neighbour,
        using each particle at most once. Returns an (m, 2) index array."""
        idx = np.asarray(idx, dtype=np.int64)
        pts = self._positions[idx]
        tree = cKDTree(pts)
        dist, nn = tree.query(pts, k=2)
        order = np.argsort(dist[:, 1])          # closest pairs first
        used = np.zeros(len(idx), dtype=bool)
        pairs = []
        for a in order:
            b = nn[a, 1]
            if used[a] or used[b]:
                continue
            used[a] = used[b] = True
            pairs.append((idx[a], idx[b]))
        return np.array(pairs, dtype=np.int64) if pairs else np.empty((0, 2), np.int64)

    def _apply_domain_bounds(self):
        """Wrap/clamp particle positions to the domain (as in ``step``)."""
        self._positions[:, 0] = self._positions[:, 0] % self.L
        self._positions[:, 1] = np.clip(self._positions[:, 1], 0.5, self.W - 0.5)
        self._positions[:, 2] = np.clip(self._positions[:, 2], 0.1, self.H - 0.1)

    def set_observation(self, center: np.ndarray, radius: float):
        """
        Set observation zone location and size.

        Parameters
        ----------
        center : np.ndarray
            Center of observation zone [x, y, z]
        radius : float
            Radius of observation zone
        """
        self.obs_center = np.asarray(center, dtype=np.float64)
        self.obs_radius = radius
        self.obs_zones = None  # revert to single-zone mode

    def set_observation_zones(self, zones: list):
        """
        Set multiple observation zones.

        Parameters
        ----------
        zones : list of (center, radius) tuples
            Each center is a 3-element array [x, y, z].
        """
        self.obs_zones = [
            (np.asarray(c, dtype=np.float64), float(r)) for c, r in zones
        ]
        self.observation_active = True

    def toggle_observation(self, active: Optional[bool] = None):
        """
        Toggle or set observation-dependent resolution.

        Parameters
        ----------
        active : bool, optional
            If provided, set to this value. Otherwise toggle.
        """
        if active is not None:
            self.observation_active = active
        else:
            self.observation_active = not self.observation_active

    def get_state(self) -> FieldState:
        """
        Get current field state for visualization.

        OPTIMIZED: Returns array views when possible.

        Returns
        -------
        FieldState
            Current state of the vortex field
        """
        n = len(self._positions)
        if n == 0:
            return FieldState(
                positions=np.zeros((0, 3)),
                vorticities=np.zeros((0, 3)),
                energies=np.zeros(0),
                sigmas=np.zeros(0),
                ages=np.zeros(0),
                obs_center=self.obs_center.copy(),
                obs_radius=self.obs_radius,
                observation_active=self.observation_active,
                domain_length=self.L,
                domain_width=self.W,
                domain_depth=self.H,
                trails=[t.copy() for t in self.trails],
            )

        return FieldState(
            positions=self._positions.copy(),
            vorticities=self._vorticities.copy(),
            energies=self._compute_energies(),
            sigmas=self._sigmas.copy(),
            ages=self._ages.copy(),
            obs_center=self.obs_center.copy(),
            obs_radius=self.obs_radius,
            observation_active=self.observation_active,
            domain_length=self.L,
            domain_width=self.W,
            domain_depth=self.H,
            trails=[t.copy() for t in self.trails],
        )

    @property
    def particles(self) -> List[VortexParticle]:
        """
        Get particles as list (for backwards compatibility).

        Note: This is slower than using the array-based interface.
        """
        return [
            VortexParticle(
                pos=self._positions[i].copy(),
                omega=self._vorticities[i].copy(),
                sigma=self._sigmas[i],
                age=self._ages[i]
            )
            for i in range(len(self._positions))
        ]

    def update_hydraulics(self, hydraulics: HydraulicsEngine):
        """
        Update hydraulics engine and reconfigure field.

        Parameters
        ----------
        hydraulics : HydraulicsEngine
            New hydraulics engine
        """
        self.hydraulics = hydraulics
        self.W = hydraulics.width
        self.H = hydraulics.depth
        self.base_sigma = self.H / 5.0
        self.min_sigma = hydraulics.eta_kolmogorov * 3
        self._build_velocity_lut()
        self._seed_field()
        self._ref_volume = self._reference_particle_volume()
        self._volumes = np.full(len(self._positions), self._ref_volume,
                                dtype=np.float64)

    def __repr__(self) -> str:
        return (
            f"VortexParticleField(n={len(self._positions)}, "
            f"L={self.L:.0f}, W={self.W:.0f}, H={self.H:.1f}, "
            f"obs_active={self.observation_active})"
        )
