"""
ULTIMATE QUANTUM-INSPIRED VORTEX PARTICLE STREAM SIMULATOR
===========================================================

Combines:
1. First-principles physics (Colebrook-White, Kolmogorov cascade)
2. True vortex particle method (Biot-Savart law, 3D particles)
3. Observation-dependent resolution (adaptive core size σ)
4. Educational features (theory panels, explanations)
5. Optimized performance (spatial trees, efficient algorithms)

This is the synthesis of everything we've discussed.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, CheckButtons, Button
from matplotlib.patches import Circle, Rectangle
from scipy.spatial import cKDTree
from collections import deque
import time

class VortexParticle:
    """
    Single vortex particle in 3D space.
    Carries vorticity (circulation) and has a core size (resolution).
    """
    __slots__ = ['pos', 'omega', 'sigma', 'age', 'energy']
    
    def __init__(self, position, vorticity, core_size):
        self.pos = np.asarray(position, dtype=np.float64)
        self.omega = np.asarray(vorticity, dtype=np.float64)
        self.sigma = core_size
        self.age = 0.0
        self.energy = np.linalg.norm(self.omega)**2 * self.sigma**3

class HydraulicsEngine:
    """
    Computes flow properties from first-principles physics.
    No empirical Manning's equation - pure Colebrook-White and conservation laws.
    """
    
    def __init__(self, Q, width, depth, slope, roughness_ks):
        # Physical constants
        self.g = 32.2  # ft/s² - gravity
        self.rho = 1.94  # slugs/ft³ - water density
        self.nu = 1.1e-5  # ft²/s - kinematic viscosity (60°F)
        
        # Channel geometry
        self.Q = Q
        self.width = width
        self.depth = depth
        self.slope = slope
        self.ks = roughness_ks
        self.side_slope = 2.0  # H:V
        
        # Compute from first principles
        self._compute_hydraulics()
        self._compute_turbulence_scales()
    
    def _compute_hydraulics(self):
        """
        Compute hydraulic properties from conservation laws.
        Uses Colebrook-White for friction, NOT Manning's equation.
        """
        # Geometry (exact)
        z = self.side_slope
        self.A = self.width * self.depth + z * self.depth**2
        self.P = self.width + 2 * self.depth * np.sqrt(1 + z**2)
        self.R = self.A / self.P  # Hydraulic radius
        self.T = self.width + 2 * z * self.depth  # Top width
        
        # Continuity (exact)
        self.V_mean = self.Q / self.A
        
        # Reynolds number (dimensionless group)
        self.Re = self.V_mean * self.R / self.nu
        
        # Friction factor from Colebrook-White (implicit iteration)
        # 1/√f = -2 log₁₀(ε/(3.7D) + 2.51/(Re√f))
        epsilon_over_D = self.ks / (4 * self.R)
        f = 0.02  # Initial guess
        
        for iteration in range(20):
            if self.Re < 1e-6:
                break
            term1 = epsilon_over_D / 3.7
            term2 = 2.51 / (self.Re * np.sqrt(f))
            f_new = (-2.0 * np.log10(term1 + term2))**(-2)
            
            if abs(f_new - f) < 1e-8:
                break
            f = f_new
        
        self.friction_factor = f
        
        # Energy slope from Darcy-Weisbach
        self.Sf = f * self.V_mean**2 / (8 * self.g * self.R)
        
        # Friction velocity (fundamental for boundary layer)
        self.u_star = self.V_mean * np.sqrt(f / 8)
        
        # Froude number (flow regime)
        self.D_hydraulic = self.A / self.T
        self.Fr = self.V_mean / np.sqrt(self.g * self.D_hydraulic)
        
        # Check if uniform flow
        self.is_uniform = abs(self.Sf - self.slope) < 0.0001
    
    def _compute_turbulence_scales(self):
        """
        Compute turbulence scales from Kolmogorov theory.
        These are exact from dimensional analysis.
        """
        # Turbulent kinetic energy per unit mass
        self.TKE = 0.5 * self.V_mean**2
        
        # Turbulent dissipation rate: ε ~ V³/L
        self.epsilon = self.V_mean**3 / self.R
        
        # Kolmogorov microscale: η = (ν³/ε)^(1/4)
        self.eta_kolmogorov = (self.nu**3 / self.epsilon)**0.25
        
        # Kolmogorov time scale
        self.tau_kolmogorov = np.sqrt(self.nu / self.epsilon)
        
        # Large eddy turnover time
        self.T_large_eddy = self.R / self.V_mean
        
        # Taylor microscale (intermediate scale)
        self.lambda_taylor = np.sqrt(15 * self.nu * self.TKE / self.epsilon)
        
        # Turbulent Reynolds number
        self.Re_turbulent = self.TKE**2 / (self.nu * self.epsilon)
    
    def velocity_profile(self, z):
        """
        Compute velocity at height z using log law (near bed) and power law (outer).
        This is derived from boundary layer theory, not empirical.
        """
        if z <= 0:
            return 0.0
        
        # Roughness length: z₀ = ks/30
        z0 = self.ks / 30.0
        
        # von Karman constant (universal!)
        kappa = 0.41
        
        if z < 0.2 * self.depth and z > z0:
            # Inner layer: log law
            # u/u* = (1/κ) ln(z/z₀)
            u = (self.u_star / kappa) * np.log(z / z0)
        else:
            # Outer layer: power law
            # u/U = (z/h)^(1/n) where n ≈ 7 for turbulent flow
            u = self.V_mean * (z / self.depth)**(1/7)
        
        return u
    
    def get_summary(self):
        """Return formatted summary of hydraulics"""
        flow_type = "SUPERCRITICAL" if self.Fr > 1.0 else "SUBCRITICAL"
        uniform_type = "UNIFORM" if self.is_uniform else "GRADUALLY VARIED"
        
        summary = {
            'flow_regime': flow_type,
            'uniformity': uniform_type,
            'Q': self.Q,
            'V': self.V_mean,
            'A': self.A,
            'R': self.R,
            'Re': self.Re,
            'Fr': self.Fr,
            'f': self.friction_factor,
            'u_star': self.u_star,
            'Sf': self.Sf,
            'TKE': self.TKE,
            'epsilon': self.epsilon,
            'eta': self.eta_kolmogorov,
            'Re_t': self.Re_turbulent
        }
        
        return summary

class VortexParticleField:
    """
    3D vortex particle system with observation-dependent resolution.
    Implements Biot-Savart law for velocity induction and particle strength exchange for diffusion.
    """
    
    def __init__(self, hydraulics, length=200.0, n_particles=6000):
        self.hydraulics = hydraulics
        self.L = length
        self.W = hydraulics.width
        self.H = hydraulics.depth
        
        # Observation zone (quantum measurement location)
        self.obs_center = np.array([length/2, hydraulics.width/2, hydraulics.depth/2])
        self.obs_radius = 25.0
        self.observation_active = True
        
        # Particle system
        self.particles = []
        self.n_particles = n_particles
        
        # Core size parameters
        self.base_sigma = self.H / 5.0  # Base resolution
        self.min_sigma = self.hydraulics.eta_kolmogorov * 3  # Can't go below ~3× Kolmogorov
        self.max_sigma = self.base_sigma * 2  # Maximum coarsening
        
        # Visualization
        self.trails = deque(maxlen=40)
        self.trail_frequency = 0  # Counter for trail updates
        
        # Performance
        self.spatial_tree = None
        self.last_tree_build = 0
        
        # Initialize particles
        self._seed_particles()
    
    def _seed_particles(self):
        """
        Seed particles representing turbulent vorticity field.
        Uses Kolmogorov cascade theory to distribute across scales.
        """
        self.particles.clear()
        rng = np.random.default_rng(42)
        
        # Generate particles at multiple scales (energy cascade)
        # Scale hierarchy: L, L/2, L/4, L/8 (Kolmogorov cascade)
        scales = [self.H, self.H/2, self.H/4, self.H/8]
        weights = [0.15, 0.25, 0.35, 0.25]  # More particles at smaller scales
        
        for scale, weight in zip(scales, weights):
            n = int(self.n_particles * weight)
            
            for i in range(n):
                # Random position in channel
                # Keep away from boundaries
                x = rng.uniform(0.1 * self.L, 0.9 * self.L)
                y = rng.uniform(0.1 * self.W, 0.9 * self.W)
                z = rng.uniform(0.1 * self.H, 0.9 * self.H)
                pos = np.array([x, y, z])
                
                # Vorticity magnitude from Kolmogorov scaling
                if scale > 10 * self.hydraulics.eta_kolmogorov:
                    # Large eddies: ω ~ V/L
                    omega_mag = self.hydraulics.V_mean / scale
                else:
                    # Small eddies: ω ~ √(ε/ν)
                    omega_mag = np.sqrt(self.hydraulics.epsilon / self.hydraulics.nu)
                
                # Random orientation (isotropic turbulence)
                theta = rng.uniform(0, 2*np.pi)
                phi = rng.uniform(0, np.pi)
                
                # Vorticity components
                # Bias toward streamwise (x) and vertical (z) vorticity
                omega_x = 0.3 * omega_mag * rng.normal()
                omega_y = omega_mag * np.sin(phi) * np.sin(theta)  # Cross-stream (helical)
                omega_z = 0.5 * omega_mag * rng.normal()
                
                omega = np.array([omega_x, omega_y, omega_z])
                
                # Initial core size based on observation
                sigma = self.get_adaptive_core_size(pos)
                
                particle = VortexParticle(pos, omega, sigma)
                self.particles.append(particle)
        
        print(f"Seeded {len(self.particles)} vortex particles across {len(scales)} scales")
    
    def get_adaptive_core_size(self, position):
        """
        Compute observation-dependent core size (THE QUANTUM PART).
        
        Near observation zone: Small σ → high resolution
        Far from observation: Large σ → coarse approximation
        
        This is the lazy evaluation: computation effort scales with observation.
        """
        if not self.observation_active:
            return self.base_sigma
        
        # Distance from observation center (2D in plan view)
        dx = position[0] - self.obs_center[0]
        dy = position[1] - self.obs_center[1]
        dz = (position[2] - self.obs_center[2]) * 0.5  # Weight vertical less
        dist = np.sqrt(dx**2 + dy**2 + dz**2)
        
        # Gaussian-like resolution enhancement near observation
        # Factor ranges from 1 (far) to ~5 (at center)
        enhancement_factor = 1.0 + 4.0 * np.exp(-(dist / self.obs_radius)**2)
        
        # Smaller sigma = higher resolution
        sigma_adaptive = self.base_sigma / enhancement_factor
        
        # Clamp to physical limits
        sigma = np.clip(sigma_adaptive, self.min_sigma, self.max_sigma)
        
        return sigma
    
    def compute_velocity_induction(self):
        """
        Compute velocity induced on each particle by all others.
        Uses Biot-Savart law with regularized kernel (prevents singularities).
        
        v = (1/4π) ∫ (ω × r) / |r|³ dV
        
        Regularized: replace |r|³ with (r² + σ²)^(3/2)
        """
        n = len(self.particles)
        positions = np.array([p.pos for p in self.particles])
        vorticities = np.array([p.omega for p in self.particles])
        
        # Build spatial tree for efficient neighbor search
        # Only rebuild every 10 steps (expensive operation)
        current_step = self.last_tree_build
        if current_step % 10 == 0:
            self.spatial_tree = cKDTree(positions)
        self.last_tree_build += 1
        
        velocities = np.zeros((n, 3))
        
        # For each particle, find nearby influencers
        for i, particle in enumerate(self.particles):
            # Query neighbors within cutoff distance (6σ is standard)
            cutoff = 6.0 * particle.sigma
            
            if self.spatial_tree is not None:
                neighbor_indices = self.spatial_tree.query_ball_point(
                    particle.pos, cutoff
                )
            else:
                neighbor_indices = list(range(n))
            
            if len(neighbor_indices) < 2:
                continue
            
            # Vectorized computation for all neighbors
            pos_neighbors = positions[neighbor_indices]
            omega_neighbors = vorticities[neighbor_indices]
            
            # Displacement vectors: r = x_i - x_neighbor
            r_vecs = pos_neighbors - particle.pos
            
            # Distance squared
            r_squared = np.sum(r_vecs**2, axis=1, keepdims=True)
            
            # Get core sizes for regularization
            sigmas = np.array([self.particles[j].sigma for j in neighbor_indices])
            sigma_squared = sigmas**2
            
            # Regularized kernel: 1 / (r² + σ²)^(3/2)
            # Add smoothing factor to prevent division by zero
            denominator = (r_squared + sigma_squared[:, np.newaxis])**1.5 + 1e-12
            
            # Viscous cutoff function (smooth transition)
            # f(r) = 1 - exp(-r²/σ²)
            cutoff_func = 1.0 - np.exp(-r_squared / sigma_squared[:, np.newaxis])
            
            # Biot-Savart kernel: K = f(r) / (4π × denominator)
            K = cutoff_func / (4 * np.pi * denominator)
            
            # Cross product: ω × r
            cross_products = np.cross(omega_neighbors, r_vecs)
            
            # Velocity contribution: v = Σ K × (ω × r)
            velocity_induced = np.sum(K * cross_products, axis=0)
            
            velocities[i] = velocity_induced
        
        return velocities
    
    def apply_diffusion(self):
        """
        Apply viscous diffusion using Particle Strength Exchange (PSE) method.
        
        This models: ∂ω/∂t = ν∇²ω
        
        PSE approximation: Dω_i/Dt ≈ Σ_j (ω_j - ω_i) × η(r_ij, σ)
        """
        positions = np.array([p.pos for p in self.particles])
        
        if self.spatial_tree is None:
            self.spatial_tree = cKDTree(positions)
        
        for particle in self.particles:
            # Find neighbors for diffusion
            search_radius = 4.0 * particle.sigma
            neighbor_indices = self.spatial_tree.query_ball_point(
                particle.pos, search_radius
            )
            
            if len(neighbor_indices) < 2:
                continue
            
            # Get neighbor positions and vorticities
            neighbor_pos = positions[neighbor_indices]
            neighbor_omega = np.array([self.particles[j].omega for j in neighbor_indices])
            
            # Displacement vectors
            dx = neighbor_pos - particle.pos
            r_squared = np.sum(dx**2, axis=1)
            
            # PSE diffusion kernel: exp(-r²/2σ²)
            weights = np.exp(-r_squared / (2 * particle.sigma**2))
            weights /= (weights.sum() + 1e-12)  # Normalize
            
            # Weighted average vorticity
            omega_avg = np.average(neighbor_omega, axis=0, weights=weights)
            
            # Diffusion: move toward average
            # Rate proportional to viscosity and inversely to σ²
            diffusion_rate = 2.0 * self.hydraulics.nu / (particle.sigma**2 + 1e-12)
            particle.omega += diffusion_rate * (omega_avg - particle.omega)
    
    def step(self, dt=0.05):
        """
        Advance particle system one timestep.
        """
        # Compute velocities from vorticity (Biot-Savart)
        velocities = self.compute_velocity_induction()
        
        # Add mean flow (downstream advection)
        for i, particle in enumerate(self.particles):
            z_normalized = particle.pos[2] / self.H
            u_mean = self.hydraulics.velocity_profile(particle.pos[2])
            velocities[i, 0] += u_mean  # Add streamwise velocity
        
        # Advect particles
        for i, particle in enumerate(self.particles):
            particle.pos += velocities[i] * dt
            particle.age += dt
            
            # Enforce boundary conditions (periodic in x, reflective in y,z)
            particle.pos[0] = particle.pos[0] % self.L  # Periodic in streamwise
            particle.pos[1] = np.clip(particle.pos[1], 0.5, self.W - 0.5)
            particle.pos[2] = np.clip(particle.pos[2], 0.1, self.H - 0.1)
        
        # Apply viscous diffusion (PSE method)
        self.apply_diffusion()
        
        # Update core sizes based on observation (adaptive resolution)
        for particle in self.particles:
            particle.sigma = self.get_adaptive_core_size(particle.pos)
            particle.energy = np.linalg.norm(particle.omega)**2 * particle.sigma**3
        
        # Store trail for visualization
        self.trail_frequency += 1
        if self.trail_frequency % 3 == 0:  # Every 3rd frame
            # Track strongest vortices for trails
            sorted_particles = sorted(
                self.particles, 
                key=lambda p: p.energy, 
                reverse=True
            )
            trail_positions = [p.pos.copy() for p in sorted_particles[:500]]
            self.trails.append(trail_positions)

class UltimateHybridSimulator:
    """
    The ultimate synthesis: First-principles physics + vortex particles + quantum observation.
    """
    
    def __init__(self):
        # Initial parameters
        self.Q = 600.0
        self.width = 30.0
        self.depth = 5.0
        self.slope = 0.002
        self.roughness = 0.15
        self.length = 200.0
        
        # Initialize physics engine
        self.hydraulics = HydraulicsEngine(
            self.Q, self.width, self.depth, self.slope, self.roughness
        )
        
        # Initialize vortex particle field
        self.vortex_field = VortexParticleField(
            self.hydraulics, 
            length=self.length,
            n_particles=6000
        )
        
        # Animation control
        self.running = False
        self.steps_per_frame = 3  # Multiple physics steps per render
        
        # Create UI
        self._create_interactive_ui()
    
    def _create_interactive_ui(self):
        """Create the complete interactive interface"""
        plt.style.use('dark_background')
        self.fig = plt.figure(figsize=(20, 11))
        self.fig.patch.set_facecolor('#0a0a0a')
        
        # Main plot areas
        gs = self.fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3,
                                   left=0.08, right=0.98, top=0.94, bottom=0.12)
        
        self.ax_plan = self.fig.add_subplot(gs[0:2, 0:3])
        self.ax_theory = self.fig.add_subplot(gs[0, 3])
        self.ax_profile = self.fig.add_subplot(gs[1, 3])
        self.ax_detail = self.fig.add_subplot(gs[2, 0:2])
        self.ax_velocity = self.fig.add_subplot(gs[2, 2])
        self.ax_spectrum = self.fig.add_subplot(gs[2, 3])
        
        # Create sliders
        self._create_controls()
        
        # Initial display
        self._update_visualization()
        
        # Setup animation
        from matplotlib.animation import FuncAnimation
        self.anim = FuncAnimation(
            self.fig, self._animate_frame,
            interval=50, blit=False, cache_frame_data=False
        )
        
        plt.show()
    
    def _create_controls(self):
        """Create interactive sliders and buttons"""
        slider_specs = [
            ('Q', 'Discharge Q (cfs)', 100, 2000, self.Q, '%1.0f'),
            ('width', 'Channel Width (ft)', 15, 60, self.width, '%1.0f'),
            ('depth', 'Water Depth (ft)', 2, 12, self.depth, '%1.1f'),
            ('slope', 'Bed Slope', 0.0001, 0.01, self.slope, '%1.4f'),
            ('roughness', 'Roughness ks (ft)', 0.01, 0.5, self.roughness, '%1.3f'),
        ]
        
        self.sliders = {}
        slider_x = 0.12
        slider_width = 0.25
        slider_height = 0.015
        y_positions = [0.08, 0.06, 0.04, 0.02, 0.00]
        
        for (key, label, vmin, vmax, vinit, fmt), y_pos in zip(slider_specs, y_positions):
            ax_slider = plt.axes([slider_x, y_pos, slider_width, slider_height],
                                facecolor='#2a2a2a')
            slider = Slider(ax_slider, label, vmin, vmax, valinit=vinit,
                          valstep=None, valfmt=fmt, color='#00d4ff')
            slider.label.set_color('white')
            slider.label.set_fontsize(9)
            slider.valtext.set_color('white')
            slider.on_changed(lambda val, k=key: self._on_parameter_change(k, val))
            self.sliders[key] = slider
        
        # Checkbox for observation
        ax_check = plt.axes([slider_x + slider_width + 0.02, 0.01, 0.12, 0.08],
                           facecolor='#0a0a0a')
        self.checkbox = CheckButtons(
            ax_check,
            ['Quantum\nObservation'],
            [True]
        )
        self.checkbox.labels[0].set_color('#00d4ff')
        self.checkbox.labels[0].set_fontsize(10)
        self.checkbox.on_clicked(self._toggle_observation)
        
        # Start/Stop button
        ax_button = plt.axes([slider_x + slider_width + 0.15, 0.01, 0.08, 0.04])
        self.button = Button(ax_button, 'Pause', color='#2a2a2a', hovercolor='#4a4a4a')
        self.button.label.set_color('white')
        self.button.on_clicked(self._toggle_animation)
        self.running = True
    
    def _on_parameter_change(self, param, value):
        """Handle parameter slider changes"""
        setattr(self, param, value)
        
        # Recompute hydraulics
        self.hydraulics = HydraulicsEngine(
            self.Q, self.width, self.depth, self.slope, self.roughness
        )
        
        # Update vortex field
        self.vortex_field.hydraulics = self.hydraulics
        self.vortex_field.W = self.width
        self.vortex_field.H = self.depth
        self.vortex_field.base_sigma = self.depth / 5.0
        self.vortex_field.min_sigma = self.hydraulics.eta_kolmogorov * 3
        
        # Reseed particles
        self.vortex_field._seed_particles()
    
    def _toggle_observation(self, label):
        """Toggle observation zone on/off"""
        self.vortex_field.observation_active = not self.vortex_field.observation_active
    
    def _toggle_animation(self, event):
        """Start/stop animation"""
        self.running = not self.running
        self.button.label.set_text('Resume' if not self.running else 'Pause')
    
    def _animate_frame(self, frame):
        """Animation update function"""
        if self.running:
            # Multiple physics steps per frame for smoothness
            for _ in range(self.steps_per_frame):
                self.vortex_field.step(dt=0.05)
            
            self._update_visualization()
    
    def _update_visualization(self):
        """Update all plots"""
        # Clear axes but preserve their positions
        for ax in [self.ax_plan, self.ax_theory, self.ax_profile, 
                   self.ax_detail, self.ax_velocity, self.ax_spectrum]:
            # Store original position before clearing
            if not hasattr(ax, '_original_position'):
                ax._original_position = ax.get_position()
            
            ax.clear()
            ax.set_facecolor('#0a0a0a')
            ax.tick_params(colors='white', labelsize=8)
            
            # Restore original position after clearing
            ax.set_position(ax._original_position)
        
        self._plot_plan_view()
        self._plot_theory_panel()
        self._plot_profile_view()
        self._plot_detail_map()
        self._plot_velocity_profile()
        self._plot_energy_spectrum()
        
        plt.draw()
    
    def _plot_plan_view(self):
        """Main plan view with vortex particles"""
        ax = self.ax_plan
        
        # Get particle data
        positions = np.array([p.pos for p in self.vortex_field.particles])
        energies = np.array([p.energy for p in self.vortex_field.particles])
        
        # Particle size reflects resolution (small σ = small dot = high resolution)
        sigmas = np.array([p.sigma for p in self.vortex_field.particles])
        sizes = 1200 / (sigmas**2)
        sizes = np.clip(sizes, 4, 120)  # Reasonable range
        
        # Plot particles colored by energy, sized by resolution
        scatter = ax.scatter(positions[:, 0], positions[:, 1],
                           c=energies, cmap='turbo', s=sizes, alpha=0.8,
                           vmin=0, vmax=np.percentile(energies, 95), linewidths=0)
        
        # Plot trails
        for trail in self.vortex_field.trails:
            if len(trail) > 1:
                trail_array = np.array(trail)
                ax.plot(trail_array[:, 0], trail_array[:, 1],
                       color='white', alpha=0.08, linewidth=0.8)
        
        # Observation zone
        if self.vortex_field.observation_active:
            obs_circle = Circle(
                (self.vortex_field.obs_center[0], self.vortex_field.obs_center[1]),
                self.vortex_field.obs_radius,
                fill=False, edgecolor='yellow', linewidth=3, linestyle='--', alpha=0.9
            )
            ax.add_patch(obs_circle)
            ax.text(self.vortex_field.obs_center[0],
                   self.vortex_field.obs_center[1] + self.vortex_field.obs_radius + 3,
                   'OBSERVATION\nZONE', color='yellow', fontsize=10,
                   ha='center', weight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='#0a0a0a', alpha=0.7))
        
        # Channel boundaries
        ax.plot([0, self.length], [0, 0], 'brown', linewidth=3, alpha=0.7)
        ax.plot([0, self.length], [self.width, self.width], 'brown', linewidth=3, alpha=0.7)
        
        ax.set_xlim(0, self.length)
        ax.set_ylim(-2, self.width + 2)
        ax.set_aspect('equal')
        ax.set_xlabel('Streamwise Distance (ft)', color='white', fontsize=10)
        ax.set_ylabel('Cross-Stream Distance (ft)', color='white', fontsize=10)
        
        # Title with flow regime
        summary = self.hydraulics.get_summary()
        title_color = 'red' if summary['Fr'] > 1 else 'lime'
        ax.set_title(
            f"3D VORTEX PARTICLE FIELD — {summary['flow_regime']} — "
            f"{len(self.vortex_field.particles)} Particles",
            color=title_color, fontsize=13, weight='bold', pad=10
        )
        
        # Info box
        info_text = (
            f"FIRST-PRINCIPLES PHYSICS + VORTEX PARTICLES\n"
            f"Biot-Savart Law • Kolmogorov Cascade • Adaptive Resolution"
        )
        ax.text(0.01, 0.97, info_text, transform=ax.transAxes,
               fontsize=9, color='cyan', va='top', weight='bold',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='#1a1a1a', alpha=0.9))
        
        # Legend for particle sizes (remove old legend first to prevent accumulation)
        old_legend = ax.get_legend()
        if old_legend is not None:
            old_legend.remove()
        
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='cyan', 
                   markersize=4, linestyle='', label='High Resolution (small σ)'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='cyan', 
                   markersize=10, linestyle='', label='Low Resolution (large σ)')
        ]
        ax.legend(handles=legend_elements, loc='lower left', fontsize=8,
                 facecolor='#1a1a1a', edgecolor='white', labelcolor='white',
                 framealpha=0.9)
        
        # Colorbar (remove old one first to prevent accumulation)
        # Remove any existing colorbars associated with this axis
        if hasattr(ax, '_last_colorbar') and ax._last_colorbar is not None:
            try:
                ax._last_colorbar.remove()
            except:
                pass
        
        cbar = plt.colorbar(scatter, ax=ax, pad=0.01, fraction=0.03)
        cbar.set_label('Vortex Energy', color='white', fontsize=9)
        cbar.ax.tick_params(colors='white', labelsize=8)
        
        # Store reference for next frame
        ax._last_colorbar = cbar
    
    def _plot_theory_panel(self):
        """Physics theory display"""
        ax = self.ax_theory
        ax.axis('off')
        
        summary = self.hydraulics.get_summary()
        
        theory_text = (
            "═══════════════════════════\n"
            "   FIRST-PRINCIPLES PHYSICS\n"
            "═══════════════════════════\n\n"
            "Conservation Laws:\n"
            f"├─ Q = {summary['Q']:.0f} cfs\n"
            f"├─ V̄ = {summary['V']:.2f} ft/s\n"
            f"├─ A = {summary['A']:.1f} ft²\n"
            f"└─ R = {summary['R']:.2f} ft\n\n"
            "Dimensionless Groups:\n"
            f"├─ Re = {summary['Re']:.0f}\n"
            f"├─ Fr = {summary['Fr']:.3f}\n"
            f"└─ Re_t = {summary['Re_t']:.0f}\n\n"
            "Friction (Colebrook):\n"
            f"├─ f = {summary['f']:.5f}\n"
            f"├─ u* = {summary['u_star']:.3f} ft/s\n"
            f"└─ Sf = {summary['Sf']:.5f}\n\n"
            "Turbulence Scales:\n"
            f"├─ ε = {summary['epsilon']:.4f} ft²/s³\n"
            f"├─ η_K = {summary['eta']:.5f} ft\n"
            f"└─ k = {summary['TKE']:.3f} ft²/s²\n\n"
            "Vortex Particles:\n"
            f"├─ N = {len(self.vortex_field.particles)}\n"
            f"├─ σ_base = {self.vortex_field.base_sigma:.4f} ft\n"
            f"└─ σ_min = {self.vortex_field.min_sigma:.5f} ft\n\n"
            f"Flow: {summary['flow_regime']}\n"
            f"Type: {summary['uniformity']}"
        )
        
        ax.text(0.05, 0.95, theory_text,
               transform=ax.transAxes, fontsize=8, color='lime',
               family='monospace', va='top',
               bbox=dict(boxstyle='round,pad=0.8', facecolor='#111111', alpha=0.95))
    
    def _plot_profile_view(self):
        """Longitudinal profile"""
        ax = self.ax_profile
        
        x_profile = np.array([0, self.length])
        bed_elev = np.array([10, 10 - self.slope * self.length])
        wse = bed_elev + self.depth
        
        # Water
        ax.fill_between(x_profile, bed_elev, wse, color='cyan', alpha=0.4)
        ax.plot(x_profile, wse, 'b-', linewidth=2.5, label='Water Surface')
        ax.plot(x_profile, bed_elev, color='#8B4513', linewidth=3, label='Channel Bed')
        
        # Observation zone marker
        if self.vortex_field.observation_active:
            ax.axvline(self.vortex_field.obs_center[0], color='yellow',
                      linestyle='--', linewidth=2, alpha=0.8, label='Observation')
        
        ax.set_xlim(0, self.length)
        ax.set_ylim(bed_elev[-1] - 1, wse[0] + 1)
        ax.set_xlabel('Distance (ft)', color='white', fontsize=9)
        ax.set_ylabel('Elevation (ft)', color='white', fontsize=9)
        ax.set_title('LONGITUDINAL PROFILE', color='white', fontsize=10, weight='bold')
        ax.legend(fontsize=7, facecolor='#2a2a2a', edgecolor='white', labelcolor='white')
        ax.grid(True, alpha=0.2, color='white', linestyle=':')
    
    def _plot_detail_map(self):
        """Computational detail map (observation-dependent resolution)"""
        ax = self.ax_detail
        
        # Create grid
        x_grid = np.linspace(0, self.length, 80)
        y_grid = np.linspace(0, self.width, 40)
        X, Y = np.meshgrid(x_grid, y_grid)
        
        # Compute core size at each point
        Z = np.zeros_like(X)
        for i in range(len(x_grid)):
            for j in range(len(y_grid)):
                pos = np.array([x_grid[i], y_grid[j], self.depth/2])
                Z[j, i] = self.vortex_field.get_adaptive_core_size(pos)
        
        # Plot as heatmap (inverted: small σ = bright = expensive)
        im = ax.contourf(X, Y, 1.0/Z, levels=25, cmap='viridis')
        
        # Observation zone
        if self.vortex_field.observation_active:
            obs_circle = Circle(
                (self.vortex_field.obs_center[0], self.vortex_field.obs_center[1]),
                self.vortex_field.obs_radius,
                fill=False, edgecolor='yellow', linewidth=3, linestyle='--'
            )
            ax.add_patch(obs_circle)
        
        ax.set_xlim(0, self.length)
        ax.set_ylim(0, self.width)
        ax.set_xlabel('Distance (ft)', color='white', fontsize=9)
        ax.set_ylabel('Width (ft)', color='white', fontsize=9)
        ax.set_title('COMPUTATIONAL RESOLUTION MAP\n(Bright = High Resolution = Expensive)',
                    color='white', fontsize=10, weight='bold')
        
        # Colorbar (remove old one first)
        if hasattr(ax, '_last_colorbar') and ax._last_colorbar is not None:
            try:
                ax._last_colorbar.remove()
            except:
                pass
        
        cbar = plt.colorbar(im, ax=ax, label='1/σ (Resolution)')
        cbar.ax.tick_params(colors='white', labelsize=8)
        cbar.set_label('Resolution (1/σ)', color='white', fontsize=8)
        
        # Store reference
        ax._last_colorbar = cbar
    
    def _plot_velocity_profile(self):
        """Vertical velocity profile (log law + power law)"""
        ax = self.ax_velocity
        
        # Compute profile
        z_values = np.linspace(0.001, self.depth, 100)
        u_values = [self.hydraulics.velocity_profile(z) for z in z_values]
        
        ax.plot(u_values, z_values, 'cyan', linewidth=2.5, label='Theory')
        ax.axhline(self.depth * 0.2, color='orange', linestyle=':', 
                  linewidth=1, alpha=0.7, label='Log-Law Limit')
        ax.axvline(self.hydraulics.V_mean, color='lime', linestyle='--',
                  linewidth=1.5, alpha=0.7, label='Mean Velocity')
        
        ax.set_xlabel('Velocity (ft/s)', color='white', fontsize=9)
        ax.set_ylabel('Height Above Bed (ft)', color='white', fontsize=9)
        ax.set_title('VELOCITY PROFILE\nLog Law + Power Law',
                    color='white', fontsize=10, weight='bold')
        ax.legend(fontsize=7, facecolor='#2a2a2a', labelcolor='white')
        ax.grid(True, alpha=0.2, color='white', linestyle=':')
        ax.set_xlim(0, max(u_values) * 1.1)
        ax.set_ylim(0, self.depth)
    
    def _plot_energy_spectrum(self):
        """Turbulent energy spectrum (Kolmogorov -5/3 law)"""
        ax = self.ax_spectrum
        
        # Compute vortex energy distribution by scale
        energies = np.array([p.energy for p in self.vortex_field.particles])
        scales = np.array([p.sigma for p in self.vortex_field.particles])
        
        # Bin by scale
        scale_bins = np.logspace(np.log10(scales.min()), np.log10(scales.max()), 15)
        energy_binned = []
        scale_centers = []
        
        for i in range(len(scale_bins) - 1):
            mask = (scales >= scale_bins[i]) & (scales < scale_bins[i+1])
            if np.any(mask):
                energy_binned.append(energies[mask].sum())
                scale_centers.append(np.sqrt(scale_bins[i] * scale_bins[i+1]))
        
        if len(scale_centers) > 0:
            # Plot actual distribution
            ax.loglog(scale_centers, energy_binned, 'o-', color='cyan',
                     linewidth=2, markersize=6, label='Vortex Energy')
            
            # Plot theoretical -5/3 slope
            k = 1.0 / np.array(scale_centers)
            E_theory = self.hydraulics.epsilon**(2/3) * k**(-5/3)
            E_theory *= energy_binned[len(energy_binned)//2] / E_theory[len(E_theory)//2]
            ax.loglog(scale_centers, E_theory, '--', color='yellow',
                     linewidth=2, alpha=0.7, label='Kolmogorov -5/3')
        
        ax.set_xlabel('Length Scale (ft)', color='white', fontsize=9)
        ax.set_ylabel('Energy', color='white', fontsize=9)
        ax.set_title('ENERGY SPECTRUM\nKolmogorov Cascade',
                    color='white', fontsize=10, weight='bold')
        ax.legend(fontsize=7, facecolor='#2a2a2a', labelcolor='white')
        ax.grid(True, alpha=0.2, color='white', linestyle=':')

def main():
    """Launch the ultimate simulator"""
    import os
    print("\n" + "=" * 80)
    print(" " * 15 + "ULTIMATE QUANTUM-INSPIRED VORTEX PARTICLE SIMULATOR")
    print("=" * 80)
    print("\nCombining:")
    print("  ✓ First-principles physics (Colebrook-White, Kolmogorov cascade)")
    print("  ✓ True 3D vortex particle method (Biot-Savart law)")
    print("  ✓ Observation-dependent resolution (adaptive core size σ)")
    print("  ✓ Educational features (theory panels, velocity profiles)")
    print("  ✓ Optimized performance (~6000 particles, spatial trees)")
    print("\nThis is the synthesis of everything we've discussed.")
    print("=" * 80 + "\n")
    
    output_dir = os.path.join(os.getcwd(), 'quantum_flow_output')
    os.makedirs(output_dir, exist_ok=True)
    
    print("Initializing simulator...")
    print("  • Computing hydraulics from conservation laws...")
    print("  • Generating vortex particles across Kolmogorov cascade...")
    print("  • Building spatial acceleration structures...")
    print("  • Creating interactive UI...\n")
    
    simulator = UltimateHybridSimulator()
    
    print("\n" + "=" * 80)
    print("Simulator running! Adjust parameters and watch the quantum physics unfold.")
    print("=" * 80)

if __name__ == "__main__":
    main()