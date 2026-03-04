"""
Analytical solutions for validation benchmarks.

These exact solutions are used to validate the vortex particle method
and hydraulics computations.
"""

import numpy as np
from typing import Tuple, Optional


def lamb_oseen_vortex(
    r: np.ndarray,
    t: float,
    gamma: float = 1.0,
    nu: float = 1.1e-5,
) -> np.ndarray:
    """
    Lamb-Oseen vortex - exact solution for decaying vortex.

    The Lamb-Oseen vortex is an exact solution to the Navier-Stokes
    equations for a single viscously decaying vortex.

    Parameters
    ----------
    r : np.ndarray
        Radial distance from vortex center
    t : float
        Time since vortex creation
    gamma : float
        Circulation strength
    nu : float
        Kinematic viscosity

    Returns
    -------
    np.ndarray
        Tangential velocity at each radius

    Notes
    -----
    The exact solution is:
        v_theta = (Gamma / (2 * pi * r)) * (1 - exp(-r^2 / (4 * nu * t)))

    As t -> 0, this approaches a point vortex.
    As t -> inf, the vortex diffuses completely.
    """
    # Avoid division by zero at r=0
    r_safe = np.maximum(r, 1e-10)

    # Core size from viscous diffusion
    r_core_sq = 4.0 * nu * t

    # Tangential velocity
    v_theta = (gamma / (2.0 * np.pi * r_safe)) * (1.0 - np.exp(-r_safe ** 2 / r_core_sq))

    return v_theta


def lamb_oseen_vorticity(
    r: np.ndarray,
    t: float,
    gamma: float = 1.0,
    nu: float = 1.1e-5,
) -> np.ndarray:
    """
    Vorticity distribution for Lamb-Oseen vortex.

    Parameters
    ----------
    r : np.ndarray
        Radial distance
    t : float
        Time
    gamma : float
        Circulation
    nu : float
        Kinematic viscosity

    Returns
    -------
    np.ndarray
        Vorticity at each radius
    """
    r_core_sq = 4.0 * nu * t

    omega = (gamma / (np.pi * r_core_sq)) * np.exp(-r ** 2 / r_core_sq)

    return omega


def poiseuille_velocity(
    y: np.ndarray,
    h: float,
    dp_dx: float,
    mu: float = 2.1e-5,
) -> np.ndarray:
    """
    Poiseuille flow - exact solution for laminar channel flow.

    Parameters
    ----------
    y : np.ndarray
        Cross-stream position (0 to h)
    h : float
        Channel half-height (total height = 2h)
    dp_dx : float
        Pressure gradient (negative for flow in +x direction)
    mu : float
        Dynamic viscosity

    Returns
    -------
    np.ndarray
        Streamwise velocity at each y position

    Notes
    -----
    Exact solution: u(y) = (1/2mu) * (-dp/dx) * (h^2 - y^2)
    Maximum velocity at centerline (y=0): u_max = (h^2 / 2mu) * (-dp/dx)
    """
    u = (1.0 / (2.0 * mu)) * (-dp_dx) * (h ** 2 - y ** 2)
    return np.maximum(u, 0.0)


def poiseuille_profile(
    y: np.ndarray,
    h: float,
    u_max: float,
) -> np.ndarray:
    """
    Poiseuille velocity profile given maximum velocity.

    Parameters
    ----------
    y : np.ndarray
        Cross-stream position (0 to h)
    h : float
        Channel half-height
    u_max : float
        Maximum (centerline) velocity

    Returns
    -------
    np.ndarray
        Velocity profile
    """
    return u_max * (1.0 - (y / h) ** 2)


def kolmogorov_spectrum(
    k: np.ndarray,
    epsilon: float,
    C_k: float = 1.5,
) -> np.ndarray:
    """
    Kolmogorov energy spectrum - the famous -5/3 law.

    Parameters
    ----------
    k : np.ndarray
        Wavenumber (1/length)
    epsilon : float
        Turbulent dissipation rate
    C_k : float
        Kolmogorov constant (typically 1.5)

    Returns
    -------
    np.ndarray
        Energy spectrum E(k)

    Notes
    -----
    In the inertial subrange:
        E(k) = C_k * epsilon^(2/3) * k^(-5/3)

    This is valid for eta << 1/k << L, where eta is Kolmogorov scale
    and L is the integral scale.
    """
    return C_k * epsilon ** (2.0 / 3.0) * k ** (-5.0 / 3.0)


def kolmogorov_scales(
    epsilon: float,
    nu: float = 1.1e-5,
) -> Tuple[float, float, float]:
    """
    Compute Kolmogorov microscales.

    Parameters
    ----------
    epsilon : float
        Turbulent dissipation rate (ft^2/s^3)
    nu : float
        Kinematic viscosity (ft^2/s)

    Returns
    -------
    eta : float
        Kolmogorov length scale (ft)
    tau : float
        Kolmogorov time scale (s)
    v : float
        Kolmogorov velocity scale (ft/s)
    """
    eta = (nu ** 3 / epsilon) ** 0.25
    tau = (nu / epsilon) ** 0.5
    v = (nu * epsilon) ** 0.25

    return eta, tau, v


def wall_vorticity(
    z: np.ndarray,
    u_star: float,
    kappa: float = 0.41,
    z0: float = 0.001,
) -> np.ndarray:
    """
    Wall vorticity from log-law velocity profile.

    Parameters
    ----------
    z : np.ndarray
        Height above wall
    u_star : float
        Friction velocity
    kappa : float
        von Karman constant (0.41)
    z0 : float
        Roughness length

    Returns
    -------
    np.ndarray
        Vorticity (du/dz)

    Notes
    -----
    From the log law u = (u*/kappa) * ln(z/z0):
    omega = du/dz = u* / (kappa * z)
    """
    z_safe = np.maximum(z, z0)
    omega = u_star / (kappa * z_safe)
    return omega


def log_law_velocity(
    z: np.ndarray,
    u_star: float,
    z0: float = 0.001,
    kappa: float = 0.41,
) -> np.ndarray:
    """
    Log-law velocity profile.

    Parameters
    ----------
    z : np.ndarray
        Height above bed
    u_star : float
        Friction velocity
    z0 : float
        Roughness length (ks/30)
    kappa : float
        von Karman constant

    Returns
    -------
    np.ndarray
        Velocity at each height
    """
    z_safe = np.maximum(z, z0)
    return (u_star / kappa) * np.log(z_safe / z0)


def colebrook_white(
    Re: float,
    epsilon_D: float,
    tol: float = 1e-8,
    max_iter: int = 100,
) -> float:
    """
    Solve Colebrook-White equation for friction factor.

    Parameters
    ----------
    Re : float
        Reynolds number
    epsilon_D : float
        Relative roughness (ks / D)
    tol : float
        Convergence tolerance
    max_iter : int
        Maximum iterations

    Returns
    -------
    float
        Darcy-Weisbach friction factor

    Notes
    -----
    Colebrook-White: 1/sqrt(f) = -2*log10(epsilon/(3.7*D) + 2.51/(Re*sqrt(f)))
    """
    if Re < 2300:
        # Laminar flow
        return 64.0 / Re

    f = 0.02  # Initial guess

    for _ in range(max_iter):
        term1 = epsilon_D / 3.7
        term2 = 2.51 / (Re * np.sqrt(f))
        f_new = (-2.0 * np.log10(term1 + term2)) ** (-2)

        if abs(f_new - f) < tol:
            return f_new
        f = f_new

    return f


def energy_balance_check(
    tke_production: float,
    tke_dissipation: float,
    tke_transport: float = 0.0,
) -> Tuple[float, bool]:
    """
    Check turbulent kinetic energy balance.

    In equilibrium: Production = Dissipation + Transport

    Parameters
    ----------
    tke_production : float
        TKE production rate
    tke_dissipation : float
        TKE dissipation rate (epsilon)
    tke_transport : float
        TKE transport rate

    Returns
    -------
    imbalance : float
        Relative imbalance
    balanced : bool
        True if imbalance < 10%
    """
    total_sink = tke_dissipation + tke_transport
    if total_sink > 0:
        imbalance = abs(tke_production - total_sink) / total_sink
    else:
        imbalance = float("inf")

    balanced = imbalance < 0.1

    return imbalance, balanced
