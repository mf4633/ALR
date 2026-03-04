"""
Quick Design Analysis Mode

Get engineering design values without the visualization overhead.
Just input your parameters and get max velocity, shear stress, scour risk, etc.
"""

import numpy as np
from typing import Dict, Optional
from dataclasses import dataclass


@dataclass
class DesignResults:
    """Engineering design results."""

    # Input parameters
    Q: float  # cfs
    width: float  # ft
    depth: float  # ft
    slope: float
    roughness_ks: float  # ft

    # Computed hydraulics
    velocity_mean: float  # ft/s
    velocity_max: float  # ft/s (from log law at surface)
    area: float  # ft2
    hydraulic_radius: float  # ft
    reynolds_number: float
    froude_number: float
    friction_factor: float

    # Turbulence
    friction_velocity: float  # ft/s
    bed_shear_stress: float  # psf
    tke: float  # ft2/s2
    kolmogorov_scale: float  # ft

    # Design recommendations
    scour_risk_index: float  # 0-1
    flow_regime: str
    scour_assessment: str
    velocity_assessment: str

    def __str__(self):
        return f"""
================================================================================
QUANTUM HYDRAULICS - DESIGN ANALYSIS RESULTS
================================================================================

INPUT PARAMETERS:
  Discharge Q:        {self.Q:.1f} cfs
  Channel Width:      {self.width:.1f} ft
  Water Depth:        {self.depth:.2f} ft
  Bed Slope:          {self.slope:.4f}
  Roughness ks:       {self.roughness_ks:.3f} ft

HYDRAULIC RESULTS:
  Mean Velocity:      {self.velocity_mean:.2f} ft/s
  Max Velocity:       {self.velocity_max:.2f} ft/s  (near surface)
  Cross-Section Area: {self.area:.1f} ft²
  Hydraulic Radius:   {self.hydraulic_radius:.2f} ft
  Reynolds Number:    {self.reynolds_number:,.0f}
  Froude Number:      {self.froude_number:.3f}
  Friction Factor:    {self.friction_factor:.5f}
  Flow Regime:        {self.flow_regime}

TURBULENCE & SHEAR:
  Friction Velocity:  {self.friction_velocity:.3f} ft/s
  Bed Shear Stress:   {self.bed_shear_stress:.3f} psf
  Turb. Kinetic Energy: {self.tke:.3f} ft²/s²
  Kolmogorov Scale:   {self.kolmogorov_scale:.6f} ft

DESIGN ASSESSMENT:
  Scour Risk Index:   {self.scour_risk_index:.2f} (0=none, 1=critical)
  Scour Assessment:   {self.scour_assessment}
  Velocity Assessment: {self.velocity_assessment}

================================================================================
"""

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'Q_cfs': self.Q,
            'width_ft': self.width,
            'depth_ft': self.depth,
            'slope': self.slope,
            'roughness_ks_ft': self.roughness_ks,
            'velocity_mean_fps': self.velocity_mean,
            'velocity_max_fps': self.velocity_max,
            'area_ft2': self.area,
            'hydraulic_radius_ft': self.hydraulic_radius,
            'reynolds_number': self.reynolds_number,
            'froude_number': self.froude_number,
            'friction_factor': self.friction_factor,
            'friction_velocity_fps': self.friction_velocity,
            'bed_shear_stress_psf': self.bed_shear_stress,
            'tke_ft2s2': self.tke,
            'kolmogorov_scale_ft': self.kolmogorov_scale,
            'scour_risk_index': self.scour_risk_index,
            'flow_regime': self.flow_regime,
        }


def analyze(
    Q: float,
    width: float,
    depth: float,
    slope: float,
    roughness_ks: float = 0.1,
    side_slope: float = 2.0,
    critical_shear: float = 0.15,
) -> DesignResults:
    """
    Quick design analysis - get engineering values without visualization.

    Parameters
    ----------
    Q : float
        Discharge in cfs
    width : float
        Channel bottom width in ft
    depth : float
        Water depth in ft
    slope : float
        Bed slope (dimensionless)
    roughness_ks : float
        Equivalent sand roughness in ft (default 0.1)
    side_slope : float
        Side slope H:V (default 2.0 for trapezoidal)
    critical_shear : float
        Critical shear stress for scour in psf (default 0.15 for sand)

    Returns
    -------
    DesignResults
        Engineering design results

    Example
    -------
    >>> results = analyze(Q=600, width=30, depth=5, slope=0.002)
    >>> print(results)
    >>> print(f"Max velocity: {results.velocity_max:.2f} ft/s")
    """
    from quantum_hydraulics.core.hydraulics import HydraulicsEngine

    # Compute hydraulics
    h = HydraulicsEngine(
        Q=Q,
        width=width,
        depth=depth,
        slope=slope,
        roughness_ks=roughness_ks,
        side_slope=side_slope,
    )

    # Max velocity from log law at surface
    velocity_max = h.velocity_profile(depth * 0.9)  # Near surface

    # Bed shear stress
    rho = 1.94  # slug/ft3
    bed_shear = rho * h.u_star ** 2

    # Scour risk (0-1)
    scour_risk = min(1.0, bed_shear / critical_shear)

    # Flow regime
    flow_regime = "SUPERCRITICAL" if h.Fr > 1.0 else "SUBCRITICAL"

    # Assessments
    if scour_risk > 0.7:
        scour_assessment = "CRITICAL - Scour protection REQUIRED"
    elif scour_risk > 0.5:
        scour_assessment = "HIGH - Scour protection recommended"
    elif scour_risk > 0.3:
        scour_assessment = "MODERATE - Monitor conditions"
    else:
        scour_assessment = "LOW - Acceptable"

    if velocity_max > 15:
        velocity_assessment = "EXTREME - Energy dissipation REQUIRED"
    elif velocity_max > 10:
        velocity_assessment = "HIGH - Energy dissipation recommended"
    elif velocity_max > 6:
        velocity_assessment = "ELEVATED - Consider energy dissipation"
    else:
        velocity_assessment = "ACCEPTABLE"

    return DesignResults(
        Q=Q,
        width=width,
        depth=depth,
        slope=slope,
        roughness_ks=roughness_ks,
        velocity_mean=h.V_mean,
        velocity_max=velocity_max,
        area=h.A,
        hydraulic_radius=h.R,
        reynolds_number=h.Re,
        froude_number=h.Fr,
        friction_factor=h.friction_factor,
        friction_velocity=h.u_star,
        bed_shear_stress=bed_shear,
        tke=h.TKE,
        kolmogorov_scale=h.eta_kolmogorov,
        scour_risk_index=scour_risk,
        flow_regime=flow_regime,
        scour_assessment=scour_assessment,
        velocity_assessment=velocity_assessment,
    )


def analyze_range(
    Q_range: tuple,
    width: float,
    depth_range: tuple,
    slope: float,
    roughness_ks: float = 0.1,
    n_points: int = 10,
) -> list:
    """
    Analyze a range of flows and depths.

    Returns list of DesignResults for each combination.
    """
    results = []

    Q_values = np.linspace(Q_range[0], Q_range[1], n_points)
    depth_values = np.linspace(depth_range[0], depth_range[1], n_points)

    for Q in Q_values:
        for depth in depth_values:
            try:
                r = analyze(Q, width, depth, slope, roughness_ks)
                results.append(r)
            except Exception:
                pass

    return results


def print_design_table(
    Q_values: list,
    width: float,
    depth: float,
    slope: float,
    roughness_ks: float = 0.1,
):
    """
    Print a table of design values for multiple discharges.
    """
    print("\n" + "=" * 90)
    print("DESIGN TABLE - Varying Discharge")
    print("=" * 90)
    print(f"Width: {width} ft | Depth: {depth} ft | Slope: {slope} | ks: {roughness_ks} ft")
    print("-" * 90)
    print(f"{'Q (cfs)':>10} {'V_mean':>10} {'V_max':>10} {'Fr':>8} {'Shear':>10} {'Scour':>8} {'Risk':<15}")
    print(f"{'':>10} {'(ft/s)':>10} {'(ft/s)':>10} {'':>8} {'(psf)':>10} {'Index':>8} {'':<15}")
    print("-" * 90)

    for Q in Q_values:
        r = analyze(Q, width, depth, slope, roughness_ks)
        risk_flag = "***" if r.scour_risk_index > 0.5 else ""
        print(f"{r.Q:>10.0f} {r.velocity_mean:>10.2f} {r.velocity_max:>10.2f} "
              f"{r.froude_number:>8.3f} {r.bed_shear_stress:>10.3f} "
              f"{r.scour_risk_index:>8.2f} {risk_flag:<15}")

    print("=" * 90)


# CLI interface
def main():
    """Command-line interface for quick analysis."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Quick hydraulic design analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m quantum_hydraulics.analysis --Q 600 --width 30 --depth 5 --slope 0.002
  python -m quantum_hydraulics.analysis --Q 600 --width 30 --depth 5 --slope 0.002 --ks 0.15
        """
    )

    parser.add_argument("--Q", type=float, required=True, help="Discharge (cfs)")
    parser.add_argument("--width", type=float, required=True, help="Channel width (ft)")
    parser.add_argument("--depth", type=float, required=True, help="Water depth (ft)")
    parser.add_argument("--slope", type=float, required=True, help="Bed slope")
    parser.add_argument("--ks", type=float, default=0.1, help="Roughness ks (ft)")
    parser.add_argument("--json", action="store_true", help="Output as JSON")

    args = parser.parse_args()

    results = analyze(
        Q=args.Q,
        width=args.width,
        depth=args.depth,
        slope=args.slope,
        roughness_ks=args.ks,
    )

    if args.json:
        import json
        print(json.dumps(results.to_dict(), indent=2))
    else:
        print(results)


if __name__ == "__main__":
    main()
