"""
QUICK HYDRAULIC ANALYSIS
========================
Simple script for design values. Run from PCSWMM or command line.

Just edit the values below and run!
"""

# ============================================================================
# EDIT YOUR VALUES HERE
# ============================================================================

Q = 600          # Discharge (cfs)
WIDTH = 30       # Channel bottom width (ft)
DEPTH = 5        # Water depth (ft)
SLOPE = 0.002    # Bed slope (ft/ft)
ROUGHNESS = 0.15 # Roughness ks (ft) - use D84 for gravel

# Optional: Analyze multiple flows at once
ANALYZE_RANGE = True
Q_VALUES = [200, 400, 600, 800, 1000, 1500]

# ============================================================================
# DON'T EDIT BELOW THIS LINE
# ============================================================================

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from quantum_hydraulics.analysis import analyze, print_design_table

def main():
    print("\n" + "=" * 70)
    print("QUANTUM HYDRAULICS - QUICK DESIGN ANALYSIS")
    print("=" * 70)

    # Single point analysis
    r = analyze(Q=Q, width=WIDTH, depth=DEPTH, slope=SLOPE, roughness_ks=ROUGHNESS)
    print(r)

    # Range analysis
    if ANALYZE_RANGE and Q_VALUES:
        print_design_table(
            Q_values=Q_VALUES,
            width=WIDTH,
            depth=DEPTH,
            slope=SLOPE,
            roughness_ks=ROUGHNESS
        )

    # Export key values
    print("\n" + "=" * 70)
    print("KEY DESIGN VALUES (copy these)")
    print("=" * 70)
    print(f"  Max Velocity:      {r.velocity_max:.2f} ft/s")
    print(f"  Mean Velocity:     {r.velocity_mean:.2f} ft/s")
    print(f"  Bed Shear Stress:  {r.bed_shear_stress:.3f} psf")
    print(f"  Froude Number:     {r.froude_number:.3f}")
    print(f"  Scour Risk Index:  {r.scour_risk_index:.2f}")
    print(f"  Assessment:        {r.scour_assessment}")
    print("=" * 70)

if __name__ == "__main__":
    main()
