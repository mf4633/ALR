"""
Synthetic scenarios for quasi-unsteady sediment transport validation.
"""

from quantum_hydraulics.integration.sediment_transport import (
    ChannelReach, GrainSizeDistribution, QuasiUnsteadyEngine,
)


def generate_clearwater_scour_scenario():
    """
    Clear-water scour below a dam: 500 ft x 40 ft rectangular channel.

    Upstream feed = 0 (dam blocks all sediment supply).
    5-step hydrograph representing ~1 year of flows:
      low (100 cfs, 2000 hrs), moderate (300 cfs, 600 hrs),
      bankfull (600 cfs, 100 hrs), flood (900 cfs, 20 hrs),
      recession (300 cfs, 200 hrs).

    Expected: bed degrades, surface coarsens, armor forms, scour self-limits.
    """
    channel = ChannelReach(
        length_ft=500.0,
        width_ft=40.0,
        slope=0.002,
        roughness_ks=0.1,
        side_slope=0.0,
        bed_elevation=0.0,
    )

    sediment_mix = GrainSizeDistribution.default_sand_gravel()

    hydrograph = [
        (100.0, 2000.0),  # low flow
        (300.0,  600.0),  # moderate
        (600.0,  100.0),  # bankfull
        (900.0,   20.0),  # flood
        (300.0,  200.0),  # recession
    ]

    return channel, sediment_mix, hydrograph, {
        "name": "Clear-Water Scour Below Dam",
        "total_hours": sum(h for _, h in hydrograph),
        "peak_Q": max(q for q, _ in hydrograph),
    }
