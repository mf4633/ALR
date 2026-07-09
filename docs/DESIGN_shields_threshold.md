# Critical-shear anchoring: Shields incipient motion vs. scour-design threshold

STATUS: IMPLEMENTED

## Problem

`SedimentProperties` carried a single hard-coded `critical_shear_psf` that was
overloaded onto two physically distinct roles:

1. the incipient-motion threshold feeding the **Shields parameter** and
   **Meyer-Peter-Muller** transport, and
2. the reference shear centring the empirical **logistic scour-risk** classifier.

The hard-coded sand value (0.10 psf) was ~15-19x higher than the true Shields
incipient-motion shear. Fed into `theta_c = tau_c / ((rho_s - rho) g d)` it
implied `theta_c ~ 0.6` for 0.5 mm sand -- physically impossible (the Shields
curve gives ~0.03) -- and suppressed MPM transport until ~15x incipient shear.

## Resolution (most-academic anchor)

Split the two thresholds and derive the physical one from first principles:

- **`critical_shear_psf`** -- incipient motion, from the **Shields curve** via
  the **Soulsby & Whitehouse (1997)** explicit fit
  `theta_c = 0.30/(1 + 1.2 D*) + 0.055 [1 - exp(-0.020 D*)]`,
  `D* = d ((s-1) g / nu^2)^(1/3)`. This is the canonical dimensionless
  threshold, calibration-free. Applied to the non-cohesive presets
  (fine_sand, sand, coarse_sand, gravel). See
  `shields_critical_shear_psf()` in `integration/swmm_node.py`.

- **`scour_design_shear_psf`** -- the empirical, HEC-18-derived design shear
  the risk logistic is centred on (defaults to the prior hard-coded values).
  Incipient motion is not a scour-design hazard, so this stays separate and the
  calibrated classifier behaviour is preserved exactly (no field data was
  re-fit or fabricated).

**Cohesion:** the Shields curve is non-cohesive. `silt` and `clay` retain
empirical cohesive erosion thresholds; Shields is not applied to them.

## Effect

- Reported Shields parameters now fall in the classic 0.02-0.10 band; MPM
  transport begins at true incipient motion.
- Scour-risk classifier output is byte-for-byte unchanged (design threshold
  preserved).
- Verified: `pytest` 284 passed; scour benchmarks 42/42; sediment 10/10;
  ALR study 25/25. New tests assert the presets imply valid Shields parameters.

## References

Soulsby, R. L., & Whitehouse, R. J. S. (1997). Threshold of sediment motion in
coastal environments. *Proc. Pacific Coasts and Ports '97*, 149-154.
