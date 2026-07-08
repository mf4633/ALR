# Review Findings — Round 3 (PCSWMM integration, visualization)

Clean correctness fixes were applied (see the commit). The items below are
documented rather than fixed because they need a testing/methodology decision or
a runtime environment (SWMM/pyswmm) not available here.

## Fixed (this round)

- **`pcswmm_script.py`:**
  - Added a unit-system guard: warns loudly when the SWMM model's `FLOW_UNITS`
    is not US (CFS/GPM/MGD), since `QuantumNode` is hardwired to imperial units
    and a metric (CMS/LPS) model would otherwise be silently mis-analyzed.
  - Errors are now surfaced once per node (previously only if they fired on
    timestep 1, when nodes are dry — so real errors on wet timesteps were never
    printed).
  - Multiple `.inp` files are now selected deterministically (sorted) with a
    loud warning, instead of arbitrary `os.listdir` order.
- **`visualization/renderers.py`:**
  - Fixed the inverted plan-view marker-size legend (marker area is
    `1200/sigma^2`, so a small sigma draws a LARGE dot; the legend labeled it
    backwards on every figure).
  - Guarded the energy colorbar `vmax` against NaN/inf (`nanpercentile` + finite
    check).
- **`visualization/export.py`:** guarded `n_frames <= 0` and `fps <= 0`
  (`interval = 1000 // fps` divided by zero).

## Documented, not fixed

1. **SI-model support (`pcswmm_script.py`, `PCSWMM_Quantum_AutoDetect.py`).** The
   guard above only *warns*. Full support requires converting `node.depth`
   (m→ft) and `node.total_inflow` (cms/lps→cfs) at ingestion, plus converting the
   `MIN_DEPTH`/`MIN_INFLOW` gates. Left unfixed because it can't be validated
   without a metric SWMM model + pyswmm here; the same guard should be added to
   `PCSWMM_Quantum_AutoDetect.py`.

2. **Frozen turbulence field across the storm (`swmm_node.py` compute_turbulence).**
   The `QuantumNode` particle cloud is injected once (deficit = 500-500 = 0
   thereafter) and its vorticity strengths are never rescaled as flow ramps, so
   the induced-velocity-derived metrics (tke, the Reynolds-stress bed-shear
   component, induced part of max velocity) stay locked to first-wet-timestep
   conditions. Peak-over-timeseries then under/mis-represents true peak
   turbulence. Touches the empirically-calibrated 2D/1D path (see
   RECALIBRATION_SPEC.md), so not fixed in isolation.

3. **Energy-spectrum figure is not an energy density (`renderers.py` plot_energy_spectrum).**
   Per-particle energy is summed over unequal (log-width) scale bins without
   dividing by bin width or population, and the Kolmogorov reference is then
   force-normalized to the data at the midpoint — so the "-5/3 comparison" is
   driven by binning/population, not a true E(k). Additionally the -5/3 reference
   is plotted against length scale (x-axis "Length Scale (ft)"), on which the law
   visually slopes +5/3 while labeled "-5/3". Making this a real spectrum needs
   density normalization and plotting versus wavenumber; it may then reveal the
   particle method does not reproduce -5/3 (same caveat as the dormant
   `benchmarks.py` spectrum test).

4. **Animation double-steps frame 0 (`export.py`).** `FuncAnimation` (no
   `init_func`, `blit=False`) draws the first frame via `update()` before the
   loop, so frame 0's physics is integrated twice and total simulated time is
   `(n_frames+1)*steps_per_frame*dt`. The `export_frames`/`export_single_frame`
   paths (manual loop) are correct and diverge from the animation for the same
   inputs. Fix needs an `init_func` and verification by rendering — deferred.

5. **Negative/reverse flows excluded (`pcswmm_script.py`).** The
   `inflow > MIN_INFLOW` gate drops reverse flows, which can still scour. A
   defensible design threshold; confirm it matches intent.
