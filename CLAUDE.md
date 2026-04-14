# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a computational physics project studying **reaction-driven self-oscillation in thermoresponsive (LCST) hydrogels**. The model is a 1D thermo-chemo-poroelastic PDE system where an exothermic reaction heats a PNIPAM-like gel above its LCST, triggering volume collapse, which cools and re-swells the gel to produce autonomous oscillation. The paper targets APS Physical Review E (RevTeX format).

## Running the Code

```bash
# Default parameter scan (Bi_c × Da, 14×14 grid, N=51 spatial points, ~20 workers)
python Code/scan_optimized.py

# Quick test scan
python Code/scan_optimized.py --nx 8 --ny 8

# High-resolution scan with custom parameters
python Code/scan_optimized.py --N 121 --workers 8 --t-end 500

# Scan a different parameter pair with physical overrides
python Code/scan_optimized.py --x-param S_chi --x-min 0.5 --x-max 2.0 --nx 14 \
                               --BiT 0.10 --Da 13.5
```

Key CLI flags: `--x-param/--y-param` (any `Params` field), `--x-min/--x-max/--nx`, `--y-min/--y-max/--ny`, `--N` (grid), `--t-end`, `--workers`, `--outdir`. Physical overrides: `--BiT`, `--BiC`, `--GammaA`, `--Schi`, `--Da`, `--alpha`, `--arrh-cap`.

Output goes to `--outdir` (default `scan_results_optimized/`): `scan_results.csv` + phase/oscillation/period/amplitude PNG maps.

## Model Architecture

**State vector** (3N unknowns): `y = [logJ₀..N₋₁, W₀..N₋₁, θ₀..N₋₁]`

- `logJ = log(J)` — log of local swelling ratio (ensures J > 0 for FD Jacobian stability)
- `W = J·u` — conserved reactant variable (prevents spurious non-conservation when J changes)
- `θ` — dimensionless temperature rise

**PDE system** (Lagrangian slab, x ∈ [0,1]):
```
J_t = -q_x,        q = -M(J,θ) · m_x
m = f(J,θ) - ℓ² J_xx          (Flory-Huggins + elastic network + Cahn-Hilliard regularization)
(Ju)_t + n_x = -Da · J · R    (reactant, upwind advection u·q, diffusion δD·u_x)
C(J)·θ_t + h_x = Da · J · R   (heat, conduction α·K·θ_x)
```

**Boundary conditions**: x=0 symmetric (zero flux); x=1 Robin with Biot numbers:
- `q = Bi_μ·(m − m_b)` — solvent exchange
- `n = Bi_c·(u − 1)` — reactant supply (bath = 1)
- `h = Bi_T·θ` — heat loss

**Key dimensionless groups** (from `Params` dataclass):
| Parameter | Physical meaning |
|-----------|-----------------|
| `Da` | Damköhler number (reaction vs. swelling timescale) |
| `Bi_c` | Reactant surface transfer (supply rate) |
| `Bi_T` | Heat Biot number (cooling rate) |
| `S_chi` | LCST sensitivity (dχ/dT × ΔT*) |
| `Gamma_A` | Arrhenius amplification |
| `alpha` | Thermal diffusivity ratio τ_s/τ_T |
| `delta` | Reactant diffusivity ratio D₀/(M₀μ*) |
| `ell` | Interface regularization (√(κ_J/μ*H₀²)) |
| `phi_p0`, `chi_inf` | Working-point proximity to LCST |

## Numerical Implementation

**Spatial**: Cell-centered finite volume on uniform grid. Face mobilities use harmonic mean (not arithmetic—critical for degenerate regimes). Advective flux `u·q` uses upwind. The Laplacian uses ghost cells for Neumann BCs: `J₀ = J₁`, `J_{N+1} = J_N`.

**Time integration**: `scipy.integrate.solve_ivp` with `method="BDF"`, using an explicit sparse Jacobian (`make_jac_sparsity` + `jac_sparse`). The Jacobian is a 3N×3N banded matrix with block bandwidth 1–2; only ~10% of columns are nonzero.

**Parallel scan**: `multiprocessing.Pool` with `spawn` context (safe for CUDA/fork issues). Each worker receives `Params` via `_worker_init`. `finalize_params` auto-sets `m_b` (bath chemical potential) from `J_init` and `theta_init` before each run.

**Oscillation detection** (`oscillation_metrics`): Detrends the tail (last 40%) of `<θ>(t)` and `<J>(t)`, finds peaks with `scipy.signal.find_peaks`, requires ≥2 peaks + ≥2 troughs + peak interval CV < 0.40.

**Classification** (`classify_run`): Labels each run as `oscillatory_nonuniform`, `oscillatory_uniform`, `steady_{cold/warm/hot}_{uniform/nonuniform}`, or `solve_failed`.

## Directory Layout

```
Code/           Active simulation code
  scan_optimized.py   Main script: parameter scan + diagnostics + plots
Data/           .npz simulation outputs (named by parameter values)
Figure/         Publication figures (.png)
Paper/          LaTeX manuscript (RevTeX4-2, targeting APS PRE)
References/     Model derivation notes (gel_model1-3.md: physics, nondimensionalization, discretization)
archive/        Earlier scripts and data (not used in current workflow)
```

## Physics Notes

Two oscillatory regimes (from paper):
- **Regime I** (weak S_chi): large-amplitude (ΔJ ≳ 0.5) relaxation oscillator
- **Regime II** (strong S_chi): small-amplitude (ΔJ ≲ 0.1) near-harmonic oscillator

The oscillation mechanism: reaction heat → T rises → χ increases → gel collapses (J↓) → reactant transport blocked → reaction quenches → T drops → gel re-swells → cycle repeats. Oscillation requires fast thermal positive feedback and slow transport-mediated negative feedback.

To enhance interior reactant penetration: reduce `Da` and increase `D0` (and `m_diff` down) while staying in the oscillatory window.

## Complex Oscillation Classification

### Phase diagram labels
The phase diagram classifies each parameter point as one of: **steady / P1 / P2 / P3 / … / Pn / chaos**. Period-n means the oscillation repeats every n macro-cycles (period-doubling cascade). MMO (mixed-mode oscillation) substructure is **not** a separate classification dimension in the phase diagram (see below).

### MMO substructure — spatial effect, not a separate phase
All oscillating states in this model exhibit MMO-like waveforms at interior nodes: each macro-cycle contains 1 large J-spike followed by a variable number of small sub-spikes. This is a **spatial phenomenon** driven by the u-diffusion front propagating inward from the x=L chemostatic boundary:

1. After each large spike, u is depleted everywhere (u ≈ 0 in the interior).
2. u diffuses inward from x=L (Robin BC: `n = Bi_c·(u−1)`).
3. As the front penetrates, it triggers sub-threshold local re-activations (small J bumps) at interior nodes before the next full macro-spike fires.
4. The number of small bumps per cycle increases toward the interior (x=0 has 0–1; x/L≈0.5–0.9 has 2–3; x=L has 0 because u is always replenished there).

Consequence: **small peaks at interior nodes are NOT period-doubling signals** — they are present in P1 and P2 alike.

### Correct probe point for classification: x=L
Use the **x=L boundary node** (node index `N-1`) for all oscillation type detection:

| Property | x=0 | **x=L** |
|----------|-----|---------|
| MMO contamination | Yes (0–1 small peaks/cycle) | **None** |
| Peak detection | Needs adaptive prominence filter | Simple `find_peaks` |
| P-n signal carrier | J amplitude alternation (noisy) | **Inter-peak interval alternation** |
| Reliability | Sensitive to threshold choice | Threshold-free |

At x=L, J oscillates cleanly between a low value (gel contracted, near bath) and a high value (gel swollen). The boundary condition prevents MMO. Period-n oscillation manifests as n distinct inter-peak intervals cycling in order.

### P2 detection criterion at x=L
Compute the inter-peak interval sequence `T = [T₁, T₂, T₃, …]` at x=L (last 50% of simulation, after transients).

- **P1**: all Tᵢ ≈ constant (CV < 0.10)
- **P2**: Tᵢ strictly alternates T_long ↔ T_short; criterion: `T_long/T_short > 1.5` AND the alternating pattern is consistent (`ac1(T) < −0.5`)
- **Pn**: n distinct interval values cycling in order (use k-means with k=n then verify cyclic order)
- **chaos**: intervals irregular, no cyclic pattern (permutation entropy > 0.8)

### Known reference cases (N=40, Bi_c=0.70, Bi_T=0.10, Gamma_A=1.5)
| Da | N | Type | T_long / T_short | Notes |
|----|---|------|-----------------|-------|
| 9.5 | 40 | P1 | — | T ≈ 32τ uniform |
| 11.0 | 40 | P1 | — | T ≈ 44τ uniform |
| 12.0 | 60 | **P2** | 50τ / 25τ = 2.0 | Stable, ac1(T)=−0.99 |
| 13.5 | 40 | transient P2 | 85τ / 40τ = 2.1 | Reverts to P1 after ~1400τ; not stable |
