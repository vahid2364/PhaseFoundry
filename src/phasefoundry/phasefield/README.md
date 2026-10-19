# phasefield

Pseudo-spectral Cahn-Hilliard phase-field solver for binary alloys, coupled to CALPHAD thermodynamics via a G_m spline surrogate.

## Why

Standard phase-field implementations use synthetic free energy models (double-well, Landau polynomials) that are not connected to real thermodynamics. This module drives the Cahn-Hilliard equation directly from a CALPHAD-derived G_m spline, so the simulated microstructure evolution is quantitatively tied to the alloy's actual free energy landscape.

## Structure

```
phasefield/
├── __init__.py       public API — solve_cahn_hilliard_binary(), plot_result(), etc.
├── binary.py         Cahn-Hilliard solver, stable_dt(), write_simulation_params()
├── free_energy.py    free energy models: double_well, calphad, calphad_gm_spline
└── spectral.py       FFT Laplacian operator, wavenumber grid
```

## Numerical method

Semi-implicit pseudo-spectral scheme on a 2-D periodic domain:

```
ĉⁿ⁺¹ = ( ĉⁿ + M Δt k² μ̂_bulk ) / ( 1 + M κ Δt k⁴ )
```

- Gradient energy term `κ k⁴` is treated **implicitly** — unconditionally stable with respect to interfacial dynamics
- Bulk chemical potential term `μ̂_bulk` is treated **explicitly** — subject to the stability criterion:

```
M · |d²G/dx²|_max · Δt / Δx²  <  0.5
```

Use `dt=None` to let `stable_dt()` select the time step automatically.

## Quick start

### Double-well benchmark (no thermodynamic data required)

```python
from phasefoundry.phasefield import solve_cahn_hilliard_binary, plot_result

result = solve_cahn_hilliard_binary(
    free_energy = 'double_well',
    N=256, Lx=200.0, kappa=0.5, M=1.0,
    c0=0.4, dt=0.1, n_steps=2000, save_every=200,
)
plot_result(result)
```

### CALPHAD-coupled (Nb-V at 500 °C)

```python
from phasefoundry.thermodynamics.fitting import build_gm_spline_surrogate
from phasefoundry.phasefield import solve_cahn_hilliard_binary, plot_free_energy_history

fit = build_gm_spline_surrogate(['data/nb_v/fine/nb_v_bcc_500C.h5'])

result = solve_cahn_hilliard_binary(
    free_energy   = 'calphad_gm_spline',
    N=256, Lx=1e-6,          # 1 μm × 1 μm domain
    kappa=1e-12,             # gradient energy coefficient  [J·m²/mol]
    M=5e-21,                 # mobility  [m²·mol/(J·s)]
    c0=0.4,                  # mean composition  (mol fraction V)
    dt=None,                 # auto-select stable dt
    n_steps=30000, save_every=5000,
    calphad_fit=fit, temperature_K=773.15,
    output_dir='results/nb_v/my_run', save_frames=True,
)
plot_free_energy_history(result, output_dir='results/nb_v/my_run')
```

## Free energy models

| `free_energy` | Description | Requires |
|---|---|---|
| `double_well` | Synthetic polynomial double-well — for benchmarking | nothing |
| `calphad` | Polynomial surrogate for Δμ | `load_fits()` output |
| `calphad_spline` | Spline surrogate for Δμ | `build_spline_surrogate()` output |
| `calphad_gm_spline` | CubicSpline fitted to G_m — **recommended** | `build_gm_spline_surrogate()` output |

## Key parameters

| Parameter | Symbol | Units | Notes |
|---|---|---|---|
| `kappa` | κ | J·m²/mol | Gradient energy coefficient. Controls interface width δ ≈ √(κ/\|d²G/dx²\|) |
| `M` | M | m²·mol/(J·s) | Prefactor of the degenerate mobility M c (1 − c); sets the time scale |
| `dt` | Δt | s | Time step. `None` = auto stable dt |
| `Lx` | L | m | Domain size. Must contain several spinodal wavelengths λ* |
| `N` | — | — | Grid points per dimension. Interface should span ≥ 5 grid points |

## Output files (when `output_dir` is set)

```
output_dir/
├── frame_000000.png  …  frame_NNNNNN.png   composition snapshots
├── free_energy_log.txt    step, time, F_total, F_bulk, F_grad, dG_mix, c_min, c_max
├── free_energy_history.png
└── sim_params.txt         δ, σ, λ*, λ_c, stability ratio, all input parameters
```

## Physical length scales

`write_simulation_params()` computes and reports these automatically:

| Quantity | Formula | Meaning |
|---|---|---|
| Interface width | δ = √(κ / \|d²G/dx²\|_max) | Must be ≥ 1–2 × Δx |
| Interface energy | σ = ∫√(2κ ΔG_mix) dc | J/m² |
| Most unstable wavelength | λ* = 4π √(κ / \|d²G/dx²\|_c₀) | Must fit inside domain |
| Critical wavelength | λ_c = 2π √(2κ / \|d²G/dx²\|_c₀) | Onset of instability |

## Requirements

```
numpy
scipy
matplotlib
tqdm
```
