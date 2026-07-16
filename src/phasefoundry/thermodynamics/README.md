# thermodynamics

Pre-compute and cache CALPHAD thermodynamic data from pycalphad into a compact HDF5 file for fast reuse in phase-field simulations.

## Why

Calling pycalphad inside a Cahn-Hilliard simulation loop is too slow (~10–100 ms per composition point). This module computes GM, μᵢ, and ∂μᵢ/∂xⱼ over a dense composition grid **once**, stores everything in a compressed HDF5 file, and lets the phase-field solver load it instantly.

## Structure

```
thermodynamics/
├── __init__.py   generate_table(), load_table() — the public API
├── grid.py       simplex_grid() — regular grid on N-component simplex
├── generate.py   compute_table() — pycalphad evaluation loop
└── storage.py    save_table(), load_table(), table_info()
```

## Quick start

```python
import sys
sys.path.insert(0, '..')          # if running from thermodynamics/

from thermodynamics import generate_table, load_table, table_info

# --- Generate (run once, takes a few minutes for n_points=20) ---
data = generate_table(
    db_path='../PhaseFoundry/HF_NB_TI_V.TDB',
    elements=['HF', 'NB', 'TI', 'V'],
    phase='BCC_A2',
    composition_ranges={
        'HF': (0.05, 0.60),
        'NB': (0.05, 0.60),
        'TI': (0.05, 0.60),
    },
    reference_element='V',
    n_points=20,           # 20³ = 8000 candidate points
    temperature=1000,      # K
    output_path='data/hfnbtiv_bcc_1000K.h5',
    n_workers=1,           # increase for parallel evaluation
    quantities=('GM', 'MU', 'dMU'),
)

# --- Load later ---
data = load_table('data/hfnbtiv_bcc_1000K.h5')
print(table_info('data/hfnbtiv_bcc_1000K.h5'))

# compositions: shape (n_valid, 4)   columns = [HF, NB, TI, V]
# data['GM']         shape (n_valid,)   J/mol
# data['MU']['HF']   shape (n_valid,)   J/mol
# data['dMU']['HF_NB']  shape (n_valid,)  J/mol  (d mu_HF / d x_NB)
```

## HDF5 layout

```
/metadata/              attributes: db_path, elements, phase, T, P, n_points, ...
/compositions           float64 (n_valid, N_elements)
/GM                     float64 (n_valid,)             J/mol
/MU/
    HF                  float64 (n_valid,)
    NB                  float64 (n_valid,)
    TI                  float64 (n_valid,)
    V                   float64 (n_valid,)
/dMU/
    HF_HF               float64 (n_valid,)    d mu_HF / d x_HF
    HF_NB               float64 (n_valid,)    d mu_HF / d x_NB
    HF_TI               float64 (n_valid,)
    NB_HF               float64 (n_valid,)
    ...
```

All datasets are gzip-compressed (level 4).

## Notes on the pycalphad calls

- **`IsolatedPhase`** is used for every evaluation. This forces pycalphad to stay on the single-phase free energy surface (no global equilibrium search), which is essential for a phase-field model that needs the Gibbs energy of a *metastable* phase.
- Chemical potentials are **absolute** (J/mol). For the Cahn-Hilliard driving force you need the exchange potential: Δμᵢ = μᵢ - μᵥ — compute this from the stored arrays.
- Grid points where pycalphad returns NaN or throws an exception are silently skipped; `metadata/n_failed` records how many.

## Requirements

```
pycalphad
h5py
numpy
tqdm
```
