<div align="center">

# PhaseFoundry

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/license/mit)
![Python](https://img.shields.io/badge/python-3.12%20%7C%203.13-blue)
![Platforms](https://img.shields.io/badge/platform-linux%20%7C%20macos-lightgrey)

[![Tests](https://github.com/vahid2364/PhaseFoundry/actions/workflows/test.yml/badge.svg)](https://github.com/vahid2364/PhaseFoundry/actions/workflows/test.yml)
[![Lint](https://github.com/vahid2364/PhaseFoundry/actions/workflows/lint.yml/badge.svg)](https://github.com/vahid2364/PhaseFoundry/actions/workflows/lint.yml)

`PhaseFoundry` couples CALPHAD thermodynamics to phase-field simulations of spinodal decomposition in binary, ternary and quaternary alloys. A YAML config selects the alloy system, so switching between them needs no code changes. Starting from a CALPHAD database, such as one generated with [PhaseForge](https://github.com/dogusariturk/PhaseForge), `pycalphad` tabulates Gibbs energies, chemical potentials and their analytic composition derivatives (Jansson derivatives). A pseudo-spectral Cahn-Hilliard solver then evolves the microstructure from those tables. An optional FFT elasticity solver adds coherency strain, using composition-dependent elastic constants from SQS calculations with machine-learned interatomic potentials (MLIPs).

<p>
  <a href="https://github.com/vahid2364/PhaseFoundry/issues/new?labels=bug">Report a Bug</a> |
  <a href="https://github.com/vahid2364/PhaseFoundry/issues/new?labels=enhancement">Request a Feature</a>
</p>

</div>

---

## Features

- Spinodal stability analysis via gauge-invariant eigenvalues and eigenvectors of the chemical and coherent Gibbs energy Hessian
- Jansson-derivative-based chemical potential evaluation directly from CALPHAD datasets, bypassing finite-difference approximations
- Semi-implicit pseudo-spectral Cahn–Hilliard solver with Eyre convex-splitting stabilization (per-component A_i)
- Elasto-chemical phase-field coupling via composition-dependent eigenstrain and Moulinec–Suquet spectral elasticity solver
- SQS/MLIP elastic constant interpolation across quaternary composition space (supports ORB-v2, GRACE-2L, and other MLIPs)
- MLIP inputs computed in-pipeline via [MaterialsFramework](https://github.com/dogusariturk/MaterialsFramework): SQS elastic constants, bond-length lattice-parameter model, and PhaseForge-style TDB generation (optional)
- Config-driven pipeline: binary, ternary, and quaternary alloy systems with no code changes required
- GPU acceleration via CuPy (optional) and incremental NPZ checkpointing for long runs
- Affine projection visualization of spinodal regions in high-dimensional composition spaces

---

## Installation

PhaseFoundry runs on Python 3.12 and 3.13.

The pipeline scripts and example configs live in the repository, so clone it and install the package from the checkout:

```sh
git clone https://github.com/vahid2364/PhaseFoundry.git
cd PhaseFoundry

# Recommended: uv
uv sync

# Alternative: pip
pip install -e .
```

Two optional extras add features that need more packages:

| Extra  | Installs                                                                 | Needed for                                      |
|--------|--------------------------------------------------------------------------|-------------------------------------------------|
| `mlip` | [MaterialsFramework](https://github.com/dogusariturk/MaterialsFramework) | Computing TDBs and elastic constants with MLIPs |
| `gpu`  | CuPy for CUDA 12 (not on macOS)                                          | Running the solver on a GPU (experimental)      |

```sh
# Using uv
uv sync --extra mlip

# Using pip
pip install -e ".[mlip]"
```

> [!NOTE]
> MaterialsFramework requires Python 3.12, and the `mlip` extra does not include an interatomic potential. Install the MaterialsFramework extra for the potential you use, for example `uv pip install "materialsframework[orb]"` or `pip install "materialsframework[orb]"`. A later `uv sync` removes packages that are not in the lock file unless you pass `--inexact`. Step 0 also needs [ATAT](https://axelvandewalle.github.io/www-avdw/atat/) with `sqs2tdb` on `PATH` and `~/.atat.rc` pointing at the ATAT installation.

---

## Pipeline

```
PhaseForge TDB
   └─ Step 0   ─  MLIP + ATAT sqs2tdb → TDB  (optional, PhaseForge approach)
   └─ Step 1   ─  pycalphad → HDF5 thermodynamic tables  (G_m, μ, ∂μ/∂x)
   └─ Step 1b  ─  SQS/MLIP  → elastic constant tables (C_ij, lattice parameter)
                  read from CSV, or computed with an MLIP; optional bond-length a(x)
   └─ Step 2   ─  HDF5 → scattered interpolant surrogate + spinodal boundaries
   └─ Step 3   ─  surrogate → Cahn–Hilliard PFM simulation (pseudo-spectral)
   └─ Step 4   ─  simulation outputs → post-processing and field extraction
   └─ Step 5   ─  surrogate + derivatives → thermodynamic validation figures  (optional)
   └─ Step 6   ─  chemical-only vs. elasto-chemical → comparison figures
```

---

## Running the Pipeline

All runs are driven by a YAML config file. To model a new alloy system, copy an existing config and edit it — no Python code changes needed.

```bash
# Step 0 (optional): generate the TDB from MLIP SQS energies (needs a tdb_generation section)
python workflows/00_generate_tdb/run.py --config configs/nb_v_mlip_c50.yaml

# Step 1: generate thermodynamic tables
python workflows/01_generate_thermo_data/run.py --config configs/nb_v_chem_c50.yaml

# Step 1b: look up (CSV) or compute (MLIP) elastic constants (elasto-chemical runs only)
python workflows/01b_lookup_elastic/run.py --config configs/nb_hf_ti_v_elastic_cNb25V25Hf25Ti25.yaml

# Step 1c: compute CALPHAD tie-line endpoints (elasto-chemical runs only)
python workflows/01c_calphad_tieline/run.py --config configs/nb_hf_ti_v_elastic_cNb25V25Hf25Ti25.yaml

# Step 2: fit surrogate interpolant + spinodal boundaries
python workflows/02_fit_surrogate/run.py --config configs/nb_v_chem_c50.yaml

# Step 3: run Cahn–Hilliard simulation
python workflows/03_simulate/run.py --config configs/nb_v_chem_c50.yaml

# Step 4: post-process simulation outputs
python workflows/04_postprocess/run.py --config configs/nb_v_chem_c50.yaml

# Step 5: validate thermodynamic tables and derivatives (optional)
python workflows/05_validate/run.py --config configs/nb_v_chem_c50.yaml

# Step 6: compare chemical-only vs. elasto-chemical results
python workflows/06_compare/run.py --config configs/nb_v_elastic_c50.yaml

# Or run the full pipeline at once
python workflows/run_all.py --config configs/nb_v_chem_c50.yaml
```

---

## Configuration Reference

All runs are configured by a single YAML file. Four modes are supported; examples below use the canonical configs shipped with the repository.

---

### Mode 1 — Binary, Chemical-Only, 2D
*Example: `configs/nb_v_chem_c50.yaml`*

```yaml
system:
  name:              nb_v
  db_path:           data/tdb/HfNbTiV_BCC_PhaseForge_v2.tdb
  elements:          [NB, V]           # V is the reference element
  phase:             BCC_A2
  reference_element: V

thermo:
  composition_ranges:
    NB: [0.001, 0.999]
  n_points:          200               # number of grid points per axis
  temperatures_C:    [500]

simulation:
  temperature_C: 500
  dimensions:    2
  N:             128                   # grid size (N × N)
  Lx:            1.0e-6               # domain side length [m]
  kappa:         1.0e-12              # gradient energy coefficient [J·m²/mol]
  M:             5.0e-21              # scalar mobility [m²·mol/(J·s)]
  c0:            0.50                 # mean x_NB (x_V = 1 − x_NB)
  noise:         0.01                 # amplitude of initial composition noise
  seed:          42
  dt:            1.0e-2               # time step [s]
  derivative_method: jansson          # jansson | finite_diff | hybrid
  use_gpu:           false
  adaptive_dt:       false
  n_steps:       100000
  save_every:    10000

output:
  data_dir:    data/HfNbTiV_BCC_PhaseForge_v2/NbV
  results_dir: results/HfNbTiV_BCC_PhaseForge_v2/NbV
  run_name:    auto                   # auto = timestamp-based
```

---

### Mode 2 — Binary, Chemical-Only, 3D
*Example: `configs/nb_v_chem_c50_3d.yaml`*

Same as Mode 1 with these changes:

```yaml
simulation:
  dimensions:  3                      # 3D simulation (N × N × N)
  dt:          null                   # null = solver computes stable dt automatically
  n_steps:     10001
  save_every:  2000
```

---

### Mode 3 — Ternary, Elasto-Chemical, 2D
*Example: `configs/nb_hf_v_elastic_cNb50V25Hf25.yaml`*

```yaml
system:
  name:              nb_hf_v
  db_path:           data/tdb/HfNbTiV_BCC_PhaseForge_v2_Tdep.tdb
  elements:          [NB, HF, V]      # V is reference; NB and HF are independent fields
  phase:             BCC_A2
  reference_element: V

thermo:
  composition_ranges:
    NB: [0.001, 0.999]
    HF: [0.001, 0.999]
  n_points:          20               # 20×20 grid → ~210 valid simplex points
  temperatures_C:    [500]

simulation:
  temperature_C: 500
  dimensions:    2
  N:             128
  Lx:            1.0e-6

  kappa:         [1.0e-12, 7.0e-12]  # [κ_NB, κ_HF]  [J·m²/mol]

  M:                                  # 2×2 mobility matrix [m²·mol/(J·s)]
    - [5.0e-21, 0.0    ]             # M_HF ~100× smaller: Hf diffuses slower at 500 °C
    - [0.0,     5.0e-23]

  c0:            [0.50, 0.25]         # [x_NB, x_HF]  (x_V = 1 − x_NB − x_HF)
  noise:         0.01
  seed:          42
  dt:            0.05
  derivative_method: jansson
  use_gpu:           false
  adaptive_dt:       false
  n_steps:       600000
  save_every:    100000
  incremental_npz: true

  elastic:
    enabled:           true
    stiffness_model:   cubic_csv        # reads C_ij from CSV, interpolates to local composition
    eigenstrain_model: local_vegard     # Vegard eigenstrain: ε* = Λ · δc
    elastic_csv:       data/HfNbTiV_BCC_PhaseForge_v2/sqs_elastic_constants_HfNbVTi-orb-v2.csv

    # Tie-line endpoint compositions (from CALPHAD equilibrium at 500 °C via step 1b)
    c_mat: {NB: 0.70, HF: 0.10, V: 0.20}   # matrix (Nb-rich)
    c_ppt: {NB: 0.15, HF: 0.35, V: 0.50}   # precipitate (V/Hf-rich)

    solve_every: 10        # re-solve elasticity every N time steps
    warm_start:  true      # initialize elastic solver from previous solution
    tol:         1.0e-5    # convergence tolerance for iterative solver
    max_iter:    500

    save_fields: true      # save stress/strain fields to NPZ

output:
  data_dir:    data/HfNbTiV_BCC_PhaseForge_v2/NbHfV
  results_dir: results/HfNbTiV_BCC_PhaseForge_v2/NbHfV
  run_name:    auto
```

---

### Mode 4 — Quaternary, Elasto-Chemical, 2D
*Example: `configs/nb_hf_ti_v_elastic_cNb25V25Hf25Ti25.yaml`*

```yaml
system:
  name:              nb_hf_ti_v
  db_path:           data/tdb/HfNbTiV_BCC_PhaseForge_v2_Tdep.tdb
  elements:          [NB, HF, TI, V]  # V is reference; NB, HF, TI are independent fields
  phase:             BCC_A2
  reference_element: V

thermo:
  composition_ranges:
    NB: [0.001, 0.999]
    HF: [0.001, 0.999]
    TI: [0.001, 0.999]
  n_points:          10               # 10×10×10 grid → ~220 valid simplex points
  temperatures_C:    [500]

simulation:
  temperature_C: 500
  dimensions:    2
  N:             128
  Lx:            1.0e-6

  kappa:         [1.0e-12, 7.0e-12, 1.0e-12]  # [κ_NB, κ_HF, κ_TI]  [J·m²/mol]

  M:                                            # 3×3 mobility matrix [m²·mol/(J·s)]
    - [5.0e-21, 0.0,     0.0    ]              # M_NbNb, M_NbHf, M_NbTi
    - [0.0,     5.0e-23, 0.0    ]              # M_HfNb, M_HfHf, M_HfTi
    - [0.0,     0.0,     5.0e-21]              # M_TiNb, M_TiHf, M_TiTi

  c0:            [0.25, 0.25, 0.25]            # [x_NB, x_HF, x_TI]  (x_V = 1 − sum)
  noise:         0.01
  seed:          42
  dt:            5.0e-1
  derivative_method: jansson
  use_gpu:           false
  adaptive_dt:       false
  n_steps:       3000000
  save_every:    50000
  incremental_npz: true

  elastic:
    enabled:           true
    stiffness_model:   cubic_csv
    eigenstrain_model: local_vegard
    elastic_csv:       data/HfNbTiV_BCC_PhaseForge_v2/sqs_elastic_constants_HfNbVTi-orb-v2.csv

    # Endpoint compositions from An et al. 2021 XRD (lever rule at f_ppt = 1/3)
    c_mat: {NB: 0.30, HF: 0.23, TI: 0.30, V: 0.16}   # β  matrix  (Nb/Ti-rich),  A_Z ≈ 1.17
    c_ppt: {NB: 0.14, HF: 0.28, TI: 0.16, V: 0.42}   # β* precipitate (Hf/V-rich), A_Z ≈ 1.12

    solve_every: 10
    warm_start:  true
    tol:         1.0e-5
    max_iter:    500

    save_fields: true

output:
  data_dir:    data/HfNbTiV_BCC_PhaseForge_v2/NbHfTiV
  results_dir: results/HfNbTiV_BCC_PhaseForge_v2/NbHfTiV
  run_name:    auto
```

---

### Mode 5 — MLIP-computed inputs (no external CSV/TDB)
*Example: `configs/nb_v_mlip_c50.yaml`*

Any elasto-chemical config can compute its atomistic inputs in-pipeline instead of reading them from files. Add an `mlip` section and switch the elastic source:

```yaml
mlip:
  calculator: orb                  # any MaterialsFramework calculator (orb, grace, mace, ...)
  params: {model: orb-v2, device: cpu, fmax: 0.01}

tdb_generation:                    # Step 0 — omit to use system.db_path as given
  enabled:  true
  lattices: [BCC_A2]
  level:    2
  fmax:     0.001                  # written to system.db_path; skipped if it exists

simulation:
  elastic:
    source:        mlip            # csv (default) | mlip
    lattice_model: bond            # sqs (default) | bond — a(x) from bond lengths (Tandoc et al. 2025)
    sqs:
      crystal_structure: bcc
      supercell:         [4, 4, 4]
      step:              0.05      # composition grid → 21 / 231 / 1771 SQS for 2 / 3 / 4 elements
    bond:
      crystal_structure: bcc
      # calculator / params override the top-level mlip section, e.g. GRACE for bonds
```

Step 1b caches the SQS table (`sqs_elastic_constants_<model>.csv`, the same schema as the shipped CSVs) and the bond matrix (`bond_lengths_<model>.csv`) under `output.data_dir`. Re-running reuses them, and an interrupted SQS sweep resumes from the last finished composition. `lattice_model: bond` can also be combined with `source: csv` to pair the shipped C_ij tables with a bond-length a(x).

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Citation

If you use `PhaseFoundry` in your research, please cite the following:

> Kunselman, C., Sarıtürk, D., Zhu, S., Attari, V., & Arróyave, R. (2026). From MLIPs to Microstructure: A High-Throughput Computational Framework to Design Spinodal Alloys in High-Dimensional Composition Spaces via Analytic Derivatives of CALPHAD Model Predictions. *Manuscript submitted for publication*.

A methods paper on PhaseFoundry itself is in preparation:

> Attari, V., & Arróyave, R. (*in preparation*). PhaseFoundry: A CALPHAD-Coupled Phase-Field Framework for Spinodal Microstructure Design in Multicomponent Alloys.

PhaseFoundry reads thermodynamic databases generated with [PhaseForge](https://github.com/dogusariturk/PhaseForge). If you use them, please also cite:

> Zhu, S., Sarıtürk, D., & Arróyave, R. (2025). Machine Learning Potentials for Alloys: A Detailed Workflow to Predict Phase Diagrams and Benchmark Accuracy. npj Computational Materials. https://doi.org/10.1038/s41524-025-01814-z

> Sarıtürk, D., Zhu, S., & Arróyave, R. (2025). PhaseForge. Zenodo. https://doi.org/10.5281/zenodo.15730911

The MLIP features (the `mlip` extra) run on [MaterialsFramework](https://github.com/dogusariturk/MaterialsFramework). If you use them, please also cite:

> Sarıtürk, D., & Arróyave, R. (2025). MaterialsFramework. Zenodo. https://doi.org/10.5281/zenodo.15731044

BibTeX:

```bibtex
@unpublished{kunselman_2026_spinodal,
  author = {Kunselman, Courtney and Sarıtürk, Doğuhan and Zhu, Siya and Attari, Vahid and Arróyave, Raymundo},
  title  = {From {MLIPs} to Microstructure: A High-Throughput Computational Framework to Design Spinodal Alloys in High-Dimensional Composition Spaces via Analytic Derivatives of {CALPHAD} Model Predictions},
  year   = 2026,
  note   = {Manuscript submitted for publication},
}

@article{zhu_2025_phaseforge,
  author  = {Zhu, Siya and Sarıtürk, Doğuhan and Arróyave, Raymundo},
  title   = {Machine Learning Potentials for Alloys: A Detailed Workflow to Predict Phase Diagrams and Benchmark Accuracy},
  journal = {npj Computational Materials},
  year    = 2025,
  doi     = {10.1038/s41524-025-01814-z},
}

@software{sariturk_2025_15730911,
  author    = {Sarıtürk, Doğuhan and Zhu, Siya and Arróyave, Raymundo},
  title     = {PhaseForge},
  year      = 2025,
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.15730911},
  url       = {https://doi.org/10.5281/zenodo.15730911},
}

@software{sariturk_2025_15731044,
  author    = {Sarıtürk, Doğuhan and Arróyave, Raymundo},
  title     = {MaterialsFramework},
  year      = 2025,
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.15731044},
  url       = {https://doi.org/10.5281/zenodo.15731044},
}
```
