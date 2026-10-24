# CLI Reference

PhaseFoundry has two interfaces to the exact same code, and they're fully
interchangeable — use whichever you prefer:

```bash
pf-simulate --config configs/nb_v_chem_c50.yaml
# is exactly the same as:
python workflows/03_simulate/run.py --config configs/nb_v_chem_c50.yaml
```

Every `pf-*` console script is a thin wrapper
([`src/phasefoundry/cli.py`](https://github.com/vahid2364/PhaseFoundry/blob/develop-new/src/phasefoundry/cli.py))
that imports the corresponding `workflows/<step>/run.py` and calls its
`main()` — nothing more. The indirection exists only because workflow
directory names start with digits (`01_generate_thermo_data`, ...) and
can't be imported as normal Python packages. All `--help` text, output
paths, and behavior below came from actually running each command on this
machine.

## Common flags

Every command takes `--config <path>` (a YAML config file — see
[Installation](installation.md) and any tutorial's "The config file"
section for examples). Most also take:

```
--set KEY=VALUE [KEY=VALUE ...]
```

Overrides a config value from the command line without editing the file,
using dotted-path keys, e.g.:

```bash
pf-simulate --config configs/nb_v_chem_c50.yaml \
  --set simulation.n_steps=2000 simulation.save_every=1000
```

This is used throughout the tutorials on this site — e.g. the
[ternary tutorial](tutorial_2d_ternary_elastic.md#step-3-run-the-simulation)'s
timing probe.

## Quick reference

| Command | Pipeline step | Workflow script |
|---|---|---|
| `pf-generate` | Step 1 | `workflows/01_generate_thermo_data/run.py` |
| `pf-elastic` | Step 1b | `workflows/01b_lookup_elastic/run.py` |
| `pf-tieline` | Step 1c | `workflows/01c_calphad_tieline/run.py` |
| `pf-phase-map` | Step 1d | `workflows/01d_phase_map/run.py` |
| `pf-surrogate` | Step 2 | `workflows/02_fit_surrogate/run.py` |
| `pf-simulate` | Step 3 | `workflows/03_simulate/run.py` |
| `pf-elastic-solve` | (standalone) | `src/phasefoundry/elasticity/fft_solver.py` |
| `pf-postprocess` | Step 4 | `workflows/04_postprocess/run.py` |
| `pf-validate` | Step 5 (optional) | `workflows/05_validate/run.py` |
| `pf-compare` | Step 6 | `workflows/06_compare/run.py` |
| `pf-run-all` | all steps | `workflows/run_all.py` |

---

## Thermodynamics (Steps 1, 1b, 1c, 1d)

### `pf-generate` — Step 1

Calls `pycalphad` to compute `G_m`, `MU`, and `dMU` (the Jansson
derivatives) at every point of the composition simplex grid, for every
temperature in `thermo.temperatures_C`, and caches the result to HDF5.

```
usage: pf-generate [-h] --config CONFIG [--set [KEY=VALUE ...]]

Step 1 — Generate thermodynamic tables (general)

options:
  -h, --help            show this help message and exit
  --config CONFIG       Path to YAML config file
  --set [KEY=VALUE ...]
```

**Output:** `<output.data_dir>/<system.name>_<phase>_{T}C.h5` — one file
per temperature.

```bash
pf-generate --config configs/nb_v_chem_c50.yaml
```

See the [2D binary tutorial's Step 1](tutorial_2d_binary.md#step-1-generate-thermodynamic-tables)
for real console output, and the
[binodal/spinodal tutorial's mathematics section](tutorial_binodal_spinodal_3T.md#the-mathematics-what-equilibrium-means-at-each-of-the-198-points)
for exactly what's being computed at each grid point.

---

### `pf-elastic` — Step 1b

Interpolates SQS/DFT elastic constants (`C11`/`C12`/`C44`, lattice
parameter) at the matrix and precipitate compositions from
`elastic.c_mat`/`elastic.c_ppt`, checks Born mechanical-stability criteria,
and derives the tensorial Vegard eigenstrain. Works for binary (1D),
ternary (2D), or quaternary (3D) interpolation automatically.

```
usage: pf-elastic [-h] --config CONFIG [--set [KEY=VALUE ...]]

Step 1b — Lookup elastic constants from SQS/DFT CSV

options:
  -h, --help            show this help message and exit
  --config CONFIG       Path to YAML config file
  --set [KEY=VALUE ...]
                        Override config values, e.g. elastic.c_ppt.NB=0.15
```

**Output:** `<output.results_dir>/elastic_params.yaml` — consumed
automatically by `pf-simulate` when `elastic.stiffness_model: cubic_csv`.

```bash
pf-elastic --config configs/nb_hf_v_elastic_cNb50V25Hf25.yaml
```

Only needed for elasto-chemical runs; see the
[ternary tutorial's Step 1b](tutorial_2d_ternary_elastic.md#step-1b-look-up-elastic-constants-and-check-born-stability)
for real output (Born stability check, Zener anisotropy, Vegard tensor).

---

### `pf-tieline` — Step 1c

Computes CALPHAD two-phase equilibrium tie-line endpoints for a given
composition via `pycalphad.equilibrium`, writing matrix/precipitate
compositions that can replace the placeholder `c_mat`/`c_ppt` values in an
elastic config.

```
usage: pf-tieline [-h] --config CONFIG

CALPHAD tie-line endpoint calculator

options:
  -h, --help       show this help message and exit
  --config CONFIG  Path to simulation config YAML
```

```bash
pf-tieline --config configs/nb_hf_v_elastic_cNb50V25Hf25.yaml
```

!!! note "This is the fix for a caveat flagged elsewhere on this site"
    The [ternary tutorial](tutorial_2d_ternary_elastic.md) and the
    `nb_hf_v_elastic_cNb50V25Hf25.yaml` config itself both flag that the
    `c_mat`/`c_ppt` endpoints used there are placeholder estimates that
    don't actually mass-balance against the parent composition. `pf-tieline`
    is the tool that replaces them with a real CALPHAD-computed tie-line —
    not yet run as part of any tutorial on this site.

---

### `pf-phase-map` — Step 1d (optional)

Reads Step 1's HDF5 tables and generates phase diagrams: classic ternary
diagrams (`python-ternary`) for 3-component systems, or an Alloy-Space UMAP
projection (Vela et al., arXiv:2408.07681) for 4+ components, controlled by
`thermo.phase_map_style` (`ternary` / `umap` / `both`).

```
usage: pf-phase-map [-h] --config CONFIG [--set [KEY=VALUE ...]]

Step 1d — Ternary phase map (general, any N-component system)

options:
  -h, --help            show this help message and exit
  --config CONFIG       Path to YAML config file
  --set [KEY=VALUE ...]
                        Override config values, e.g. --set
                        thermo.temperatures_C=[500]
```

```bash
pf-phase-map --config configs/nb_hf_v.yaml
```

Binary systems don't produce a phase map here — this is for 3+ components.
`python-ternary` is an optional dependency; see
[Installation](installation.md#optional-dependencies) if this errors with
`ImportError: ternary`.

---

## Surrogate & boundaries (Step 2)

### `pf-surrogate` — Step 2

Fits a `G_m` surrogate to Step 1's HDF5 tables (1D cubic spline for binary,
CT2D or RBF→RGI for ternary/quaternary), then computes binodal and
spinodal boundaries.

```
usage: pf-surrogate [-h] --config CONFIG [--set [KEY=VALUE ...]]

Step 2 — Build G_m interpolant (general)

options:
  -h, --help            show this help message and exit
  --config CONFIG
  --set [KEY=VALUE ...]
```

**Output** (temperature-first folder convention):

```
<results_dir>/phase_diagram.png          (binary only)
<results_dir>/GM_all.png                 (binary only)
<results_dir>/<T>C/thermodynamics/
    GM_spline.png                        (binary: spline verification)
    GM_surface.png                       (ternary: G_m surface heatmap)
    spinodal.png
```

```bash
pf-surrogate --config configs/nb_v_binodal_spinodal_3T.yaml
```

The [binodal/spinodal tutorial](tutorial_binodal_spinodal_3T.md) is built
entirely around this command's output, including a caveat worth knowing
before you rely on its phase-diagram figure: it globs *every* cached
temperature table in `data_dir`, not just the ones in your current config
(see [that tutorial's Step 2 note](tutorial_binodal_spinodal_3T.md#step-2-fit-the-surrogate-and-compute-both-boundaries)).

---

## Simulation (Step 3)

### `pf-simulate` — Step 3

Builds the `G_m` interpolant and runs the general N-component Cahn-Hilliard
solver (pseudo-spectral, semi-implicit, Eyre convex-splitting
stabilization). The same command runs binary, ternary, or quaternary
systems — only the config changes.

```
usage: pf-simulate [-h] --config CONFIG [--set [KEY=VALUE ...]]

Step 3 — Phase-field simulation (general)

options:
  -h, --help            show this help message and exit
  --config CONFIG       Path to YAML config file
  --set [KEY=VALUE ...]
```

**Output:** `<results_dir>/<T>C/kinetics/<run_name>/` — composition-field
frames, `free_energy_log.txt`/`.png`, `sim_params.txt` (the diagnostics
dump explained in full in the
[Hessian stability tutorial](tutorial_hessian_stability_binary.md#from-eigenvalue-to-solver-parameters)),
and `fields.npz` (raw snapshots for post-processing).

```bash
pf-simulate --config configs/nb_v_chem_c50.yaml
```

This is the command behind both the
[2D binary tutorial](tutorial_2d_binary.md) and the
[2D ternary elasto-chemical tutorial](tutorial_2d_ternary_elastic.md) — see
either for a full walkthrough of what the output actually looks like,
including run time and honest caveats about numerical limits.

---

### `pf-elastic-solve` — standalone

A standalone elasticity solver, separate from the numbered pipeline steps:
solves mechanical equilibrium for a single microstructure/eigenstrain field
via FFT-based Lippmann-Schwinger iteration (Gururajan & Abinandanan 2007,
*Acta Materialia* 55, 5015-5026), without running a phase-field
simulation. The same Moulinec-Suquet-family solver that `pf-simulate` calls
automatically every `solve_every` steps when `elastic.enabled: true` — this
command exposes it directly.

```
usage: pf-elastic-solve [-h] --config CONFIG

Solve mechanical equilibrium via Lippmann-Schwinger iteration.

options:
  -h, --help       show this help message and exit
  --config CONFIG  Path to YAML configuration file.
```

```bash
pf-elastic-solve --config configs/nb_hf_v_elastic_cNb50V25Hf25.yaml
```

Not currently used by any tutorial on this site — useful if you want to
inspect a single elastic-equilibrium solve in isolation, e.g. for debugging
a stiffness/eigenstrain model before running a full simulation with it.

---

## Post-processing & validation (Steps 4, 5, 6)

### `pf-postprocess` — Step 4

Re-renders the free-energy history and generates the STEM-style
composition microstructure figure (main panel + per-element insets)
directly from the saved `fields.npz`.

```
usage: pf-postprocess [-h] [--config CONFIG] [--set [KEY=VALUE ...]]

Step 4 — Post-processing (general, any N-component system)

options:
  -h, --help            show this help message and exit
  --config CONFIG       Path to YAML config file
  --set [KEY=VALUE ...]
                        Override config values, e.g. --set
                        simulation.temperature_C=600
```

```bash
pf-postprocess --config configs/nb_v_chem_c50.yaml
```

See [the binary tutorial's Step 4](tutorial_2d_binary.md#step-4-post-process)
for the actual STEM-style figure this produces.

---

### `pf-validate` — Step 5 (optional)

Compares the `fd` / `jansson` / `hybrid` derivative methods against raw
CALPHAD data for your system, and checks for other thermodynamic-table
issues.

```
usage: pf-validate [-h] [--config CONFIG] [--set [KEY=VALUE ...]]

Step 5 — Validate thermodynamic tables

options:
  -h, --help            show this help message and exit
  --config CONFIG
  --set [KEY=VALUE ...]
```

**Output:** `<results_dir>/<T>C/thermodynamics/thermo_properties.pdf`
(`G_m`, `μ_i`, `H_ii` vs. composition) and
`derivative_comparison.pdf` (FD/Jansson/hybrid vs. raw CALPHAD data).

```bash
pf-validate --config configs/nb_v_chem_c50.yaml
```

The [Hessian stability tutorial](tutorial_hessian_stability_binary.md#validating-against-finite-differences)
runs this for real and works through why its RMSE summary table looks
alarming at first glance (and why it isn't) — worth reading before trusting
this command's numbers at face value.

---

### `pf-compare` — Step 6

Builds a 2×N panel figure comparing final-frame composition fields between
matched chemical-only and elasto-chemical runs, at a chosen set of mean
compositions.

```
usage: pf-compare [-h] --config CONFIG [--set [KEY=VALUE ...]] [--snap SNAP]
                  [--compositions COMPOSITIONS [COMPOSITIONS ...]]

Step 6 — Chemical vs. elasto-chemical comparison figure

options:
  -h, --help            show this help message and exit
  --config CONFIG
  --set [KEY=VALUE ...]
  --snap SNAP           Snapshot index to plot (-1 = final, default)
  --compositions COMPOSITIONS [COMPOSITIONS ...]
                        x_NB values to compare (default: 0.5 0.6 0.7 0.8)
```

**Output:** `<results_dir>/500C/comparison/chem_vs_elastic_final.{pdf,png}`

```bash
pf-compare --config configs/nb_v_chem_c60.yaml --snap -1
```

Requires matching chemical-only and elasto-chemical runs to already exist
at each composition in `--compositions` — not yet run as part of any
tutorial on this site.

---

## Orchestration

### `pf-run-all`

Runs Steps 1 → 2 → 3 → 4 → 5 end-to-end from one config (its own internal
numbering is 1-5: generate → phase-map → fit surrogate → simulate →
postprocess → validate — it does **not** include Steps 1b, 1c, or 6, which
are elasto-chemical/comparison extras invoked separately via the commands
above).

```
usage: pf-run-all [-h] [--config CONFIG] [--from N] [--only N]
                  [--set [KEY=VALUE ...]]

Integrated CALPHAD / phase-field pipeline

options:
  -h, --help            show this help message and exit
  --config CONFIG       Path to YAML config file (default: configs/nb_v.yaml)
  --from N              Start from step N (1–6)
  --only N              Run only step N (1–6)
  --set [KEY=VALUE ...]
                        Override config values, e.g. --set
                        simulation.kappa=1e-10
```

```bash
# Full pipeline, binary Nb-V
pf-run-all --config configs/nb_v_chem_c50.yaml

# Resume from step 3 (skip re-generating thermo data + re-fitting the surrogate)
pf-run-all --config configs/nb_v_chem_c50.yaml --from 3

# Run only step 3
pf-run-all --config configs/nb_v_chem_c50.yaml --only 3
```

Every tutorial on this site runs the individual `pf-*` commands one at a
time instead of `pf-run-all`, specifically so each step's real console
output and figures can be shown and explained — `pf-run-all` is the faster
path once you already understand what each step does.
