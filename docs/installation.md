# Installation

## Requirements

- **Python ≥ 3.10** (developed and tested with 3.13).
- **OS:** Linux or macOS. There's no Windows-specific blocker in the code,
  but it isn't tested there.
- A C/C++ build toolchain available to `pip` — `pycalphad`'s dependency
  chain (via `symengine`) ships pre-built wheels for common platforms, but
  if your platform/Python combination lacks one, pip will try to compile
  from source and needs a compiler present.

No GPU, no external services, and no license files are required for the
core package.

## Quick install

```bash
git clone https://github.com/vahid2364/PhaseFoundry.git
cd PhaseFoundry
pip install -e .
```

`-e` (editable) installs the package in-place from the repo so edits to
`src/phasefoundry/` take effect immediately — the right mode for working
with configs and workflows directly out of the checkout, which is how every
tutorial on this site is written. A regular `pip install .` works too if you
just want the `pf-*` commands available and don't plan to edit the source.

**Verify it worked:**

```bash
pf-simulate --help   # should print usage, not an error
python -c "import phasefoundry; print(phasefoundry.__file__)"
```

## What gets installed

`pip install -e .` pulls in these core dependencies (pinned as minimum
versions in `pyproject.toml`):

| Package | Minimum version | What it's for |
|---|---|---|
| `numpy` | 1.24 | field arrays, FFT-based spectral solver |
| `scipy` | 1.10 | interpolation, sparse/linear-algebra utilities |
| `matplotlib` | 3.7 | all figures (spinodal maps, microstructures, free-energy history) |
| `h5py` | 3.8 | thermodynamic-table storage (Step 1 output) |
| `tqdm` | 4.65 | the progress bar you see during Step 3 |
| `scikit-learn` | 1.2 | surrogate interpolants (Step 2) |
| `pycalphad` | 0.11 | CALPHAD equilibrium + the Jansson analytic-derivative machinery this whole toolkit is built on |
| `colorama` | 0.4 | colored terminal output |
| `psutil` | 5.9 | peak-memory tracking in the structured logger |

And it registers the `pf-*` console scripts (`pf-generate`, `pf-elastic`,
`pf-elastic-solve`, `pf-tieline`, `pf-phase-map`, `pf-surrogate`,
`pf-simulate`, `pf-postprocess`, `pf-validate`, `pf-compare`, `pf-run-all`)
as thin wrappers around the corresponding `workflows/<step>/run.py` script —
see `src/phasefoundry/cli.py` if you want the exact mapping.

**Thermodynamic databases (TDB files) are bundled in the repo** under
`data/tdb/` (a few tens of KB) — there's no separate data download for the
example systems in the tutorials and configs. Generating a *new* TDB for a
different alloy system is a separate step via
[PhaseForge](https://github.com/dogusariturk/PhaseForge), not something
`pip install` handles.

## Optional dependencies

Two things are imported lazily (only inside the specific function that
needs them), not listed as hard dependencies, and not installed by
`pip install -e .`:

- **`python-ternary`** — needed only for ternary/quaternary composition
  diagram plots (used in `workflows/02_fit_surrogate/run.py` and
  `src/phasefoundry/thermodynamics/plot.py`). Binary-only workflows (like
  the [2D binary tutorial](tutorial_2d_binary.md)) never touch this code
  path. Install if you're working with 3+ component systems and hit an
  `ImportError: ternary`:

  ```bash
  pip install python-ternary
  ```

- **`cupy`** — optional GPU backend for the FFT-based phase-field solver
  (`simulation.use_gpu: true` in a config). The solver falls back to NumPy
  transparently if `cupy` isn't installed and `use_gpu` is left `false`
  (the default in every example config). Install the build matching your
  CUDA version if you want GPU acceleration:

  ```bash
  pip install cupy-cuda12x   # adjust the suffix for your CUDA version
  ```

Neither is needed to run any tutorial on this site as written.

## Running the test suite (optional, for development)

```bash
pip install pytest
pytest tests/
```

`pytest` itself isn't a runtime dependency of the package — only needed if
you're developing `phasefoundry` itself, not for running the pipeline.

## Troubleshooting

- **`pip install -e .` is slow or fails while building `symengine`/
  `python-flint`:** these are transitive dependencies pulled in by
  `pycalphad`. If pip is trying to compile them from source instead of
  pulling a wheel, you're likely on a platform/Python combination without a
  prebuilt wheel upstream. Installing `pycalphad` from `conda-forge` first
  (`conda install -c conda-forge pycalphad`), then `pip install -e . --no-deps`
  to just register the `phasefoundry` package itself, sidesteps this.
- **`pf-simulate: command not found` after installing:** the console
  scripts are installed into whatever Python environment you ran `pip
  install` in — check `pip show phasefoundry` points at the environment
  you expect to be active, and that its `bin/`/`Scripts/` directory is on
  your `PATH`.
- **`ImportError: ternary`** when running Step 2 on a ternary/quaternary
  config: see [Optional dependencies](#optional-dependencies) above.

## Next step

[2D binary tutorial](tutorial_2d_binary.md) — the first full pipeline run, start to finish. Or see the [CLI Reference](cli_reference.md) for every `pf-*` command.
