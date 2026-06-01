calphad-pf — Integrated CALPHAD / Phase-Field Modelling Toolkit
================================================================

A Python toolkit that couples `pycalphad <https://pycalphad.org>`_ thermodynamics
with a pseudo-spectral Cahn-Hilliard phase-field solver for multicomponent alloys.

.. contents:: Quick links
   :depth: 1
   :local:

Overview
--------

The pipeline takes a TDB thermodynamic database as input and produces
phase-field microstructure simulations driven by real CALPHAD free energies:

.. code-block:: text

    TDB file
       └─ Step 1 ─ pycalphad → HDF5 thermodynamic tables  (GM, μ, ∂μ/∂x)
       └─ Step 2 ─ HDF5 → G_m spline surrogate + spinodal/binodal boundaries
       └─ Step 3 ─ surrogate → Cahn-Hilliard PFM simulation (pseudo-spectral)
       └─ Step 4 ─ simulation outputs → figures and analysis


Project structure
-----------------

.. code-block:: text

    calphad-pf/
    ├── src/calphad_pf/          installable Python package
    │   ├── thermodynamics/      generate, fit, spinodal, storage, grid, plot
    │   └── phasefield/          C-H solver, free energy models, FFT ops,
    │                            visualize (2D slices + PyVista 3D renders)
    │
    ├── runs/
    │   ├── v2/                  run scripts — TDB PhaseForge v2
    │   └── v3/                  run scripts — TDB PhaseForge v3
    │       ├── NbV_500C.py           2D binary, 500 °C
    │       ├── NbHfV_500C.py         2D ternary, 500 °C
    │       ├── run_3d_nbv_500C.py    3D binary, 500 °C
    │       └── run_3d_nbhfv_500C.py  3D ternary, 500 °C
    │
    ├── workflows/               end-to-end pipeline scripts
    │   ├── run_all.py           ← single entry point for the full pipeline
    │   ├── 01_generate_thermo_data/run_nb_v.py
    │   ├── 02_fit_surrogate/run_nb_v.py
    │   ├── 03_simulate/run_nb_v.py
    │   └── 04_postprocess/run_nb_v.py
    │
    ├── data/
    │   ├── tdb/                 TDB input files
    │   └── nb_v/                HDF5 thermodynamic tables (fine / coarse / kelvin)
    │
    ├── results/                 simulation outputs (gitignored)
    ├── notebooks/               exploratory Jupyter notebooks
    └── archive/                 legacy code


Requirements
------------

- Python ≥ 3.10
- `pycalphad <https://pycalphad.org>`_ ≥ 0.11
- numpy, scipy, matplotlib, h5py, scikit-learn, tqdm
- **3D visualization** (optional): ``pip install pyvista``
- **GPU acceleration** (optional): ``pip install cupy-cuda12x`` (adjust suffix for your CUDA version)

Recommended: install inside a conda environment.

.. code-block:: bash

    conda create -n calphad-pf python=3.11
    conda activate calphad-pf
    conda install -c conda-forge pycalphad


Installation
------------

Clone and install the package in editable mode:

.. code-block:: bash

    git clone https://github.com/cjkunselman18/pycalphad.git
    cd pycalphad-phase-field-coupling
    pip install -e .

Verify:

.. code-block:: bash

    python -c "from calphad_pf.phasefield import solve_cahn_hilliard_binary; print('OK')"


Running the pipeline
--------------------

All runs are driven by a YAML config file.  The default is ``configs/nb_v.yaml``.
To model a new alloy system, copy the config file and edit it — no Python code changes needed.

Full pipeline (all 4 steps)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    python workflows/run_all.py --config configs/nb_v.yaml

This runs steps 1–4 in sequence. **Step 1 calls pycalphad and can take
several minutes** depending on grid resolution and number of temperatures.
Steps 2–4 are fast (seconds to a few minutes).

Resume from a specific step
~~~~~~~~~~~~~~~~~~~~~~~~~~~

If step 1 has already been run and the HDF5 files exist, skip it:

.. code-block:: bash

    python workflows/run_all.py --config configs/nb_v.yaml --from 2

Run only one step
~~~~~~~~~~~~~~~~~

.. code-block:: bash

    python workflows/run_all.py --config configs/nb_v.yaml --only 3

Override parameters without editing the config file
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use ``--set key.subkey=value`` to override any config value on the fly.
Multiple overrides can be chained.  Type coercion (float, int, bool, list)
is automatic.

.. code-block:: bash

    # Test with a different kappa and fewer steps
    python workflows/run_all.py --config configs/nb_v.yaml --only 3 \
        --set simulation.kappa=1e-10 \
              simulation.n_steps=10000 \
              output.run_name=pf_nb_v_500C_kappa1e10_test

    # Run at 800 °C
    python workflows/run_all.py --config configs/nb_v.yaml --from 3 \
        --set simulation.temperature_C=800 \
              output.run_name=pf_nb_v_800C_kappa1e12

    # Use a finer composition grid when regenerating data
    python workflows/run_all.py --config configs/nb_v.yaml --only 1 \
        --set thermo.n_points=500

    # Use auto-stable dt (set dt to None)
    python workflows/run_all.py --config configs/nb_v.yaml --only 3 \
        --set simulation.dt=None

Run a step directly (without run_all.py)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Each step script is fully self-contained:

.. code-block:: bash

    python workflows/01_generate_thermo_data/run_nb_v.py --config configs/nb_v.yaml
    python workflows/02_fit_surrogate/run_nb_v.py        --config configs/nb_v.yaml
    python workflows/03_simulate/run_nb_v.py             --config configs/nb_v.yaml
    python workflows/04_postprocess/run_nb_v.py          --config configs/nb_v.yaml

    # --set works here too
    python workflows/03_simulate/run_nb_v.py --config configs/nb_v.yaml \
        --set simulation.kappa=1e-10

Adding a new alloy system
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    cp configs/nb_v.yaml configs/my_alloy.yaml
    # Edit configs/my_alloy.yaml — change db_path, elements, phase, etc.
    python workflows/run_all.py --config configs/my_alloy.yaml


3D phase-field simulations
--------------------------

The ``runs/v3/`` scripts run fully 3D Cahn-Hilliard simulations on an N³ grid
using the same CALPHAD-coupled solver as the 2D runs.  Adaptive time-stepping
is enabled by default.  PyVista 3D renders are saved automatically alongside
the standard 2D mid-plane slice figures.

Binary system — Nb\ :sub:`50`\ V\ :sub:`50` at 500 °C
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    # Smoke test  (N=32, fast)
    python runs/v3/run_3d_nbv_500C.py --N 32 --t 50

    # Default  (N=64, 200 s physical time)
    python runs/v3/run_3d_nbv_500C.py

    # Production quality  (N=128, long run)
    python runs/v3/run_3d_nbv_500C.py --N 128 --t 1000

    # Full options
    python runs/v3/run_3d_nbv_500C.py --N 128 --t 1000 --snapshots 10 --safety 0.5 --adaptive --gpu

Ternary system — Nb\ :sub:`50`\ Hf\ :sub:`5`\ V\ :sub:`45` at 500 °C
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    python runs/v3/run_3d_nbhfv_500C.py --N 128 --t 1000 --snapshots 10 --safety 0.5 --adaptive --gpu

CLI flags
~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - Flag
     - Default
     - Description
   * - ``--N``
     - 64
     - Grid points per side (64 → 64³ = 262k points; 128 → 128³ = 2M points)
   * - ``--t``
     - 200.0
     - Physical end time in seconds
   * - ``--snapshots``
     - 5
     - Number of composition snapshots saved during the run
   * - ``--safety``
     - 1.0
     - dt safety factor relative to the stability criterion (< 1.0 = more conservative)
   * - ``--adaptive`` / ``--no-adaptive``
     - on
     - Adaptive time-stepping: doubles dt when dynamics are slow, halves when fast
   * - ``--gpu``
     - off
     - Use CuPy GPU-accelerated FFT (requires ``pip install cupy-cuda12x``)

Outputs (per run)
~~~~~~~~~~~~~~~~~

.. code-block:: text

    results/<TDB>/<SYSTEM>/500C/kinetics/<RUN_NAME>/
        slices_xNb_t0s.png            2D XY/XZ/YZ mid-plane slices — initial
        slices_xNb_t<T>s.png          2D XY/XZ/YZ mid-plane slices — final
        pyvista_slices_xNb_t<T>s.png  PyVista orthogonal slice mesh (3D)
        pyvista_surface_xNb_t<T>s.png PyVista smooth-shaded outer surface (3D)
        pyvista_iso_xNb_t<T>s.png     PyVista isosurface at mean composition
        free_energy_3d.png            Free energy vs physical time
        final_field_3d.npz            Compressed NumPy archive of the final field

Verification checks (binary run)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The binary run reports three quantitative checks at the end:

.. code-block:: text

    Mass conservation   [PASS]  max |mean(c) − mean₀| < 1e-10
    Energy monotonicity [PASS]  violations = 0
    Phase separation    [PASS]  composition range widens by > 0.05


Expected outputs
----------------

After a full run you will find:

.. code-block:: text

    data/nb_v/fine/
        nb_v_bcc_100C.h5  …  nb_v_bcc_1000C.h5    (thermodynamic tables)

    results/nb_v/
        spinodal.png
        phase_diagram.png
        GM_all.png
        GM_500C.png
        d2G_vs_T.png
        pf_nb_v_500C_kappa1e12_dt1e2/
            frame_000000.png  …  frame_030000.png   (microstructure snapshots)
            free_energy_log.txt
            free_energy_history.png
            sim_params.txt


Configuration reference
-----------------------

All parameters live in ``configs/<system>.yaml``:

.. code-block:: yaml

    system:
      name:              nb_v
      db_path:           data/tdb/HfNbVTi.tdb
      elements:          [NB, V]
      phase:             BCC_A2
      reference_element: V

    thermo:
      composition_ranges:
        NB: [0.001, 0.999]
      n_points:       200          # grid resolution per temperature
      temperatures_C: [100, ..., 1000]

    simulation:
      temperature_C:  500
      N:              256          # grid points (NxN domain)
      Lx:             1.0e-6       # domain size [m]
      kappa:          1.0e-12      # gradient energy coefficient [J·m²/mol]
      M:              5.0e-21      # mobility [m⁵/(J·s)]
      c0:             0.4          # mean composition (mol fraction)
      dt:             1.0e-2       # time step [s]  (null = auto stable dt)
      n_steps:        30000
      save_every:     5000

    output:
      data_dir:    data/nb_v/fine
      results_dir: results/nb_v
      run_name:    pf_nb_v_500C_kappa1e12_dt1e2   # subfolder for this run


Using the package directly
--------------------------

.. code-block:: python

    from calphad_pf.thermodynamics.fitting import build_gm_spline_surrogate
    from calphad_pf.phasefield import solve_cahn_hilliard_binary

    fit = build_gm_spline_surrogate(["data/nb_v/fine/nb_v_bcc_500C.h5"])

    result = solve_cahn_hilliard_binary(
        free_energy   = "calphad_gm_spline",
        N=256, Lx=1e-6, kappa=1e-12, M=5e-21,
        c0=0.4, dt=1e-2, n_steps=10_000, save_every=2000,
        calphad_fit=fit, temperature_K=773.15,
        output_dir="results/my_run", save_frames=True,
    )


License
-------

MIT — see ``LICENSE.txt``.
