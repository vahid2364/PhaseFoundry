"""
Step 3 — Cahn-Hilliard phase-field simulation (general, any N-component)
=========================================================================
Reads config, builds a G_m interpolant, and runs the general CH solver.
Works for binary, ternary, quaternary, … — no code changes needed, only
the config file changes.

Output (temperature-first folder convention)
--------------------------------------------
<results_dir>/<T>C/kinetics/<run_name>/
    frame_XXXXXXX.png      composition fields at each saved step
    free_energy_log.txt
    free_energy_history.png
    sim_params.txt

Run standalone
--------------
    # Binary Nb-V
    python workflows/03_simulate/run.py --config configs/nb_v.yaml

    # Ternary Nb-Hf-V
    python workflows/03_simulate/run.py --config configs/nb_hf_v.yaml

    # Override kappa
    python workflows/03_simulate/run.py \
        --config configs/nb_v.yaml \
        --set simulation.kappa=1e-10 output.run_name=test_kappa1e10
"""

from __future__ import annotations
import argparse, sys, warnings
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from calphad_pf.config import load_config, apply_overrides, resolve_paths, summary
from calphad_pf.thermodynamics.storage import load_table
from calphad_pf.thermodynamics.interpolant import build_interpolant
from calphad_pf.phasefield.solver import solve_cahn_hilliard, SimResult


def _glob_tables(cfg: dict) -> list[Path]:
    data_dir = Path(cfg["output"]["data_dir"])
    name     = cfg["system"]["name"]
    # match any file starting with {name}_ and ending with C.h5
    # (tolerates both old naming nb_v_bcc_*C.h5 and new nb_v_bcc_a2_*C.h5)
    paths    = sorted(
        data_dir.glob(f"{name}_*C.h5"),
        key=lambda p: float(load_table(p)["metadata"]["temperature"]),
    )
    if not paths:
        raise FileNotFoundError(
            f"No HDF5 tables matching '{name}_*C.h5' in {data_dir}. "
            "Run step 1 first.")
    return paths


def run(cfg: dict) -> SimResult:
    """Run phase-field simulation. Returns SimResult."""
    data_dir    = Path(cfg["output"]["data_dir"])
    results_dir = Path(cfg["output"]["results_dir"])
    sim         = cfg["simulation"]
    elements    = cfg["system"]["elements"]
    ref_el      = cfg["system"]["reference_element"]
    n_ind       = len(elements) - 1

    paths = _glob_tables(cfg)

    # match requested temperature
    T_target = sim["temperature_C"] + 273.15
    all_T    = sorted(
        float(load_table(p)["metadata"]["temperature"]) for p in paths
    )
    T_K = min(all_T, key=lambda t: abs(t - T_target))
    print(f"  Temperature: {sim['temperature_C']} °C  →  matched "
          f"{T_K - 273.15:.1f} °C in data")

    # build interpolant
    interp = build_interpolant(
        paths,
        elements          = elements,
        reference_element = ref_el,
        temperature_K     = T_K,
    )

    # output directories
    T_folder   = f"{sim['temperature_C']}C"
    output_dir = results_dir / T_folder / "kinetics" / cfg["output"]["run_name"]
    thermo_dir = results_dir / T_folder / "thermodynamics"
    thermo_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Output → {output_dir}")

    # handle dt
    dt = sim.get("dt", None)
    if dt in (None, "None", "none"):
        dt = None

    # normalise M and kappa from config
    # M in config may be scalar, list (diagonal), or list-of-lists (matrix)
    M_cfg     = sim["M"]
    kappa_cfg = sim["kappa"]
    c0_cfg    = sim["c0"]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = solve_cahn_hilliard(
            interpolant = interp,
            M           = M_cfg,
            kappa       = kappa_cfg,
            c0          = c0_cfg,
            N           = sim["N"],
            Lx          = sim["Lx"],
            noise       = sim["noise"],
            seed        = sim["seed"],
            dt          = dt,
            n_steps     = sim["n_steps"],
            save_every  = sim["save_every"],
            output_dir  = output_dir,
            save_frames = True,
        )

    print(f"  Elapsed: {result.elapsed_seconds:.1f} s")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Step 3 — Phase-field simulation (general)")
    parser.add_argument("--config", required=True,
                        help="Path to YAML config file")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg = apply_overrides(cfg, args.set)
    cfg = resolve_paths(cfg, _ROOT)

    n_el  = len(cfg["system"]["elements"])
    n_ind = n_el - 1
    print(f"── Step 3: Phase-field simulation ──")
    print(f"   System : {cfg['system']['name']}  "
          f"({n_el} components, {n_ind} independent field(s))")
    print(summary(cfg))

    result = run(cfg)
    print(f"\nDone.")
    print(f"  c_min = {result.c_min:.4f}")
    print(f"  c_max = {result.c_max:.4f}")
    for i, el in enumerate(result.ind_elements):
        ci = result.compositions[i]
        print(f"  x_{el:<4}: [{ci.min():.4f}, {ci.max():.4f}]")
