"""
Step 3 — Cahn-Hilliard phase-field simulation (general, any N-component)
=========================================================================
Reads config, builds a G_m interpolant, and runs the general CH solver.
Works for binary, ternary, quaternary, … — no code changes needed, only
the config file changes.

Output (temperature-first folder convention)
--------------------------------------------
<results_dir>/<T>C/kinetics/<run_name>/
    frame_XXXXXXX.png           composition fields at each saved step (2D)
    free_energy_log.txt
    free_energy_history.png
    sim_params.txt
    fields.npz                  raw composition snapshots + metadata
                                  keys: x_{el} (n_snaps, N[,N[,N]] float32)
                                        x_{el}_final (N[,N[,N]] float32)
                                        time, free_energy (float64)
                                        T_C, T_K, N, ndim, Lx, elements
    slices_x{el}_*.png          2D mid-plane slices of final frame
    pyvista_slices_x{el}_*.png  3D orthogonal slice renders  (3D only)
    pyvista_surface_x{el}_*.png 3D smooth outer-surface render (3D only)
    pyvista_iso_x{el}_*.png     3D isosurface render          (3D only)

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
from calphad_pf.phasefield.visualize import save_slices_2d, save_pyvista_3d
import numpy as np


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
    deriv_method = sim.get("derivative_method", "fd")
    print(f"  Derivative method: {deriv_method!r}")
    interp = build_interpolant(
        paths,
        elements          = elements,
        reference_element = ref_el,
        temperature_K     = T_K,
        derivative_method = deriv_method,
    )

    # output directories
    T_folder   = f"{sim['temperature_C']}C"
    output_dir = results_dir / T_folder / "kinetics" / cfg["output"]["run_name"]
    thermo_dir = results_dir / T_folder / "thermodynamics"
    thermo_dir.mkdir(parents=True, exist_ok=True)
    ndim        = int(sim.get("dimensions", 2))
    use_gpu     = bool(sim.get("use_gpu", False))
    adaptive_dt = bool(sim.get("adaptive_dt", False))
    grid_str    = "×".join([str(sim["N"])] * ndim)
    print(f"  Dimensions: {ndim}D  ({grid_str} grid)")
    print(f"  GPU FFT: {use_gpu}  |  Adaptive dt: {adaptive_dt}")
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
            ndim        = ndim,
            use_gpu     = use_gpu,
            adaptive_dt = adaptive_dt,
        )

    print(f"  Elapsed: {result.elapsed_seconds:.1f} s")

    # ── post-simulation visualization ─────────────────────────────────────────
    ind_elements = [e for e in elements if e != ref_el]
    t_final      = result.time_history[-1] if result.time_history else 0.0

    # ── save raw composition fields ───────────────────────────────────────────
    # All snapshots saved as float32 for compactness; shape (n_steps, N, N)
    # or (n_steps, N, N, N) for 3D.  Reload with np.load("fields.npz").
    save_dict: dict[str, np.ndarray] = {}
    for i, el in enumerate(ind_elements):
        # full history: shape (n_snapshots, N[, N[, N]])
        history = np.array(
            [snap[i] for snap in result.compositions_history],
            dtype=np.float32,
        )
        save_dict[f"x_{el.lower()}"]         = history
        save_dict[f"x_{el.lower()}_final"]   = result.compositions[i].astype(np.float32)
    save_dict["time"]        = np.array(result.time_history,      dtype=np.float64)
    save_dict["free_energy"] = np.array(result.free_energy_history, dtype=np.float64)
    save_dict["T_C"]         = np.float32(sim["temperature_C"])
    save_dict["T_K"]         = np.float32(T_K)
    save_dict["N"]           = np.int32(sim["N"])
    save_dict["ndim"]        = np.int32(ndim)
    save_dict["Lx"]          = np.float64(sim["Lx"])
    save_dict["elements"]    = np.array(ind_elements)

    npz_path = output_dir / "fields.npz"
    np.savez_compressed(npz_path, **save_dict)
    size_mb = npz_path.stat().st_size / 1e6
    print(f"  Raw fields saved → fields.npz  ({size_mb:.1f} MB, "
          f"{len(result.time_history)} snapshots)")

    print(f"  Saving final-frame visualizations ...")
    if ndim == 2:
        for i, el in enumerate(ind_elements):
            c_final = result.compositions[i]
            p = save_slices_2d(
                c_final, t_final, output_dir,
                label=f"x_{el.lower()}",
                title=f"{cfg['system']['name'].upper()} {sim['temperature_C']}C",
            )
            print(f"    Saved: {p.name}")
    else:
        for i, el in enumerate(ind_elements):
            c_final = result.compositions[i]
            paths = save_pyvista_3d(
                c_final, t_final, output_dir,
                label      = f"x_{el.lower()}",
                bar_title  = f"{el.capitalize()} Composition",
                title      = f"{cfg['system']['name'].upper()} {sim['temperature_C']}C",
            )
            for p in paths:
                print(f"    Saved: {p.name}")
            # also save mid-plane 2D slices for quick inspection
            p = save_slices_2d(
                c_final, t_final, output_dir,
                label=f"x_{el.lower()}",
                title=f"{cfg['system']['name'].upper()} {sim['temperature_C']}C",
            )
            print(f"    Saved: {p.name}")

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
