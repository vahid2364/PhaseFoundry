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

import yaml

from phasefoundry.config import load_config, apply_overrides, resolve_paths, summary
from phasefoundry.thermodynamics.storage import load_table
from phasefoundry.thermodynamics.interpolant import build_interpolant
from phasefoundry.phasefield.solver import solve_cahn_hilliard, SimResult
from phasefoundry.phasefield.visualize import save_slices_2d, save_pyvista_3d
from phasefoundry.elasticity import (
    build_cubic_stiffness, build_isotropic_stiffness, build_vegard_tensor,
    build_eigenstrain_model,
)
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

    # elastic coupling (optional; off by default if not in config)
    el_cfg = sim.get("elastic", {})
    elastic_kwargs = {}
    if el_cfg.get("enabled", False):
        stiffness_model   = el_cfg.get("stiffness_model", "isotropic")
        eigenstrain_model = el_cfg.get("eigenstrain_model", "uniform")

        # ── stiffness tensors ─────────────────────────────────────────────────
        if stiffness_model == "cubic_csv":
            # Load pre-computed elastic_params.yaml written by step 1b
            T_folder     = f"{sim['temperature_C']}C"
            params_path  = (Path(cfg["output"]["results_dir"])
                            / T_folder / "kinetics" / cfg["output"]["run_name"]
                            / "elastic_params.yaml")
            if not params_path.exists():
                raise FileNotFoundError(
                    f"elastic_params.yaml not found at {params_path}.\n"
                    "Run step 1b first:\n"
                    "  python workflows/01b_lookup_elastic/run.py "
                    f"--config {args.config if 'args' in dir() else '<config>'}"
                )
            with open(params_path) as f:
                ep = yaml.safe_load(f)
            cm, cp = ep["C_mat"], ep["C_ppt"]
            C_mat_tens = build_cubic_stiffness(
                cm["C11_GPa"] * 1e9, cm["C12_GPa"] * 1e9, cm["C44_GPa"] * 1e9)
            C_ppt_tens = build_cubic_stiffness(
                cp["C11_GPa"] * 1e9, cp["C12_GPa"] * 1e9, cp["C44_GPa"] * 1e9)
            print(f"  Stiffness (cubic_csv):")
            print(f"    C_mat: C11={cm['C11_GPa']:.1f}  C12={cm['C12_GPa']:.1f}  "
                  f"C44={cm['C44_GPa']:.1f} GPa  A_Z={cm['A_Z']:.3f}")
            print(f"    C_ppt: C11={cp['C11_GPa']:.1f}  C12={cp['C12_GPa']:.1f}  "
                  f"C44={cp['C44_GPa']:.1f} GPa  A_Z={cp['A_Z']:.3f}")
        else:
            # isotropic fallback — build from G_matrix / delta / nu
            G_mat  = float(el_cfg["G_matrix"])
            delta  = float(el_cfg.get("delta", 1.0))
            nu_val = float(el_cfg.get("nu", 0.3))
            C_mat_tens = build_isotropic_stiffness(G_mat, nu_val)
            C_ppt_tens = build_isotropic_stiffness(delta * G_mat, nu_val)
            ep = {}   # no elastic_params.yaml in isotropic mode

        # ── Vegard tensor Λ and unit conversion ──────────────────────────────
        ind_elements = [e for e in elements if e != ref_el]
        es_data = ep.get("eigenstrain", {}) if stiffness_model == "cubic_csv" else {}
        if stiffness_model == "cubic_csv" and "alpha" in es_data:
            # Tensorial Vegard: α_k from step 1b finite-difference
            alpha_dict = {k.upper(): v for k, v in es_data["alpha"].items()}
            a_ref_A    = float(es_data.get("a_ref_A", es_data.get("a_mat_A", 3.3)))
            N_A = 6.02214076e23
            a_ref_m = a_ref_A * 1e-10
            V_m = a_ref_m**3 * N_A / 2.0          # BCC: 2 atoms/cell
            N_V = float(el_cfg.get("N_V", 1.0 / V_m))
            Lambda = build_vegard_tensor(alpha_dict, ind_elements, ndim=ndim)
            print(f"  Vegard tensor Λ (tensorial):")
            for k, el in enumerate(ind_elements):
                print(f"    α_{el} = {alpha_dict[el.upper()]:.6f}  "
                      f"→  Λ[{k}] = α × I")
            print(f"  a_ref = {a_ref_A:.4f} Å  "
                  f"→  V_m = {V_m:.4e} m³/mol  N_V = {N_V:.4e} mol/m³")
        else:
            # Legacy scalar Vegard fallback (binary only)
            if stiffness_model == "cubic_csv":
                a_mat_A = float(es_data.get("a_mat_A", es_data.get("a_matrix_A", 3.3)))
                a_ppt_A = float(es_data.get("a_ppt_A", 3.3))
                alpha_dict = {ind_elements[0].upper():
                              (a_ppt_A - a_mat_A) / a_mat_A}
                a_ref_A = a_mat_A
            else:
                a_mat_A = float(el_cfg.get("a_matrix", 0.0))
                a_ppt_A = float(el_cfg.get("a_ppt", 0.0))
                alpha_dict = {ind_elements[0].upper():
                              (a_ppt_A - a_mat_A) / a_mat_A}
                a_ref_A = a_mat_A
            N_A = 6.02214076e23
            a_ref_m = a_ref_A * 1e-10
            V_m = a_ref_m**3 * N_A / 2.0
            N_V = float(el_cfg.get("N_V", 1.0 / V_m))
            Lambda = build_vegard_tensor(alpha_dict, ind_elements, ndim=ndim)
            print(f"  Vegard fallback (scalar): α_{ind_elements[0]} = "
                  f"{list(alpha_dict.values())[0]:.6f}")

        # Build eigenstrain model (fixed_vegard default; local_vegard optional)
        model_name = el_cfg.get("eigenstrain_model", "fixed_vegard")
        _model = build_eigenstrain_model(
            model_name    = model_name,
            el_cfg        = el_cfg,
            elastic_params = ep if stiffness_model == "cubic_csv" else {},
            ind_elements  = ind_elements,
            c0            = c0_cfg,
            ndim          = ndim,
            Lambda        = Lambda,
            C_mat         = C_mat_tens,
            C_ppt         = C_ppt_tens,
            all_elements  = [e.upper() for e in elements],
        )

        elastic_kwargs = dict(
            elastic_enabled      = True,
            elastic_C_mat        = C_mat_tens,
            elastic_C_ppt        = C_ppt_tens,
            elastic_Lambda       = Lambda,
            elastic_N_V          = N_V,
            elastic_model        = _model,
            elastic_solve_every  = int  (el_cfg.get("solve_every", 1)),
            elastic_warm_start   = bool (el_cfg.get("warm_start",  True)),
            elastic_tol          = float(el_cfg.get("tol",         1e-8)),
            elastic_max_iter     = int  (el_cfg.get("max_iter",    500)),
            elastic_save_fields  = bool (el_cfg.get("save_fields", False)),
        )

    incremental_npz = bool(sim.get("incremental_npz", False))
    npz_meta = dict(
        T_C      = np.float32(sim["temperature_C"]),
        T_K      = np.float32(T_K),
        N        = np.int32(sim["N"]),
        ndim     = np.int32(ndim),
        Lx       = np.float64(sim["Lx"]),
        elements = np.array([e for e in elements if e != ref_el]),
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = solve_cahn_hilliard(
            interpolant     = interp,
            M               = M_cfg,
            kappa           = kappa_cfg,
            c0              = c0_cfg,
            N               = sim["N"],
            Lx              = sim["Lx"],
            noise           = sim["noise"],
            seed            = sim["seed"],
            dt              = dt,
            n_steps         = sim["n_steps"],
            save_every      = sim["save_every"],
            output_dir      = output_dir,
            save_frames     = True,
            ndim            = ndim,
            use_gpu         = use_gpu,
            adaptive_dt     = adaptive_dt,
            incremental_npz = incremental_npz,
            npz_meta        = npz_meta,
            **elastic_kwargs,
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
    checkpoint = output_dir / "fields_checkpoint.npz"
    if incremental_npz and checkpoint.exists():
        # checkpoint already has the complete history written during the run;
        # update _final fields to the last composition and rename
        ckpt = dict(np.load(checkpoint, allow_pickle=True))
        for i, el in enumerate(ind_elements):
            ckpt[f"x_{el.lower()}_final"] = result.compositions[i].astype(np.float32)
        np.savez_compressed(npz_path, **ckpt)
        checkpoint.unlink()
    else:
        np.savez_compressed(npz_path, **save_dict)
    size_mb = npz_path.stat().st_size / 1e6
    print(f"  Raw fields saved → fields.npz  ({size_mb:.1f} MB, "
          f"{len(result.time_history)} snapshots)")

    # ── save elastic field snapshots (optional) ───────────────────────────────
    if el_cfg.get("enabled", False) and el_cfg.get("save_fields", False):
        if result.mu_el_history:
            el_dict = {
                "mu_ch":  np.array(result.mu_ch_history,  dtype=np.float32),
                "mu_el":  np.array(result.mu_el_history,  dtype=np.float32),
                "stress": np.array(result.stress_history, dtype=np.float32),
                "strain": np.array(result.strain_history, dtype=np.float32),
                "time":   np.array(result.time_history,   dtype=np.float64),
            }
            el_path = output_dir / "elastic_fields.npz"
            np.savez_compressed(el_path, **el_dict)
            size_mb = el_path.stat().st_size / 1e6
            print(f"  Elastic fields saved → elastic_fields.npz  ({size_mb:.1f} MB, "
                  f"{len(result.mu_el_history)} snapshots)")
        else:
            print("  Warning: save_fields=true but no elastic snapshots were collected.")

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
