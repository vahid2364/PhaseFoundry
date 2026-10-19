"""Workflow step 3: run the Cahn-Hilliard phase-field simulation.

Builds a G_m interpolant from the step-1 tables at the table temperature closest to
``simulation.temperature_C`` and runs ``solve_cahn_hilliard`` for any number of
components, with elastic coupling when ``simulation.elastic.enabled`` is set. The run
directory ``<results_dir>/<T>C/kinetics/<run_name>/`` receives the composition frames
(``frame_*.png``), ``free_energy_log.txt``, ``free_energy_history.png``,
``composition_evolution.png``, ``sim_params.txt``, ``fields.npz`` and images of the final
field: ``slices_x<el>_*.png``, plus ``pyvista_slices``, ``pyvista_surface`` and
``pyvista_iso`` renders in 3-D. When elastic coupling is on and ``save_fields`` is set in
``simulation.elastic``, it also receives ``elastic_fields.npz``.

``fields.npz`` holds ``x_<el>`` (every snapshot, float32), ``x_<el>_final``, ``time``,
``free_energy``, ``T_C``, ``T_K``, ``N``, ``ndim``, ``Lx`` and ``elements``.

```bash
python workflows/03_simulate/run.py --config configs/nb_v_pipeline_test.yaml
python workflows/03_simulate/run.py --config configs/nb_hf_v_pipeline_test.yaml
python workflows/03_simulate/run.py --config configs/nb_v_pipeline_test.yaml --set simulation.kappa=1e-10 output.run_name=k1e10
```
"""

from __future__ import annotations

import argparse
import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import yaml

from phasefoundry.config import apply_overrides, load_config, resolve_paths, summary
from phasefoundry.elasticity import (
    build_cubic_stiffness,
    build_eigenstrain_model,
    build_isotropic_stiffness,
    build_vegard_tensor,
)
from phasefoundry.elasticity.sources import elastic_table_path, load_lattice_model
from phasefoundry.phasefield.solver import SimResult, solve_cahn_hilliard
from phasefoundry.phasefield.visualize import save_pyvista_3d, save_slices_2d
from phasefoundry.thermodynamics.interpolant import build_interpolant
from phasefoundry.thermodynamics.storage import load_table

_ROOT = Path(__file__).resolve().parents[2]


def _glob_tables(cfg: dict) -> list[Path]:
    """Return the step-1 tables of this config's system, sorted by temperature.

    Any ``<system name>_*C.h5`` file matches, so tables from older and newer naming
    schemes are both picked up.

    Args:
        cfg: Resolved config.

    Returns:
        Paths of the tables.

    Raises:
        FileNotFoundError: If no table matches.
    """
    data_dir = Path(cfg["output"]["data_dir"])
    name = cfg["system"]["name"]
    paths = sorted(
        data_dir.glob(f"{name}_*C.h5"),
        key=lambda p: float(load_table(p)["metadata"]["temperature"]),
    )
    if not paths:
        raise FileNotFoundError(f"No HDF5 tables matching '{name}_*C.h5' in {data_dir}. Run step 1 first.")
    return paths


def run(cfg: dict) -> SimResult:
    """Run the phase-field simulation described by the config and save its outputs.

    Args:
        cfg: Resolved config.

    Returns:
        The solver result.

    Raises:
        FileNotFoundError: If the step-1 tables are missing, or the stiffness model is
            ``cubic_csv`` and step 1b has not written ``elastic_params.yaml``.
    """
    results_dir = Path(cfg["output"]["results_dir"])
    sim = cfg["simulation"]
    elements = cfg["system"]["elements"]
    ref_el = cfg["system"]["reference_element"]

    paths = _glob_tables(cfg)

    T_target = sim["temperature_C"] + 273.15
    all_T = sorted(float(load_table(p)["metadata"]["temperature"]) for p in paths)
    T_K = min(all_T, key=lambda t: abs(t - T_target))
    print(f"  Temperature: {sim['temperature_C']} °C  →  matched {T_K - 273.15:.1f} °C in data")

    deriv_method = sim.get("derivative_method", "fd")
    print(f"  Derivative method: {deriv_method!r}")
    interp = build_interpolant(
        paths,
        elements=elements,
        reference_element=ref_el,
        temperature_K=T_K,
        derivative_method=deriv_method,
    )

    T_folder = f"{sim['temperature_C']}C"
    output_dir = results_dir / T_folder / "kinetics" / cfg["output"]["run_name"]
    thermo_dir = results_dir / T_folder / "thermodynamics"
    thermo_dir.mkdir(parents=True, exist_ok=True)
    ndim = int(sim.get("dimensions", 2))
    use_gpu = bool(sim.get("use_gpu", False))
    adaptive_dt = bool(sim.get("adaptive_dt", False))
    grid_str = "×".join([str(sim["N"])] * ndim)
    print(f"  Dimensions: {ndim}D  ({grid_str} grid)")
    print(f"  GPU FFT: {use_gpu}  |  Adaptive dt: {adaptive_dt}")
    print(f"  Output → {output_dir}")

    dt = sim.get("dt", None)
    if dt in (None, "None", "none"):
        dt = None

    M_cfg = sim["M"]
    kappa_cfg = sim["kappa"]
    c0_cfg = sim["c0"]

    el_cfg = sim.get("elastic", {})
    elastic_kwargs = {}
    if el_cfg.get("enabled", False):
        stiffness_model = el_cfg.get("stiffness_model", "isotropic")

        if stiffness_model == "cubic_csv":
            T_folder = f"{sim['temperature_C']}C"
            params_path = (
                Path(cfg["output"]["results_dir"]) / T_folder / "kinetics" / cfg["output"]["run_name"] / "elastic_params.yaml"
            )
            if not params_path.exists():
                raise FileNotFoundError(
                    f"elastic_params.yaml not found at {params_path}.\n"
                    "Run step 1b first:\n"
                    "  python workflows/01b_lookup_elastic/run.py "
                    f"--config {args.config if 'args' in dir() else '<config>'}"
                )
            with params_path.open() as f:
                ep = yaml.safe_load(f)
            cm, cp = ep["C_mat"], ep["C_ppt"]
            C_mat_tens = build_cubic_stiffness(cm["C11_GPa"] * 1e9, cm["C12_GPa"] * 1e9, cm["C44_GPa"] * 1e9)
            C_ppt_tens = build_cubic_stiffness(cp["C11_GPa"] * 1e9, cp["C12_GPa"] * 1e9, cp["C44_GPa"] * 1e9)
            print("  Stiffness (cubic_csv):")
            print(
                f"    C_mat: C11={cm['C11_GPa']:.1f}  C12={cm['C12_GPa']:.1f}  C44={cm['C44_GPa']:.1f} GPa  A_Z={cm['A_Z']:.3f}"
            )
            print(
                f"    C_ppt: C11={cp['C11_GPa']:.1f}  C12={cp['C12_GPa']:.1f}  C44={cp['C44_GPa']:.1f} GPa  A_Z={cp['A_Z']:.3f}"
            )
        else:
            G_mat = float(el_cfg["G_matrix"])
            delta = float(el_cfg.get("delta", 1.0))
            nu_val = float(el_cfg.get("nu", 0.3))
            C_mat_tens = build_isotropic_stiffness(G_mat, nu_val)
            C_ppt_tens = build_isotropic_stiffness(delta * G_mat, nu_val)
            ep = {}

        ind_elements = [e for e in elements if e != ref_el]
        es_data = ep.get("eigenstrain", {}) if stiffness_model == "cubic_csv" else {}
        if stiffness_model == "cubic_csv" and "alpha" in es_data:
            alpha_dict = {k.upper(): v for k, v in es_data["alpha"].items()}
            a_ref_A = float(es_data.get("a_ref_A", es_data.get("a_mat_A", 3.3)))
            N_A = 6.02214076e23
            a_ref_m = a_ref_A * 1e-10
            V_m = a_ref_m**3 * N_A / 2.0  # 2 atoms per BCC cell
            N_V = float(el_cfg.get("N_V", 1.0 / V_m))
            Lambda = build_vegard_tensor(alpha_dict, ind_elements, ndim=ndim)
            print("  Vegard tensor Λ (tensorial):")
            for k, el in enumerate(ind_elements):
                print(f"    α_{el} = {alpha_dict[el.upper()]:.6f}  →  Λ[{k}] = α × I")
            print(f"  a_ref = {a_ref_A:.4f} Å  →  V_m = {V_m:.4e} m³/mol  N_V = {N_V:.4e} mol/m³")
        else:
            if stiffness_model == "cubic_csv":
                a_mat_A = float(es_data.get("a_mat_A", es_data.get("a_matrix_A", 3.3)))
                a_ppt_A = float(es_data.get("a_ppt_A", 3.3))
                alpha_dict = {ind_elements[0].upper(): (a_ppt_A - a_mat_A) / a_mat_A}
                a_ref_A = a_mat_A
            else:
                a_mat_A = float(el_cfg.get("a_matrix", 0.0))
                a_ppt_A = float(el_cfg.get("a_ppt", 0.0))
                alpha_dict = {ind_elements[0].upper(): (a_ppt_A - a_mat_A) / a_mat_A}
                a_ref_A = a_mat_A
            N_A = 6.02214076e23
            a_ref_m = a_ref_A * 1e-10
            V_m = a_ref_m**3 * N_A / 2.0
            N_V = float(el_cfg.get("N_V", 1.0 / V_m))
            Lambda = build_vegard_tensor(alpha_dict, ind_elements, ndim=ndim)
            print(f"  Vegard fallback (scalar): α_{ind_elements[0]} = {list(alpha_dict.values())[0]:.6f}")

        model_name = el_cfg.get("eigenstrain_model", "fixed_vegard")
        table_path = lattice_model = None
        if stiffness_model == "cubic_csv" and model_name == "local_vegard":
            table_path = elastic_table_path(cfg, _ROOT)
            lattice_model = load_lattice_model(cfg, _ROOT)
        _model = build_eigenstrain_model(
            model_name=model_name,
            el_cfg=el_cfg,
            elastic_params=ep if stiffness_model == "cubic_csv" else {},
            ind_elements=ind_elements,
            c0=c0_cfg,
            ndim=ndim,
            Lambda=Lambda,
            C_mat=C_mat_tens,
            C_ppt=C_ppt_tens,
            all_elements=[e.upper() for e in elements],
            table_path=table_path,
            lattice_model=lattice_model,
            reference_element=ref_el,
        )

        elastic_kwargs = {
            "elastic_enabled": True,
            "elastic_C_mat": C_mat_tens,
            "elastic_C_ppt": C_ppt_tens,
            "elastic_Lambda": Lambda,
            "elastic_N_V": N_V,
            "elastic_model": _model,
            "elastic_solve_every": int(el_cfg.get("solve_every", 1)),
            "elastic_warm_start": bool(el_cfg.get("warm_start", True)),
            "elastic_tol": float(el_cfg.get("tol", 1e-8)),
            "elastic_max_iter": int(el_cfg.get("max_iter", 500)),
            "elastic_save_fields": bool(el_cfg.get("save_fields", False)),
        }

    incremental_npz = bool(sim.get("incremental_npz", False))
    npz_meta = {
        "T_C": np.float32(sim["temperature_C"]),
        "T_K": np.float32(T_K),
        "N": np.int32(sim["N"]),
        "ndim": np.int32(ndim),
        "Lx": np.float64(sim["Lx"]),
        "elements": np.array([e for e in elements if e != ref_el]),
    }

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = solve_cahn_hilliard(
            interpolant=interp,
            M=M_cfg,
            kappa=kappa_cfg,
            c0=c0_cfg,
            N=sim["N"],
            Lx=sim["Lx"],
            noise=sim["noise"],
            seed=sim["seed"],
            dt=dt,
            n_steps=sim["n_steps"],
            save_every=sim["save_every"],
            output_dir=output_dir,
            save_frames=True,
            ndim=ndim,
            use_gpu=use_gpu,
            adaptive_dt=adaptive_dt,
            incremental_npz=incremental_npz,
            npz_meta=npz_meta,
            **elastic_kwargs,
        )

    print(f"  Elapsed: {result.elapsed_seconds:.1f} s")

    ind_elements = [e for e in elements if e != ref_el]
    t_final = result.time_history[-1] if result.time_history else 0.0

    save_dict: dict[str, np.ndarray] = {}
    for i, el in enumerate(ind_elements):
        history = np.array(
            [snap[i] for snap in result.compositions_history],
            dtype=np.float32,
        )
        save_dict[f"x_{el.lower()}"] = history
        save_dict[f"x_{el.lower()}_final"] = result.compositions[i].astype(np.float32)
    save_dict["time"] = np.array(result.time_history, dtype=np.float64)
    save_dict["free_energy"] = np.array(result.free_energy_history, dtype=np.float64)
    save_dict["T_C"] = np.float32(sim["temperature_C"])
    save_dict["T_K"] = np.float32(T_K)
    save_dict["N"] = np.int32(sim["N"])
    save_dict["ndim"] = np.int32(ndim)
    save_dict["Lx"] = np.float64(sim["Lx"])
    save_dict["elements"] = np.array(ind_elements)

    npz_path = output_dir / "fields.npz"
    checkpoint = output_dir / "fields_checkpoint.npz"
    if incremental_npz and checkpoint.exists():
        # The checkpoint already holds the full history; only the final fields need updating.
        ckpt = dict(np.load(checkpoint, allow_pickle=True))
        for i, el in enumerate(ind_elements):
            ckpt[f"x_{el.lower()}_final"] = result.compositions[i].astype(np.float32)
        np.savez_compressed(npz_path, **ckpt)
        checkpoint.unlink()
    else:
        np.savez_compressed(npz_path, **save_dict)
    size_mb = npz_path.stat().st_size / 1e6
    print(f"  Raw fields saved → fields.npz  ({size_mb:.1f} MB, {len(result.time_history)} snapshots)")

    if el_cfg.get("enabled", False) and el_cfg.get("save_fields", False):
        if result.mu_el_history:
            el_dict = {
                "mu_ch": np.array(result.mu_ch_history, dtype=np.float32),
                "mu_el": np.array(result.mu_el_history, dtype=np.float32),
                "stress": np.array(result.stress_history, dtype=np.float32),
                "strain": np.array(result.strain_history, dtype=np.float32),
                "time": np.array(result.time_history, dtype=np.float64),
            }
            el_path = output_dir / "elastic_fields.npz"
            np.savez_compressed(el_path, **el_dict)
            size_mb = el_path.stat().st_size / 1e6
            print(f"  Elastic fields saved → elastic_fields.npz  ({size_mb:.1f} MB, {len(result.mu_el_history)} snapshots)")
        else:
            print("  Warning: save_fields=true but no elastic snapshots were collected.")

    print("  Saving final-frame visualizations ...")
    if ndim == 2:
        for i, el in enumerate(ind_elements):
            c_final = result.compositions[i]
            p = save_slices_2d(
                c_final,
                t_final,
                output_dir,
                label=f"x_{el.lower()}",
                title=f"{cfg['system']['name'].upper()} {sim['temperature_C']}C",
            )
            print(f"    Saved: {p.name}")
    else:
        for i, el in enumerate(ind_elements):
            c_final = result.compositions[i]
            paths = save_pyvista_3d(
                c_final,
                t_final,
                output_dir,
                label=f"x_{el.lower()}",
                bar_title=f"{el.capitalize()} Composition",
                title=f"{cfg['system']['name'].upper()} {sim['temperature_C']}C",
            )
            for p in paths:
                print(f"    Saved: {p.name}")
            p = save_slices_2d(
                c_final,
                t_final,
                output_dir,
                label=f"x_{el.lower()}",
                title=f"{cfg['system']['name'].upper()} {sim['temperature_C']}C",
            )
            print(f"    Saved: {p.name}")

    return result


if __name__ == "__main__":
    logging.basicConfig(format="%(message)s", stream=sys.stdout)
    logging.getLogger("phasefoundry").setLevel(logging.INFO)
    parser = argparse.ArgumentParser(description="Step 3 — Phase-field simulation (general)")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg = apply_overrides(cfg, args.set)
    cfg = resolve_paths(cfg, _ROOT)

    n_el = len(cfg["system"]["elements"])
    n_ind = n_el - 1
    print("── Step 3: Phase-field simulation ──")
    print(f"   System : {cfg['system']['name']}  ({n_el} components, {n_ind} independent field(s))")
    print(summary(cfg))

    result = run(cfg)
    print("\nDone.")
    print(f"  c_min = {result.c_min:.4f}")
    print(f"  c_max = {result.c_max:.4f}")
    for i, el in enumerate(result.ind_elements):
        ci = result.compositions[i]
        print(f"  x_{el:<4}: [{ci.min():.4f}, {ci.max():.4f}]")
