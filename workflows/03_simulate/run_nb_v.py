"""
Step 3 — Cahn-Hilliard phase-field simulation
=============================================
Uses the G_m spline surrogate to drive spinodal decomposition
on a 2-D periodic domain.

Output
------
<output.results_dir>/<output.run_name>/
    frame_*.png
    free_energy_log.txt
    free_energy_history.png
    sim_params.txt

Run standalone
--------------
    python workflows/03_simulate/run_nb_v.py --config configs/nb_v.yaml

    # Quick test with different kappa
    python workflows/03_simulate/run_nb_v.py \
        --config configs/nb_v.yaml \
        --set simulation.kappa=1e-10 simulation.n_steps=10000 \
              output.run_name=pf_nb_v_500C_kappa1e10_test
"""

from __future__ import annotations
import argparse, sys, warnings
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from calphad_pf.config import load_config, apply_overrides, resolve_paths, summary
from calphad_pf.thermodynamics.fitting import build_gm_spline_surrogate
from calphad_pf.phasefield import solve_cahn_hilliard_binary


def run(cfg: dict) -> object:
    """Run phase-field simulation. Returns CHResult."""
    data_dir    = Path(cfg["output"]["data_dir"])
    results_dir = Path(cfg["output"]["results_dir"])
    sim         = cfg["simulation"]

    # Load surrogate fit
    paths = sorted(data_dir.glob("nb_v_bcc_*C.h5"))
    if not paths:
        raise FileNotFoundError(f"No HDF5 tables in {data_dir}. Run step 1 first.")

    fit = build_gm_spline_surrogate(paths, reference_element=cfg["system"]["reference_element"])

    # Match requested temperature
    T_target = sim["temperature_C"] + 273.15
    T_K = float(min(fit["temperatures_K"], key=lambda t: abs(t - T_target)))
    print(f"  Temperature: {sim['temperature_C']} °C  →  matched {T_K - 273.15:.1f} °C in data")

    output_dir = results_dir / cfg["output"]["run_name"]
    print(f"  Output → {output_dir}")

    dt = sim["dt"] if sim["dt"] != "None" and sim["dt"] is not None else None

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = solve_cahn_hilliard_binary(
            free_energy   = "calphad_gm_spline",
            N             = sim["N"],
            Lx            = sim["Lx"],
            kappa         = sim["kappa"],
            M             = sim["M"],
            c0            = sim["c0"],
            noise         = sim["noise"],
            seed          = sim["seed"],
            dt            = dt,
            n_steps       = sim["n_steps"],
            save_every    = sim["save_every"],
            calphad_fit   = fit,
            temperature_K = T_K,
            output_dir    = output_dir,
            save_frames   = True,
        )
    print(f"  Elapsed: {result.elapsed_seconds:.1f} s")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 3 — Phase-field simulation")
    parser.add_argument("--config", default="configs/nb_v.yaml")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg = apply_overrides(cfg, args.set)
    cfg = resolve_paths(cfg, _ROOT)

    print("── Step 3: Phase-field simulation ──")
    print(summary(cfg))
    result = run(cfg)
    print(f"\nDone. c_min={result.composition.min():.3f}  c_max={result.composition.max():.3f}")
