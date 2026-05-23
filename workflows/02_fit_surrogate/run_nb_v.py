"""
Step 2 — Fit G_m spline surrogate + spinodal/binodal
=====================================================
Reads HDF5 tables from step 1, fits a CubicSpline to G_m(x_V),
and computes spinodal + binodal boundaries.

Output (temperature-first convention)
--------------------------------------
<results_dir>/phase_diagram.png          ← multi-temperature overview
<results_dir>/GM_all.png                 ← G_m curves at all temperatures
<results_dir>/<T>C/thermodynamics/       ← per-temperature thermo outputs
    spinodal.png
    GM_spline.png

Run standalone
--------------
    python workflows/02_fit_surrogate/run_nb_v.py --config configs/nb_v.yaml
"""

from __future__ import annotations
import argparse, sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from calphad_pf.config import load_config, apply_overrides, resolve_paths, summary
from calphad_pf.thermodynamics.fitting import build_gm_spline_surrogate
from calphad_pf.thermodynamics.spinodal import (
    spinodal_boundary_binary, binodal_boundary_binary,
    plot_spinodal, plot_phase_diagram,
)
from calphad_pf.thermodynamics.plot import plot_free_energy_vs_composition


def run(cfg: dict) -> dict:
    """Fit surrogate and compute phase boundaries. Returns fit dict."""
    data_dir    = Path(cfg["output"]["data_dir"])
    results_dir = Path(cfg["output"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    temps_C     = cfg["thermo"]["temperatures_C"]

    paths = sorted(data_dir.glob("nb_v_bcc_*C.h5"))
    if not paths:
        raise FileNotFoundError(f"No HDF5 tables in {data_dir}. Run step 1 first.")
    print(f"  Found {len(paths)} temperature tables.")

    # Fit G_m spline
    print("  Fitting G_m spline surrogate ...")
    fit = build_gm_spline_surrogate(paths, reference_element=cfg["system"]["reference_element"])
    print(f"  Surrogate covers {len(fit['temperatures_K'])} temperatures.")

    # Spinodal + binodal
    print("  Computing spinodal boundaries ...")
    raw_sp = spinodal_boundary_binary(paths, independent_element="NB", temperatures_celsius=temps_C)
    sp = dict(raw_sp)
    sp["x_spinodal_low"]  = [1-h if h is not None else None for h in raw_sp["x_spinodal_high"]]
    sp["x_spinodal_high"] = [1-l if l is not None else None for l in raw_sp["x_spinodal_low"]]

    print("  Computing binodal boundaries ...")
    raw_bn = binodal_boundary_binary(paths, independent_element="NB", temperatures_celsius=temps_C)
    bn = dict(raw_bn)
    bn["x_binodal_low"]  = [1-h if h is not None else None for h in raw_bn["x_binodal_high"]]
    bn["x_binodal_high"] = [1-l if l is not None else None for l in raw_bn["x_binodal_low"]]

    x_el = cfg["system"]["reference_element"].capitalize()
    sim_T = cfg["simulation"]["temperature_C"]

    # Multi-temperature overview plots → results_dir root
    plot_phase_diagram(spinodal_data=sp, binodal_data=bn, x_element=x_el,
                       system_name=f"{cfg['system']['name'].upper().replace('_','-')}  {cfg['system']['phase']}",
                       save_path=results_dir / "phase_diagram.png", show=False)
    plot_free_energy_vs_composition(paths, x_element=x_el, quantity="GM",
                                    temperatures_celsius=temps_C,
                                    save_path=results_dir / "GM_all.png", show=False)

    # Per-temperature outputs → results_dir / {T}C / thermodynamics /
    for T_C, path in zip(temps_C, paths):
        thermo_dir = results_dir / f"{T_C}C" / "thermodynamics"
        thermo_dir.mkdir(parents=True, exist_ok=True)
        plot_free_energy_vs_composition([path], x_element=x_el, quantity="GM",
                                        temperatures_celsius=[T_C],
                                        save_path=thermo_dir / "GM_spline.png", show=False)

    # Spinodal diagram also at simulation temperature thermodynamics folder
    thermo_sim = results_dir / f"{sim_T}C" / "thermodynamics"
    thermo_sim.mkdir(parents=True, exist_ok=True)
    plot_spinodal(sp, x_element=x_el,
                  save_path=thermo_sim / "spinodal.png", show=False)

    print(f"  Plots saved → {results_dir}")
    return fit


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 2 — Fit G_m spline surrogate")
    parser.add_argument("--config", default="configs/nb_v.yaml")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg = apply_overrides(cfg, args.set)
    cfg = resolve_paths(cfg, _ROOT)

    print("── Step 2: Fit G_m spline surrogate ──")
    print(summary(cfg))
    run(cfg)
    print("\nDone.")
