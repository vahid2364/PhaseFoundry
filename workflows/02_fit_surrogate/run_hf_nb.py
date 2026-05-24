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
    python workflows/02_fit_surrogate/run_nb_v.py --config configs/hf_nb.yaml
"""

from __future__ import annotations
import argparse, sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from calphad_pf.config import load_config, apply_overrides, resolve_paths, summary
from calphad_pf.thermodynamics.fitting import build_gm_spline_surrogate
from calphad_pf.thermodynamics.spinodal import (
    spinodal_boundary_binary, binodal_boundary_binary,
    plot_spinodal, plot_phase_diagram,
)
from calphad_pf.thermodynamics.plot import plot_free_energy_vs_composition
from calphad_pf.thermodynamics.storage import load_table


def run(cfg: dict) -> dict:
    """Fit surrogate and compute phase boundaries. Returns fit dict."""
    data_dir    = Path(cfg["output"]["data_dir"])
    results_dir = Path(cfg["output"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    temps_C     = cfg["thermo"]["temperatures_C"]

    paths = sorted(data_dir.glob("hf_nb_bcc_*C.h5"))
    if not paths:
        raise FileNotFoundError(f"No HDF5 tables in {data_dir}. Run step 1 first.")
    print(f"  Found {len(paths)} temperature tables.")

    # Fit G_m spline
    print("  Fitting G_m spline surrogate ...")
    fit = build_gm_spline_surrogate(paths, reference_element=cfg["system"]["reference_element"])
    print(f"  Surrogate covers {len(fit['temperatures_K'])} temperatures.")

    # Spinodal + binodal
    # IMPORTANT: do NOT pass temperatures_celsius from config — paths are sorted
    # alphabetically so the list order doesn't match the config order. Each
    # function reads T from the HDF5 file metadata to get the correct label.
    print("  Computing spinodal boundaries ...")
    raw_sp = spinodal_boundary_binary(paths, independent_element="HF")
    sp = dict(raw_sp)
    sp["x_spinodal_low"]  = [1-h if h is not None else None for h in raw_sp["x_spinodal_high"]]
    sp["x_spinodal_high"] = [1-l if l is not None else None for l in raw_sp["x_spinodal_low"]]

    print("  Computing binodal boundaries ...")
    raw_bn = binodal_boundary_binary(paths, independent_element="HF")
    bn = dict(raw_bn)
    bn["x_binodal_low"]  = [1-h if h is not None else None for h in raw_bn["x_binodal_high"]]
    bn["x_binodal_high"] = [1-l if l is not None else None for l in raw_bn["x_binodal_low"]]

    x_el = cfg["system"]["reference_element"]
    sim_T = cfg["simulation"]["temperature_C"]

    # Multi-temperature overview plots → results_dir root
    plot_phase_diagram(spinodal_data=sp, binodal_data=bn, x_element=x_el,
                       system_name=f"{cfg['system']['name'].upper().replace('_','-')}  {cfg['system']['phase']}",
                       save_path=results_dir / "phase_diagram.png", show=False)
    plot_free_energy_vs_composition(paths, x_element=x_el, quantity="GM",
                                    temperatures_celsius=None,   # read T from metadata
                                    save_path=results_dir / "GM_all.png", show=False)

    # Per-temperature outputs → results_dir / {T}C / thermodynamics /
    # NOTE: paths are sorted alphabetically (1000C before 100C before 25C),
    # so we always read T from the file metadata — never from the config list.
    x_ref = cfg["system"]["reference_element"]
    for path in paths:
        data     = load_table(path)
        T_K      = float(data["metadata"]["temperature"])
        T_C_file = T_K - 273.15
        T_label  = f"{T_C_file:.0f}"

        thermo_dir = results_dir / f"{T_label}C" / "thermodynamics"
        thermo_dir.mkdir(parents=True, exist_ok=True)

        # --- scatter (raw CALPHAD points) + spline overlay ---
        elements = data["elements"]
        col_idx  = elements.index(x_ref)
        x_raw    = data["compositions"][:, col_idx]
        y_raw    = data["GM"]

        # find matching spline (keys are floats from file metadata, exact match)
        T_key_found = min(fit["fits"].keys(), key=lambda k: abs(float(k) - T_K))
        spline_fn = fit["fits"][T_key_found]["models"][x_ref]["spline_gm"]
        x_fine = np.linspace(x_raw.min(), x_raw.max(), 400)
        y_fine = spline_fn(x_fine)

        fig_sp, ax_sp = plt.subplots(figsize=(7, 4))
        ax_sp.scatter(x_raw, y_raw, s=18, alpha=0.6, color="#1f77b4",
                      zorder=3, label="CALPHAD data")
        ax_sp.plot(x_fine, y_fine, color="#d62728", linewidth=2,
                   label="CubicSpline fit")
        ax_sp.set_xlabel(f"$x_{{\\rm {x_el}}}$  (mol fraction)", fontsize=12)
        ax_sp.set_ylabel("$G_m$  [J/mol]", fontsize=12)
        ax_sp.set_title(f"$G_m$ spline verification — {T_label} °C", fontsize=12)
        ax_sp.legend(fontsize=9)
        ax_sp.grid(False)
        fig_sp.tight_layout()
        fig_sp.savefig(thermo_dir / "GM_spline.png", dpi=150, bbox_inches="tight")
        plt.close(fig_sp)
        print(f"    Saved GM_spline.png → {thermo_dir}")

    # Spinodal diagram also at simulation temperature thermodynamics folder
    thermo_sim = results_dir / f"{sim_T}C" / "thermodynamics"
    thermo_sim.mkdir(parents=True, exist_ok=True)
    plot_spinodal(sp, x_element=x_el,
                  save_path=thermo_sim / "spinodal.png", show=False)

    print(f"  Plots saved → {results_dir}")
    return fit


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 2 — Fit G_m spline surrogate")
    parser.add_argument("--config", default="configs/hf_nb.yaml")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg = apply_overrides(cfg, args.set)
    cfg = resolve_paths(cfg, _ROOT)

    print("── Step 2: Fit G_m spline surrogate ──")
    print(summary(cfg))
    run(cfg)
    print("\nDone.")
