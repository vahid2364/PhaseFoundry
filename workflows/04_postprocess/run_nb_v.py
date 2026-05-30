"""
Step 4 — Post-processing and figures
======================================
Reads simulation outputs from step 3 and produces figures.

Output
------
<output.results_dir>/<temperature_C>C/kinetics/<output.run_name>/
    free_energy_history.png   (re-generated)
<output.results_dir>/<temperature_C>C/thermodynamics/
    d2G_vs_T.png

Run standalone
--------------
    python workflows/04_postprocess/run_nb_v.py --config configs/nb_v.yaml
"""

from __future__ import annotations
import argparse, sys
from pathlib import Path

import h5py
import numpy as np
import matplotlib.pyplot as plt

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from calphad_pf.config import load_config, apply_overrides, resolve_paths, summary
from calphad_pf.phasefield import plot_free_energy_history


def plot_d2G_vs_T(cfg: dict, thermo_dir: Path) -> None:
    """Plot ∂²G/∂x²_V vs temperature at x_V = 0.3."""
    data_dir = Path(cfg["output"]["data_dir"])
    paths    = sorted(data_dir.glob("nb_v_bcc_*C.h5"))
    temps_C  = cfg["thermo"]["temperatures_C"]
    if not paths:
        print("  Skipping d²G/dT plot — no HDF5 tables found.")
        return

    x_V_target, d2G_vals = 0.3, []
    for p in paths:
        with h5py.File(p, "r") as f:
            x_V = f["compositions"][:, 1]
            d2G = f["dMU/NB_NB"][:]
        d2G_vals.append(d2G[int(np.argmin(np.abs(x_V - x_V_target)))])

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(temps_C[:len(d2G_vals)], d2G_vals, "o-", color="#d62728", lw=1.8)
    ax.axhline(0, color="black", lw=1.2, ls="--")
    ax.set_xlabel("Temperature (°C)", fontsize=12)
    ax.set_ylabel(r"$\partial^2 G_m / \partial x_{\rm V}^2$  (J/mol)", fontsize=12)
    ax.set_title(r"Spinodal criterion at $x_{\rm V} = 0.3$", fontsize=12)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    thermo_dir.mkdir(parents=True, exist_ok=True)
    p = thermo_dir / "d2G_vs_T.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {p}")


def run(cfg: dict) -> None:
    T_label     = f"{cfg['simulation']['temperature_C']}C"
    results_dir = Path(cfg["output"]["results_dir"])
    run_dir     = results_dir / T_label / "kinetics" / cfg["output"]["run_name"]
    thermo_dir  = results_dir / T_label / "thermodynamics"

    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}\nRun step 3 first.")

    # Re-plot free energy history from log file
    log = run_dir / "free_energy_log.txt"
    if log.exists():
        import pandas as pd
        df = pd.read_csv(log, sep=r"\s+")

        class _R:
            time_history        = df["time_s"].tolist()
            free_energy_history = df["F_total"].tolist()
            bulk_energy_history = df["F_bulk"].tolist()
            grad_energy_history = df["F_grad"].tolist()
            mix_energy_history  = df["dG_mix"].tolist() if "dG_mix" in df.columns else []
            composition_history = []

        plot_free_energy_history(_R(), output_dir=run_dir)
        print(f"  Free energy history → {run_dir / 'free_energy_history.png'}")

    plot_d2G_vs_T(cfg, thermo_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 4 — Post-processing")
    parser.add_argument("--config", default="configs/nb_v.yaml")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg = apply_overrides(cfg, args.set)
    cfg = resolve_paths(cfg, _ROOT)

    print("── Step 4: Post-processing ──")
    print(summary(cfg))
    run(cfg)
    print("\nDone.")
