"""
Step 4 — Post-processing and figures  (general, any N-component system)
=======================================================================
Reads simulation outputs from step 3 and produces figures.

Output
------
<output.results_dir>/<temperature_C>C/kinetics/<output.run_name>/
    free_energy_history.png   (re-generated from log)
<output.results_dir>/<temperature_C>C/thermodynamics/
    d2G_vs_T.png              (spinodal criterion vs temperature)

Run standalone
--------------
    python workflows/04_postprocess/run.py --config configs/nb_v.yaml
    python workflows/04_postprocess/run.py --config configs/nb_hf_v.yaml

    # Override temperature
    python workflows/04_postprocess/run.py --config configs/nb_v.yaml \
        --set simulation.temperature_C=600
"""

from __future__ import annotations
import argparse, sys
import logging
from pathlib import Path
import sys

import h5py
import numpy as np
import matplotlib.pyplot as plt

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from phasefoundry.config import load_config, apply_overrides, resolve_paths, summary
from phasefoundry.phasefield import plot_free_energy_history


def plot_d2G_vs_T(cfg: dict, thermo_dir: Path) -> None:
    """
    Plot ∂²G/∂x²_{el0} vs temperature at a fixed reference composition.

    Works for any binary or multicomponent system — the first independent
    element and the HDF5 dMU key are derived from the config.
    """
    data_dir  = Path(cfg["output"]["data_dir"])
    temps_C   = cfg["thermo"]["temperatures_C"]
    elements  = cfg["system"]["elements"]           # e.g. ["NB", "V"]
    ref_el    = cfg["system"]["reference_element"]  # e.g. "V"
    phase     = cfg["system"]["phase"].lower().replace("_", "")  # e.g. "bcca2"
    sys_name  = cfg["system"]["name"]               # e.g. "nb_v"

    ind_els = [e for e in elements if e != ref_el]
    el0 = ind_els[0]                                # e.g. "NB"
    dmu_key = f"dMU/{el0}_{el0}"                   # e.g. "dMU/NB_NB"

    paths = sorted(data_dir.glob(f"{sys_name}_*C.h5"))
    if not paths:
        paths = sorted(data_dir.glob("*.h5"))
    if not paths:
        print(f"  Skipping d²G/dT plot — no HDF5 tables found in {data_dir}")
        return

    x_ref_target = 0.3
    d2G_vals, t_vals = [], []
    for p in paths:
        try:
            with h5py.File(p, "r") as f:
                if dmu_key not in f:
                    continue
                comps = f["compositions"][:]
                # composition of reference element is the last column
                x_ref = comps[:, -1]
                d2G   = f[dmu_key][:]
            d2G_vals.append(d2G[int(np.argmin(np.abs(x_ref - x_ref_target)))])
            # extract temperature from filename: ..._500C.h5 → 500
            try:
                t_vals.append(int(p.stem.split("_")[-1].replace("C", "")))
            except ValueError:
                t_vals.append(None)
        except Exception as exc:
            print(f"  Warning: skipped {p.name} ({exc})")

    if not d2G_vals:
        print(f"  Skipping d²G/dT plot — key '{dmu_key}' not found in tables.")
        return

    x_label = ref_el.capitalize()
    fig, ax = plt.subplots(figsize=(7, 4))
    t_axis  = t_vals if all(t is not None for t in t_vals) else list(range(len(d2G_vals)))
    ax.plot(t_axis, d2G_vals, "o-", color="#d62728", lw=1.8)
    ax.axhline(0, color="black", lw=1.2, ls="--")
    ax.set_xlabel("Temperature (°C)", fontsize=12)
    ax.set_ylabel(
        rf"$\partial^2 G_m / \partial x_{{\rm {el0.capitalize()}}}^2$  (J/mol)",
        fontsize=12,
    )
    ax.set_title(
        rf"Spinodal criterion at $x_{{\rm {x_label}}} = {x_ref_target}$",
        fontsize=12,
    )
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    thermo_dir.mkdir(parents=True, exist_ok=True)
    out = thermo_dir / "d2G_vs_T.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out}")


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
    else:
        print(f"  No free_energy_log.txt found in {run_dir} — skipping.")

    plot_d2G_vs_T(cfg, thermo_dir)


if __name__ == "__main__":
    logging.basicConfig(format="%(message)s", stream=sys.stdout)
    logging.getLogger("phasefoundry").setLevel(logging.INFO)
    parser = argparse.ArgumentParser(
        description="Step 4 — Post-processing (general, any N-component system)"
    )
    parser.add_argument("--config", default="configs/nb_v.yaml",
                        help="Path to YAML config file")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                        help="Override config values, e.g. --set simulation.temperature_C=600")
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg = apply_overrides(cfg, args.set)
    cfg = resolve_paths(cfg, _ROOT)

    print("── Step 4: Post-processing ──")
    print(summary(cfg))
    run(cfg)
    print("\nDone.")
