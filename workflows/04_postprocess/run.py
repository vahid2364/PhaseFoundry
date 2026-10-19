"""Workflow step 4: redraw the free-energy history and plot the spinodal criterion.

Reads ``free_energy_log.txt`` from the step-3 run directory and redraws
``free_energy_history.png`` there. It then plots ∂μ/∂x of the first independent element,
the stored ``dMU/<el>_<el>``, against temperature at the table point closest to a
reference-element fraction of 0.3, and saves it as
``<results_dir>/<T>C/thermodynamics/d2G_vs_T.png``.

```bash
python workflows/04_postprocess/run.py --config configs/nb_v_pipeline_test.yaml
python workflows/04_postprocess/run.py --config configs/nb_v_pipeline_test.yaml --set simulation.temperature_C=600
```
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

from phasefoundry.config import apply_overrides, load_config, resolve_paths, summary
from phasefoundry.phasefield import plot_free_energy_history

_ROOT = Path(__file__).resolve().parents[2]


def plot_d2G_vs_T(cfg: dict, thermo_dir: Path) -> None:
    """Save ``d2G_vs_T.png``, the spinodal criterion of the first independent element against temperature.

    The plotted value is the stored ``dMU/<el>_<el>`` at the table point whose
    reference-element fraction is closest to 0.3. Tables without that key are skipped, and
    the plot is skipped if no table has it.

    Args:
        cfg: Resolved config.
        thermo_dir: Output directory, created if needed.
    """
    data_dir = Path(cfg["output"]["data_dir"])
    elements = cfg["system"]["elements"]
    ref_el = cfg["system"]["reference_element"]
    sys_name = cfg["system"]["name"]

    ind_els = [e for e in elements if e != ref_el]
    el0 = ind_els[0]
    dmu_key = f"dMU/{el0}_{el0}"

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
                # step 1 stores the reference element in the last column
                x_ref = comps[:, -1]
                d2G = f[dmu_key][:]
            d2G_vals.append(d2G[int(np.argmin(np.abs(x_ref - x_ref_target)))])
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
    t_axis = t_vals if all(t is not None for t in t_vals) else list(range(len(d2G_vals)))
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
    """Redraw the free-energy history of the step-3 run and plot the spinodal criterion.

    Args:
        cfg: Resolved config.

    Raises:
        FileNotFoundError: If the step-3 run directory does not exist.
    """
    T_label = f"{cfg['simulation']['temperature_C']}C"
    results_dir = Path(cfg["output"]["results_dir"])
    run_dir = results_dir / T_label / "kinetics" / cfg["output"]["run_name"]
    thermo_dir = results_dir / T_label / "thermodynamics"

    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}\nRun step 3 first.")

    log = run_dir / "free_energy_log.txt"
    if log.exists():
        import pandas as pd

        df = pd.read_csv(log, sep=r"\s+")

        class _R:
            """Stand-in for ``CHResult`` with the energy histories read from the log."""

            time_history = df["time_s"].tolist()
            free_energy_history = df["F_total"].tolist()
            bulk_energy_history = df["F_bulk"].tolist()
            grad_energy_history = df["F_grad"].tolist()
            mix_energy_history = df["dG_mix"].tolist() if "dG_mix" in df.columns else []
            composition_history = []

        plot_free_energy_history(_R(), output_dir=run_dir)
        print(f"  Free energy history → {run_dir / 'free_energy_history.png'}")
    else:
        print(f"  No free_energy_log.txt found in {run_dir} — skipping.")

    plot_d2G_vs_T(cfg, thermo_dir)


if __name__ == "__main__":
    logging.basicConfig(format="%(message)s", stream=sys.stdout)
    logging.getLogger("phasefoundry").setLevel(logging.INFO)
    parser = argparse.ArgumentParser(description="Step 4 — Post-processing (general, any N-component system)")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument(
        "--set",
        nargs="*",
        default=[],
        metavar="KEY=VALUE",
        help="Override config values, e.g. --set simulation.temperature_C=600",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg = apply_overrides(cfg, args.set)
    cfg = resolve_paths(cfg, _ROOT)

    print("── Step 4: Post-processing ──")
    print(summary(cfg))
    run(cfg)
    print("\nDone.")
