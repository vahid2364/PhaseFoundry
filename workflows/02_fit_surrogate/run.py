"""Workflow step 2: fit G_m surrogates to the step-1 tables and map the spinodal.

For a binary system the step fits a G_m spline surrogate, computes the spinodal and
binodal boundaries over temperature, and writes ``phase_diagram.png`` and ``GM_all.png``
to ``<results_dir>``, a spline check ``GM_spline.png`` for each temperature, and the
spinodal against temperature as ``spinodal.png`` in the folder of the simulation
temperature. For three or more components it builds
a ``GmInterpolant`` for each temperature and writes ``spinodal.png`` (the sign of the
smallest Hessian eigenvalue on a composition grid), plus ``GM_surface.png`` for a
ternary. Per-temperature figures go to ``<results_dir>/<T>C/thermodynamics/``.

```bash
python workflows/02_fit_surrogate/run.py --config configs/nb_v_pipeline_test.yaml
python workflows/02_fit_surrogate/run.py --config configs/nb_hf_v_pipeline_test.yaml
```
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from phasefoundry.config import apply_overrides, load_config, resolve_paths, summary
from phasefoundry.thermodynamics.interpolant import build_interpolant
from phasefoundry.thermodynamics.storage import load_table

_ROOT = Path(__file__).resolve().parents[2]


def _glob_tables(cfg: dict) -> list[Path]:
    """Return one step-1 table per temperature, sorted by temperature.

    Tables from older and newer naming schemes can sit side by side in the data
    directory. When two tables have the same temperature, the most recently modified one
    is kept and a warning is printed.

    Args:
        cfg: Resolved config.

    Returns:
        Paths of the selected tables.

    Raises:
        FileNotFoundError: If no table matches ``<system name>_*C.h5``.
    """
    data_dir = Path(cfg["output"]["data_dir"])
    name = cfg["system"]["name"]
    all_paths = sorted(data_dir.glob(f"{name}_*C.h5"))
    if not all_paths:
        raise FileNotFoundError(f"No HDF5 tables matching '{name}_*C.h5' in {data_dir}. Run step 1 first.")

    by_temp: dict[float, Path] = {}
    for p in all_paths:
        T = float(load_table(p)["metadata"]["temperature"])
        if T not in by_temp or p.stat().st_mtime > by_temp[T].stat().st_mtime:
            by_temp[T] = p

    n_dropped = len(all_paths) - len(by_temp)
    if n_dropped:
        print(f"  Warning: {n_dropped} duplicate table(s) found and skipped (kept most recently modified file per temperature).")

    return sorted(by_temp.values(), key=lambda p: float(load_table(p)["metadata"]["temperature"]))


def run(cfg: dict) -> None:
    """Fit the surrogates and draw the spinodal figures for the configured system.

    Args:
        cfg: Resolved config.

    Raises:
        FileNotFoundError: If the step-1 tables are missing.
    """
    results_dir = Path(cfg["output"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    elements = cfg["system"]["elements"]
    ref_el = cfg["system"]["reference_element"]
    n_ind = len(elements) - 1
    sim_T_C = cfg["simulation"]["temperature_C"]

    paths = _glob_tables(cfg)
    print(f"  Found {len(paths)} temperature table(s).")

    if n_ind == 1:
        _run_binary(cfg, paths, results_dir, ref_el, sim_T_C)
    else:
        _run_multicomponent(cfg, paths, results_dir, elements, ref_el, n_ind, sim_T_C)

    print(f"  Outputs saved → {results_dir}")


def _run_binary(cfg, paths, results_dir, ref_el, sim_T_C):
    """Fit the G_m spline and draw the binary phase diagram and spinodal figures.

    The boundaries are converted from the independent element's mole fraction to the
    reference element's, which is the x-axis of every figure.

    Args:
        cfg: Resolved config.
        paths: Step-1 tables, one per temperature.
        results_dir: Root output directory.
        ref_el: Reference element.
        sim_T_C: Simulation temperature in °C; ``spinodal.png`` goes to its folder.
    """
    from phasefoundry.thermodynamics.fitting import build_gm_spline_surrogate
    from phasefoundry.thermodynamics.plot import plot_free_energy_vs_composition, plot_phase_diagram, plot_spinodal
    from phasefoundry.thermodynamics.spinodal import (
        binodal_boundary_binary,
        spinodal_boundary_binary,
    )

    ind_el = [e for e in cfg["system"]["elements"] if e != ref_el][0]
    x_el = ref_el.capitalize()

    print("  Fitting G_m spline surrogate ...")
    fit = build_gm_spline_surrogate(paths, reference_element=ref_el)

    print("  Computing spinodal/binodal boundaries ...")
    raw_sp = spinodal_boundary_binary(paths, independent_element=ind_el)
    sp = dict(raw_sp)
    sp["x_spinodal_low"] = [1 - hi if hi is not None else None for hi in raw_sp["x_spinodal_high"]]
    sp["x_spinodal_high"] = [1 - lo if lo is not None else None for lo in raw_sp["x_spinodal_low"]]

    raw_bn = binodal_boundary_binary(paths, independent_element=ind_el)
    bn = dict(raw_bn)
    bn["x_binodal_low"] = [1 - hi if hi is not None else None for hi in raw_bn["x_binodal_high"]]
    bn["x_binodal_high"] = [1 - lo if lo is not None else None for lo in raw_bn["x_binodal_low"]]

    system_label = f"{cfg['system']['name'].upper().replace('_', '-')}  {cfg['system']['phase']}"
    plot_phase_diagram(
        spinodal_data=sp,
        binodal_data=bn,
        x_element=x_el,
        system_name=system_label,
        save_path=results_dir / "phase_diagram.png",
        show=False,
    )
    plot_free_energy_vs_composition(
        paths, x_element=x_el, quantity="GM", temperatures_celsius=None, save_path=results_dir / "GM_all.png", show=False
    )

    for path in paths:
        data = load_table(path)
        T_K = float(data["metadata"]["temperature"])
        T_C_file = T_K - 273.15
        T_label = f"{T_C_file:.0f}"
        thermo_dir = results_dir / f"{T_label}C" / "thermodynamics"
        thermo_dir.mkdir(parents=True, exist_ok=True)

        elements_stored = data["elements"]
        col_idx = list(elements_stored).index(ref_el)
        x_raw = data["compositions"][:, col_idx]
        y_raw = data["GM"]

        T_key = min(fit["fits"].keys(), key=lambda k: abs(float(k) - T_K))
        spl = fit["fits"][T_key]["models"][ref_el]["spline_gm"]
        x_fine = np.linspace(x_raw.min(), x_raw.max(), 400)

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.scatter(x_raw, y_raw, s=18, alpha=0.6, color="#1f77b4", zorder=3, label="CALPHAD data")
        ax.plot(x_fine, spl(x_fine), color="#d62728", linewidth=2, label="CubicSpline fit")
        ax.set_xlabel(f"$x_{{\\rm {x_el}}}$", fontsize=12)
        ax.set_ylabel("$G_m$  [J/mol]", fontsize=12)
        ax.set_title(f"$G_m$ spline — {T_label} °C", fontsize=12)
        ax.legend(fontsize=9)
        fig.tight_layout()
        fig.savefig(thermo_dir / "GM_spline.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    thermo_sim = results_dir / f"{sim_T_C}C" / "thermodynamics"
    thermo_sim.mkdir(parents=True, exist_ok=True)
    plot_spinodal(sp, x_element=x_el, save_path=thermo_sim / "spinodal.png", show=False)


def _run_multicomponent(cfg, paths, results_dir, elements, ref_el, n_ind, sim_T_C):
    """Build an interpolant per temperature and draw its spinodal map and G_m surface.

    Args:
        cfg: Resolved config.
        paths: Step-1 tables, one per temperature.
        results_dir: Root output directory.
        elements: All elements of the system.
        ref_el: Reference element.
        n_ind: Number of independent elements.
        sim_T_C: Simulation temperature in °C (not used).
    """
    ind_elements = [e for e in elements if e != ref_el]

    for path in paths:
        data = load_table(path)
        T_K = float(data["metadata"]["temperature"])
        T_C = T_K - 273.15
        T_label = f"{T_C:.0f}"
        thermo_dir = results_dir / f"{T_label}C" / "thermodynamics"
        thermo_dir.mkdir(parents=True, exist_ok=True)

        interp = build_interpolant(
            [path],
            elements=elements,
            reference_element=ref_el,
            temperature_K=T_K,
        )

        if n_ind == 2:
            _plot_gm_surface(data, interp, ind_elements, thermo_dir, T_label)

        _plot_spinodal_map(interp, n_ind, ind_elements, thermo_dir, T_label)
        print(f"    {T_label} °C  →  saved to {thermo_dir}")


def _plot_gm_surface(data, interp, ind_elements, thermo_dir, T_label):
    """Save ``GM_surface.png``, a scatter of the table's G_m over the two independent fractions.

    Args:
        data: Table from ``load_table``.
        interp: Interpolant of the table (not used).
        ind_elements: The two independent elements.
        thermo_dir: Output directory.
        T_label: Temperature label in °C for the title.
    """
    comps = data["compositions"]
    GM = data["GM"]

    el0, el1 = ind_elements[0], ind_elements[1]
    elements_stored = data["elements"]
    col0 = list(elements_stored).index(el0)
    col1 = list(elements_stored).index(el1)

    x0 = comps[:, col0]
    x1 = comps[:, col1]

    fig, ax = plt.subplots(figsize=(6, 5))
    sc = ax.scatter(x0, x1, c=GM, cmap="viridis", s=10)
    ax.set_xlabel(f"$x_{{\\rm {el0.capitalize()}}}$", fontsize=12)
    ax.set_ylabel(f"$x_{{\\rm {el1.capitalize()}}}$", fontsize=12)
    ax.set_title(f"$G_m$ landscape — {T_label} °C", fontsize=12)
    plt.colorbar(sc, ax=ax, label="$G_m$ [J/mol]")
    fig.tight_layout()
    fig.savefig(thermo_dir / "GM_surface.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_spinodal_map(interp, n_ind, ind_elements, thermo_dir, T_label):
    """Save ``spinodal.png``, marking grid compositions where the Hessian has a negative eigenvalue.

    The grid has 30 points per independent axis from 0.05 to 0.90 / ``n_ind``, limited to
    compositions whose independent fractions sum to less than 0.95. If the eigenvalue
    evaluation fails, every point is treated as undetermined and the figure is empty.

    Args:
        interp: Interpolant whose ``min_eigenvalue`` is evaluated.
        n_ind: Number of independent elements.
        ind_elements: Independent elements, used for the axis labels.
        thermo_dir: Output directory.
        T_label: Temperature label in °C for the title.
    """
    n_scan = 30
    axis = np.linspace(0.05, 0.90 / n_ind, n_scan)

    if n_ind == 1:
        c_scan = axis.reshape(1, -1)
    elif n_ind == 2:
        g0, g1 = np.meshgrid(axis, axis)
        mask = (g0 + g1) < 0.95
        c_scan = np.stack([g0[mask], g1[mask]], axis=0)
    else:
        grids = np.meshgrid(*[axis] * n_ind, indexing="ij")
        pts = np.stack([g.ravel() for g in grids], axis=0)
        mask = pts.sum(axis=0) < 0.95
        c_scan = pts[:, mask]

    try:
        lmin = interp.min_eigenvalue(c_scan)
    except Exception:
        lmin = np.full(c_scan.shape[1], np.nan)
    valid = ~np.isnan(lmin)
    unstable = valid & (lmin < 0)
    stable = valid & (lmin >= 0)

    fig, ax = plt.subplots(figsize=(6, 5))
    if n_ind == 2:
        x0 = c_scan[0]
        x1 = c_scan[1]
        ax.scatter(x0[stable], x1[stable], c="#1f77b4", s=15, label="stable", alpha=0.6)
        ax.scatter(x0[unstable], x1[unstable], c="#d62728", s=15, label="spinodal (λ_min < 0)", alpha=0.8)
        ax.set_xlabel(f"$x_{{\\rm {ind_elements[0].capitalize()}}}$", fontsize=12)
        ax.set_ylabel(f"$x_{{\\rm {ind_elements[1].capitalize()}}}$", fontsize=12)
        ax.legend(fontsize=9)
    else:
        ax.scatter(c_scan[0, stable], np.zeros(np.sum(stable)), c="#1f77b4", s=20, label="stable")
        ax.scatter(c_scan[0, unstable], np.zeros(np.sum(unstable)), c="#d62728", s=20, label="spinodal")
        ax.set_xlabel(f"$x_{{\\rm {ind_elements[0].capitalize()}}}$", fontsize=12)
        ax.set_yticks([])
        ax.legend(fontsize=9)

    ax.set_title(f"Spinodal map — {T_label} °C", fontsize=12)
    fig.tight_layout()
    fig.savefig(thermo_dir / "spinodal.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    logging.basicConfig(format="%(message)s", stream=sys.stdout)
    logging.getLogger("phasefoundry").setLevel(logging.INFO)
    parser = argparse.ArgumentParser(description="Step 2 — Build G_m interpolant (general)")
    parser.add_argument("--config", required=True)
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg = apply_overrides(cfg, args.set)
    cfg = resolve_paths(cfg, _ROOT)

    n_el = len(cfg["system"]["elements"])
    n_ind = n_el - 1
    print("── Step 2: Build G_m interpolant ──")
    print(f"   System : {cfg['system']['name']}  ({n_el} components, {n_ind} independent field(s))")
    print(summary(cfg))
    run(cfg)
    print("\nDone.")
