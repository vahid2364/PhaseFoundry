"""
Step 2 — Build G_m interpolant + spinodal analysis (general, any N-component)
===============================================================================
Reads HDF5 tables from step 1, builds a G_m interpolant (1-D CubicSpline for
binary, N-D RegularGridInterpolator for ternary/quaternary), and computes
spinodal boundaries.

Output (temperature-first folder convention)
--------------------------------------------
<results_dir>/phase_diagram.png          (binary only)
<results_dir>/GM_all.png                 (binary only)
<results_dir>/<T>C/thermodynamics/
    GM_spline.png                        (binary: spline verification)
    GM_surface.png                       (ternary: G_m surface heatmap)
    spinodal.png

Run standalone
--------------
    python workflows/02_fit_surrogate/run.py --config configs/nb_v.yaml
    python workflows/02_fit_surrogate/run.py --config configs/nb_hf_v.yaml
"""

from __future__ import annotations
import argparse, sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from phasefoundry.config import load_config, apply_overrides, resolve_paths, summary
from phasefoundry.thermodynamics.storage import load_table
from phasefoundry.thermodynamics.interpolant import build_interpolant


def _glob_tables(cfg: dict) -> list[Path]:
    """
    Return one HDF5 table per unique temperature, sorted by temperature.

    The data directory may contain files from multiple naming conventions
    (e.g. nb_v_bcc_500C.h5 and nb_v_bcc_a2_500C.h5).  When duplicates
    exist for the same temperature, the most recently modified file is kept
    and a warning is printed.
    """
    data_dir = Path(cfg["output"]["data_dir"])
    name     = cfg["system"]["name"]
    all_paths = sorted(data_dir.glob(f"{name}_*C.h5"))
    if not all_paths:
        raise FileNotFoundError(
            f"No HDF5 tables matching '{name}_*C.h5' in {data_dir}. "
            "Run step 1 first.")

    by_temp: dict[float, Path] = {}
    for p in all_paths:
        T = float(load_table(p)["metadata"]["temperature"])
        if T not in by_temp or p.stat().st_mtime > by_temp[T].stat().st_mtime:
            by_temp[T] = p

    n_dropped = len(all_paths) - len(by_temp)
    if n_dropped:
        print(f"  Warning: {n_dropped} duplicate table(s) found and skipped "
              f"(kept most recently modified file per temperature).")

    return sorted(by_temp.values(),
                  key=lambda p: float(load_table(p)["metadata"]["temperature"]))


def run(cfg: dict) -> None:
    results_dir = Path(cfg["output"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    elements  = cfg["system"]["elements"]
    ref_el    = cfg["system"]["reference_element"]
    n_ind     = len(elements) - 1
    sim_T_C   = cfg["simulation"]["temperature_C"]

    paths = _glob_tables(cfg)
    print(f"  Found {len(paths)} temperature table(s).")

    # binary: use existing spline + phase diagram code
    if n_ind == 1:
        _run_binary(cfg, paths, results_dir, ref_el, sim_T_C)
    else:
        _run_multicomponent(cfg, paths, results_dir, elements, ref_el,
                            n_ind, sim_T_C)

    print(f"  Outputs saved → {results_dir}")


def _run_binary(cfg, paths, results_dir, ref_el, sim_T_C):
    from phasefoundry.thermodynamics.fitting import build_gm_spline_surrogate
    from phasefoundry.thermodynamics.spinodal import (
        spinodal_boundary_binary, binodal_boundary_binary,
        plot_spinodal, plot_phase_diagram,
    )
    from phasefoundry.thermodynamics.plot import plot_free_energy_vs_composition

    ind_el = [e for e in cfg["system"]["elements"] if e != ref_el][0]
    x_el   = ref_el.capitalize()

    print("  Fitting G_m spline surrogate ...")
    fit = build_gm_spline_surrogate(paths, reference_element=ref_el)

    print("  Computing spinodal/binodal boundaries ...")
    raw_sp = spinodal_boundary_binary(paths, independent_element=ind_el)
    sp = dict(raw_sp)
    sp["x_spinodal_low"]  = [1-h if h is not None else None
                              for h in raw_sp["x_spinodal_high"]]
    sp["x_spinodal_high"] = [1-l if l is not None else None
                              for l in raw_sp["x_spinodal_low"]]

    raw_bn = binodal_boundary_binary(paths, independent_element=ind_el)
    bn = dict(raw_bn)
    bn["x_binodal_low"]  = [1-h if h is not None else None
                             for h in raw_bn["x_binodal_high"]]
    bn["x_binodal_high"] = [1-l if l is not None else None
                             for l in raw_bn["x_binodal_low"]]

    system_label = (f"{cfg['system']['name'].upper().replace('_','-')}  "
                    f"{cfg['system']['phase']}")
    plot_phase_diagram(spinodal_data=sp, binodal_data=bn, x_element=x_el,
                       system_name=system_label,
                       save_path=results_dir / "phase_diagram.png", show=False)
    plot_free_energy_vs_composition(paths, x_element=x_el, quantity="GM",
                                    temperatures_celsius=None,
                                    save_path=results_dir / "GM_all.png", show=False)

    # per-temperature spline verification plots
    for path in paths:
        data     = load_table(path)
        T_K      = float(data["metadata"]["temperature"])
        T_C_file = T_K - 273.15
        T_label  = f"{T_C_file:.0f}"
        thermo_dir = results_dir / f"{T_label}C" / "thermodynamics"
        thermo_dir.mkdir(parents=True, exist_ok=True)

        elements_stored = data["elements"]
        col_idx  = list(elements_stored).index(ref_el)
        x_raw    = data["compositions"][:, col_idx]
        y_raw    = data["GM"]

        T_key   = min(fit["fits"].keys(), key=lambda k: abs(float(k) - T_K))
        spl     = fit["fits"][T_key]["models"][ref_el]["spline_gm"]
        x_fine  = np.linspace(x_raw.min(), x_raw.max(), 400)

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.scatter(x_raw, y_raw, s=18, alpha=0.6, color="#1f77b4",
                   zorder=3, label="CALPHAD data")
        ax.plot(x_fine, spl(x_fine), color="#d62728", linewidth=2,
                label="CubicSpline fit")
        ax.set_xlabel(f"$x_{{\\rm {x_el}}}$", fontsize=12)
        ax.set_ylabel("$G_m$  [J/mol]", fontsize=12)
        ax.set_title(f"$G_m$ spline — {T_label} °C", fontsize=12)
        ax.legend(fontsize=9)
        fig.tight_layout()
        fig.savefig(thermo_dir / "GM_spline.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    thermo_sim = results_dir / f"{sim_T_C}C" / "thermodynamics"
    thermo_sim.mkdir(parents=True, exist_ok=True)
    plot_spinodal(sp, x_element=x_el,
                  save_path=thermo_sim / "spinodal.png", show=False)


def _run_multicomponent(cfg, paths, results_dir, elements, ref_el,
                        n_ind, sim_T_C):
    ind_elements = [e for e in elements if e != ref_el]

    for path in paths:
        data    = load_table(path)
        T_K     = float(data["metadata"]["temperature"])
        T_C     = T_K - 273.15
        T_label = f"{T_C:.0f}"
        thermo_dir = results_dir / f"{T_label}C" / "thermodynamics"
        thermo_dir.mkdir(parents=True, exist_ok=True)

        interp = build_interpolant(
            [path], elements=elements,
            reference_element=ref_el, temperature_K=T_K,
        )

        if n_ind == 2:
            _plot_gm_surface(data, interp, ind_elements, thermo_dir, T_label)

        # spinodal: scan compositions, find where min eigenvalue < 0
        _plot_spinodal_map(interp, n_ind, ind_elements, thermo_dir, T_label)
        print(f"    {T_label} °C  →  saved to {thermo_dir}")


def _plot_gm_surface(data, interp, ind_elements, thermo_dir, T_label):
    """Heatmap of G_m(c₁, c₂) for ternary systems."""
    comps = data["compositions"]
    GM    = data["GM"]

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
    """Spinodal map: min eigenvalue of Hessian over composition space."""
    n_scan = 30
    axis   = np.linspace(0.05, 0.90 / n_ind, n_scan)

    if n_ind == 1:
        c_scan = axis.reshape(1, -1)
    elif n_ind == 2:
        g0, g1 = np.meshgrid(axis, axis)
        mask   = (g0 + g1) < 0.95
        c_scan = np.stack([g0[mask], g1[mask]], axis=0)
    else:
        grids  = np.meshgrid(*[axis] * n_ind, indexing="ij")
        pts    = np.stack([g.ravel() for g in grids], axis=0)
        mask   = pts.sum(axis=0) < 0.95
        c_scan = pts[:, mask]

    try:
        lmin = interp.min_eigenvalue(c_scan)   # (n_pts,)
    except Exception:
        lmin = np.full(c_scan.shape[1], np.nan)
    valid    = ~np.isnan(lmin)
    unstable = valid & (lmin < 0)
    stable   = valid & (lmin >= 0)

    fig, ax = plt.subplots(figsize=(6, 5))
    if n_ind == 2:
        x0 = c_scan[0]
        x1 = c_scan[1]
        ax.scatter(x0[stable],   x1[stable],   c="#1f77b4", s=15,
                   label="stable", alpha=0.6)
        ax.scatter(x0[unstable], x1[unstable], c="#d62728", s=15,
                   label="spinodal (λ_min < 0)", alpha=0.8)
        ax.set_xlabel(f"$x_{{\\rm {ind_elements[0].capitalize()}}}$", fontsize=12)
        ax.set_ylabel(f"$x_{{\\rm {ind_elements[1].capitalize()}}}$", fontsize=12)
        ax.legend(fontsize=9)
    else:
        ax.scatter(c_scan[0, stable],   np.zeros(np.sum(stable)),
                   c="#1f77b4", s=20, label="stable")
        ax.scatter(c_scan[0, unstable], np.zeros(np.sum(unstable)),
                   c="#d62728", s=20, label="spinodal")
        ax.set_xlabel(f"$x_{{\\rm {ind_elements[0].capitalize()}}}$", fontsize=12)
        ax.set_yticks([])
        ax.legend(fontsize=9)

    ax.set_title(f"Spinodal map — {T_label} °C", fontsize=12)
    fig.tight_layout()
    fig.savefig(thermo_dir / "spinodal.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Step 2 — Build G_m interpolant (general)")
    parser.add_argument("--config", required=True)
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg = apply_overrides(cfg, args.set)
    cfg = resolve_paths(cfg, _ROOT)

    n_el  = len(cfg["system"]["elements"])
    n_ind = n_el - 1
    print(f"── Step 2: Build G_m interpolant ──")
    print(f"   System : {cfg['system']['name']}  "
          f"({n_el} components, {n_ind} independent field(s))")
    print(summary(cfg))
    run(cfg)
    print("\nDone.")
