"""Pipeline step 5: validation figures for the thermodynamic tables.

Reads the step-1 table in ``output.data_dir`` whose temperature is closest to
``simulation.temperature_C`` and writes two figures, as PDF and PNG, to
``<output.results_dir>/<temperature_C>C/thermodynamics/``: ``thermo_properties`` (G_m, the
chemical potentials and ∂μ_i/∂x_i against composition) and ``derivative_comparison`` (the
FD, Jansson and hybrid interpolants against the raw CALPHAD points). It also prints the RMSE
of each interpolant against the table.

```bash
python workflows/05_validate/run.py --config configs/nb_v_pipeline_test.yaml
python workflows/05_validate/run.py --config configs/nb_v_pipeline_test.yaml --set simulation.temperature_C=600
python workflows/run_all.py --config configs/nb_v_pipeline_test.yaml --only 5
```
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker
from matplotlib.lines import Line2D

from phasefoundry.config import apply_overrides, load_config, resolve_paths, summary
from phasefoundry.thermodynamics.interpolant import build_interpolant
from phasefoundry.thermodynamics.storage import load_table

mpl.use("Agg")
_ROOT = Path(__file__).resolve().parents[2]

mpl.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 9,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "lines.linewidth": 1.6,
        "axes.linewidth": 0.8,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "figure.dpi": 150,
        "savefig.dpi": 300,
    }
)

SPINODAL_COLOR = "#ffd966"
SPINODAL_ALPHA = 0.40


def _load_best_table(data_dir: Path, T_target: float) -> tuple[Path, dict]:
    """Load the table in ``data_dir`` whose temperature is closest to ``T_target``.

    Args:
        data_dir: Directory with the step-1 HDF5 tables.
        T_target: Target temperature in K.

    Returns:
        The path of the chosen table and its contents from ``load_table``.

    Raises:
        FileNotFoundError: If ``data_dir`` has no ``.h5`` files.
    """
    tables = sorted(data_dir.glob("*.h5"))
    if not tables:
        raise FileNotFoundError(
            f"No HDF5 tables found in {data_dir}.\n"
            f"Run step 1 first: python workflows/01_generate_thermo_data/run.py "
            f"--config <your_config>"
        )
    best = min(tables, key=lambda p: abs(float(load_table(p)["metadata"]["temperature"]) - T_target))
    return best, load_table(best)


def _spinodal_bounds(x_axis: np.ndarray, h: np.ndarray, x_min: float = 0.02, x_max: float = 0.98):
    """Return the smallest and largest ``x_axis`` value with ``h < 0`` inside (``x_min``, ``x_max``).

    Returns ``(None, None)`` when no interior point has ``h < 0``.
    """
    interior = (x_axis > x_min) & (x_axis < x_max)
    sp = x_axis[interior & (h < 0)]
    if sp.size == 0:
        return None, None
    return sp.min(), sp.max()


def _shade_spinodal(ax, x_lo, x_hi, label=True):
    """Shade the interval from ``x_lo`` to ``x_hi`` on ``ax``; do nothing if ``x_lo`` is None."""
    if x_lo is None:
        return
    ax.axvspan(x_lo, x_hi, color=SPINODAL_COLOR, alpha=SPINODAL_ALPHA, zorder=0, label="Spinodal" if label else "_nolegend_")


def plot_thermo_properties(cfg: dict, out_dir: Path) -> None:
    """Plot G_m, the chemical potentials and ∂μ_i/∂x_i against composition.

    i is the first independent element and the x axis is the reference element's mole
    fraction. Compositions where ∂μ_i/∂x_i < 0 are shaded as the spinodal.

    Args:
        cfg: Resolved config dict.
        out_dir: Directory for ``thermo_properties.pdf`` and ``thermo_properties.png``.
    """
    data_dir = Path(cfg["output"]["data_dir"])
    T_target = cfg["simulation"]["temperature_C"] + 273.15
    elements = cfg["system"]["elements"]
    ref_el = cfg["system"]["reference_element"]
    ind_els = [e for e in elements if e != ref_el]
    primary = ind_els[0]

    best, data = _load_best_table(data_dir, T_target)
    T_C = float(data["metadata"]["temperature"]) - 273.15
    print(f"  [thermo_properties] table: {best.name}  ({T_C:.0f} °C)")

    el_list = list(data["elements"])
    col_pri = el_list.index(primary)
    x_pri = data["compositions"][:, col_pri]
    x_ref = 1.0 - x_pri if len(ind_els) == 1 else data["compositions"][:, el_list.index(ref_el)]
    gm = data["GM"]
    mu_pri = data["MU"][primary]
    mu_ref = data["MU"][ref_el]
    h_pri = data["dMU"][f"{primary}_{primary}"]

    idx = np.argsort(x_ref)
    x_ref = x_ref[idx]
    gm = gm[idx]
    mu_pri = mu_pri[idx]
    mu_ref = mu_ref[idx]
    h_pri = h_pri[idx]

    sp_lo, sp_hi = _spinodal_bounds(x_ref, h_pri)

    tdb_stem = Path(cfg["system"]["db_path"]).stem
    system = cfg["system"]["name"].upper().replace("_", "–")
    xlabel = f"$x_{{\\rm {ref_el.capitalize()}}}$"
    pri_label = ref_el.capitalize()
    sec_label = primary.capitalize()

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.4))
    fig.subplots_adjust(wspace=0.38)

    for ax in axes:
        ax.set_xlabel(xlabel)
        ax.set_xlim(0, 1)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(0.2))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.1))

    ax = axes[0]
    gm_plot = gm - gm.min()
    _shade_spinodal(ax, sp_lo, sp_hi)
    ax.plot(x_ref, gm_plot / 1e3, color="#1f4e79", lw=1.6)
    ax.axvline(0.50, color="gray", lw=0.8, ls="--", alpha=0.7)
    ax.set_ylabel(r"$G_m - G_m^{\min}$  (kJ mol$^{-1}$)")
    ax.set_title("Molar Gibbs Energy")
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))
    ax.legend(loc="upper right", frameon=True, framealpha=0.85, edgecolor="0.8")

    ax = axes[1]
    _shade_spinodal(ax, sp_lo, sp_hi, label=False)
    ax.plot(x_ref, mu_pri / 1e3, color="#1f4e79", lw=1.6, label=f"$\\mu_{{\\rm {sec_label}}}$")
    ax.plot(x_ref, mu_ref / 1e3, color="#c00000", lw=1.6, ls="--", label=f"$\\mu_{{\\rm {pri_label}}}$")
    ax.axvline(0.50, color="gray", lw=0.8, ls="--", alpha=0.7)
    ax.set_ylabel(r"$\mu_i$  (kJ mol$^{-1}$)")
    ax.set_title("Chemical Potentials")
    ax.legend(loc="lower right", frameon=True, framealpha=0.85, edgecolor="0.8")

    ax = axes[2]
    x_mask = (x_ref > 0.05) & (x_ref < 0.95)
    h_int = h_pri[x_mask]
    y_lo = h_int.min() / 1e3 * 1.15
    y_hi = max(h_int.max() / 1e3 * 1.15, abs(y_lo) * 0.3)
    _shade_spinodal(ax, sp_lo, sp_hi, label=False)
    if sp_lo is not None:
        ax.fill_between(x_ref, h_pri / 1e3, 0, where=(h_pri < 0), color=SPINODAL_COLOR, alpha=0.4)
    ax.plot(x_ref, h_pri / 1e3, color="#375623", lw=1.6, label=f"$H_{{\\rm {sec_label}{sec_label}}}$")
    ax.axhline(0, color="black", lw=0.8)
    ax.axvline(0.50, color="gray", lw=0.8, ls="--", alpha=0.7)
    ax.set_ylabel(r"$\partial\mu / \partial x$  (kJ mol$^{-1}$)")
    ax.set_title(r"Hessian $\partial^2 G_m / \partial x^2$")
    ax.set_xlim(0.05, 0.95)
    ax.set_ylim(y_lo, y_hi)
    ax.legend(loc="upper left", frameon=True, framealpha=0.85, edgecolor="0.8")

    fig.suptitle(f"{system}  BCC$_{{\\rm A2}}$  |  $T = {T_C:.0f}\\,^{{\\circ}}$C  |  {tdb_stem}", fontsize=10, y=1.01)

    for ext in ("pdf", "png"):
        out = out_dir / f"thermo_properties.{ext}"
        fig.savefig(out, bbox_inches="tight", dpi=300)
        print(f"  Saved: {out}")
    plt.close(fig)


def plot_derivative_comparison(cfg: dict, out_dir: Path) -> None:
    """Compare the FD, Jansson and hybrid interpolants with the raw CALPHAD points.

    Plots G_m, μ_i − μ_ref and d²G/dx_i² from each interpolant over the table points,
    shades the spinodal from the Jansson curve, and prints each method's RMSE against the
    table. The scan runs along the single independent mole fraction, so this works for
    binary systems only.

    Args:
        cfg: Resolved config dict.
        out_dir: Directory for ``derivative_comparison.pdf`` and ``derivative_comparison.png``.
    """
    data_dir = Path(cfg["output"]["data_dir"])
    T_target = cfg["simulation"]["temperature_C"] + 273.15
    elements = cfg["system"]["elements"]
    ref_el = cfg["system"]["reference_element"]
    ind_els = [e for e in elements if e != ref_el]
    primary = ind_els[0]

    best, data = _load_best_table(data_dir, T_target)
    T_C = float(data["metadata"]["temperature"]) - 273.15
    T_K = T_C + 273.15
    print(f"  [derivative_comparison] table: {best.name}  ({T_C:.0f} °C)")

    el_list = list(data["elements"])
    col_pri = el_list.index(primary)
    col_ref = el_list.index(ref_el)
    x_nb_raw = data["compositions"][:, col_pri]
    x_ref_raw = data["compositions"][:, col_ref]
    gm_raw = data["GM"]
    mu_raw = data["MU"][primary] - data["MU"][ref_el]
    h_raw = data["dMU"][f"{primary}_{primary}"] - data["dMU"][f"{ref_el}_{primary}"]

    order = np.argsort(x_ref_raw)
    x_ref_raw, x_nb_raw, gm_raw, mu_raw, h_raw = (x_ref_raw[order], x_nb_raw[order], gm_raw[order], mu_raw[order], h_raw[order])

    methods = ["fd", "jansson", "hybrid"]
    print("  Building interpolants ...")
    interps = {}
    for m in methods:
        interps[m] = build_interpolant(
            [best], elements=elements, reference_element=ref_el, temperature_K=T_K, derivative_method=m
        )

    xnb_scan = np.linspace(x_nb_raw.min() + 1e-4, x_nb_raw.max() - 1e-4, 500)
    xref_scan = 1.0 - xnb_scan
    c_scan = xnb_scan.reshape(1, -1)

    curves = {}
    for m, interp in interps.items():
        curves[m] = {
            "gm": interp.gm(c_scan).squeeze(),
            "mu": interp.mu(c_scan).squeeze(),
            "h": interp.d2g(c_scan).squeeze(),
        }

    h_ref = curves["jansson"]["h"]
    sp_lo, sp_hi = _spinodal_bounds(xref_scan, h_ref)

    COLORS = {"calphad": "#888888", "fd": "#0077BB", "jansson": "#EE7733", "hybrid": "#009988"}
    LABELS = {
        "calphad": "CALPHAD data",
        "fd": "Finite difference (FD)",
        "jansson": "Jansson (direct)",
        "hybrid": "Hybrid (Jansson $\\mu$, FD $H$)",
    }
    LS = {"fd": "-", "jansson": "--", "hybrid": "-."}
    ZO = {"fd": 3, "jansson": 4, "hybrid": 3}

    xlabel = f"$x_{{\\rm {ref_el.capitalize()}}}$"
    sec_cap = primary.capitalize()
    pri_cap = ref_el.capitalize()

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.32))
    fig.subplots_adjust(left=0.09, right=0.98, top=0.98, bottom=0.38, wspace=0.38)
    ax_gm, ax_mu, ax_h = axes

    def _shade(ax, label=False):
        """Shade the spinodal interval on ``ax``, with a legend entry if ``label`` is set."""
        if sp_lo is not None:
            ax.axvspan(sp_lo, sp_hi, color="#FFCCCC", alpha=0.55, zorder=0, label="Spinodal ($H<0$)" if label else "_nolegend_")

    _shade(ax_gm, label=True)
    ax_gm.scatter(
        x_ref_raw, gm_raw / 1e3, s=14, color=COLORS["calphad"], alpha=0.4, linewidths=0, zorder=5, label=LABELS["calphad"]
    )
    for m in methods:
        ax_gm.plot(xref_scan, curves[m]["gm"] / 1e3, ls=LS[m], color=COLORS[m], zorder=ZO[m], label=LABELS[m])
    ax_gm.set_xlabel(xlabel)
    ax_gm.set_ylabel("$G_m$  (kJ mol$^{-1}$)")
    ax_gm.text(0.97, 0.05, "(a)", transform=ax_gm.transAxes, ha="right", va="bottom", fontsize=9, fontweight="bold")

    _shade(ax_mu)
    ax_mu.axhline(0, color="0.6", lw=0.7, ls=":", zorder=0)
    ax_mu.scatter(x_ref_raw, mu_raw / 1e3, s=14, color=COLORS["calphad"], alpha=0.4, linewidths=0, zorder=5)
    for m in methods:
        ax_mu.plot(xref_scan, curves[m]["mu"] / 1e3, ls=LS[m], color=COLORS[m], zorder=ZO[m])
    ax_mu.set_xlabel(xlabel)
    ax_mu.set_ylabel(f"$\\mu_{{\\rm {sec_cap}}} - \\mu_{{\\rm {pri_cap}}}$  (kJ mol$^{{-1}}$)")
    ax_mu.text(0.97, 0.05, "(b)", transform=ax_mu.transAxes, ha="right", va="bottom", fontsize=9, fontweight="bold")

    _shade(ax_h)
    ax_h.axhline(0, color="0.35", lw=1.0, ls="--", zorder=1)
    ax_h.scatter(x_ref_raw, h_raw / 1e3, s=14, color=COLORS["calphad"], alpha=0.4, linewidths=0, zorder=5)
    for m in methods:
        ax_h.plot(xref_scan, curves[m]["h"] / 1e3, ls=LS[m], color=COLORS[m], zorder=ZO[m])
    ax_h.set_ylim(-150, 150)
    ax_h.set_xlabel(xlabel)
    ax_h.set_ylabel(f"$\\partial^2 G / \\partial x_{{\\rm {sec_cap}}}^2$  (kJ mol$^{{-1}}$)")
    ax_h.text(0.97, 0.05, "(c)", transform=ax_h.transAxes, ha="right", va="bottom", fontsize=9, fontweight="bold")
    for x_note in [0.04, 0.96]:
        ax_h.annotate(
            "$H\\to+\\infty$", xy=(x_note, 0.93), xycoords="axes fraction", ha="center", va="top", fontsize=6.5, color="0.45"
        )

    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=COLORS["calphad"], markersize=5, label=LABELS["calphad"]),
        *[Line2D([0], [0], ls=LS[m], color=COLORS[m], lw=1.8, label=LABELS[m]) for m in methods],
        mpatches.Patch(facecolor="#FFCCCC", alpha=0.7, edgecolor="none", label="Spinodal ($H<0$)"),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncols=3,
        frameon=True,
        fontsize=7.5,
        handlelength=2.0,
        columnspacing=1.2,
    )

    print(f"\n  {'Method':<10}  {'RMSE G_m':>12}  {'RMSE μ':>12}  {'RMSE H':>12}  J/mol")
    x_eval = x_nb_raw.reshape(1, -1)
    for m, interp in interps.items():
        gm_c = interp.gm(x_eval).squeeze()
        mu_c = interp.mu(x_eval).squeeze()
        h_c = interp.d2g(x_eval).squeeze()
        print(
            f"  {m:<10}  "
            f"{np.sqrt(np.mean((gm_c - gm_raw) ** 2)):>12.1f}  "
            f"{np.sqrt(np.mean((mu_c - mu_raw) ** 2)):>12.1f}  "
            f"{np.sqrt(np.mean((h_c - h_raw) ** 2)):>12.1f}"
        )

    for ext in ("pdf", "png"):
        out = out_dir / f"derivative_comparison.{ext}"
        fig.savefig(out, bbox_inches="tight", dpi=300)
        print(f"  Saved: {out}")
    plt.close(fig)


def run(cfg: dict) -> None:
    """Write both validation figures for ``cfg``.

    Args:
        cfg: Resolved config dict.
    """
    T_label = f"{cfg['simulation']['temperature_C']}C"
    out_dir = Path(cfg["output"]["results_dir"]) / T_label / "thermodynamics"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Output → {out_dir}")

    print("\n  [1/2] Thermodynamic properties plot ...")
    plot_thermo_properties(cfg, out_dir)

    print("\n  [2/2] Derivative method comparison ...")
    plot_derivative_comparison(cfg, out_dir)


if __name__ == "__main__":
    logging.basicConfig(format="%(message)s", stream=sys.stdout)
    logging.getLogger("phasefoundry").setLevel(logging.INFO)
    parser = argparse.ArgumentParser(description="Step 5 — Validate thermodynamic tables")
    parser.add_argument("--config", default="configs/nb_v.yaml")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg = apply_overrides(cfg, args.set or [])
    cfg = resolve_paths(cfg, _ROOT)

    print("── Step 5: Validate thermodynamic tables ──")
    print(summary(cfg))
    run(cfg)
    print("\nDone.")
