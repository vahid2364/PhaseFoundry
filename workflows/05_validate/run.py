"""
Step 5 — Validation  (optional)
================================
Validates the thermodynamic tables and derivative methods for the
configured alloy system. Produces paper-quality figures saved under:

    <output.results_dir>/<temperature_C>C/thermodynamics/
        thermo_properties.pdf       — G_m, mu_i, H_ii vs composition
        derivative_comparison.pdf   — FD / Jansson / hybrid vs raw CALPHAD data

Run standalone
--------------
    python workflows/05_validate/run.py --config configs/nb_v.yaml

    # Ternary system
    python workflows/05_validate/run.py --config configs/nb_hf_v.yaml

    # Override temperature
    python workflows/05_validate/run.py --config configs/nb_v.yaml \
        --set thermo.temperatures_C=[600]

Called from the main pipeline
------------------------------
    python workflows/run_all.py --config configs/nb_v.yaml --validate
"""

from __future__ import annotations
import argparse, sys, time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from phasefoundry.config import load_config, apply_overrides, resolve_paths, summary
from phasefoundry.logger import get_logger
from phasefoundry.thermodynamics.storage import load_table
from phasefoundry.thermodynamics.interpolant import build_interpolant

matplotlib.rcParams.update({
    "font.family":       "serif",
    "font.serif":        ["Times New Roman", "DejaVu Serif"],
    "font.size":         9,
    "axes.labelsize":    10,
    "axes.titlesize":    10,
    "legend.fontsize":   8,
    "xtick.labelsize":   8,
    "ytick.labelsize":   8,
    "lines.linewidth":   1.6,
    "axes.linewidth":    0.8,
    "xtick.direction":   "in",
    "ytick.direction":   "in",
    "xtick.top":         True,
    "ytick.right":       True,
    "figure.dpi":        150,
    "savefig.dpi":       300,
})

SPINODAL_COLOR = "#ffd966"
SPINODAL_ALPHA = 0.40


# ── helpers ───────────────────────────────────────────────────────────────────

def _load_best_table(data_dir: Path, T_target: float) -> tuple[Path, dict]:
    tables = sorted(data_dir.glob("*.h5"))
    if not tables:
        raise FileNotFoundError(
            f"No HDF5 tables found in {data_dir}.\n"
            f"Run step 1 first: python workflows/01_generate_thermo_data/run.py "
            f"--config <your_config>"
        )
    best = min(tables, key=lambda p: abs(
        float(load_table(p)["metadata"]["temperature"]) - T_target))
    return best, load_table(best)


def _spinodal_bounds(x_axis: np.ndarray, h: np.ndarray,
                     x_min: float = 0.02, x_max: float = 0.98):
    """Return (x_lo, x_hi) of the spinodal region in the interior.

    Near-pure-element compositions cause Jansson derivatives to diverge to
    ±∞, which falsely satisfy H < 0.  Filter these out by requiring H to be
    finite and within 50× the median |H| of interior points.
    """
    interior = (x_axis > x_min) & (x_axis < x_max) & np.isfinite(h)
    if not interior.any():
        return None, None
    h_med = np.median(np.abs(h[interior]))
    outlier_free = interior & (np.abs(h) <= 50.0 * max(h_med, 1.0))
    sp = x_axis[outlier_free & (h < 0)]
    if sp.size == 0:
        return None, None
    return sp.min(), sp.max()


def _shade_spinodal(ax, x_lo, x_hi, label=True):
    if x_lo is None:
        return
    ax.axvspan(x_lo, x_hi,
               color=SPINODAL_COLOR, alpha=SPINODAL_ALPHA,
               zorder=0, label="Spinodal" if label else "_nolegend_")


# ── plot 1: thermodynamic properties ─────────────────────────────────────────

def plot_thermo_properties(cfg: dict, out_dir: Path) -> None:
    """G_m, chemical potentials, and Hessian diagonal for the primary element."""
    data_dir = Path(cfg["output"]["data_dir"])
    T_target = cfg["simulation"]["temperature_C"] + 273.15
    elements = cfg["system"]["elements"]
    ref_el   = cfg["system"]["reference_element"]
    ind_els  = [e for e in elements if e != ref_el]
    primary  = ind_els[0]       # first independent element (e.g. NB)

    best, data = _load_best_table(data_dir, T_target)
    T_C = float(data["metadata"]["temperature"]) - 273.15
    print(f"  [thermo_properties] table: {best.name}  ({T_C:.0f} °C)")

    el_list  = list(data["elements"])
    col_pri  = el_list.index(primary)
    x_pri    = data["compositions"][:, col_pri]
    x_ref    = 1.0 - x_pri if len(ind_els) == 1 else data["compositions"][:, el_list.index(ref_el)]
    gm       = data["GM"]
    mu_pri   = data["MU"][primary]
    mu_ref   = data["MU"][ref_el]
    # d²G/dx_i² = dMU[i_i] − dMU[ref_i]  (Gibbs-Duhem correction)
    h_pri    = data["dMU"][f"{primary}_{primary}"] - data["dMU"][f"{ref_el}_{primary}"]

    # sort by x_ref (ascending) for clean lines; drop non-finite rows
    finite = np.isfinite(gm) & np.isfinite(mu_pri) & np.isfinite(mu_ref) & np.isfinite(h_pri)
    x_ref  = x_ref[finite]
    gm     = gm[finite]
    mu_pri = mu_pri[finite]
    mu_ref = mu_ref[finite]
    h_pri  = h_pri[finite]
    idx    = np.argsort(x_ref)
    x_ref  = x_ref[idx]
    gm     = gm[idx]
    mu_pri = mu_pri[idx]
    mu_ref = mu_ref[idx]
    h_pri  = h_pri[idx]

    sp_lo, sp_hi = _spinodal_bounds(x_ref, h_pri)

    tdb_stem = Path(cfg["system"]["db_path"]).stem
    system   = cfg["system"]["name"].upper().replace("_", "–")
    xlabel   = f"$x_{{\\rm {ref_el.capitalize()}}}$"
    pri_label = ref_el.capitalize()
    sec_label = primary.capitalize()

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.4))
    fig.subplots_adjust(wspace=0.38)

    for ax in axes:
        ax.set_xlabel(xlabel)
        ax.set_xlim(0, 1)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(0.2))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.1))

    # Panel 1: G_m
    ax = axes[0]
    gm_plot = gm - gm.min()
    _shade_spinodal(ax, sp_lo, sp_hi)
    ax.plot(x_ref, gm_plot / 1e3, color="#1f4e79", lw=1.6)
    ax.axvline(0.50, color="gray", lw=0.8, ls="--", alpha=0.7)
    ax.set_ylabel(r"$G_m - G_m^{\min}$  (kJ mol$^{-1}$)")
    ax.set_title("Molar Gibbs Energy")
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))
    ax.legend(loc="upper right", frameon=True, framealpha=0.85, edgecolor="0.8")

    # Panel 2: Chemical potentials
    ax = axes[1]
    _shade_spinodal(ax, sp_lo, sp_hi, label=False)
    ax.plot(x_ref, mu_pri / 1e3, color="#1f4e79", lw=1.6,
            label=f"$\\mu_{{\\rm {sec_label}}}$")
    ax.plot(x_ref, mu_ref / 1e3, color="#c00000", lw=1.6, ls="--",
            label=f"$\\mu_{{\\rm {pri_label}}}$")
    ax.axvline(0.50, color="gray", lw=0.8, ls="--", alpha=0.7)
    ax.set_ylabel(r"$\mu_i$  (kJ mol$^{-1}$)")
    ax.set_title("Chemical Potentials")
    ax.legend(loc="lower right", frameon=True, framealpha=0.85, edgecolor="0.8")

    # Panel 3: Hessian
    ax = axes[2]
    x_mask    = (x_ref > 0.05) & (x_ref < 0.95) & np.isfinite(h_pri)
    h_int     = h_pri[x_mask]
    if len(h_int) == 0 or not np.isfinite(h_int).any():
        y_lo, y_hi = -1.0, 1.0
    else:
        y_lo = h_int.min() / 1e3 * 1.15
        y_hi = max(h_int.max() / 1e3 * 1.15, abs(y_lo) * 0.3)
    _shade_spinodal(ax, sp_lo, sp_hi, label=False)
    if sp_lo is not None:
        ax.fill_between(x_ref, h_pri / 1e3, 0,
                        where=(h_pri < 0),
                        color=SPINODAL_COLOR, alpha=0.4)
    ax.plot(x_ref, h_pri / 1e3, color="#375623", lw=1.6,
            label=f"$H_{{\\rm {sec_label}{sec_label}}}$")
    ax.axhline(0, color="black", lw=0.8)
    ax.axvline(0.50, color="gray", lw=0.8, ls="--", alpha=0.7)
    ax.set_ylabel(r"$\partial\mu / \partial x$  (kJ mol$^{-1}$)")
    ax.set_title(r"Hessian $\partial^2 G_m / \partial x^2$")
    ax.set_xlim(0.05, 0.95)
    ax.set_ylim(y_lo, y_hi)
    ax.legend(loc="upper left", frameon=True, framealpha=0.85, edgecolor="0.8")

    fig.suptitle(
        f"{system}  BCC$_{{\\rm A2}}$  |  "
        f"$T = {T_C:.0f}\\,^{{\\circ}}$C  |  {tdb_stem}",
        fontsize=10, y=1.01)

    for ext in ("pdf", "png"):
        out = out_dir / f"thermo_properties.{ext}"
        fig.savefig(out, bbox_inches="tight", dpi=300)
        print(f"  Saved: {out}")
    plt.close(fig)


# ── plot 2: derivative method comparison ─────────────────────────────────────

def plot_derivative_comparison(cfg: dict, out_dir: Path) -> dict | None:
    """Compare FD / Jansson / hybrid interpolants against raw CALPHAD data."""
    data_dir = Path(cfg["output"]["data_dir"])
    T_target = cfg["simulation"]["temperature_C"] + 273.15
    elements = cfg["system"]["elements"]
    ref_el   = cfg["system"]["reference_element"]
    ind_els  = [e for e in elements if e != ref_el]
    n_ind    = len(ind_els)
    primary  = ind_els[0]

    best, data = _load_best_table(data_dir, T_target)
    T_C  = float(data["metadata"]["temperature"]) - 273.15
    T_K  = T_C + 273.15
    print(f"  [derivative_comparison] table: {best.name}  ({T_C:.0f} °C)")

    el_list  = list(data["elements"])
    col_pri  = el_list.index(primary)
    col_ref  = el_list.index(ref_el)
    x_nb_raw = data["compositions"][:, col_pri]
    x_ref_raw = data["compositions"][:, col_ref]
    gm_raw   = data["GM"]
    mu_raw   = data["MU"][primary] - data["MU"][ref_el]
    h_raw    = data["dMU"][f"{primary}_{primary}"] - data["dMU"][f"{ref_el}_{primary}"]

    # For ternary+ systems, restrict raw scatter to a slice near the fixed
    # secondary-element values so the 1-D scan is comparable to the raw data.
    if n_ind > 1:
        c0_cfg_local = cfg.get("simulation", {}).get("c0", [0.5] * n_ind)
        if isinstance(c0_cfg_local, (int, float)):
            c0_cfg_local = [c0_cfg_local]
        slice_mask = np.ones(len(x_nb_raw), dtype=bool)
        for i, el in enumerate(ind_els):
            if el == primary:
                continue
            col_sec = el_list.index(el)
            x_sec   = data["compositions"][:, col_sec]
            sec_val = float(c0_cfg_local[i]) if i < len(c0_cfg_local) else 0.1
            # Use ~1 grid-step tolerance: snap to the nearest sampled value
            # so we get a clean 1-D slice instead of a wide band
            nearest = x_sec[np.argmin(np.abs(x_sec - sec_val))]
            unique_sec = np.unique(np.round(x_sec, 6))
            if len(unique_sec) > 1:
                tol = np.diff(unique_sec).min() * 0.6
            else:
                tol = 0.01
            slice_mask &= np.abs(x_sec - nearest) <= tol
        if slice_mask.sum() < 5:
            slice_mask = np.ones(len(x_nb_raw), dtype=bool)  # fall back to all
        x_nb_raw  = x_nb_raw[slice_mask]
        x_ref_raw = x_ref_raw[slice_mask]
        gm_raw    = gm_raw[slice_mask]
        mu_raw    = mu_raw[slice_mask]
        h_raw     = h_raw[slice_mask]

    order = np.argsort(x_ref_raw)
    x_ref_raw, x_nb_raw, gm_raw, mu_raw, h_raw = (
        x_ref_raw[order], x_nb_raw[order],
        gm_raw[order], mu_raw[order], h_raw[order])

    # build interpolants
    methods = ["fd", "jansson", "hybrid"]
    print("  Building interpolants ...")
    interps = {}
    for m in methods:
        interps[m] = build_interpolant(
            [best], elements=elements, reference_element=ref_el,
            temperature_K=T_K, derivative_method=m)

    # dense scan along the primary element axis; hold secondary elements
    # at their c0 values so c_scan is always (N_ind, 500)
    n_ind    = len(ind_els)
    pri_idx  = ind_els.index(primary)
    xnb_scan = np.linspace(x_nb_raw.min() + 1e-4, x_nb_raw.max() - 1e-4, 500)

    c0_cfg = cfg.get("simulation", {}).get("c0", [0.5] * n_ind)
    if isinstance(c0_cfg, (int, float)):
        c0_cfg = [c0_cfg]

    rows = []
    for i, el in enumerate(ind_els):
        if el == primary:
            rows.append(xnb_scan)
        else:
            rows.append(np.full(500, float(c0_cfg[i]) if i < len(c0_cfg) else 0.1))
    c_scan    = np.array(rows)                    # (N_ind, 500)
    xref_scan = 1.0 - c_scan.sum(axis=0)         # x_ref along the scan

    curves = {}
    for m, interp in interps.items():
        mu_all = interp.mu(c_scan)                # (N_ind, 500)
        h_all  = interp.d2g(c_scan)              # (N_ind, N_ind, 500)
        curves[m] = dict(
            gm = interp.gm(c_scan).squeeze(),
            mu = mu_all[pri_idx],                 # primary element row
            h  = h_all[pri_idx, pri_idx],         # primary diagonal H_ii
        )

    h_ref         = curves["jansson"]["h"]
    spinodal_mask = h_ref < 0
    # x-axis of the plot is xref_scan (x_V), so spinodal bounds must use the
    # same axis — not xnb_scan — to avoid a left-right mirror artifact.
    sp_lo, sp_hi  = _spinodal_bounds(xref_scan, h_ref)

    # style
    COLORS = {"calphad": "#888888", "fd": "#0077BB",
               "jansson": "#EE7733", "hybrid": "#009988"}
    LABELS = {"calphad": "CALPHAD data",
               "fd":      "Finite difference (FD)",
               "jansson": "Jansson (direct)",
               "hybrid":  "Hybrid (Jansson $\\mu$, FD $H$)"}
    LS     = {"fd": "-", "jansson": "--", "hybrid": "-."}
    ZO     = {"fd": 3,   "jansson": 4,    "hybrid": 3}

    xlabel   = f"$x_{{\\rm {ref_el.capitalize()}}}$"
    sec_cap  = primary.capitalize()
    pri_cap  = ref_el.capitalize()

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.32))
    fig.subplots_adjust(left=0.09, right=0.98, top=0.98,
                        bottom=0.38, wspace=0.38)
    ax_gm, ax_mu, ax_h = axes

    def _shade(ax, label=False):
        if sp_lo is not None:
            ax.axvspan(sp_lo, sp_hi, color="#FFCCCC", alpha=0.55,
                       zorder=0, label="Spinodal ($H<0$)" if label else "_nolegend_")

    # Panel A: G_m
    _shade(ax_gm, label=True)
    ax_gm.scatter(x_ref_raw, gm_raw / 1e3, s=14,
                  color=COLORS["calphad"], alpha=0.4, linewidths=0,
                  zorder=5, label=LABELS["calphad"])
    for m in methods:
        ax_gm.plot(xref_scan, curves[m]["gm"] / 1e3,
                   ls=LS[m], color=COLORS[m], zorder=ZO[m], label=LABELS[m])
    ax_gm.set_xlabel(xlabel)
    ax_gm.set_xlim(-0.05, 1.05)
    ax_gm.set_ylabel("$G_m$  (kJ mol$^{-1}$)")
    ax_gm.text(0.97, 0.05, "(a)", transform=ax_gm.transAxes,
               ha="right", va="bottom", fontsize=9, fontweight="bold")

    # Panel B: μ
    _shade(ax_mu)
    ax_mu.axhline(0, color="0.6", lw=0.7, ls=":", zorder=0)
    ax_mu.scatter(x_ref_raw, mu_raw / 1e3, s=14,
                  color=COLORS["calphad"], alpha=0.4, linewidths=0, zorder=5)
    for m in methods:
        ax_mu.plot(xref_scan, curves[m]["mu"] / 1e3,
                   ls=LS[m], color=COLORS[m], zorder=ZO[m])
    ax_mu.set_xlabel(xlabel)
    ax_mu.set_xlim(-0.05, 1.05)
    ax_mu.set_ylabel(
        f"$\\mu_{{\\rm {sec_cap}}} - \\mu_{{\\rm {pri_cap}}}$  (kJ mol$^{{-1}}$)")
    ax_mu.text(0.97, 0.05, "(b)", transform=ax_mu.transAxes,
               ha="right", va="bottom", fontsize=9, fontweight="bold")

    # Panel C: H
    # Mask divergent near-pure-element values before plotting so they don't
    # produce oscillating vertical stripes at the edges.
    H_LIM = 150.0  # kJ/mol display clip
    _shade(ax_h)
    ax_h.axhline(0, color="0.35", lw=1.0, ls="--", zorder=1)
    h_raw_plot = h_raw / 1e3
    h_raw_plot = np.where(np.abs(h_raw_plot) <= H_LIM, h_raw_plot, np.nan)
    ax_h.scatter(x_ref_raw, h_raw_plot, s=14,
                 color=COLORS["calphad"], alpha=0.4, linewidths=0, zorder=5)
    for m in methods:
        h_curve = curves[m]["h"] / 1e3
        h_curve = np.where(np.abs(h_curve) <= H_LIM, h_curve, np.nan)
        ax_h.plot(xref_scan, h_curve,
                  ls=LS[m], color=COLORS[m], zorder=ZO[m])
    ax_h.set_ylim(-H_LIM, H_LIM)
    ax_h.set_xlabel(xlabel)
    ax_h.set_xlim(-0.05, 1.05)
    ax_h.set_ylabel(
        f"$\\partial^2 G / \\partial x_{{\\rm {sec_cap}}}^2$  (kJ mol$^{{-1}}$)")
    ax_h.text(0.97, 0.05, "(c)", transform=ax_h.transAxes,
              ha="right", va="bottom", fontsize=9, fontweight="bold")
    for x_note in [0.04, 0.96]:
        ax_h.annotate("$H\\to+\\infty$",
                      xy=(x_note, 0.93), xycoords="axes fraction",
                      ha="center", va="top", fontsize=6.5, color="0.45")

    # shared legend
    handles = [
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor=COLORS["calphad"], markersize=5,
               label=LABELS["calphad"]),
        *[Line2D([0], [0], ls=LS[m], color=COLORS[m], lw=1.8,
                 label=LABELS[m]) for m in methods],
        mpatches.Patch(facecolor="#FFCCCC", alpha=0.7,
                       edgecolor="none", label="Spinodal ($H<0$)"),
    ]
    fig.legend(handles=handles, loc="lower center",
               bbox_to_anchor=(0.5, 0.01), ncols=3,
               frameon=True, fontsize=7.5,
               handlelength=2.0, columnspacing=1.2)

    # residual summary — rebuild eval grid at raw primary compositions,
    # holding secondary elements fixed at their c0 values (same as c_scan)
    eval_rows = []
    for i, el in enumerate(ind_els):
        if el == primary:
            eval_rows.append(x_nb_raw)
        else:
            eval_rows.append(np.full(len(x_nb_raw),
                                     float(c0_cfg[i]) if i < len(c0_cfg) else 0.1))
    x_eval = np.array(eval_rows)   # (N_ind, n_raw_pts)

    # RMSE over the full raw-data range is dominated by a handful of points
    # near the pure-element edges, where H (and to a lesser extent mu) diverge
    # as x -> 0 or 1 — a single squared-and-averaged number there is not
    # representative of typical accuracy. Report median/P90/max *relative*
    # error instead: scale-invariant, and the median/P90 are insensitive to
    # the small minority of edge (or H~0 zero-crossing) points that dominate
    # RMSE, without needing an arbitrary composition cutoff. The legacy
    # absolute RMSE table is kept below for reference.
    EPS = 1e-9  # avoids literal division by zero; negligible next to real J/mol magnitudes
    secondary_note = ""
    if n_ind > 1:
        fixed = ", ".join(
            f"x_{el}={float(c0_cfg[i]) if i < len(c0_cfg) else 0.1:.3f}"
            for i, el in enumerate(ind_els) if el != primary
        )
        secondary_note = f", {fixed} held fixed"
    print(f"\n  Residuals computed over x_{primary} in "
          f"[{x_nb_raw.min():.4f}, {x_nb_raw.max():.4f}] "
          f"({len(x_nb_raw)} raw CALPHAD grid points, {T_C:.0f} °C{secondary_note})")
    print(f"\n  {'Method':<10}  {'median |Δμ|/|μ|':>14}  {'P90':>8}  {'max':>8}"
          f"  {'median |ΔH|/|H|':>16}  {'P90':>8}  {'max':>8}")
    rmse_by_method: dict[str, dict[str, float]] = {}
    rel_err_by_method: dict[str, dict[str, float]] = {}
    for m, interp in interps.items():
        gm_c = interp.gm(x_eval).squeeze()
        mu_c = interp.mu(x_eval)[pri_idx]
        h_c  = interp.d2g(x_eval)[pri_idx, pri_idx]

        rmse_gm = float(np.sqrt(np.mean((gm_c - gm_raw)**2)))
        rmse_mu = float(np.sqrt(np.mean((mu_c - mu_raw)**2)))
        rmse_h  = float(np.sqrt(np.mean((h_c  - h_raw )**2)))
        rmse_by_method[m] = {"rmse_gm": rmse_gm, "rmse_mu": rmse_mu, "rmse_h": rmse_h}

        rel_mu = np.abs(mu_c - mu_raw) / (np.abs(mu_raw) + EPS) * 100.0
        rel_h  = np.abs(h_c  - h_raw ) / (np.abs(h_raw ) + EPS) * 100.0
        mu_med, mu_p90, mu_max = (float(np.median(rel_mu)), float(np.percentile(rel_mu, 90)), float(np.max(rel_mu)))
        h_med,  h_p90,  h_max  = (float(np.median(rel_h)),  float(np.percentile(rel_h, 90)),  float(np.max(rel_h)))
        rel_err_by_method[m] = {
            "mu_median_pct": mu_med, "mu_p90_pct": mu_p90, "mu_max_pct": mu_max,
            "h_median_pct":  h_med,  "h_p90_pct":  h_p90,  "h_max_pct":  h_max,
        }
        print(f"  {m:<10}  {mu_med:>13.4f}%  {mu_p90:>7.4f}%  {mu_max:>7.4f}%"
              f"  {h_med:>15.4f}%  {h_p90:>7.4f}%  {h_max:>7.4f}%")

    print(f"\n  For reference, absolute RMSE over this same full range "
          f"(dominated by divergence near the pure-element edges — the "
          f"median/P90 figures above are the representative accuracy "
          f"numbers, not these):")
    print(f"  {'Method':<10}  {'RMSE G_m':>12}  {'RMSE μ':>12}  {'RMSE H':>12}  J/mol")
    for m in interps:
        r = rmse_by_method[m]
        print(f"  {m:<10}  {r['rmse_gm']:>12.1f}  {r['rmse_mu']:>12.1f}  {r['rmse_h']:>12.1f}")

    for ext in ("pdf", "png"):
        out = out_dir / f"derivative_comparison.{ext}"
        fig.savefig(out, bbox_inches="tight", dpi=300)
        print(f"  Saved: {out}")
    plt.close(fig)

    # Report accuracy for whichever method the simulation step will actually
    # use. gm_rel_err_pct / hessian_rel_err_pct now carry the median relative
    # error (see note above), not RMSE normalized by the raw-data range.
    sim_method = cfg.get("simulation", {}).get("derivative_method", "fd")
    if sim_method not in rmse_by_method:
        sim_method = next(iter(rmse_by_method), None)
    if sim_method is None:
        return {}
    m_rmse = rmse_by_method[sim_method]
    m_rel  = rel_err_by_method[sim_method]
    return {
        "derivative_method":     sim_method,
        "composition_range":     [float(x_nb_raw.min()), float(x_nb_raw.max())],
        "rmse_by_method":        rmse_by_method,
        "rel_err_by_method":     rel_err_by_method,
        "gm_rel_err_pct":        round(100.0 * m_rmse["rmse_gm"] / ((float(np.ptp(gm_raw)) or 1.0)), 4),
        "hessian_rel_err_pct":   round(m_rel["h_median_pct"], 4),
        "hessian_p90_rel_err_pct": round(m_rel["h_p90_pct"], 4),
        "mu_rel_err_pct":        round(m_rel["mu_median_pct"], 4),
    }


# ── main entry point ──────────────────────────────────────────────────────────

def run(cfg: dict) -> None:
    _t0 = time.perf_counter()
    results_dir = Path(cfg["output"]["results_dir"])
    T_label = f"{cfg['simulation']['temperature_C']}C"
    out_dir = results_dir / T_label / "thermodynamics"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Output → {out_dir}")

    pf_log = get_logger(cfg, step="validate", results_dir=out_dir)

    print("\n  [1/2] Thermodynamic properties plot ...")
    plot_thermo_properties(cfg, out_dir)

    print("\n  [2/2] Derivative method comparison ...")
    result = plot_derivative_comparison(cfg, out_dir)

    _elapsed = time.perf_counter() - _t0
    if pf_log.enabled:
        record: dict = {"label": "validate", "elapsed_s": round(_elapsed, 2)}
        # plot_derivative_comparison returns accuracy metrics when available
        if isinstance(result, dict):
            record.update(result)
        pf_log.log_timing_record(record)
        if isinstance(result, dict) and "gm_rel_err_pct" in result:
            pf_log.info(
                f"Surrogate accuracy  │  "
                f"G_m: {result['gm_rel_err_pct']:.3f}%  "
                f"Hessian: {result.get('hessian_rel_err_pct', 'n/a')}%"
            )
        pf_log.info(f"Validation complete  │  {_elapsed:.1f} s")


def main():
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


if __name__ == "__main__":
    main()
