"""
Derivative-method comparison test — binary Nb-V system at 500 °C
=================================================================
Builds three GmInterpolant objects (fd / jansson / hybrid) from the
pre-computed HDF5 table and plots, side by side:

  Panel A  G_m(x_NB)          — shared free energy surface (should match)
  Panel B  μ(x_NB)            — chemical potential ∂G/∂x_NB
  Panel C  H(x_NB) = d²G/dx²  — second derivative (Hessian)

Raw CALPHAD data points are shown as reference in every panel.
The spinodal region (H < 0) is highlighted.

Usage
-----
    python tests/test_derivative_methods.py
    python tests/test_derivative_methods.py --save results/derivative_method_comparison.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from calphad_pf.thermodynamics.storage import load_table
from calphad_pf.thermodynamics.interpolant import build_interpolant


# ── style ─────────────────────────────────────────────────────────────────────

# Consistent with Nature/Acta Materialia style: clean, no chartjunk
matplotlib.rcParams.update({
    "font.family":        "serif",
    "font.serif":         ["Times New Roman", "DejaVu Serif"],
    "font.size":          9,
    "axes.titlesize":     10,
    "axes.labelsize":     9,
    "xtick.labelsize":    8,
    "ytick.labelsize":    8,
    "legend.fontsize":    8,
    "legend.framealpha":  0.92,
    "lines.linewidth":    1.8,
    "axes.linewidth":     0.8,
    "xtick.major.width":  0.8,
    "ytick.major.width":  0.8,
    "figure.dpi":         150,
    "savefig.dpi":        300,
    # do NOT set savefig.bbox here — we control it per-call
})

# Colour palette (colour-blind safe: Paul Tol "bright")
COLORS = {
    "calphad": "#888888",
    "fd":      "#0077BB",   # blue
    "jansson": "#EE7733",   # orange
    "hybrid":  "#009988",   # teal
}
LABELS = {
    "calphad": "CALPHAD data",
    "fd":      "Finite difference (FD)",
    "jansson": "Jansson (direct)",
    "hybrid":  "Hybrid (Jansson $\\mu$, FD $H$)",
}
LINE_STYLES = {
    "fd":      "-",
    "jansson": "--",
    "hybrid":  "-.",
}
ZORDER = {"fd": 3, "jansson": 4, "hybrid": 3}


# ── helpers ───────────────────────────────────────────────────────────────────

def _si_prefix(val: float) -> tuple[float, str]:
    """Return (scaled_value, unit_prefix) for neat axis labels."""
    a = abs(val)
    if a >= 1e3:
        return val / 1e3, "k"
    return val, ""


def _format_energy(arr: np.ndarray) -> tuple[np.ndarray, str]:
    """Scale an energy array and return (scaled_array, ylabel_units)."""
    mx = np.nanmax(np.abs(arr))
    if mx >= 1e3:
        return arr / 1e3, "kJ mol$^{-1}$"
    return arr, "J mol$^{-1}$"


# ── main ──────────────────────────────────────────────────────────────────────

def run(save_prefix: str | None = None, show: bool = True) -> None:
    """
    Parameters
    ----------
    save_prefix : str or None
        Path stem for the three panel PDFs. When given, the script writes
        ``<save_prefix>_a.pdf``, ``<save_prefix>_b.pdf``, ``<save_prefix>_c.pdf``
        — one file per panel, identical in construction to the eigenvalues
        figure panels (eigenvalues_Hf1.pdf, Hf25.pdf, Hf4.pdf).
    show : bool
        Whether to open an interactive window after saving.
    """
    # ── locate table ──────────────────────────────────────────────────────────
    # Search known data directories for NbV HDF5 tables, preferring v2 then v1.
    candidates = [
        _ROOT / "data" / "HfNbTiV_BCC_PhaseForge_v2_Tdep" / "NbV",
        _ROOT / "data" / "HfNbTiV_BCC_PhaseForge_v1_Tindep" / "NbV",
        _ROOT / "data" / "HfNbTiV_BCC_PhaseForge_v3" / "NbV",
        _ROOT / "data" / "nb_v" / "fine",
    ]
    tables = []
    table_dir = None
    for d in candidates:
        tables = sorted(d.glob("*.h5"))
        if tables:
            table_dir = d
            break
    if not tables:
        raise FileNotFoundError(
            "No NbV HDF5 tables found. Searched:\n"
            + "\n".join(f"  {d}" for d in candidates)
            + "\nRun step 1 first."
        )

    T_target = 773.15   # 500 °C
    best = min(tables, key=lambda p: abs(
        float(load_table(p)["metadata"]["temperature"]) - T_target
    ))
    data  = load_table(best)
    T_K   = float(data["metadata"]["temperature"])
    T_C   = T_K - 273.15
    el_list = list(data["elements"])   # ['NB', 'V']

    elements  = el_list
    ref_el    = "V"
    ind_el    = "NB"

    print(f"Table      : {best.name}")
    print(f"Temperature: {T_C:.0f} °C  ({T_K:.2f} K)")
    print(f"Elements   : {elements}  →  reference = {ref_el}")
    print(f"Points     : {len(data['compositions'])}")

    # ── raw CALPHAD data ──────────────────────────────────────────────────────
    col_nb  = el_list.index(ind_el)
    col_v   = el_list.index(ref_el)
    x_nb_raw = data["compositions"][:, col_nb]   # x_NB (independent variable)
    x_raw    = data["compositions"][:, col_v]    # x_V  = 1 − x_NB  (plot axis)
    gm_raw   = data["GM"]

    # raw μ_NB (relative chemical potential = MU[NB] - MU[V])
    mu_raw  = data["MU"][ind_el] - data["MU"][ref_el]

    # raw Hessian H = d(μ_NB - μ_V)/dx_NB = dMU[NB][NB] - dMU[V][NB]
    # Note: sign convention — H w.r.t. x_NB; plotting against x_V flips x-axis
    # but does not change H (it's still ∂²G/∂x_NB²).
    h_raw   = data["dMU"]["NB_NB"] - data["dMU"]["V_NB"]

    # sort by x_V (ascending)
    order   = np.argsort(x_raw)
    x_raw, x_nb_raw, gm_raw, mu_raw, h_raw = (
        x_raw[order], x_nb_raw[order],
        gm_raw[order], mu_raw[order], h_raw[order]
    )

    # ── build interpolants ────────────────────────────────────────────────────
    methods = ["fd", "jansson", "hybrid"]
    print("\nBuilding interpolants …")
    interps = {}
    for m in methods:
        print(f"  [{m}]")
        interps[m] = build_interpolant(
            [best], elements=elements, reference_element=ref_el,
            temperature_K=T_K, derivative_method=m,
        )

    # ── dense composition scan (in x_NB, the independent variable) ──────────
    xnb_scan = np.linspace(x_nb_raw.min() + 1e-4, x_nb_raw.max() - 1e-4, 500)
    xv_scan  = 1.0 - xnb_scan          # x_V for the plot axis
    c_scan   = xnb_scan.reshape(1, -1)  # shape (N_ind=1, 500)

    curves = {}
    for m, interp in interps.items():
        # For binary (N_ind=1) the interpolant accepts a (1, n_pts) array.
        # gm → spatial shape; mu/d2g carry leading element axes → squeeze to 1-D.
        gm_c = interp.gm(c_scan).squeeze()          # (500,)
        mu_c = interp.mu(c_scan).squeeze()          # (500,)
        h_c  = interp.d2g(c_scan).squeeze()         # (500,)
        curves[m] = dict(gm=gm_c, mu=mu_c, h=h_c)

    # ── detect spinodal region from Jansson (most physically accurate) ────────
    h_ref         = curves["jansson"]["h"]
    spinodal_mask = h_ref < 0

    # ── per-panel helpers ─────────────────────────────────────────────────────
    CALPHAD_ALPHA = 0.40

    def _shade_spinodal(ax):
        """Light red fill where H < 0."""
        in_region = False
        x_start   = None
        for k, is_sp in enumerate(spinodal_mask):
            xv = xv_scan[k]
            if is_sp and not in_region:
                x_start   = xv
                in_region = True
            elif not is_sp and in_region:
                ax.axvspan(min(x_start, xv_scan[k - 1]),
                           max(x_start, xv_scan[k - 1]),
                           color="#FFCCCC", alpha=0.55, zorder=0,
                           label="_spinodal")
                in_region = False
        if in_region:
            lo = min(x_start, xv_scan[-1])
            hi = max(x_start, xv_scan[-1])
            ax.axvspan(lo, hi, color="#FFCCCC", alpha=0.55, zorder=0)

    def _add_panel_label(ax, label):
        ax.text(0.97, 0.05, label,
                transform=ax.transAxes,
                ha="right", va="bottom",
                fontsize=9, fontweight="bold")

    # Legend handles (placed in panel a only)
    legend_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=COLORS["calphad"],
               markersize=5, label=LABELS["calphad"]),
        Line2D([0], [0], ls=LINE_STYLES["fd"],      color=COLORS["fd"],      lw=1.8, label=LABELS["fd"]),
        Line2D([0], [0], ls=LINE_STYLES["jansson"],  color=COLORS["jansson"], lw=1.8, label=LABELS["jansson"]),
        Line2D([0], [0], ls=LINE_STYLES["hybrid"],   color=COLORS["hybrid"],  lw=1.8, label=LABELS["hybrid"]),
        mpatches.Patch(facecolor="#FFCCCC", alpha=0.7, edgecolor="none",
                       label="Spinodal region ($H < 0$)"),
    ]

    gm_scale = 1e3   # J → kJ

    # ── Panel A: G_m — individual figure ────────────────────────────────────
    fig_a, ax_gm = plt.subplots(1, 1)
    ax_gm.scatter(x_raw, gm_raw / gm_scale,
                  s=14, color=COLORS["calphad"], zorder=5,
                  label=LABELS["calphad"], alpha=CALPHAD_ALPHA, linewidths=0)
    for m in methods:
        ax_gm.plot(xv_scan, curves[m]["gm"] / gm_scale,
                   ls=LINE_STYLES[m], color=COLORS[m],
                   zorder=ZORDER[m], label=LABELS[m])
    _shade_spinodal(ax_gm)
    ax_gm.set_xlabel("$x_{\\mathrm{V}}$")
    ax_gm.set_ylabel("$G_m$  (kJ mol$^{-1}$)")
    _add_panel_label(ax_gm, "(a)")
    ax_gm.legend(handles=legend_handles, fontsize=7.5, handlelength=2.0,
                 framealpha=0.92, loc="upper right")

    # ── Panel B: μ — individual figure ──────────────────────────────────────
    fig_b, ax_mu = plt.subplots(1, 1)
    ax_mu.axhline(0, color="0.6", lw=0.7, ls=":", zorder=0)
    ax_mu.scatter(x_raw, mu_raw / gm_scale,
                  s=14, color=COLORS["calphad"], zorder=5,
                  label=LABELS["calphad"], alpha=CALPHAD_ALPHA, linewidths=0)
    for m in methods:
        ax_mu.plot(xv_scan, curves[m]["mu"] / gm_scale,
                   ls=LINE_STYLES[m], color=COLORS[m],
                   zorder=ZORDER[m], label=LABELS[m])
    _shade_spinodal(ax_mu)
    ax_mu.set_xlabel("$x_{\\mathrm{V}}$")
    ax_mu.set_ylabel("$\\mu_{\\mathrm{Nb}} - \\mu_{\\mathrm{V}}$  (kJ mol$^{-1}$)")
    _add_panel_label(ax_mu, "(b)")

    # ── Panel C: H = d²G/dx_NB² — individual figure ──────────────────────────
    h_clip = 150.0   # kJ/mol
    fig_c, ax_h = plt.subplots(1, 1)
    ax_h.axhline(0, color="0.35", lw=1.0, ls="--", zorder=1)
    ax_h.scatter(x_raw, h_raw / gm_scale,
                 s=14, color=COLORS["calphad"], zorder=5,
                 label=LABELS["calphad"], alpha=CALPHAD_ALPHA, linewidths=0)
    for m in methods:
        ax_h.plot(xv_scan, curves[m]["h"] / gm_scale,
                  ls=LINE_STYLES[m], color=COLORS[m],
                  zorder=ZORDER[m], label=LABELS[m])
    _shade_spinodal(ax_h)
    ax_h.set_xlabel("$x_{\\mathrm{V}}$")
    ax_h.set_ylabel("$\\partial^2 G / \\partial x_{\\mathrm{Nb}}^2$  (kJ mol$^{-1}$)")
    ax_h.set_ylim(-h_clip, h_clip)
    for x_note in [0.04, 0.96]:
        ax_h.annotate(
            "$H \\to +\\infty$",
            xy=(x_note, 0.93), xycoords="axes fraction",
            ha="center", va="top", fontsize=6.5, color="0.45",
        )
    _add_panel_label(ax_h, "(c)")

    all_figs = [fig_a, fig_b, fig_c]

    # ── quantitative residual summary (printed, not on figure) ───────────────
    print("\n── Residuals vs CALPHAD raw data (at raw x_NB points) ──────────────")
    x_eval   = x_nb_raw.reshape(1, -1)
    ref_data = dict(gm=gm_raw, mu=mu_raw, h=h_raw)
    for m, interp in interps.items():
        gm_c = interp.gm(x_eval).squeeze()
        mu_c = interp.mu(x_eval).squeeze()
        h_c  = interp.d2g(x_eval).squeeze()
        rmse_gm = np.sqrt(np.mean((gm_c - ref_data["gm"])**2))
        rmse_mu = np.sqrt(np.mean((mu_c - ref_data["mu"])**2))
        rmse_h  = np.sqrt(np.mean((h_c  - ref_data["h"])**2))
        print(f"  [{m:<8}]  RMSE G_m = {rmse_gm:8.1f}  "
              f"RMSE μ = {rmse_mu:9.1f}  "
              f"RMSE H = {rmse_h:10.1f}  J/mol")

    # ── save / show ───────────────────────────────────────────────────────────
    if save_prefix is not None:
        stem = Path(save_prefix)
        stem.parent.mkdir(parents=True, exist_ok=True)
        for suffix, fig in zip(["_a", "_b", "_c"], all_figs):
            out = stem.parent / (stem.name + suffix + ".pdf")
            fig.savefig(out, dpi=300, bbox_inches="tight")
            print(f"Panel saved → {out}")

    if show:
        plt.show()
    else:
        for fig in all_figs:
            plt.close(fig)


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Derivative-method comparison for binary Nb-V"
    )
    parser.add_argument(
        "--save", default=None,
        help=(
            "Path stem for the three panel PDFs.  E.g. "
            "'paper/2025_Kunselman_Design_of_Spinodal_Microstructures_in_HEAs"
            "/figs/derivative_method_comparison' writes "
            "'..._a.pdf', '..._b.pdf', '..._c.pdf'."
        ),
    )
    parser.add_argument("--no-show", action="store_true",
                        help="Do not open an interactive window")
    args = parser.parse_args()

    run(save_prefix=args.save, show=not args.no_show)
