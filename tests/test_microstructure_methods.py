"""
Microstructure comparison — binary Nb-V, FD vs Jansson derivative methods
=========================================================================
Runs two identical Cahn-Hilliard simulations (fd / jansson) for the Nb-V
BCC_A2 binary at 500 °C and compares:

  Row 1  Final x_V composition field: (a) FD, (b) Jansson
  Row 2  (c) |FD − Jansson| difference map, (d) statistics table

Results are cached to a .npz file so re-running the figure does not
re-run the simulation.

Usage
-----
    python tests/test_microstructure_methods.py
    python tests/test_microstructure_methods.py --save results/microstructure_methods.pdf
    python tests/test_microstructure_methods.py --no-cache   # force re-simulation
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from calphad_pf.thermodynamics.storage import load_table
from calphad_pf.thermodynamics.interpolant import build_interpolant
from calphad_pf.phasefield.solver import solve_cahn_hilliard

# ── style ─────────────────────────────────────────────────────────────────────

matplotlib.rcParams.update({
    "font.family":       "serif",
    "font.serif":        ["Times New Roman", "DejaVu Serif"],
    "font.size":         9,
    "axes.labelsize":    9,
    "xtick.labelsize":   8,
    "ytick.labelsize":   8,
    "axes.linewidth":    0.7,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "figure.dpi":        150,
    "savefig.dpi":       300,
})

METHODS      = ["fd", "jansson"]
METHOD_LABEL = {
    "fd":      "Finite difference (FD)",
    "jansson": "Jansson (direct)",
}

# ── simulation parameters ─────────────────────────────────────────────────────

SIM = dict(
    N          = 256,
    Lx         = 1.0e-6,
    c0         = 0.5,
    noise      = 0.01,
    seed       = 42,
    kappa      = 1.0e-12,
    M          = 5.0e-21,
    dt         = 1.0e-2,
    n_steps    = 80000,   # ~800 s → well past τ≈30 s, coarsened spinodal domains
    save_every = 80000,
)

_CACHE = _ROOT / "tests" / "_cache_microstructure_nbv.npz"


# ── helpers ───────────────────────────────────────────────────────────────────

def _run_simulations() -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """Run FD and Jansson CH simulations; return ({method: x_V}, {method: wall_s})."""
    table_dir = _ROOT / "data" / "nb_v" / "fine"
    tables    = sorted(table_dir.glob("*.h5"))
    best = min(tables, key=lambda p: abs(
        float(load_table(p)["metadata"]["temperature"]) - 773.15
    ))
    T_K = float(load_table(best)["metadata"]["temperature"])
    print(f"Table: {best.name}  ({T_K - 273.15:.0f} °C)")

    fields  = {}
    timings = {}
    for m in METHODS:
        print(f"\n── [{m}] ──────────────────────────────────────────────────────")
        interp = build_interpolant(
            [best], elements=["NB", "V"], reference_element="V",
            temperature_K=T_K, derivative_method=m,
        )
        res = solve_cahn_hilliard(
            interpolant=interp, output_dir=None, save_frames=False, **SIM,
        )
        xv = 1.0 - res.compositions[0]   # x_V = 1 − x_NB
        print(f"  x_V : [{xv.min():.4f}, {xv.mean():.4f}, {xv.max():.4f}]")
        print(f"  F   : {res.free_energy_history[-1]:.4f} J/mol")
        print(f"  Wall: {res.elapsed_seconds:.1f} s")
        fields[m]  = xv
        timings[m] = res.elapsed_seconds
    return fields, timings


def _load_or_run(no_cache: bool = False
                 ) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    if not no_cache and _CACHE.exists():
        print(f"Loading cached fields from {_CACHE.name}")
        d       = np.load(_CACHE)
        fields  = {m: d[m] for m in METHODS}
        timings = {m: float(d[f"t_{m}"]) for m in METHODS}
        return fields, timings
    fields, timings = _run_simulations()
    save_dict = {m: fields[m] for m in METHODS}
    save_dict.update({f"t_{m}": np.float64(timings[m]) for m in METHODS})
    np.savez_compressed(_CACHE, **save_dict)
    print(f"\nCached fields → {_CACHE}")
    return fields, timings


# ── main ──────────────────────────────────────────────────────────────────────

def run(save_path: str | None = None, show: bool = True,
        no_cache: bool = False) -> None:

    fields, timings = _load_or_run(no_cache)

    # ── colour range and difference ───────────────────────────────────────────
    all_vals = np.concatenate([f.ravel() for f in fields.values()])
    vlo, vhi = np.percentile(all_vals, 1), np.percentile(all_vals, 99)

    diff = np.abs(fields["fd"] - fields["jansson"])
    dmax = diff.max()

    # ── figure: row 0 = 3 micrographs, row 1 = statistics table ─────────────
    Lx_um  = SIM["Lx"] * 1e6
    extent = [0, Lx_um, 0, Lx_um]

    # ── fully manual layout — every axis placed by exact figure coordinates ──
    # This guarantees all three image panels are identical in size.
    #
    # Figure: 7.2 × 4.2 in
    # Image row occupies [left=0.07, right=0.99, bottom=0.31, top=0.99]
    # Three images share that space equally; colorbars sit beside (b) and (c).
    #
    #  im_w + gap + im_w + cb_w + cb_pad + gap + im_w + cb_w + cb_pad = avail_w
    #  let im_w = x, gap=0.010, cb_w=0.014, cb_pad=0.004
    #  3x + 2*0.010 + 2*(0.014+0.004) = 0.92  →  x = (0.92 - 0.056)/3 = 0.2880

    FIG_W = 7.2
    L, R, BOT, TOP = 0.07, 0.96, 0.26, 0.99
    # Compute IM_W first (depends only on FIG_W), then size FIG_H so
    # that IM_H_inches ≈ IM_W_inches (square panels, no padding).
    GAP_pre   = 9 / (FIG_W * 25.4)
    CB_W_pre  = 0.014 * 0.60 * 0.70
    CB_PAD    = -0.002
    IM_W_pre  = ((R - L) - 2*GAP_pre - 3*(CB_W_pre + CB_PAD)) / 3
    IM_W_in   = IM_W_pre * FIG_W          # target image height in inches
    FIG_H     = IM_W_in / (TOP - BOT)     # force square panels
    IM_H   = TOP - BOT
    GAP    = GAP_pre
    CB_W   = CB_W_pre
    AVAIL  = R - L
    IM_W   = IM_W_pre

    CB_H    = IM_H * 0.60                  # colorbar is 60% of image height
    CB_VOFF = (IM_H - CB_H) / 2           # centre vertically

    x0 = L
    x1 = x0 + IM_W + CB_W + CB_PAD + GAP
    x2 = x1 + IM_W + CB_W + CB_PAD + GAP

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax_fd   = fig.add_axes([x0,                      BOT,           IM_W, IM_H])
    cax0    = fig.add_axes([x0 + IM_W + CB_PAD,      BOT + CB_VOFF, CB_W, CB_H])
    ax_jan  = fig.add_axes([x1,                      BOT,           IM_W, IM_H])
    cax1    = fig.add_axes([x1 + IM_W + CB_PAD,      BOT + CB_VOFF, CB_W, CB_H])
    ax_diff = fig.add_axes([x2,                      BOT,           IM_W, IM_H])
    cax2    = fig.add_axes([x2 + IM_W + CB_PAD,      BOT + CB_VOFF, CB_W, CB_H])
    TBL_GAP = 0.028 + 7 / (FIG_H * 25.4)   # extra 7 mm gap below images
    TBL_H   = 0.09    # table height (fig coords)
    TBL_W   = (R - L) * 0.93
    TBL_L   = L + (R - L - TBL_W) / 2    # centred
    ax_tbl  = fig.add_axes([TBL_L, BOT - TBL_GAP - TBL_H, TBL_W, TBL_H])

    cmap_comp = "coolwarm"
    cmap_diff = "Oranges"

    # ── Row 0: composition micrographs (a) FD and (b) Jansson ────────────────
    for ax, m, label in [(ax_fd, "fd", "(a)"), (ax_jan, "jansson", "(b)")]:
        im = ax.imshow(
            fields[m], origin="lower", extent=extent,
            cmap=cmap_comp, vmin=vlo, vmax=vhi, interpolation="antialiased",
            aspect="auto",
        )
        ax.set_xlabel("$L_x$ (µm)", labelpad=2, fontsize=6.3)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(0.25))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(0.25))
        ax.tick_params(axis="both", labelsize=5.6)
        if ax is ax_fd:
            ax.set_ylabel("$L_y$ (µm)", labelpad=2, fontsize=6.3)
        else:
            ax.tick_params(labelleft=False)
        ax.text(0.97, 0.04, label,
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=9, fontweight="bold", color="white")
        ax.text(0.03, 0.97, METHOD_LABEL[m],
                transform=ax.transAxes, ha="left", va="top",
                fontsize=7.5, color="white",
                bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.45, lw=0))
        if m == "fd":
            im_fd = im
        if m == "jansson":
            im_comp = im

    # colorbars for (a) and (b) — same scale
    t_lo  = round(vlo  + (vhi - vlo) * 0.02, 2)
    t_mid = round((vlo + vhi) / 2,            2)
    t_hi  = round(vhi  - (vhi - vlo) * 0.02, 2)
    ax_fd.spines["right"].set_visible(False)
    ax_jan.spines["right"].set_visible(False)
    for cax, im_src in [(cax0, im_fd), (cax1, im_comp)]:
        cb = fig.colorbar(im_src, cax=cax, ticks=[t_lo, t_mid, t_hi])
        cb.set_label("$x_{\\mathrm{V}}$", fontsize=7, labelpad=2)
        cb.ax.tick_params(labelsize=6, pad=1, length=2)
        cb.ax.set_yticklabels([f"{t_lo:.2f}", f"{t_mid:.2f}", f"{t_hi:.2f}"])
        cb.ax.spines["left"].set_visible(False)

    # ── Row 0, panel (c): difference map ─────────────────────────────────────
    vmax_diff = max(dmax, 1e-10)
    im_diff   = ax_diff.imshow(
        diff, origin="lower", extent=extent,
        cmap=cmap_diff, vmin=0, vmax=vmax_diff, interpolation="antialiased",
        aspect="auto",
    )
    ax_diff.set_xlabel("$L_x$ (µm)", labelpad=2, fontsize=6.3)
    ax_diff.tick_params(axis="both", labelsize=5.6, labelleft=False)
    ax_diff.xaxis.set_major_locator(ticker.MultipleLocator(0.25))
    ax_diff.yaxis.set_major_locator(ticker.MultipleLocator(0.25))
    ax_diff.text(0.97, 0.04, "(c)",
                 transform=ax_diff.transAxes, ha="right", va="bottom",
                 fontsize=9, fontweight="bold")
    ax_diff.text(0.03, 0.97, "$|$FD $-$ Jansson$|$",
                 transform=ax_diff.transAxes, ha="left", va="top",
                 fontsize=7.5,
                 bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.6, lw=0))
    rmse = float(np.sqrt(np.mean(diff**2)))
    ax_diff.text(0.03, 0.04,
                 f"max $|\\Delta|$ = {dmax:.2e}\nRMSE = {rmse:.2e}",
                 transform=ax_diff.transAxes, ha="left", va="bottom",
                 fontsize=8, color="0.2",
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.75, lw=0.5))

    # colorbar for (c)
    if dmax > 0:
        cb2_exp    = int(np.floor(np.log10(dmax)))
        cb2_scale  = 10 ** cb2_exp
        cb2_ticks  = [0, dmax / 2, dmax]
        cb2_labels = ["0", f"{dmax/2/cb2_scale:.1f}", f"{dmax/cb2_scale:.1f}"]
        cb2_unit   = f"$\\times10^{{{cb2_exp}}}$"
    else:
        cb2_ticks = [0, 5e-11, 1e-10]; cb2_labels = ["0", "0", "0"]; cb2_unit = ""
    ax_diff.spines["right"].set_visible(False)
    cb2   = fig.colorbar(im_diff, cax=cax2, ticks=cb2_ticks)
    cb2.set_label(f"$|\\Delta x_{{\\mathrm{{V}}}}|$ {cb2_unit}", fontsize=6, labelpad=2)
    cb2.ax.tick_params(labelsize=6, pad=1, length=2)
    cb2.ax.set_yticklabels(cb2_labels)
    cb2.ax.spines["left"].set_visible(False)

    # ── Table: pre-placed ax_tbl ──────────────────────────────────────────────
    ax_tbl.axis("off")

    col_labels = ["Method", "min $x_V$", "max $x_V$", "RMSE vs Jansson", "Simulation time (s)"]
    tbl_rows   = []
    for m in METHODS:
        xv       = fields[m]
        rmse_val = (float(np.sqrt(np.mean((fields[m] - fields["jansson"])**2)))
                    if m != "jansson" else None)
        rmse_str = f"{rmse_val:.2e}" if rmse_val is not None else "—"
        t_str    = f"{timings[m]:.0f}" if m in timings else "—"
        short    = {"fd": "FD", "jansson": "Jansson"}
        tbl_rows.append([short[m], f"{xv.min():.4f}", f"{xv.max():.4f}",
                         rmse_str, t_str])

    ROW_H  = 0.45
    n_rows = len(tbl_rows) + 1          # data rows + header
    tbl_h  = n_rows * ROW_H             # fraction of ax height the cells occupy
    tbl = ax_tbl.table(
        cellText  = tbl_rows,
        colLabels = col_labels,
        loc       = "upper center",
        cellLoc   = "center",
        bbox      = [0.0, 1.0 - tbl_h, 1.0, tbl_h],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7.5)
    tbl.auto_set_column_width(list(range(len(col_labels))))
    for (r, c), cell in tbl.get_celld().items():
        cell.set_height(ROW_H)
        cell.set_linewidth(0.4)
        if r == 0:
            cell.set_facecolor("#DDDDDD")
            cell.set_text_props(fontweight="bold")


    # ── save / show ───────────────────────────────────────────────────────────
    if save_path is not None:
        out = Path(save_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=300)
        print(f"\nFigure saved → {out}")

    if show:
        plt.show()
    else:
        plt.close(fig)


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Microstructure comparison — FD vs Jansson derivative methods"
    )
    parser.add_argument("--save",     default=None)
    parser.add_argument("--no-show",  action="store_true")
    parser.add_argument("--no-cache", action="store_true",
                        help="Force re-run of simulations")
    args = parser.parse_args()
    run(save_path=args.save, show=not args.no_show, no_cache=args.no_cache)
