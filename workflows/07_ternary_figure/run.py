"""
Figure: Ternary Nb--Hf--V microstructure summary.

Layout:
  Left block  — 2 rows of phase-field panels:
    Row 1 (2 panels) — Chemical vs elasto-chemical, 25%Hf, 500°C
    Row 2 (2 panels) — Temperature dependence, Nb50Hf5V45: 500°C | 600°C

  Right block — pfm_ternary.png (ternary miscibility-gap phase diagram)

  One shared colorbar on the far right of the left block.
  Row labels as figure-level text, left of the left block.

The Hf composition sweep at 500°C (Hf=5%, 10%, 20%, 25%, 35%, 45%) is drawn
as separate single-row figures, chemical only and elasto-chemical.

Output: paper/.../figs/ternary_summary.{pdf,png}
        paper/.../figs/hf_sweep.{pdf,png}
        paper/.../figs/hf_sweep_elastic.{pdf,png}

Data sources:
  Hf sweep   : results/HfNbTiV_BCC_PhaseForge_v2/NbHfV/500C/kinetics/
                  pf_2d_c50Nb{05,10,20,25,35,45}Hf_kappa1e12_M5e21_tdbv2[_elastic]/fields.npz
  Elastic    : results/HfNbTiV_BCC_PhaseForge_v2/NbHfV/500C/kinetics/
                  pf_2d_c50Nb25Hf_kappa1e12_M5e21_tdbv2_elastic/fields.npz
  T-dep 500°C: results/HfNbTiV_BCC_PhaseForge_v2/NbHfV/500C/kinetics/
                  pf_2d_c50Nb05Hf_kappa1e12_M5e21_tdbv2/fields.npz  (snap idx 3 → t≈3000s)
  T-dep 600°C: results/HfNbTiV_BCC_PhaseForge_v2_Tdep/NbHfV/600C/snapshots.npz
"""

from __future__ import annotations
import logging
import pathlib
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.image import imread
from mpl_toolkits.axes_grid1 import make_axes_locatable
import sys

matplotlib.rcParams.update({
    "font.family":       "sans-serif",
    "font.size":         8,
    "axes.linewidth":    0.6,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.major.size":  2.5,
    "ytick.major.size":  2.5,
    "axes.labelsize":    8,
    "xtick.labelsize":   6.5,
    "ytick.labelsize":   6.5,
})

ROOT     = pathlib.Path(__file__).resolve().parents[2]
RES_V2   = ROOT / "results" / "HfNbTiV_BCC_PhaseForge_v2" / "NbHfV" / "500C" / "kinetics"
RES_TDEP = ROOT / "results" / "HfNbTiV_BCC_PhaseForge_v2_Tdep" / "NbHfV"
FIGS     = ROOT / "paper" / "2025_Kunselman_Design_of_Spinodal_Microstructures_in_HEAs" / "figs"

HF_SWEEP_PATHS = [
    ("Nb$_{50}$Hf$_{5}$V$_{45}$",  RES_V2 / "pf_2d_c50Nb05Hf_kappa1e12_M5e21_tdbv2" / "fields.npz", -1, None),
    ("Nb$_{50}$Hf$_{10}$V$_{40}$", RES_V2 / "pf_2d_c50Nb10Hf_kappa1e12_M5e21_tdbv2" / "fields.npz", -1, None),
    ("Nb$_{50}$Hf$_{20}$V$_{30}$", RES_V2 / "pf_2d_c50Nb20Hf_kappa1e12_M5e21_tdbv2" / "fields.npz", -1, None),
    ("Nb$_{50}$Hf$_{25}$V$_{25}$", RES_V2 / "pf_2d_c50Nb25Hf_kappa1e12_M5e21_tdbv2" / "fields.npz", -1, None),
    ("Nb$_{50}$Hf$_{35}$V$_{15}$", RES_V2 / "pf_2d_c50Nb35Hf_kappa1e12_M5e21_tdbv2" / "fields.npz", -1, None),
    ("Nb$_{50}$Hf$_{45}$V$_{5}$",  RES_V2 / "pf_2d_c50Nb45Hf_kappa1e12_M5e21_tdbv2" / "fields.npz", -1, None),
]
HF_SWEEP_ELASTIC_PATHS = [
    ("Nb$_{50}$Hf$_{5}$V$_{45}$",  RES_V2 / "pf_2d_c50Nb05Hf_kappa1e12_M5e21_tdbv2_elastic" / "fields.npz", 3, None),
    ("Nb$_{50}$Hf$_{10}$V$_{40}$", RES_V2 / "pf_2d_c50Nb10Hf_kappa1e12_M5e21_tdbv2_elastic" / "fields.npz", 3, None),
    ("Nb$_{50}$Hf$_{20}$V$_{30}$", RES_V2 / "pf_2d_c50Nb20Hf_kappa1e12_M5e21_tdbv2_elastic" / "fields.npz", 6, None),
    ("Nb$_{50}$Hf$_{25}$V$_{25}$", RES_V2 / "pf_2d_c50Nb25Hf_kappa1e12_M5e21_tdbv2_elastic" / "fields.npz", -1, None),
    ("Nb$_{50}$Hf$_{35}$V$_{15}$", RES_V2 / "pf_2d_c50Nb35Hf_kappa1e12_M5e21_tdbv2_elastic" / "fields.npz", 6, None),
    ("Nb$_{50}$Hf$_{45}$V$_{5}$",  RES_V2 / "pf_2d_c50Nb45Hf_kappa1e12_M5e21_tdbv2_elastic" / "fields.npz", 6, None),
]

CHEM_25  = RES_V2 / "pf_2d_c50Nb25Hf_kappa1e12_M5e21_tdbv2"          / "fields.npz"
ELAS_25  = RES_V2 / "pf_2d_c50Nb25Hf_kappa1e12_M5e21_tdbv2_elastic"  / "fields.npz"
TDEP_500 = RES_V2 / "pf_2d_c50Nb05Hf_kappa1e12_M5e21_tdbv2"          / "fields.npz"
TDEP_600 = RES_TDEP / "600C" / "snapshots.npz"
PFM_IMG  = FIGS / "pfm_ternary.png"


def _load_fields(path: pathlib.Path, snap_idx: int = -1):
    d   = np.load(path)
    idx = snap_idx % len(d["x_nb"])
    xnb = np.clip(d["x_nb"][idx], 0.0, 1.0)
    t_s = float(d["time"][idx])
    Lx  = float(d["Lx"]) * 1e6   # m → µm
    return xnb, t_s, Lx


def _load_tdep(path: pathlib.Path, snap_idx: int = -1):
    d    = np.load(path)
    idx  = snap_idx % len(d["compositions"])
    comp = d["compositions"][idx]           # (2, N, N)
    nb_i = list(d["elements"]).index("NB")
    xnb  = np.clip(comp[nb_i], 0.0, 1.0)
    t_s  = float(d["times"][idx])
    Lx   = float(d["Lx"]) * 1e6 if "Lx" in d else 1.0
    return xnb, t_s, Lx


def _draw_panel(ax, field, Lx_um, t_s, cmap, vmin, vmax,
                show_xlabel=False, show_ylabel=False, title=""):
    nm = Lx_um * 1e3
    im = ax.imshow(field, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax,
                   extent=[0, nm, 0, nm])
    ax.set_aspect("equal")
    ax.tick_params(labelsize=6, length=2, pad=2)

    if show_xlabel:
        ax.set_xlabel("nm", fontsize=7, labelpad=2)
    else:
        ax.set_xticklabels([])
        ax.tick_params(bottom=False)

    if show_ylabel:
        ax.set_ylabel("nm", fontsize=7, labelpad=2)
    else:
        ax.set_yticklabels([])
        ax.tick_params(left=False)

    if title:
        ax.set_title(title, fontsize=7, pad=3)

    tlabel = f"$t$ = {t_s:.0f} s" if t_s < 3600 else f"$t$ = {t_s/3600:.1f} h"
    ax.text(0.97, 0.97, tlabel, transform=ax.transAxes,
            ha="right", va="top", fontsize=5.5, color="white",
            bbox=dict(boxstyle="round,pad=0.15", fc="0.15", ec="none", alpha=0.75))
    return im


def make_figure(out_dir: pathlib.Path = FIGS) -> pathlib.Path:
    out_dir.mkdir(parents=True, exist_ok=True)

    cmap = "coolwarm"

    sweep = []
    for entry in HF_SWEEP_PATHS:
        lbl, p = entry[0], entry[1]
        if p is None:
            sweep.append((lbl, None, None, None))   # placeholder
        else:
            xnb, t_s, Lx = _load_fields(p, entry[2])
            sweep.append((lbl, xnb, t_s, Lx))
    # sweep items: (label, xnb, t_s, Lx)
    xnb_c25, t_c25, Lx_c25 = _load_fields(CHEM_25, -1)
    xnb_e25, t_e25, Lx_e25 = _load_fields(ELAS_25, -1)
    xnb_500, t_500, Lx_500 = _load_fields(TDEP_500, 3)   # t≈3000s, more developed
    xnb_600, t_600, Lx_600 = _load_tdep(TDEP_600, -1)
    pfm_img = imread(PFM_IMG)

    # shared colormap range, centred on 0.5 (skip placeholders)
    all_f = [s[1] for s in sweep if s[1] is not None] + [xnb_c25, xnb_e25, xnb_500, xnb_600]
    vmin  = min(f.min() for f in all_f)
    vmax  = max(f.max() for f in all_f)
    half  = max(0.5 - vmin, vmax - 0.5)
    vmin, vmax = 0.5 - half, 0.5 + half

    FIG_W, FIG_H = 7.2, 5.6

    # horizontal zones (figure fractions)
    ROW_LBL_W = 0.055   # width reserved for row labels
    LEFT_L    = 0.10    # left edge of panel grid (extra room for row labels)
    LEFT_R    = 0.60    # right edge of panel grid (before shared colorbar)
    CB_W      = 0.018   # colorbar width
    CB_GAP    = 0.012   # gap between grid and colorbar
    RIGHT_L   = 0.665   # left edge of pfm image
    RIGHT_R   = 0.99

    # vertical zones
    TOP  = 0.96
    BOT  = 0.04
    RGAP = 0.035   # gap between rows

    # 2 rows of equal height
    n_rows  = 2
    total_h = TOP - BOT - (n_rows - 1) * RGAP
    ROW_H   = total_h / n_rows

    def row_bottom(row):   # row 0 = top
        return TOP - (row + 1) * ROW_H - row * RGAP

    def make_axes_row(row, n_cols, last_row=False):
        """Create n_cols axes for a given row, evenly spaced across LEFT_L..LEFT_R."""
        bot  = row_bottom(row)
        grid_w = LEFT_R - LEFT_L
        gap_w  = 0.012
        pw     = (grid_w - (n_cols - 1) * gap_w) / n_cols
        axes   = []
        for c in range(n_cols):
            lft = LEFT_L + c * (pw + gap_w)
            ax  = fig.add_axes([lft, bot, pw, ROW_H])
            axes.append(ax)
        return axes, bot, ROW_H

    fig = plt.figure(figsize=(FIG_W, FIG_H))

    panel_label = iter("abcdefghij")

    def label_ax(ax):
        lbl = next(panel_label)
        ax.text(-0.04, 1.03, f"({lbl})", transform=ax.transAxes,
                fontsize=7, fontweight="bold", va="bottom", ha="right")

    def row_label(row, text):
        bot  = row_bottom(row)
        mid  = bot + ROW_H / 2
        fig.text(LEFT_L - 0.055, mid, text, ha="center", va="center",
                 fontsize=6.5, fontweight="bold", rotation=90,
                 transform=fig.transFigure)

    axes2, bot2, h2 = make_axes_row(0, 2)
    for ax, (xnb, t_s, Lx, title) in zip(axes2, [
        (xnb_c25, t_c25, Lx_c25, "Chemical only"),
        (xnb_e25, t_e25, Lx_e25, "Elasto-chemical"),
    ]):
        is_first = (ax is axes2[0])
        im = _draw_panel(ax, xnb, Lx, t_s, cmap, vmin, vmax,
                         show_xlabel=False, show_ylabel=is_first,
                         title=title)
        label_ax(ax)
        last_im = im
    row_label(0, "Elastic effect\n25%Hf, 500°C")

    axes3, bot3, h3 = make_axes_row(1, 2)
    for ax, (xnb, t_s, Lx, title) in zip(axes3, [
        (xnb_500, t_500, Lx_500, "500°C"),
        (xnb_600, t_600, Lx_600, "600°C"),
    ]):
        is_first = (ax is axes3[0])
        im = _draw_panel(ax, xnb, Lx, t_s, cmap, vmin, vmax,
                         show_xlabel=True, show_ylabel=is_first,
                         title=title)
        label_ax(ax)
        last_im = im
    row_label(1, "T dependence\nNb$_{50}$Hf$_5$V$_{45}$")

    cb_ax = fig.add_axes([LEFT_R + CB_GAP, BOT, CB_W, TOP - BOT])
    cb    = fig.colorbar(last_im, cax=cb_ax)
    cb.set_label("$x_{\\rm Nb}$", fontsize=8, labelpad=3)
    cb.ax.tick_params(labelsize=6.5, length=2)
    cb.set_ticks([round(vmin + 0.02, 2),
                  round((vmin + vmax) / 2, 2),
                  round(vmax - 0.02, 2)])

    ax_r = fig.add_axes([RIGHT_L, BOT + 0.05, RIGHT_R - RIGHT_L, TOP - BOT - 0.05])
    ax_r.imshow(pfm_img, aspect="equal")
    ax_r.set_axis_off()
    ax_r.set_title("Ternary miscibility gap\n(BCC$_{A2}$, Hf--Nb--V)",
                   fontsize=7.5, pad=5)
    label_ax(ax_r)

    for ext in ("pdf", "png"):
        p = out_dir / f"ternary_summary.{ext}"
        fig.savefig(p, dpi=300, bbox_inches="tight",
                    bbox_extra_artists=[])
        print(f"Saved: {p}")
    plt.close(fig)
    return out_dir / "ternary_summary.pdf"


def make_hf_sweep_figure(out_dir: pathlib.Path = FIGS) -> pathlib.Path:
    """Standalone figure: Hf composition sweep row (6 panels + colorbar)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cmap = "coolwarm"

    sweep = []
    for entry in HF_SWEEP_PATHS:
        lbl, p = entry[0], entry[1]
        if p is None:
            sweep.append((lbl, None, None, None))
        else:
            xnb, t_s, Lx = _load_fields(p, entry[2])
            sweep.append((lbl, xnb, t_s, Lx))

    all_f = [s[1] for s in sweep if s[1] is not None]
    vmin  = min(f.min() for f in all_f)
    vmax  = max(f.max() for f in all_f)
    half  = max(0.5 - vmin, vmax - 0.5)
    vmin, vmax = 0.5 - half, 0.5 + half

    n     = len(sweep)
    panel = 1.35          # inches per panel (square field)
    cb_w  = 0.25
    gap   = 0.08
    FIG_W = n * panel + cb_w + 0.5
    FIG_H = panel + 0.45  # panel + title + xlabel

    fig, axes = plt.subplots(1, n, figsize=(FIG_W, FIG_H))
    fig.subplots_adjust(left=0.06, right=(n * panel) / FIG_W,
                        bottom=0.14, top=0.88, wspace=0.11)

    last_im = None
    for ax, (lbl, xnb, t_s, Lx) in zip(axes, sweep):
        is_first = (ax is axes[0])
        if xnb is None:
            ax.set_facecolor("0.92")
            ax.set_xlim(0, 1); ax.set_ylim(0, 1)
            ax.set_aspect("equal")
            ax.text(0.5, 0.5, "simulation\npending", transform=ax.transAxes,
                    ha="center", va="center", fontsize=6.5, color="0.5",
                    style="italic")
            ax.set_xticks([]); ax.set_yticks([])
        else:
            im = _draw_panel(ax, xnb, Lx, t_s, cmap, vmin, vmax,
                             show_xlabel=True, show_ylabel=is_first)
            last_im = im
        ax.set_title(lbl, fontsize=7.5, pad=3)

    # colorbar to the right of the last axis
    cb_ax = fig.add_axes([
        axes[-1].get_position().x1 + gap / FIG_W,
        axes[-1].get_position().y0,
        cb_w / FIG_W,
        axes[-1].get_position().height,
    ])
    cb = fig.colorbar(last_im, cax=cb_ax)
    cb.set_label("$x_{\\rm Nb}$", fontsize=8, labelpad=3)
    cb.ax.tick_params(labelsize=6.5, length=2)
    cb.set_ticks([round(vmin + 0.02, 2),
                  round((vmin + vmax) / 2, 2),
                  round(vmax - 0.02, 2)])

    for ext in ("pdf", "png"):
        p = out_dir / f"hf_sweep.{ext}"
        fig.savefig(p, dpi=300, bbox_inches="tight")
        print(f"Saved: {p}")
    plt.close(fig)
    return out_dir / "hf_sweep.pdf"


def make_hf_sweep_elastic_figure(sweep=None, out_dir=None):
    """Single-row elasto-chemical Hf sweep, mirrors make_hf_sweep_figure()."""
    if sweep is None:
        sweep = []
        for entry in HF_SWEEP_ELASTIC_PATHS:
            lbl, p = entry[0], entry[1]
            if p is None:
                sweep.append((lbl, None, None, None))
            else:
                xnb, t_s, Lx = _load_fields(p, entry[2])
                sweep.append((lbl, xnb, t_s, Lx))
    if out_dir is None:
        out_dir = FIGS

    cmap = "RdBu_r"

    all_f = [s[1] for s in sweep if s[1] is not None]
    if all_f:
        vmin  = min(f.min() for f in all_f)
        vmax  = max(f.max() for f in all_f)
        half  = max(0.5 - vmin, vmax - 0.5)
        vmin, vmax = 0.5 - half, 0.5 + half
    else:
        vmin, vmax = 0.0, 1.0

    n     = len(sweep)
    panel = 1.35
    cb_w  = 0.25
    FIG_W = n * panel + cb_w + 0.5
    FIG_H = panel + 0.45

    fig, axes = plt.subplots(1, n, figsize=(FIG_W, FIG_H))
    fig.subplots_adjust(left=0.06, right=(n * panel) / FIG_W,
                        bottom=0.14, top=0.88, wspace=0.11)

    last_im = None
    for ax, (lbl, xnb, t_s, Lx) in zip(axes, sweep):
        is_first = (ax is axes[0])
        if xnb is None:
            ax.set_facecolor("0.92")
            ax.set_xlim(0, 1); ax.set_ylim(0, 1)
            ax.set_aspect("equal")
            ax.text(0.5, 0.5, "simulation\npending", transform=ax.transAxes,
                    ha="center", va="center", fontsize=6.5, color="0.5",
                    style="italic")
            ax.set_xticks([]); ax.set_yticks([])
        else:
            im = _draw_panel(ax, xnb, Lx, t_s, cmap, vmin, vmax,
                             show_xlabel=True, show_ylabel=is_first)
            last_im = im
        ax.set_title(lbl, fontsize=7.5, pad=3)

    # colorbar (use a placeholder mappable if all panels are placeholders)
    if last_im is None:
        import matplotlib.cm as cm
        import matplotlib.colors as mcolors
        sm = plt.cm.ScalarMappable(cmap=cmap,
                                   norm=mcolors.Normalize(vmin=vmin, vmax=vmax))
        sm.set_array([])
        last_im = sm

    cb_ax = fig.add_axes([
        axes[-1].get_position().x1 + 0.08 / FIG_W,
        axes[-1].get_position().y0,
        cb_w / FIG_W,
        axes[-1].get_position().height,
    ])
    cb = fig.colorbar(last_im, cax=cb_ax)
    cb.set_label("$x_{\\rm Nb}$", fontsize=8, labelpad=3)
    cb.ax.tick_params(labelsize=6.5, length=2)
    cb.set_ticks([round(vmin + 0.02, 2),
                  round((vmin + vmax) / 2, 2),
                  round(vmax - 0.02, 2)])

    for ext in ("pdf", "png"):
        p = out_dir / f"hf_sweep_elastic.{ext}"
        fig.savefig(p, dpi=300, bbox_inches="tight")
        print(f"Saved: {p}")
    plt.close(fig)
    return out_dir / "hf_sweep_elastic.pdf"


if __name__ == "__main__":
    logging.basicConfig(format="%(message)s", stream=sys.stdout)
    logging.getLogger("phasefoundry").setLevel(logging.INFO)
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-sweep",         action="store_true")
    ap.add_argument("--hf-sweep-elastic", action="store_true")
    args = ap.parse_args()
    if args.hf_sweep:
        make_hf_sweep_figure()
    elif args.hf_sweep_elastic:
        make_hf_sweep_elastic_figure()
    else:
        make_figure()
        make_hf_sweep_figure()
        make_hf_sweep_elastic_figure()
