"""
Compact 3-system microstructure comparison figure
==================================================
Produces a single figure with one panel per system showing the primary
composition field at the end of the Cahn-Hilliard simulation:

  (a) Binary   Nb₅₀V₅₀     → x_V   (fully decomposed)
  (b) Ternary  Nb₅₀Hf₁₀V₄₀ → x_NB  (fully decomposed)
  (c) Quaternary Nb₅₀Hf₅Ti₅V₄₀ → x_NB (no decomposition – stiffness-limited)

Binary and ternary data loaded from existing simulation caches.
Quaternary run at the stable dt (1.32e-3 s) for 30 000 steps (~40 s physical).

Output: paper/.../figs/stiffness_microstructures.pdf
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

matplotlib.rcParams.update({
    "font.family":       "serif",
    "font.serif":        ["Times New Roman", "DejaVu Serif"],
    "font.size":         7,
    "axes.labelsize":    7,
    "xtick.labelsize":   6.2,
    "ytick.labelsize":   6.2,
    "axes.linewidth":    0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "figure.dpi":        150,
    "savefig.dpi":       300,
})

_CACHE_BIN = _ROOT / "tests" / "_cache_microstructure_nbv.npz"
_CACHE_TER = _ROOT / "tests" / "_cache_microstructure_nbhfv.npz"
_CACHE_QUA = _ROOT / "tests" / "_cache_microstructure_nbhftiv.npz"

_PAPER_FIGS = (_ROOT / "paper" /
               "2025_Kunselman_Design_of_Spinodal_Microstructures_in_HEAs" /
               "figs")
OUT_PDF = _PAPER_FIGS / "stiffness_microstructures.pdf"

def _get_quaternary_field() -> tuple[np.ndarray, float]:
    """Run quaternary at stable dt for 30 000 steps; cache result."""
    if _CACHE_QUA.exists():
        print(f"  Quaternary: loading cache {_CACHE_QUA.name}")
        d = np.load(_CACHE_QUA)
        return d["jansson_nb"], float(d["t_jansson"])

    from phasefoundry.phasefield.solver import solve_cahn_hilliard, stable_dt
    from phasefoundry.thermodynamics.interpolant import build_interpolant
    from phasefoundry.thermodynamics.storage import load_table

    ELEMENTS  = ["NB", "HF", "TI", "V"]
    REF_EL    = "V"
    T_TARGET  = 773.15  # 500 °C
    N_STEPS   = 30_000

    table_dir = _ROOT / "data" / "nb_hf_ti_v"
    tables    = sorted(table_dir.glob("*.h5"))
    best = min(tables, key=lambda p: abs(
        float(load_table(p)["metadata"]["temperature"]) - T_TARGET))
    T_K = float(load_table(best)["metadata"]["temperature"])
    print(f"  Quaternary table: {best.name}  ({T_K - 273.15:.0f} °C)")

    c0  = [0.50, 0.05, 0.05]   # x_NB, x_HF, x_TI  (x_V = 0.40)
    M   = [[5e-21, 0, 0], [0, 5e-21, 0], [0, 0, 5e-21]]
    dx  = 1e-6 / 256
    M_diag = np.array([M[i][i] for i in range(3)])

    interp = build_interpolant(
        [best], elements=ELEMENTS, reference_element=REF_EL,
        temperature_K=T_K, derivative_method="jansson")
    dt = stable_dt(M_diag, dx, interp, c0=np.array(c0))
    print(f"  Quaternary stable dt = {dt:.4e} s  "
          f"→ physical time = {dt * N_STEPS:.1f} s")

    t0  = time.time()
    res = solve_cahn_hilliard(
        interpolant=interp, output_dir=None, save_frames=False,
        N=256, Lx=1e-6, c0=c0, noise=0.01, seed=42,
        kappa=[1e-12, 1e-12, 1e-12], M=M,
        dt=dt, n_steps=N_STEPS, save_every=N_STEPS,
    )
    elapsed = time.time() - t0
    x_nb = res.compositions[0]
    print(f"  Quaternary done: x_NB [{x_nb.min():.4f}, {x_nb.max():.4f}]"
          f"  wall={elapsed:.0f} s")

    np.savez_compressed(_CACHE_QUA, jansson_nb=x_nb, t_jansson=elapsed)
    return x_nb, elapsed


def make_figure():
    print("Loading binary cache …")
    b = np.load(_CACHE_BIN)
    x_v_bin   = b["jansson"]          # x_V  (256×256)
    t_bin     = float(b["t_jansson"])

    print("Loading ternary cache …")
    t_ = np.load(_CACHE_TER)
    x_nb_ter  = t_["jansson"]         # x_NB (256×256)
    t_ter     = float(t_["t_jansson"])

    print("Getting quaternary field …")
    x_nb_qua, t_qua = _get_quaternary_field()

    FIG_W  = 10.8         # inches (1.5× text width)
    MM     = 1 / (FIG_W * 25.4)   # 1 mm in figure-fraction units

    L      = 0.06 * FIG_W * MM * 25.4   # left margin ~6% ≈ left in frac
    L      = 0.06
    R      = 0.97
    BOT    = 0.16
    TOP    = 0.95

    CB_W   = 2  * MM      # colorbar width 2 mm
    CB_PAD = 0.5 * MM     # 0.5 mm gap image→colorbar
    GAP    = 5  * MM      # 5 mm between colorbar and next image

    n = 3
    IM_W = ((R - L) - n * CB_W - n * CB_PAD - (n - 1) * GAP) / n
    IM_W_in = IM_W * FIG_W
    FIG_H   = IM_W_in / (TOP - BOT)   # forces square image panels

    fig = plt.figure(figsize=(FIG_W, FIG_H))

    xs = []
    xs.append(L)
    xs.append(xs[0] + IM_W + CB_PAD + CB_W + GAP)
    xs.append(xs[1] + IM_W + CB_PAD + CB_W + GAP)

    IM_H_norm = TOP - BOT          # axes height spans full vertical range → square
    CB_H      = IM_H_norm * 0.65
    CB_VOFF   = (IM_H_norm - CB_H) / 2

    panels = [
        dict(field=x_v_bin,
             label=r"$x_{\rm V}$",
             title=r"(a) Binary  Nb$_{50}$V$_{50}$",
             cmap="RdBu_r",
             t_phys=0.01 * 80000,          # dt * n_steps
             t_wall=t_bin),
        dict(field=x_nb_ter,
             label=r"$x_{\rm Nb}$",
             title=r"(b) Ternary  Nb$_{50}$Hf$_{10}$V$_{40}$",
             cmap="RdBu_r",
             t_phys=1.713e-3 * 30000,
             t_wall=t_ter),
        dict(field=x_nb_qua,
             label=r"$x_{\rm Nb}$",
             title=r"(c) Quaternary  Nb$_{50}$Hf$_{5}$Ti$_{5}$V$_{40}$",
             cmap="RdBu_r",
             t_phys=1.32e-3 * 30000,
             t_wall=t_qua),
    ]

    Lx_um = 1.0   # domain side in µm

    for i, (x0, p) in enumerate(zip(xs, panels)):
        ax  = fig.add_axes([x0, BOT, IM_W, IM_H_norm])
        cax = fig.add_axes([x0 + IM_W + CB_PAD, BOT + CB_VOFF, CB_W, CB_H])

        vmin, vmax = p["field"].min(), p["field"].max()
        # symmetric colourmap around mean
        vmid = p["field"].mean()
        vspan = max(abs(vmin - vmid), abs(vmax - vmid))
        im = ax.imshow(p["field"],
                       origin="lower",
                       extent=[0, Lx_um, 0, Lx_um],
                       cmap=p["cmap"],
                       vmin=vmid - vspan, vmax=vmid + vspan,
                       interpolation="nearest",
                       aspect="auto")

        cb = fig.colorbar(im, cax=cax)
        cb.ax.tick_params(labelsize=5.5)
        cb.set_label(p["label"], fontsize=6.0, labelpad=2)
        cb.ax.yaxis.set_major_locator(plt.MaxNLocator(4))

        ax.set_xlabel(r"$L_x$ (µm)", fontsize=6.5, labelpad=2)
        if i == 0:
            ax.set_ylabel(r"$L_y$ (µm)", fontsize=6.5, labelpad=2)
        else:
            ax.set_yticklabels([])
        ax.tick_params(labelsize=5.8)

        ax.set_title(p["title"], fontsize=7.0, pad=3)

        # time annotation: white box inside top-left of image
        note = f"$t = {p['t_phys']:.0f}$ s"
        ax.text(0.03, 0.97, note,
                transform=ax.transAxes,
                ha="left", va="top", fontsize=6.0,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.75))

    fig.savefig(OUT_PDF, bbox_inches="tight")
    print(f"\nSaved → {OUT_PDF}")
    plt.close(fig)


if __name__ == "__main__":
    make_figure()
