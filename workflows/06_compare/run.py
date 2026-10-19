"""
Step 6 — Chemical vs. Elasto-Chemical Comparison Figure
========================================================
Produces a 2×N panel figure comparing final-frame composition fields from
chemical-only and elasto-chemical simulations at matched compositions.

Layout (landscape, paper-ready):
    Row 0 — Chemical only:     c50  c60  c70  c80
    Row 1 — Elasto-chemical:   c50  c60  c70  c80

Colormap: RdBu_r  (blue = V-rich, red = Nb-rich)
Colorbar: shared per row, labelled x_NB.

Output
------
<results_dir>/500C/comparison/
    chem_vs_elastic_final.pdf   — vector, for Overleaf
    chem_vs_elastic_final.png   — raster, 300 dpi

Run standalone
--------------
    python workflows/06_compare/run.py --config configs/nb_v_chem_c60.yaml

    # Override snapshot index (0 = initial, -1 = final)
    python workflows/06_compare/run.py --config configs/nb_v_chem_c60.yaml --snap -1
"""

from __future__ import annotations
import argparse, sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from phasefoundry.config import load_config, apply_overrides, resolve_paths


def _run_dir(results_dir: Path, T_C: int, run_name: str) -> Path:
    return results_dir / f"{T_C}C" / "kinetics" / run_name


def _load_snap(run_dir: Path, snap_idx: int = -1) -> tuple[np.ndarray, float] | None:
    """Return (field_2d, time_s) for the requested snapshot, or None if missing."""
    npz = run_dir / "fields.npz"
    if not npz.exists():
        return None
    d    = np.load(npz)
    key  = [k for k in d.keys() if k.startswith("x_") and not k.endswith("_final")][0]
    arr  = d[key]          # (n_snaps, N, N)
    t    = d["time"]       # (n_snaps,)
    idx  = snap_idx % len(arr)
    return arr[idx], float(t[idx])


def _sci(val: float) -> str:
    """1e-12 → '$10^{-12}$'  for matplotlib math text."""
    from math import log10, floor
    exp = int(floor(log10(abs(val))))
    return f"$10^{{{exp}}}$"


def make_comparison_figure(
    results_dir: Path,
    T_C:         int,
    compositions: list[float],
    snap_idx:    int = -1,
    Lx_um:       float = 1.0,
    kappa:       float = 1e-12,
    M:           float = 5e-21,
    tdb_tag:     str   = "tdbv2",
    out_dir:     Path | None = None,
) -> Path:
    """
    Build the comparison figure.

    Parameters
    ----------
    results_dir   Root results directory (e.g. results/HfNbTiV_BCC_PhaseForge_v2/NbV)
    T_C           Simulation temperature in °C
    compositions  List of x_NB values, e.g. [0.5, 0.6, 0.7, 0.8]
    snap_idx      Snapshot index (-1 = final frame)
    Lx_um         Domain size in µm (for axis labels)
    kappa, M      Simulation parameters (for run_name reconstruction)
    tdb_tag       TDB version tag used in run_name
    out_dir       Output directory (defaults to results_dir/T_CC/comparison/)
    """
    from phasefoundry.config import _sci_tag   # reuse compact tag helper

    def _run_name(c0: float, elastic: bool) -> str:
        c_pct   = round(c0 * 100)
        k_tag   = "kappa" + _sci_tag(kappa)
        m_tag   = "M"     + _sci_tag(M)
        suffix  = "_elastic" if elastic else ""
        return f"pf_2d_c{c_pct:02d}Nb_{k_tag}_{m_tag}_{tdb_tag}{suffix}"

    n_comp   = len(compositions)
    fig_w    = 2.0 * n_comp + 0.6   # ~8.6 in for 4 compositions
    fig_h    = 4.8
    fig, axes = plt.subplots(
        2, n_comp,
        figsize=(fig_w, fig_h),
        constrained_layout=False,
    )
    fig.subplots_adjust(left=0.085, right=0.88, top=0.91, bottom=0.10,
                        wspace=0.08, hspace=0.00)

    row_labels = ["Chemical only", "Elasto-chemical"]
    row_elastic = [False, True]

    # determine global colormap range across all available fields
    all_fields = []
    for elastic in row_elastic:
        for c0 in compositions:
            rdir = _run_dir(results_dir, T_C, _run_name(c0, elastic))
            res  = _load_snap(rdir, snap_idx)
            if res is not None:
                all_fields.append(res[0])

    if not all_fields:
        raise FileNotFoundError(
            f"No fields.npz found under {results_dir}/{T_C}C/kinetics/. "
            "Run step 3 first."
        )

    vmin = min(f.min() for f in all_fields)
    vmax = max(f.max() for f in all_fields)
    # center colormap symmetrically around the midpoint of the range
    vcen = 0.5 * (vmin + vmax)
    vhalf = max(vcen - vmin, vmax - vcen)
    vmin, vmax = vcen - vhalf, vcen + vhalf

    extent = [0, Lx_um * 1e3, 0, Lx_um * 1e3]   # nm

    for row, (elastic, row_label) in enumerate(zip(row_elastic, row_labels)):
        for col, c0 in enumerate(compositions):
            ax   = axes[row, col]
            rdir = _run_dir(results_dir, T_C, _run_name(c0, elastic))
            res  = _load_snap(rdir, snap_idx)

            if res is None:
                ax.text(0.5, 0.5, "no data", ha="center", va="center",
                        transform=ax.transAxes, fontsize=8, color="gray")
                ax.set_xticks([]); ax.set_yticks([])
            else:
                field, t_s = res
                im = ax.imshow(
                    field, origin="lower", cmap="RdBu_r",
                    vmin=vmin, vmax=vmax,
                    extent=extent, aspect="equal",
                )
                ax.xaxis.set_major_locator(ticker.MaxNLocator(3, integer=True))
                ax.yaxis.set_major_locator(ticker.MaxNLocator(3, integer=True))

                # time stamp top-right
                t_label = f"t = {t_s:.0f} s" if t_s < 3600 else f"t = {t_s/3600:.1f} h"
                ax.text(0.97, 0.97, t_label, transform=ax.transAxes,
                        ha="right", va="top", fontsize=6.5,
                        color="white",
                        bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.45, ec="none"))

            # column header (top row only)
            if row == 0:
                ax.set_title(f"$x_{{\\rm Nb}}^0 = {c0:.2f}$", fontsize=9, pad=4)

            # x-axis label (bottom row only)
            if row == 1:
                ax.set_xlabel("nm", fontsize=8)
            else:
                ax.set_xticklabels([])

            # y-axis label (leftmost column only)
            if col == 0:
                ax.set_ylabel("nm", fontsize=8)
            else:
                ax.set_yticklabels([])

            ax.tick_params(labelsize=7)

        # shared colorbar on the right of each row
        # each row occupies ~0.405 of figure height; colorbar is 80% of that
        # row 0 is the top row, as in the axes grid
        row_bottom = 0.10 + (1 - row) * 0.405
        row_height = 0.405
        cb_height  = row_height * 0.80
        cb_bottom  = row_bottom + (row_height - cb_height) / 2
        cbar_ax = fig.add_axes([0.895, cb_bottom, 0.018, cb_height])
        sm = plt.cm.ScalarMappable(cmap="RdBu_r",
                                    norm=plt.Normalize(vmin=vmin, vmax=vmax))
        sm.set_array([])
        cb = fig.colorbar(sm, cax=cbar_ax)
        cb.set_label("$x_{\\rm Nb}$", fontsize=8)
        cb.ax.tick_params(labelsize=7)

        # row label on the far left
        fig.text(0.005, row_bottom + row_height / 2, row_label,
                 va="center", ha="left", fontsize=8.5, fontweight="bold",
                 rotation=90)

    from math import log10, floor
    k_exp = int(floor(log10(abs(kappa))))
    M_exp = int(floor(log10(abs(M))))
    fig.suptitle(
        f"Nb–V  |  {T_C} °C  |  "
        r"$\kappa=10^{" + str(k_exp) + r"}$ J$\cdot$m$^2$/mol  "
        r"$M=10^{" + str(M_exp) + r"}$ m$^5$/(J$\cdot$s)",
        fontsize=9, y=0.97,
    )

    if out_dir is None:
        out_dir = results_dir / f"{T_C}C" / "comparison"
    out_dir.mkdir(parents=True, exist_ok=True)

    snap_tag = "final" if snap_idx == -1 else f"snap{snap_idx:03d}"
    for ext in ("pdf", "png"):
        p = out_dir / f"chem_vs_elastic_{snap_tag}.{ext}"
        fig.savefig(p, dpi=300, bbox_inches="tight")
        print(f"  Saved: {p}")
    plt.close(fig)
    return out_dir / f"chem_vs_elastic_{snap_tag}.pdf"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Step 6 — Chemical vs. elasto-chemical comparison figure")
    parser.add_argument("--config", required=True)
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    parser.add_argument("--snap", type=int, default=-1,
                        help="Snapshot index to plot (-1 = final, default)")
    parser.add_argument("--compositions", nargs="+", type=float,
                        default=[0.50, 0.60, 0.70, 0.80],
                        help="x_NB values to compare (default: 0.5 0.6 0.7 0.8)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg = apply_overrides(cfg, args.set)
    cfg = resolve_paths(cfg, _ROOT)

    sim         = cfg["simulation"]
    results_dir = Path(cfg["output"]["results_dir"])
    T_C         = int(sim["temperature_C"])
    kappa       = sim["kappa"] if not isinstance(sim["kappa"], list) else sim["kappa"][0]
    M_cfg       = sim["M"]
    M           = M_cfg if not isinstance(M_cfg, list) else (
                      M_cfg[0][0] if isinstance(M_cfg[0], list) else M_cfg[0])

    import re
    db_path = str(cfg["system"].get("db_path", ""))
    m = re.search(r"_v(\d+)", Path(db_path).stem, re.IGNORECASE)
    tdb_tag = f"tdbv{m.group(1)}" if m else "tdb"

    print(f"── Step 6: Chem vs. elastic comparison ──")
    print(f"   Compositions: {args.compositions}")
    print(f"   Snapshot:     {args.snap} ({'final' if args.snap == -1 else args.snap})")
    print(f"   Results dir:  {results_dir}")

    make_comparison_figure(
        results_dir  = results_dir,
        T_C          = T_C,
        compositions = args.compositions,
        snap_idx     = args.snap,
        Lx_um        = float(sim["Lx"]) * 1e6,
        kappa        = float(kappa),
        M            = float(M),
        tdb_tag      = tdb_tag,
    )
    print("Done.")
