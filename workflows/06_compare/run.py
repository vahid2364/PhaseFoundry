"""Pipeline step 6: chemical against elasto-chemical composition fields for Nb-V.

Reads the step-3 ``fields.npz`` of the chemical and the elasto-chemical run at each
composition from ``<output.results_dir>/<temperature_C>C/kinetics/``, rebuilding the run
names from the config's kappa, M and TDB version. It writes a two-row figure (chemical on
top, elasto-chemical below, one column per composition, x_Nb on a shared colour scale) to
``<output.results_dir>/<temperature_C>C/comparison/chem_vs_elastic_<snapshot>.pdf`` and
``.png``, where ``<snapshot>`` is ``final`` for the last snapshot and ``snapNNN`` otherwise.

```bash
python workflows/06_compare/run.py --config configs/nb_v_chem_c60.yaml
python workflows/06_compare/run.py --config configs/nb_v_chem_c60.yaml --snap 0
python workflows/06_compare/run.py --config configs/nb_v_chem_c60.yaml --compositions 0.5 0.6
```
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker

from phasefoundry.config import apply_overrides, load_config, resolve_paths

mpl.use("Agg")
_ROOT = Path(__file__).resolve().parents[2]


def _run_dir(results_dir: Path, T_C: int, run_name: str) -> Path:
    """Return the step-3 output directory of one run."""
    return results_dir / f"{T_C}C" / "kinetics" / run_name


def _load_snap(run_dir: Path, snap_idx: int = -1) -> tuple[np.ndarray, float] | None:
    """Load one snapshot of the first composition field in ``run_dir/fields.npz``.

    Args:
        run_dir: Step-3 run directory.
        snap_idx: Snapshot index; negative values count from the end.

    Returns:
        The 2-D field and its time in s, or ``None`` if ``fields.npz`` is missing.
    """
    npz = run_dir / "fields.npz"
    if not npz.exists():
        return None
    d = np.load(npz)
    key = [k for k in d if k.startswith("x_") and not k.endswith("_final")][0]
    arr = d[key]
    t = d["time"]
    idx = snap_idx % len(arr)
    return arr[idx], float(t[idx])


def _sci(val: float) -> str:
    """Format ``val`` as a power of ten in matplotlib math text, e.g. ``"$10^{-12}$"``."""
    from math import floor, log10

    exp = int(floor(log10(abs(val))))
    return f"$10^{{{exp}}}$"


def make_comparison_figure(
    results_dir: Path,
    T_C: int,
    compositions: list[float],
    snap_idx: int = -1,
    Lx_um: float = 1.0,
    kappa: float = 1e-12,
    M: float = 5e-21,
    tdb_tag: str = "tdbv2",
    out_dir: Path | None = None,
) -> Path:
    """Build the chemical against elasto-chemical comparison figure.

    Missing runs leave a "no data" panel. The colour scale spans every field that was found.

    Args:
        results_dir: The config's ``output.results_dir``.
        T_C: Simulation temperature in °C.
        compositions: Initial x_Nb of each column, e.g. ``[0.5, 0.6, 0.7, 0.8]``.
        snap_idx: Snapshot index; ``-1`` is the last one.
        Lx_um: Domain size in µm, for the axis extent.
        kappa: Gradient energy coefficient, used to rebuild the run names.
        M: Mobility, used to rebuild the run names.
        tdb_tag: TDB version tag in the run names, e.g. ``"tdbv2"``.
        out_dir: Output directory. Defaults to ``results_dir/<T_C>C/comparison``.

    Returns:
        Path of the saved PDF.

    Raises:
        FileNotFoundError: If none of the runs has a ``fields.npz``.
    """
    from phasefoundry.config import _sci_tag

    def _run_name(c0: float, elastic: bool) -> str:
        """Return the folder name ``config.build_run_name`` gives a 2-D Nb-V run at this composition."""
        c_pct = round(c0 * 100)
        k_tag = "kappa" + _sci_tag(kappa)
        m_tag = "M" + _sci_tag(M)
        suffix = "_elastic" if elastic else ""
        return f"pf_2d_c{c_pct:02d}Nb_{k_tag}_{m_tag}_{tdb_tag}{suffix}"

    n_comp = len(compositions)
    fig_w = 2.0 * n_comp + 0.6
    fig_h = 4.8
    fig, axes = plt.subplots(
        2,
        n_comp,
        figsize=(fig_w, fig_h),
        constrained_layout=False,
    )
    fig.subplots_adjust(left=0.085, right=0.88, top=0.91, bottom=0.10, wspace=0.08, hspace=0.00)

    row_labels = ["Chemical only", "Elasto-chemical"]
    row_elastic = [False, True]

    all_fields = []
    for elastic in row_elastic:
        for c0 in compositions:
            rdir = _run_dir(results_dir, T_C, _run_name(c0, elastic))
            res = _load_snap(rdir, snap_idx)
            if res is not None:
                all_fields.append(res[0])

    if not all_fields:
        raise FileNotFoundError(f"No fields.npz found under {results_dir}/{T_C}C/kinetics/. Run step 3 first.")

    vmin = min(f.min() for f in all_fields)
    vmax = max(f.max() for f in all_fields)
    vcen = 0.5 * (vmin + vmax)
    vhalf = max(vcen - vmin, vmax - vcen)
    vmin, vmax = vcen - vhalf, vcen + vhalf

    extent = [0, Lx_um * 1e3, 0, Lx_um * 1e3]

    for row, (elastic, row_label) in enumerate(zip(row_elastic, row_labels, strict=True)):
        for col, c0 in enumerate(compositions):
            ax = axes[row, col]
            rdir = _run_dir(results_dir, T_C, _run_name(c0, elastic))
            res = _load_snap(rdir, snap_idx)

            if res is None:
                ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes, fontsize=8, color="gray")
                ax.set_xticks([])
                ax.set_yticks([])
            else:
                field, t_s = res
                ax.imshow(
                    field,
                    origin="lower",
                    cmap="RdBu_r",
                    vmin=vmin,
                    vmax=vmax,
                    extent=extent,
                    aspect="equal",
                )
                ax.xaxis.set_major_locator(ticker.MaxNLocator(3, integer=True))
                ax.yaxis.set_major_locator(ticker.MaxNLocator(3, integer=True))

                t_label = f"t = {t_s:.0f} s" if t_s < 3600 else f"t = {t_s / 3600:.1f} h"
                ax.text(
                    0.97,
                    0.97,
                    t_label,
                    transform=ax.transAxes,
                    ha="right",
                    va="top",
                    fontsize=6.5,
                    color="white",
                    bbox={"boxstyle": "round,pad=0.2", "fc": "black", "alpha": 0.45, "ec": "none"},
                )

            if row == 0:
                ax.set_title(f"$x_{{\\rm Nb}}^0 = {c0:.2f}$", fontsize=9, pad=4)

            if row == 1:
                ax.set_xlabel("nm", fontsize=8)
            else:
                ax.set_xticklabels([])

            if col == 0:
                ax.set_ylabel("nm", fontsize=8)
            else:
                ax.set_yticklabels([])

            ax.tick_params(labelsize=7)

        # row 0 is the top axes row, and figure coordinates run bottom-up
        row_bottom = 0.10 + (1 - row) * 0.405
        row_height = 0.405
        cb_height = row_height * 0.80
        cb_bottom = row_bottom + (row_height - cb_height) / 2
        cbar_ax = fig.add_axes([0.895, cb_bottom, 0.018, cb_height])
        sm = plt.cm.ScalarMappable(cmap="RdBu_r", norm=plt.Normalize(vmin=vmin, vmax=vmax))
        sm.set_array([])
        cb = fig.colorbar(sm, cax=cbar_ax)
        cb.set_label("$x_{\\rm Nb}$", fontsize=8)
        cb.ax.tick_params(labelsize=7)

        fig.text(
            0.005, row_bottom + row_height / 2, row_label, va="center", ha="left", fontsize=8.5, fontweight="bold", rotation=90
        )

    from math import floor, log10

    k_exp = int(floor(log10(abs(kappa))))
    M_exp = int(floor(log10(abs(M))))
    fig.suptitle(
        f"Nb–V  |  {T_C} °C  |  "
        r"$\kappa=10^{" + str(k_exp) + r"}$ J$\cdot$m$^2$/mol  "
        r"$M=10^{" + str(M_exp) + r"}$ m$^5$/(J$\cdot$s)",
        fontsize=9,
        y=0.97,
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
    logging.basicConfig(format="%(message)s", stream=sys.stdout)
    logging.getLogger("phasefoundry").setLevel(logging.INFO)
    parser = argparse.ArgumentParser(description="Step 6 — Chemical vs. elasto-chemical comparison figure")
    parser.add_argument("--config", required=True)
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    parser.add_argument("--snap", type=int, default=-1, help="Snapshot index to plot (-1 = final, default)")
    parser.add_argument(
        "--compositions",
        nargs="+",
        type=float,
        default=[0.50, 0.60, 0.70, 0.80],
        help="x_NB values to compare (default: 0.5 0.6 0.7 0.8)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg = apply_overrides(cfg, args.set)
    cfg = resolve_paths(cfg, _ROOT)

    sim = cfg["simulation"]
    results_dir = Path(cfg["output"]["results_dir"])
    T_C = int(sim["temperature_C"])
    kappa = sim["kappa"] if not isinstance(sim["kappa"], list) else sim["kappa"][0]
    M_cfg = sim["M"]
    M = M_cfg if not isinstance(M_cfg, list) else (M_cfg[0][0] if isinstance(M_cfg[0], list) else M_cfg[0])

    import re

    db_path = str(cfg["system"].get("db_path", ""))
    m = re.search(r"_v(\d+)", Path(db_path).stem, re.IGNORECASE)
    tdb_tag = f"tdbv{m.group(1)}" if m else "tdb"

    print("── Step 6: Chem vs. elastic comparison ──")
    print(f"   Compositions: {args.compositions}")
    print(f"   Snapshot:     {args.snap} ({'final' if args.snap == -1 else args.snap})")
    print(f"   Results dir:  {results_dir}")

    make_comparison_figure(
        results_dir=results_dir,
        T_C=T_C,
        compositions=args.compositions,
        snap_idx=args.snap,
        Lx_um=float(sim["Lx"]) * 1e6,
        kappa=float(kappa),
        M=float(M),
        tdb_tag=tdb_tag,
    )
    print("Done.")
