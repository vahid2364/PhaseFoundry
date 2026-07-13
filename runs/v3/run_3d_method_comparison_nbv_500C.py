"""
3D derivative-method comparison — any binary system
=====================================================
Runs two identical 3D Cahn-Hilliard simulations that differ only in the
derivative method used to evaluate chemical-potential driving forces:

  * fd      — finite differences on the G_m spline
  * jansson — direct CALPHAD interpolation (MU / dMU datasets)

All parameters are read from a YAML config file (same format as the main
pipeline).  The simulation section's ``dimensions`` key is overridden to 3
automatically; use ``--set simulation.N=64`` to control grid size.

Produces:
  <results_dir>/<T>C/kinetics/pf_3d_method_comparison/
    fd/      pyvista_surface_*.png
    jansson/ pyvista_surface_*.png
    microstructure_methods_3d.pdf    (paper figure, mid-plane slices)

The PDF is also copied to paper/figs/ for direct \\includegraphics inclusion.

Usage
-----
    python runs/v3/run_3d_method_comparison_nbv_500C.py --config configs/nb_v_2d.yaml
    python runs/v3/run_3d_method_comparison_nbv_500C.py \\
        --config configs/nb_v_2d.yaml --set simulation.N=128 simulation.n_steps=5000
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from calphad_pf.config                     import load_config, apply_overrides, resolve_paths, summary
from calphad_pf.thermodynamics.storage     import load_table
from calphad_pf.thermodynamics.interpolant import build_interpolant
from calphad_pf.phasefield.solver          import solve_cahn_hilliard, stable_dt
from calphad_pf.phasefield.visualize       import save_pyvista_3d

PAPER_FIGS = _ROOT / "paper" / "2025_Kunselman_Design_of_Spinodal_Microstructures_in_HEAs" / "figs"


# ── simulation helper ─────────────────────────────────────────────────────────

def _run_sim(cfg: dict, interp, use_gpu: bool, adaptive_dt: bool):
    sim = cfg["simulation"]
    dt  = sim.get("dt", None)
    if dt in (None, "None", "none"):
        dt = None
    return solve_cahn_hilliard(
        interpolant = interp,
        M           = sim["M"],
        kappa       = sim["kappa"],
        c0          = sim["c0"],
        N           = sim["N"],
        Lx          = sim["Lx"],
        noise       = sim["noise"],
        seed        = sim["seed"],
        dt          = dt,
        n_steps     = sim["n_steps"],
        save_every  = sim["save_every"],
        output_dir  = None,
        save_frames = False,
        ndim        = 3,           # always 3D for this script
        use_gpu     = use_gpu,
        adaptive_dt = adaptive_dt,
    )


# ── figure assembly ───────────────────────────────────────────────────────────

def _midplane_slice(c3d: np.ndarray) -> np.ndarray:
    return c3d[:, :, c3d.shape[2] // 2]


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

METHOD_LABEL_3D = {
    "fd":      "Finite difference (FD)",
    "jansson": "Jansson (direct)",
}


def _add_cube_inset(fig: plt.Figure, ax_pos: list[float], slice_frac: float = 0.5):
    """Add a schematic 3D wireframe cube inset with the mid-plane highlighted.

    ax_pos : [left, bottom, width, height] of the *parent* panel in figure coords.
    slice_frac : fractional z-position of the highlighted plane (0–1).
    """
    from matplotlib.patches import FancyBboxPatch

    # 3D axes need extra room for internal whitespace — the cube itself
    # appears in roughly the central 60 % of the axes bounding box.
    S   = ax_pos[2] * 0.22          # inset axes width  (22 % of panel width)
    SH  = ax_pos[3] * 0.24          # inset axes height (24 % of panel height)
    PAD = ax_pos[2] * 0.010

    inset_pos = [ax_pos[0] + PAD, ax_pos[1] + PAD, S, SH]

    ax3 = fig.add_axes(inset_pos, projection="3d")
    # Semi-transparent white background for the whole inset axes area
    ax3.set_facecolor("white")
    ax3.patch.set_alpha(0.85)

    # Wireframe cube edges
    edges = [
        ([0,0,0],[1,0,0]), ([1,0,0],[1,1,0]), ([1,1,0],[0,1,0]), ([0,1,0],[0,0,0]),
        ([0,0,1],[1,0,1]), ([1,0,1],[1,1,1]), ([1,1,1],[0,1,1]), ([0,1,1],[0,0,1]),
        ([0,0,0],[0,0,1]), ([1,0,0],[1,0,1]), ([1,1,0],[1,1,1]), ([0,1,0],[0,1,1]),
    ]
    for s, e in edges:
        ax3.plot3D(*zip(s, e), color="0.55", lw=0.7, zorder=2)

    # Highlighted mid-plane as a filled semi-transparent quad
    xx, yy = np.meshgrid([0, 1], [0, 1])
    zz     = np.full_like(xx, slice_frac, dtype=float)
    ax3.plot_surface(xx, yy, zz, alpha=0.40, color="#4393C3",
                     linewidth=0, antialiased=False, zorder=3)
    sq = [[0,0],[1,0],[1,1],[0,1],[0,0]]
    ax3.plot([p[0] for p in sq], [p[1] for p in sq],
             [slice_frac]*5, color="#2166AC", lw=1.2, zorder=4)

    ax3.set_xlim(0, 1); ax3.set_ylim(0, 1); ax3.set_zlim(0, 1)
    ax3.set_axis_off()
    ax3.view_init(elev=22, azim=-55)
    for pane in (ax3.xaxis.pane, ax3.yaxis.pane, ax3.zaxis.pane):
        pane.fill = False
        pane.set_edgecolor("none")

    # Axis orientation labels
    ax3.text(0.5, -0.12, 0,   "x", fontsize=5.5, color="0.35",
             ha="center", va="top",    zorder=5)
    ax3.text(1.12, 0.5,  0,   "y", fontsize=5.5, color="0.35",
             ha="left",   va="center", zorder=5)
    ax3.text(1.08, 1.0,  0.5, "z", fontsize=5.5, color="0.35",
             ha="left",   va="center", zorder=5)

    return ax3


def _save_comparison_figure(
    c_fd: np.ndarray,
    c_jan: np.ndarray,
    t_s: float,
    cfg: dict,
    wall_fd: float,
    wall_jan: float,
    out_dir: Path,
) -> Path:
    """Save a single composite PDF matching the 2D microstructure_methods style."""
    sim      = cfg["simulation"]
    Lx       = sim["Lx"]
    ref_el   = cfg["system"]["reference_element"]

    # mid-plane x_V fields  (c arrays store x_NB, the independent component)
    xv_fd  = 1.0 - _midplane_slice(c_fd)
    xv_jan = 1.0 - _midplane_slice(c_jan)
    diff   = np.abs(xv_fd - xv_jan)

    all_vals = np.concatenate([xv_fd.ravel(), xv_jan.ravel()])
    vlo, vhi = np.percentile(all_vals, 1), np.percentile(all_vals, 99)
    dmax     = diff.max()
    rmse     = float(np.sqrt(np.mean(diff**2)))

    Lx_um  = Lx * 1e6
    extent = [0, Lx_um, 0, Lx_um]

    # ── figure geometry matching the 2D figure exactly ────────────────────────
    FIG_W = 7.2
    L, R, BOT, TOP = 0.07, 0.96, 0.26, 0.99

    GAP_pre  = 9 / (FIG_W * 25.4)
    CB_W_pre = 0.014 * 0.60 * 0.70
    CB_PAD   = -0.002
    IM_W_pre = ((R - L) - 2*GAP_pre - 3*(CB_W_pre + CB_PAD)) / 3
    IM_W_in  = IM_W_pre * FIG_W
    FIG_H    = IM_W_in / (TOP - BOT)
    IM_H     = TOP - BOT
    GAP      = GAP_pre
    CB_W     = CB_W_pre
    IM_W     = IM_W_pre

    CB_H    = IM_H * 0.60
    CB_VOFF = (IM_H - CB_H) / 2

    x0 = L
    x1 = x0 + IM_W + CB_W + CB_PAD + GAP
    x2 = x1 + IM_W + CB_W + CB_PAD + GAP

    fig    = plt.figure(figsize=(FIG_W, FIG_H))
    ax_fd  = fig.add_axes([x0, BOT, IM_W, IM_H])
    cax0   = fig.add_axes([x0 + IM_W + CB_PAD, BOT + CB_VOFF, CB_W, CB_H])
    ax_jan = fig.add_axes([x1, BOT, IM_W, IM_H])
    cax1   = fig.add_axes([x1 + IM_W + CB_PAD, BOT + CB_VOFF, CB_W, CB_H])
    ax_dif = fig.add_axes([x2, BOT, IM_W, IM_H])
    cax2   = fig.add_axes([x2 + IM_W + CB_PAD, BOT + CB_VOFF, CB_W, CB_H])

    TBL_GAP = 0.028 + 7 / (FIG_H * 25.4)
    TBL_H   = 0.09
    TBL_W   = (R - L) * 0.93
    TBL_L   = L + (R - L - TBL_W) / 2
    ax_tbl  = fig.add_axes([TBL_L, BOT - TBL_GAP - TBL_H, TBL_W, TBL_H])

    cmap_comp = "coolwarm"
    cmap_diff = "Oranges"

    # ── panels (a) FD and (b) Jansson ─────────────────────────────────────────
    fields_dict = {"fd": xv_fd, "jansson": xv_jan}
    axes_dict   = {"fd": (ax_fd, cax0, "(a)"), "jansson": (ax_jan, cax1, "(b)")}
    im_srcs     = {}
    for m, (ax, cax, plabel) in axes_dict.items():
        im = ax.imshow(
            fields_dict[m], origin="lower", extent=extent,
            cmap=cmap_comp, vmin=vlo, vmax=vhi,
            interpolation="antialiased", aspect="auto",
        )
        im_srcs[m] = im
        ax.set_xlabel("$L_x$ (µm)", labelpad=2, fontsize=6.3)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(0.25))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(0.25))
        ax.tick_params(axis="both", labelsize=5.6)
        if ax is ax_fd:
            ax.set_ylabel("$L_y$ (µm)", labelpad=2, fontsize=6.3)
        else:
            ax.tick_params(labelleft=False)
        ax.text(0.97, 0.04, plabel,
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=9, fontweight="bold", color="white")
        ax.text(0.03, 0.97, METHOD_LABEL_3D[m],
                transform=ax.transAxes, ha="left", va="top",
                fontsize=7.5, color="white",
                bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.45, lw=0))

    # ── 3D cube insets for panels (a) and (b) ────────────────────────────────
    slice_frac = 0.5   # mid-plane
    _add_cube_inset(fig, [x0, BOT, IM_W, IM_H], slice_frac)
    _add_cube_inset(fig, [x1, BOT, IM_W, IM_H], slice_frac)

    # colorbars (a) and (b) — shared scale
    t_lo  = round(vlo + (vhi - vlo) * 0.02, 2)
    t_mid = round((vlo + vhi) / 2,           2)
    t_hi  = round(vhi - (vhi - vlo) * 0.02, 2)
    ax_fd.spines["right"].set_visible(False)
    ax_jan.spines["right"].set_visible(False)
    for cax, m in [(cax0, "fd"), (cax1, "jansson")]:
        cb = fig.colorbar(im_srcs[m], cax=cax, ticks=[t_lo, t_mid, t_hi])
        cb.set_label(f"$x_{{\\mathrm{{{ref_el.capitalize()}}}}}$",
                     fontsize=7, labelpad=2)
        cb.ax.tick_params(labelsize=6, pad=1, length=2)
        cb.ax.set_yticklabels([f"{t_lo:.2f}", f"{t_mid:.2f}", f"{t_hi:.2f}"])
        cb.ax.spines["left"].set_visible(False)

    # ── panel (c): difference map ──────────────────────────────────────────────
    vmax_diff = max(dmax, 1e-10)
    im_diff   = ax_dif.imshow(
        diff, origin="lower", extent=extent,
        cmap=cmap_diff, vmin=0, vmax=vmax_diff,
        interpolation="antialiased", aspect="auto",
    )
    ax_dif.set_xlabel("$L_x$ (µm)", labelpad=2, fontsize=6.3)
    ax_dif.tick_params(axis="both", labelsize=5.6, labelleft=False)
    ax_dif.xaxis.set_major_locator(ticker.MultipleLocator(0.25))
    ax_dif.yaxis.set_major_locator(ticker.MultipleLocator(0.25))
    ax_dif.text(0.97, 0.04, "(c)",
                transform=ax_dif.transAxes, ha="right", va="bottom",
                fontsize=9, fontweight="bold")
    ax_dif.text(0.03, 0.97, "$|$FD $-$ Jansson$|$",
                transform=ax_dif.transAxes, ha="left", va="top",
                fontsize=7.5,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.6, lw=0))
    ax_dif.text(0.03, 0.04,
                f"max $|\\Delta|$ = {dmax:.2e}\nRMSE = {rmse:.2e}",
                transform=ax_dif.transAxes, ha="left", va="bottom",
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
    ax_dif.spines["right"].set_visible(False)
    cb2 = fig.colorbar(im_diff, cax=cax2, ticks=cb2_ticks)
    cb2.set_label(f"$|\\Delta x_{{\\mathrm{{{ref_el.capitalize()}}}}}|$ {cb2_unit}",
                  fontsize=6, labelpad=2)
    cb2.ax.tick_params(labelsize=6, pad=1, length=2)
    cb2.ax.set_yticklabels(cb2_labels)
    cb2.ax.spines["left"].set_visible(False)

    # ── statistics table ──────────────────────────────────────────────────────
    ax_tbl.axis("off")
    ref_cap = ref_el.capitalize()
    col_labels = [
        "Method",
        f"min $x_{{{ref_cap}}}$",
        f"max $x_{{{ref_cap}}}$",
        "RMSE vs Jansson",
        "Simulation time (s)",
    ]
    tbl_rows = []
    for m, xv, wall in [("FD", xv_fd, wall_fd), ("Jansson", xv_jan, wall_jan)]:
        rmse_val = (float(np.sqrt(np.mean((xv_fd - xv_jan)**2)))
                    if m == "FD" else None)
        rmse_str = f"{rmse_val:.2e}" if rmse_val is not None else "—"
        tbl_rows.append([m, f"{xv.min():.4f}", f"{xv.max():.4f}",
                         rmse_str, f"{wall:.0f}"])
    ROW_H  = 0.45
    n_rows = len(tbl_rows) + 1
    tbl_h  = n_rows * ROW_H
    tbl = ax_tbl.table(
        cellText=tbl_rows, colLabels=col_labels,
        loc="upper center", cellLoc="center",
        bbox=[0.0, 1.0 - tbl_h, 1.0, tbl_h],
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

    # ── save single composite PDF ─────────────────────────────────────────────
    out_pdf = out_dir / "microstructure_methods_3d.pdf"
    fig.savefig(out_pdf, dpi=300)
    plt.close(fig)
    print(f"  Saved: {out_pdf.name}")
    print(f"  RMSE (mid-plane) = {rmse:.2e}   max |Δ| = {dmax:.2e}")
    return out_pdf


# ── main ──────────────────────────────────────────────────────────────────────

def run(cfg: dict, use_gpu: bool = False, adaptive_dt: bool = False) -> None:
    sim         = cfg["simulation"]
    results_dir = Path(cfg["output"]["results_dir"])
    T_C         = sim["temperature_C"]
    elements    = cfg["system"]["elements"]
    ref_el      = cfg["system"]["reference_element"]

    # force 3D
    cfg["simulation"]["dimensions"] = 3

    base_dir = results_dir / f"{T_C}C" / "kinetics" / "pf_3d_method_comparison"
    dir_fd   = base_dir / "fd"
    dir_jan  = base_dir / "jansson"
    for d in (base_dir, dir_fd, dir_jan):
        d.mkdir(parents=True, exist_ok=True)

    # ── load thermodynamic table ──────────────────────────────────────────────
    data_dir = Path(cfg["output"]["data_dir"])
    name     = cfg["system"]["name"]
    tables   = sorted(data_dir.glob(f"{name}_*C.h5"))
    if not tables:
        raise FileNotFoundError(
            f"No HDF5 tables matching '{name}_*C.h5' in {data_dir}. "
            "Run step 1 first.")
    T_target = T_C + 273.15
    best = min(tables, key=lambda p: abs(
        float(load_table(p)["metadata"]["temperature"]) - T_target))
    T_K = float(load_table(best)["metadata"]["temperature"])
    print(f"Table     : {best.name}  ({T_K - 273.15:.0f} °C)")

    # ── build interpolants ────────────────────────────────────────────────────
    print("Building fd interpolant ...")
    interp_fd  = build_interpolant(
        [best], elements=elements, reference_element=ref_el,
        temperature_K=T_K, derivative_method="fd")
    print("Building jansson interpolant ...")
    interp_jan = build_interpolant(
        [best], elements=elements, reference_element=ref_el,
        temperature_K=T_K, derivative_method="jansson")

    # ── run fd ────────────────────────────────────────────────────────────────
    print("\n" + "─" * 50)
    print("Running fd simulation ...")
    t0 = time.time()
    res_fd = _run_sim(cfg, interp_fd, use_gpu, adaptive_dt)
    wall_fd = time.time() - t0
    print(f"  Wall time: {wall_fd:.1f} s")

    # ── run jansson ───────────────────────────────────────────────────────────
    print("─" * 50)
    print("Running jansson simulation ...")
    t0 = time.time()
    res_jan = _run_sim(cfg, interp_jan, use_gpu, adaptive_dt)
    wall_jan = time.time() - t0
    print(f"  Wall time: {wall_jan:.1f} s")

    t_s   = float(res_fd.time_history[-1])
    c_fd  = res_fd.compositions[0]
    c_jan = res_jan.compositions[0]

    ind_el = [e for e in elements if e != ref_el][0]

    # ── save raw composition arrays ───────────────────────────────────────────
    np.savez_compressed(
        base_dir / "compositions_3d.npz",
        c_fd=c_fd, c_jan=c_jan,
        t_s=t_s, Lx=sim["Lx"], T_C=sim["temperature_C"],
        N=sim["N"], n_steps=sim["n_steps"],
    )
    print(f"  Raw data saved → {base_dir / 'compositions_3d.npz'}")

    # ── 3D PyVista renders ────────────────────────────────────────────────────
    print("\n── 3D PyVista renders ───────────────────────────────────────────────")
    for p in save_pyvista_3d(c_fd, t_s, dir_fd,
                              label=f"x_{ind_el.lower()}",
                              bar_title=f"{ind_el.capitalize()} Composition",
                              title=f"{name.upper()} {T_C}C — fd"):
        print(f"  Saved: fd/{p.name}")

    for p in save_pyvista_3d(c_jan, t_s, dir_jan,
                              label=f"x_{ind_el.lower()}",
                              bar_title=f"{ind_el.capitalize()} Composition",
                              title=f"{name.upper()} {T_C}C — jansson"):
        print(f"  Saved: jansson/{p.name}")

    # ── paper comparison figure ───────────────────────────────────────────────
    print("\n── Comparison figure ────────────────────────────────────────────────")
    out_pdf = _save_comparison_figure(
        c_fd, c_jan, t_s, cfg, wall_fd, wall_jan, base_dir)

    PAPER_FIGS.mkdir(parents=True, exist_ok=True)
    dst = PAPER_FIGS / out_pdf.name
    dst.write_bytes(out_pdf.read_bytes())
    print(f"  Copied to paper/figs/{dst.name}")

    print("\n" + "=" * 60)
    print(f"  Results → {base_dir}")
    print(f"  Paper fig → {dst}")
    print("=" * 60)


# ── entry point ───────────────────────────────────────────────────────────────

def _run_from_npz(npz_path: Path, cfg: dict) -> None:
    """Regenerate the comparison figure from saved raw data without re-running."""
    d      = np.load(npz_path)
    c_fd   = d["c_fd"]
    c_jan  = d["c_jan"]
    t_s    = float(d["t_s"])
    out_dir = npz_path.parent

    print(f"Loaded raw data from {npz_path}")
    print(f"  c_fd   shape={c_fd.shape}  range=[{c_fd.min():.4f}, {c_fd.max():.4f}]")
    print(f"  c_jan  shape={c_jan.shape}  range=[{c_jan.min():.4f}, {c_jan.max():.4f}]")
    print(f"  t_s    = {t_s:.1f} s")

    out_pdf = _save_comparison_figure(
        c_fd, c_jan, t_s, cfg,
        wall_fd=0.0, wall_jan=0.0,
        out_dir=out_dir,
    )
    PAPER_FIGS.mkdir(parents=True, exist_ok=True)
    dst = PAPER_FIGS / out_pdf.name
    dst.write_bytes(out_pdf.read_bytes())
    print(f"  Copied to paper/figs/{dst.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="3D derivative-method comparison — any binary system"
    )
    parser.add_argument("--config", default="configs/nb_v_2d.yaml",
                        help="Path to YAML config file")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                        help="Override config values, e.g. --set simulation.N=128")
    parser.add_argument("--gpu", action="store_true", default=False,
                        help="Use CuPy GPU-accelerated FFT")
    parser.add_argument("--adaptive", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="Adaptive time-stepping (default: enabled)")
    parser.add_argument("--from-npz", default=None, metavar="NPZ",
                        help="Regenerate figure from saved .npz without re-running")
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg = apply_overrides(cfg, args.set or [])
    cfg = resolve_paths(cfg, _ROOT)

    if args.from_npz:
        _run_from_npz(Path(args.from_npz), cfg)
    else:
        print("── 3D Derivative-method comparison ──")
        print(f"   System : {cfg['system']['name'].upper()}")
        print(summary(cfg))
        run(cfg, use_gpu=args.gpu, adaptive_dt=args.adaptive)
