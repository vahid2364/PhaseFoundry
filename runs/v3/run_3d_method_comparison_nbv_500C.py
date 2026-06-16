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
import matplotlib.gridspec as gridspec
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


def _save_comparison_figure(
    c_fd: np.ndarray,
    c_jan: np.ndarray,
    t_s: float,
    cfg: dict,
    wall_fd: float,
    wall_jan: float,
    out_dir: Path,
) -> Path:
    sim      = cfg["simulation"]
    N        = sim["N"]
    n_steps  = sim["n_steps"]
    Lx       = sim["Lx"]
    T_C      = sim["temperature_C"]
    elements = cfg["system"]["elements"]
    ref_el   = cfg["system"]["reference_element"]
    ind_el   = [e for e in elements if e != ref_el][0]
    label    = f"x_{{{ref_el.capitalize()}}}"   # e.g. x_V

    sl_fd  = _midplane_slice(c_fd)
    sl_jan = _midplane_slice(c_jan)
    sl_dif = np.abs(sl_fd - sl_jan)

    rmse = float(np.sqrt(np.mean(sl_dif**2)))
    maxd = float(sl_dif.max())

    vmin = min(sl_fd.min(), sl_jan.min())
    vmax = max(sl_fd.max(), sl_jan.max())
    extent = [0, Lx * 1e6, 0, Lx * 1e6]   # µm

    stats = [
        ["", "fd", "jansson"],
        [f"${label}$ min",  f"{sl_fd.min():.4f}",  f"{sl_jan.min():.4f}"],
        [f"${label}$ mean", f"{sl_fd.mean():.4f}", f"{sl_jan.mean():.4f}"],
        [f"${label}$ max",  f"{sl_fd.max():.4f}",  f"{sl_jan.max():.4f}"],
        ["RMSE",    f"{rmse:.2e}", "—"],
        ["max |Δ|", f"{maxd:.2e}", "—"],
        ["Wall (s)", f"{wall_fd:.0f}", f"{wall_jan:.0f}"],
    ]

    suptitle = (
        f"3D {ind_el.capitalize()}$_{{50}}${ref_el.capitalize()}$_{{50}}$ "
        f"BCC$_{{\\rm A2}}$ — {T_C} °C — "
        f"$t = {t_s:.0f}$ s  ({n_steps:,} steps, {N}³ grid, $L = 1$ µm)"
    )

    def _make_fig():
        fig = plt.figure(figsize=(14, 5.5))
        gs  = gridspec.GridSpec(1, 4, figure=fig,
                                width_ratios=[1, 1, 1, 0.55], wspace=0.05)
        ax_fd  = fig.add_subplot(gs[0, 0])
        ax_jan = fig.add_subplot(gs[0, 1])
        ax_dif = fig.add_subplot(gs[0, 2])
        ax_tab = fig.add_subplot(gs[0, 3])

        im = ax_fd.imshow(sl_fd.T, origin="lower", cmap="RdBu_r",
                          vmin=vmin, vmax=vmax, extent=extent, aspect="equal")
        ax_fd.set_title(r"(a) \texttt{fd}", fontsize=11)
        ax_fd.set_xlabel(r"$x\;(\mu\mathrm{m})$", fontsize=10)
        ax_fd.set_ylabel(r"$y\;(\mu\mathrm{m})$", fontsize=10)
        plt.colorbar(im, ax=ax_fd, fraction=0.046, pad=0.04).set_label(
            f"${label}$", fontsize=10)

        im2 = ax_jan.imshow(sl_jan.T, origin="lower", cmap="RdBu_r",
                            vmin=vmin, vmax=vmax, extent=extent, aspect="equal")
        ax_jan.set_title(r"(b) \texttt{jansson}", fontsize=11)
        ax_jan.set_xlabel(r"$x\;(\mu\mathrm{m})$", fontsize=10)
        ax_jan.tick_params(labelleft=False)
        plt.colorbar(im2, ax=ax_jan, fraction=0.046, pad=0.04).set_label(
            f"${label}$", fontsize=10)

        im3 = ax_dif.imshow(sl_dif.T, origin="lower", cmap="Reds",
                            vmin=0, vmax=maxd, extent=extent, aspect="equal")
        ax_dif.set_title(
            r"(c) $|x_\mathrm{V}^{\rm fd} - x_\mathrm{V}^{\rm jansson}|$",
            fontsize=11)
        ax_dif.set_xlabel(r"$x\;(\mu\mathrm{m})$", fontsize=10)
        ax_dif.tick_params(labelleft=False)
        plt.colorbar(im3, ax=ax_dif, fraction=0.046, pad=0.04).set_label(
            "Abs. difference", fontsize=10)

        ax_tab.axis("off")
        tbl = ax_tab.table(cellText=stats[1:], colLabels=stats[0],
                           loc="center", cellLoc="center")
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(9)
        tbl.scale(1.0, 1.5)
        ax_tab.set_title("Statistics\n(mid-plane slice)", fontsize=10, pad=4)

        fig.suptitle(suptitle, fontsize=12, y=1.01)
        fig.tight_layout()
        return fig

    out_png = out_dir / "microstructure_methods_3d_slices.png"
    fig = _make_fig()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_png.name}")

    out_pdf = out_dir / "microstructure_methods_3d.pdf"
    fig = _make_fig()
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_pdf.name}")
    print(f"  RMSE (mid-plane) = {rmse:.2e}   max |Δ| = {maxd:.2e}")
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
    dst = PAPER_FIGS / "microstructure_methods_3d.pdf"
    dst.write_bytes(out_pdf.read_bytes())
    print(f"  Copied to paper/figs/{dst.name}")

    print("\n" + "=" * 60)
    print(f"  Results → {base_dir}")
    print(f"  Paper fig → {dst}")
    print("=" * 60)


# ── entry point ───────────────────────────────────────────────────────────────

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
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg = apply_overrides(cfg, args.set or [])
    cfg = resolve_paths(cfg, _ROOT)

    print("── 3D Derivative-method comparison ──")
    print(f"   System : {cfg['system']['name'].upper()}")
    print(summary(cfg))

    run(cfg, use_gpu=args.gpu, adaptive_dt=args.adaptive)
