"""
Visualization utilities for Cahn-Hilliard phase-field simulations.

Functions
---------
save_slices_2d      Save three orthogonal mid-plane slices as a matplotlib figure.
save_pyvista_3d     Save PyVista renders: orthogonal slices, smooth outer surface,
                    and isosurface at the mean composition.

PyVista is an optional dependency.  All PyVista functions degrade gracefully
with an informative message when the package is not installed.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ── 2D slice helpers ──────────────────────────────────────────────────────────

def _mid_slices(field_3d: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return XY, XZ, YZ mid-plane slices of a 3-D field."""
    N = field_3d.shape[0]
    mid = N // 2
    return field_3d[:, :, mid], field_3d[:, mid, :], field_3d[mid, :, :]


def save_slices_2d(
    c_3d:    np.ndarray,
    t_s:     float,
    out_dir: Path,
    label:   str  = "x_Nb",
    title:   str  = "",
    cmap:    str  = "RdBu_r",
    dpi:     int  = 150,
) -> Path:
    """
    Save three orthogonal mid-plane slices of a 3D composition field.

    Parameters
    ----------
    c_3d    : np.ndarray  shape (N, N, N)
    t_s     : float       physical time in seconds
    out_dir : Path        output directory (created if absent)
    label   : str         axis / colorbar label
    title   : str         figure suptitle prefix (e.g. "Nb50V50 500 °C")
    cmap    : str         matplotlib colormap
    dpi     : int         figure resolution

    Returns
    -------
    Path to the saved PNG file.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    N = c_3d.shape[0]
    xy, xz, yz = _mid_slices(c_3d)
    vmin, vmax = float(c_3d.min()), float(c_3d.max())

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    plane_titles = [
        "XY plane  (z = N/2)",
        "XZ plane  (y = N/2)",
        "YZ plane  (x = N/2)",
    ]
    for ax, data, ptitle in zip(axes, [xy, xz, yz], plane_titles):
        im = ax.imshow(data, origin="lower", cmap=cmap,
                       vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.set_title(f"{ptitle}\n{label},  t = {t_s:.1f} s", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    sup = f"{title}  |  N={N}  |  t={t_s:.1f} s" if title else f"N={N}  |  t={t_s:.1f} s"
    fig.suptitle(sup, fontsize=11)
    fig.tight_layout()

    fname = out_dir / f"slices_{label.replace('_','').replace(' ','')}_t{t_s:.0f}s.png"
    fig.savefig(fname, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return fname


# ── PyVista 3D renders ────────────────────────────────────────────────────────

def _scalar_bar_kwargs(bar_title: str) -> dict:
    """Consistent horizontal bottom-right scalar bar for all 3D PyVista plots."""
    return dict(
        title=f"{bar_title}\n", vertical=False, fmt="%.2f",
        title_font_size=14, label_font_size=11,
        width=0.361, height=0.0475,
        position_x=0.595, position_y=0.04,
        color="black",
    )


def _show_bounds(pl: object) -> None:
    pl.show_bounds(
        grid=False, location="outer", ticks="both",
        font_size=7, color="black",
        xtitle="x", ytitle="y", ztitle="z",
    )


def save_pyvista_3d(
    c_3d:       np.ndarray,
    t_s:        float,
    out_dir:    Path,
    label:      str  = "x_Nb",
    bar_title:  str  = "Nb Composition",
    title:      str  = "",
    cmap:       str  = "RdBu_r",
    window_size: tuple[int, int] = (900, 800),
) -> list[Path]:
    """
    Render the 3D composition field with PyVista and save PNG files.

    Three renders are produced:

    1. ``pyvista_slices_*``  — three orthogonal mid-plane meshes in 3D space
    2. ``pyvista_surface_*`` — smooth-shaded outer surface of the cube
    3. ``pyvista_iso_*``     — isosurface at the mean composition

    Parameters
    ----------
    c_3d       : np.ndarray  shape (N, N, N)
    t_s        : float       physical time in seconds
    out_dir    : Path        output directory (created if absent)
    label      : str         scalar field key used in file names
    bar_title  : str         colorbar title shown in the figure
    title      : str         figure title prefix
    cmap       : str         PyVista colormap name
    window_size: tuple       off-screen render resolution in pixels

    Returns
    -------
    List of Path objects for each saved PNG (empty if PyVista is unavailable).
    """
    try:
        import pyvista as pv
    except ImportError:
        print("  [PyVista] not installed — skipping 3D renders.  "
              "Install with:  pip install pyvista")
        return []

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pv.global_theme.background = "white"
    pv.global_theme.font.color = "black"

    N = c_3d.shape[0]
    grid = pv.ImageData(
        dimensions=(N, N, N),
        spacing=(1.0, 1.0, 1.0),
        origin=(0.0, 0.0, 0.0),
    )
    grid.point_data[label] = c_3d.ravel(order="F")

    vmin, vmax = float(c_3d.min()), float(c_3d.max())
    clim  = [vmin, vmax]
    mid   = float(N) / 2.0
    sbkw  = _scalar_bar_kwargs(bar_title)
    saved: list[Path] = []

    fig_title = f"{title}  |  N={N}  |  t={t_s:.1f} s" if title else f"N={N}  |  t={t_s:.1f} s"

    # ── 1. orthogonal slices ──────────────────────────────────────────────
    slices = grid.slice_orthogonal(x=mid, y=mid, z=mid)
    pl = pv.Plotter(off_screen=True, window_size=list(window_size))
    pl.add_mesh(slices, scalars=label, cmap=cmap, clim=clim,
                show_edges=False, lighting=True, scalar_bar_args=sbkw)
    pl.add_title(fig_title, font_size=10, color="black")
    pl.camera_position = "iso"
    _show_bounds(pl)
    fname = out_dir / f"pyvista_slices_{label.replace('_','')}_t{t_s:.0f}s.png"
    pl.screenshot(str(fname), transparent_background=False)
    pl.close()
    saved.append(fname)

    # ── 2. smooth-shaded outer surface ────────────────────────────────────
    outer = grid.extract_surface(algorithm="dataset_surface")
    pl2 = pv.Plotter(off_screen=True, window_size=list(window_size))
    pl2.add_mesh(outer, scalars=label, cmap=cmap, clim=clim,
                 smooth_shading=True, show_edges=False, scalar_bar_args=sbkw)
    pl2.add_title(fig_title, font_size=10, color="black")
    pl2.camera_position = "iso"
    fname = out_dir / f"pyvista_surface_{label.replace('_','')}_t{t_s:.0f}s.png"
    pl2.screenshot(str(fname), transparent_background=False)
    pl2.close()
    saved.append(fname)

    # ── 3. isosurface at mean composition ─────────────────────────────────
    iso_val = float(c_3d.mean())
    try:
        iso_mesh = grid.contour(isosurfaces=[iso_val], scalars=label)
        pl3 = pv.Plotter(off_screen=True, window_size=list(window_size))
        pl3.add_mesh(iso_mesh, scalars=label, cmap=cmap, clim=clim,
                     smooth_shading=True, scalar_bar_args=sbkw)
        pl3.add_title(
            f"Isosurface  c = {iso_val:.2f}  |  N={N}  |  t={t_s:.1f} s",
            font_size=10, color="black",
        )
        pl3.camera_position = "iso"
        fname = out_dir / f"pyvista_iso_{label.replace('_','')}_t{t_s:.0f}s.png"
        pl3.screenshot(str(fname), transparent_background=False)
        pl3.close()
        saved.append(fname)
    except Exception as exc:
        print(f"  [PyVista] isosurface skipped ({exc})")

    return saved
