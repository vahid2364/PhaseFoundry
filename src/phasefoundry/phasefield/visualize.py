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
import matplotlib.patches as mpatches
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

    c_3d = np.asarray(c_3d)
    N    = c_3d.shape[0]
    vmin, vmax = float(c_3d.min()), float(c_3d.max())

    if c_3d.ndim == 2:
        fig, ax = plt.subplots(1, 1, figsize=(5, 4.5))
        im = ax.imshow(c_3d.T, origin="lower", cmap=cmap,
                       vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.set_title(f"{label},  t = {t_s:.1f} s", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        sup = f"{title}  |  N={N}  |  t={t_s:.1f} s" if title else f"N={N}  |  t={t_s:.1f} s"
        fig.suptitle(sup, fontsize=11)
    else:
        xy, xz, yz = _mid_slices(c_3d)
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


# ── composition microstructure figure (STEM-style) ───────────────────────────

def plot_microstructure_composition(
    fields: dict[str, np.ndarray],
    Lx_nm: float,
    main_element: str | None = None,
    display_names: dict[str, str] | None = None,
    crop: tuple[int, int, int, int] | None = None,
    t_s: float | None = None,
    scalebar_nm: float = 200,
    cmap: str = "coolwarm",
    fig_size: float = 4.0,
    dpi: int = 200,
    save_path: Path | str | None = None,
    show: bool = True,
) -> plt.Figure:
    """
    STEM-style composition figure: one large main panel for the reference
    element and a row of cropped inset panels for every other element,
    connected by yellow lines from the highlighted crop rectangle.

    Scales automatically with the number of elements:
      2 elements → main + 1 inset
      3 elements → main + 2 insets
      N elements → main + (N-1) insets

    Parameters
    ----------
    fields : dict[str, ndarray]
        Element key → 2-D composition field (Ny × Nx).  Dict order sets the
        inset order; ``main_element`` is pulled out to the large panel.
    Lx_nm : float
        Physical domain side length in nm (square domain assumed).
    main_element : str, optional
        Key from *fields* to display as the large panel.
        Defaults to the last key (typically the reference element).
    display_names : dict[str, str], optional
        Map from field keys to human-readable labels, e.g.
        ``{'NB': 'Nb', 'HF': 'Hf'}``.  Keys not present fall back to the
        raw key title-cased.
    crop : (row0, row1, col0, col1), optional
        Pixel-index slice of the highlighted region shown in the insets.
        Defaults to a portrait strip covering the centre of the domain.
    t_s : float, optional
        Simulation time in seconds; appended to the panel label.
    scalebar_nm : float
        Length of the scale bar in nm.
    cmap : str
        Matplotlib colormap applied to all panels.
    fig_size : float
        Square figure side in inches.
    dpi : int
        Output resolution.
    save_path : Path or str, optional
        Destination file.  Extension selects format (PDF or PNG).
    show : bool
        Call ``plt.show()`` when True.

    Returns
    -------
    matplotlib.figure.Figure
    """
    from matplotlib.patches import ConnectionPatch

    keys = list(fields.keys())
    if main_element is None:
        main_element = keys[-1]
    inset_keys = [k for k in keys if k != main_element]
    n_insets = len(inset_keys)

    def _label(key: str) -> str:
        if display_names and key in display_names:
            return display_names[key]
        return key.title()

    main_field = np.asarray(fields[main_element], dtype=float)
    N = main_field.shape[0]

    # Default crop: portrait strip on the right half.
    # r0 must be above the inset group top in figure fraction (inset_bot + inset_h_approx).
    # We use 0.46 as a safe lower bound (well above the ~0.39 inset top).
    if crop is None:
        r0, r1 = int(0.46 * N), int(0.94 * N)
        c0, c1 = int(0.52 * N), int(0.80 * N)
        crop = (r0, r1, c0, c1)
    r0, r1, c0, c1 = crop

    nm_per_px = Lx_nm / N
    extent    = [0, Lx_nm, 0, Lx_nm]

    # ── figure ───────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(fig_size, fig_size), facecolor="white")
    ax_main = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax_main.set_aspect("equal")
    ax_main.axis("off")
    ax_main.set_facecolor("white")

    ax_main.imshow(
        main_field, origin="lower", cmap=cmap,
        vmin=main_field.min(), vmax=main_field.max(),
        extent=extent, interpolation="bilinear",
    )

    # Element label (top-left, white bg box)
    ax_main.text(
        0.02, 0.87, _label(main_element),
        transform=ax_main.transAxes,
        color="black", fontsize=18.6, fontweight="bold",
        va="top", ha="left", zorder=6,
        bbox=dict(facecolor="white", alpha=0.6, edgecolor="none", pad=1.5),
    )

    # Colorbar — inset inside the main axes (top-left corner, clear of rectangle)
    sm = plt.cm.ScalarMappable(
        cmap=cmap, norm=plt.Normalize(vmin=main_field.min(), vmax=main_field.max())
    )
    # Colorbar: top-level axes (not inset_axes) so tick labels are never clipped
    # by ax_main's clip path.  Positioned to the right of the crop rectangle.
    # Colorbar: near right image boundary, ticks on the left side.
    # fig.add_axes keeps it as a top-level axes so labels are never clipped.
    cb_ax = fig.add_axes([0.93, 0.70, 0.035, 0.26])
    cb = fig.colorbar(sm, cax=cb_ax)
    cb.set_label(rf"$x_{{\rm {_label(main_element)}}}$", fontsize=8.5, labelpad=2)
    cb.ax.tick_params(labelsize=7.5, colors="white", length=2, pad=1)
    cb.ax.yaxis.set_label_position("left")
    cb.ax.yaxis.set_ticks_position("left")
    cb.outline.set_edgecolor("white")
    cb.ax.yaxis.label.set_color("white")

    # Scale bar — right-aligned with margin so it always fits inside the domain
    sb_margin = Lx_nm * 0.05
    sb_x0 = Lx_nm - scalebar_nm - sb_margin
    sb_y0 = Lx_nm * 0.04
    ax_main.plot([sb_x0, sb_x0 + scalebar_nm], [sb_y0, sb_y0],
                 color="black", lw=4.9, zorder=5)
    ax_main.text(sb_x0 + scalebar_nm / 2, sb_y0 + Lx_nm * 0.018,
                 f"{scalebar_nm:.0f} nm",
                 color="black", fontsize=13.0,
                 ha="center", va="bottom", fontweight="bold", zorder=6)

    # Yellow crop rectangle
    rect_x = c0 * nm_per_px
    rect_y = r0 * nm_per_px
    rect_w = (c1 - c0) * nm_per_px
    rect_h = (r1 - r0) * nm_per_px
    ax_main.add_patch(mpatches.Rectangle(
        (rect_x, rect_y), rect_w, rect_h,
        linewidth=1.8, edgecolor="yellow", facecolor="none", zorder=5,
    ))

    # ── inset panels ─────────────────────────────────────────────────────────
    inset_w   = 0.155
    inset_gap = 0.005
    inset_left = 0.01
    inset_bot  = 0.01
    # Derive inset height from crop aspect ratio so images are not stretched
    crop_aspect = (r1 - r0) / max(c1 - c0, 1)   # height / width
    inset_h = inset_w * crop_aspect
    inset_top = inset_bot + inset_h

    # Guard: if the crop rectangle's bottom is below the inset group top (in figure
    # fraction), the connecting lines would go upward — visually wrong.  Clamp r0.
    r0_min = int(np.ceil((inset_top + 0.02) * N))
    if r0 < r0_min:
        r0 = min(r0_min, r1 - 4)  # keep at least 4 rows of crop
        r1 = min(r1, N - 1)
        crop_aspect = (r1 - r0) / max(c1 - c0, 1)
        inset_h = inset_w * crop_aspect
        inset_top = inset_bot + inset_h

    for i, key in enumerate(inset_keys):
        left   = inset_left + i * (inset_w + inset_gap)
        ax_ins = fig.add_axes([left, inset_bot, inset_w, inset_h])
        crop_field = np.asarray(fields[key], dtype=float)[r0:r1, c0:c1]
        ax_ins.imshow(
            crop_field, origin="lower", cmap=cmap,
            vmin=float(np.asarray(fields[key]).min()),
            vmax=float(np.asarray(fields[key]).max()),
            interpolation="bilinear", aspect="auto",
        )
        ax_ins.axis("off")
        ax_ins.text(
            0.08, 0.97, _label(key),
            transform=ax_ins.transAxes,
            color="black", fontsize=11.4, fontweight="bold",
            va="top", ha="left",
            bbox=dict(facecolor="white", alpha=0.6, edgecolor="none", pad=1.5),
        )

    # Outer border and vertical dividers around inset group
    inset_right = inset_left + n_insets * inset_w + (n_insets - 1) * inset_gap
    lw_outer, lw_div = 1.5, 0.8
    for xs, ys in [
        ([inset_left,  inset_right], [inset_bot, inset_bot]),
        ([inset_left,  inset_right], [inset_top, inset_top]),
        ([inset_left,  inset_left ], [inset_bot, inset_top]),
        ([inset_right, inset_right], [inset_bot, inset_top]),
    ]:
        fig.add_artist(plt.Line2D(xs, ys, transform=fig.transFigure,
                                  color="black", lw=lw_outer,
                                  clip_on=False, zorder=10))
    for j in range(1, n_insets):
        x_div = inset_left + j * inset_w + (j - 1) * inset_gap
        fig.add_artist(plt.Line2D(
            [x_div, x_div], [inset_bot, inset_top],
            transform=fig.transFigure, color="black", lw=lw_div,
            clip_on=False, zorder=10,
        ))

    # Yellow connecting lines from rectangle bottom corners → inset group top
    fig.add_artist(ConnectionPatch(
        xyA=(rect_x, rect_y),           coordsA=ax_main.transData,
        xyB=(inset_left, inset_top),     coordsB=fig.transFigure,
        color="yellow", lw=0.9, zorder=7, clip_on=False,
    ))
    fig.add_artist(ConnectionPatch(
        xyA=(rect_x + rect_w, rect_y),  coordsA=ax_main.transData,
        xyB=(inset_right, inset_top),    coordsB=fig.transFigure,
        color="yellow", lw=0.9, zorder=7, clip_on=False,
    ))

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight", facecolor="white")
        print(f"Saved microstructure figure to {save_path}")

    if show:
        plt.show()

    return fig


def plot_microstructure_from_npz(
    npz_path: Path | str,
    elements: list[str],
    ref_element: str,
    time_index: int = -1,
    display_names: dict[str, str] | None = None,
    crop: tuple[int, int, int, int] | None = None,
    scalebar_nm: float = 200,
    cmap: str = "coolwarm",
    fig_size: float = 4.0,
    dpi: int = 200,
    save_path: Path | str | None = None,
    show: bool = True,
) -> plt.Figure:
    """
    Load a ``fields.npz`` produced by the phase-field solver and call
    :func:`plot_microstructure_composition`.

    The NPZ must contain keys ``x_<el.lower()>`` for each independent element
    (e.g. ``x_nb``, ``x_hf``, ``x_ti``).  The reference-element field is
    computed as ``1 - sum(independent fields)``.

    Parameters
    ----------
    npz_path : Path or str
        Path to the ``fields.npz`` file.
    elements : list[str]
        All elements including the reference, upper-case, e.g.
        ``['NB', 'HF', 'TI', 'V']``.
    ref_element : str
        Reference (dependent) element; placed in the main panel.
    time_index : int
        Snapshot index to plot (default: -1 = latest).
    display_names : dict[str, str], optional
        Override labels, e.g. ``{'NB': 'Nb'}``.
    crop, scalebar_nm, cmap, fig_size, dpi, save_path, show
        Forwarded to :func:`plot_microstructure_composition`.

    Returns
    -------
    matplotlib.figure.Figure
    """
    npz_path = Path(npz_path)
    d   = np.load(npz_path, allow_pickle=True)
    t_s = float(d["time"][time_index])
    Lx_nm = float(d["Lx"]) * 1e9

    independent = [e for e in elements if e != ref_element]
    fields: dict[str, np.ndarray] = {}
    x_ref = np.ones(d[f"x_{independent[0].lower()}"][time_index].shape, dtype=float)
    for el in independent:
        key = f"x_{el.lower()}"
        arr = np.asarray(d[key][time_index], dtype=float)
        fields[el] = arr
        x_ref -= arr
    fields[ref_element] = x_ref  # reference element last → main panel by default

    return plot_microstructure_composition(
        fields=fields,
        Lx_nm=Lx_nm,
        main_element=ref_element,
        display_names=display_names,
        crop=crop,
        t_s=t_s,
        scalebar_nm=scalebar_nm,
        cmap=cmap,
        fig_size=fig_size,
        dpi=dpi,
        save_path=save_path,
        show=show,
    )


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
