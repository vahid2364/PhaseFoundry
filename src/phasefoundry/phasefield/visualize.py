"""Images of phase-field results.

``save_slices_2d`` saves a 2-D field, or three mid-plane slices of a 3-D field, as a
matplotlib PNG, and ``save_pyvista_3d`` saves PyVista renders of a 3-D field.
``plot_result``, ``plot_free_energy_history`` and ``plot_gm_with_pfm`` plot the output of
``solve_cahn_hilliard_binary``. The frames and energy plots that both solvers write during
a run are drawn here too.

Importing this module switches matplotlib to the non-interactive Agg backend.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    import pyvista as pv
    from pyvista.plotting._typing import ColormapOptions, ScalarBarArgs

    from phasefoundry.phasefield.binary import CHResult

_logger = logging.getLogger(__name__)

_CMAP_FIELD = "RdBu_r"
"""Colormap of the composition frames."""


def _mid_slices(field_3d: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the XY, XZ and YZ mid-plane slices of a cubic 3-D field."""
    N = field_3d.shape[0]
    mid = N // 2
    return field_3d[:, :, mid], field_3d[:, mid, :], field_3d[mid, :, :]


def save_slices_2d(
    c_3d: np.ndarray,
    t_s: float,
    out_dir: Path,
    label: str = "x_Nb",
    title: str = "",
    cmap: str = "RdBu_r",
    dpi: int = 150,
) -> Path:
    """Save a composition field as a PNG: the field itself in 2-D, or three mid-plane slices in 3-D.

    All panels share one colour scale, from the field's minimum to its maximum.

    Args:
        c_3d: Composition field, shape ``(N, N)`` or ``(N, N, N)``.
        t_s: Physical time in seconds, shown in the titles and the file name.
        out_dir: Output directory, created if missing.
        label: Field name used in the panel titles and the file name.
        title: Prefix of the figure title, e.g. ``"Nb50V50 500 °C"``.
        cmap: Matplotlib colormap name.
        dpi: Image resolution.

    Returns:
        Path of the saved file, ``slices_<label>_t<t_s>s.png`` with underscores and spaces
        removed from the label.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    c_3d = np.asarray(c_3d)
    N = c_3d.shape[0]
    vmin, vmax = float(c_3d.min()), float(c_3d.max())

    if c_3d.ndim == 2:
        fig, ax = plt.subplots(1, 1, figsize=(5, 4.5))
        im = ax.imshow(c_3d.T, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.set_title(f"{label},  t = {t_s:.1f} s", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
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
        for ax, data, ptitle in zip(axes, [xy, xz, yz], plane_titles, strict=True):
            im = ax.imshow(data, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
            ax.set_title(f"{ptitle}\n{label},  t = {t_s:.1f} s", fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        sup = f"{title}  |  N={N}  |  t={t_s:.1f} s" if title else f"N={N}  |  t={t_s:.1f} s"
        fig.suptitle(sup, fontsize=11)
    fig.tight_layout()

    fname = out_dir / f"slices_{label.replace('_', '').replace(' ', '')}_t{t_s:.0f}s.png"
    fig.savefig(fname, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return fname


def _scalar_bar_kwargs(bar_title: str) -> ScalarBarArgs:
    """Return PyVista scalar-bar settings: a horizontal bar at the bottom right titled ``bar_title``."""
    return {
        "title": f"{bar_title}\n",
        "vertical": False,
        "fmt": "%.2f",
        "title_font_size": 14,
        "label_font_size": 11,
        "width": 0.361,
        "height": 0.0475,
        "position_x": 0.595,
        "position_y": 0.04,
        "color": "black",
    }


def _show_bounds(pl: pv.Plotter) -> None:
    """Show the outer axis bounds, labelled x, y and z, on a PyVista plotter."""
    pl.show_bounds(  # ty: ignore[missing-argument]
        grid=False,
        location="outer",
        ticks="both",
        font_size=7,
        color="black",
        xtitle="x",
        ytitle="y",
        ztitle="z",
    )


def save_pyvista_3d(
    c_3d: np.ndarray,
    t_s: float,
    out_dir: Path,
    label: str = "x_Nb",
    bar_title: str = "Nb Composition",
    title: str = "",
    cmap: ColormapOptions = "RdBu_r",
    window_size: tuple[int, int] = (900, 800),
) -> list[Path]:
    """Render a 3-D composition field off-screen with PyVista and save PNG files.

    Three images are saved: ``pyvista_slices_*`` with the three orthogonal mid-planes in
    3-D, ``pyvista_surface_*`` with the smooth-shaded outer surface of the cube, and
    ``pyvista_iso_*`` with the isosurface at the mean composition. If the isosurface
    fails, a message is printed and that image is skipped. The grid has unit spacing, so
    the axes are in grid points.

    Args:
        c_3d: Composition field, shape ``(N, N, N)``.
        t_s: Physical time in seconds, shown in the titles and the file names.
        out_dir: Output directory, created if missing.
        label: Name of the scalar field on the grid, also used in the file names.
        bar_title: Title of the colour bar.
        title: Prefix of the image titles.
        cmap: Colormap name.
        window_size: Image size in pixels.

    Returns:
        Paths of the saved images.
    """
    import pyvista as pv

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
    clim = [vmin, vmax]
    mid = float(N) / 2.0
    sbkw = _scalar_bar_kwargs(bar_title)
    saved: list[Path] = []

    fig_title = f"{title}  |  N={N}  |  t={t_s:.1f} s" if title else f"N={N}  |  t={t_s:.1f} s"

    slices = grid.slice_orthogonal(x=mid, y=mid, z=mid)
    pl = pv.Plotter(off_screen=True, window_size=list(window_size))
    pl.add_mesh(slices, scalars=label, cmap=cmap, clim=clim, show_edges=False, lighting=True, scalar_bar_args=sbkw)
    pl.add_title(fig_title, font_size=10, color="black")
    pl.camera_position = "iso"
    _show_bounds(pl)
    fname = out_dir / f"pyvista_slices_{label.replace('_', '')}_t{t_s:.0f}s.png"
    pl.screenshot(str(fname), transparent_background=False)
    pl.close()
    saved.append(fname)

    outer = grid.extract_surface(algorithm="dataset_surface")
    pl2 = pv.Plotter(off_screen=True, window_size=list(window_size))
    pl2.add_mesh(outer, scalars=label, cmap=cmap, clim=clim, smooth_shading=True, show_edges=False, scalar_bar_args=sbkw)
    pl2.add_title(fig_title, font_size=10, color="black")
    pl2.camera_position = "iso"
    fname = out_dir / f"pyvista_surface_{label.replace('_', '')}_t{t_s:.0f}s.png"
    pl2.screenshot(str(fname), transparent_background=False)
    pl2.close()
    saved.append(fname)

    iso_val = float(c_3d.mean())
    try:
        iso_mesh = grid.contour(isosurfaces=[iso_val], scalars=label)
        pl3 = pv.Plotter(off_screen=True, window_size=list(window_size))
        pl3.add_mesh(iso_mesh, scalars=label, cmap=cmap, clim=clim, smooth_shading=True, scalar_bar_args=sbkw)
        pl3.add_title(
            f"Isosurface  c = {iso_val:.2f}  |  N={N}  |  t={t_s:.1f} s",
            font_size=10,
            color="black",
        )
        pl3.camera_position = "iso"
        fname = out_dir / f"pyvista_iso_{label.replace('_', '')}_t{t_s:.0f}s.png"
        pl3.screenshot(str(fname), transparent_background=False)
        pl3.close()
        saved.append(fname)
    except Exception as exc:
        _logger.warning("  [PyVista] isosurface skipped (%s)", exc)

    return saved


def plot_result(
    result: CHResult,
    output_dir: str | Path | None = None,
    show: bool = True,
) -> None:
    """Plot the final composition field and the free energy history.

    The left panel shows the final field; the right panel shows the total, bulk and
    gradient energies against time.

    Args:
        result: Output of ``solve_cahn_hilliard_binary``.
        output_dir: If given, the figure is saved there as ``result_summary.png``.
        show: Call ``plt.show()`` before closing the figure.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    im = axes[0].imshow(
        result.composition,
        cmap="RdBu_r",
        origin="lower",
    )
    plt.colorbar(im, ax=axes[0], label="Composition  (mol fraction)")
    axes[0].set_title(
        f"Composition field  —  $t$ = {result.time_history[-1]:.3g}\n[{result.params['free_energy']} model]",
        fontsize=12,
    )
    axes[0].set_xlabel("")
    axes[0].set_ylabel("")
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    axes[0].grid(False)

    mk = {"marker": "o", "markersize": 4, "markerfacecolor": "white", "markeredgewidth": 1.2}
    axes[1].plot(result.time_history, result.free_energy_history, color="#1f77b4", linewidth=1.8, label="$F_{total}$", **mk)
    axes[1].plot(
        result.time_history, result.bulk_energy_history, color="#d62728", linewidth=1.5, linestyle="--", label="$F_{bulk}$", **mk
    )
    axes[1].plot(
        result.time_history, result.grad_energy_history, color="#2ca02c", linewidth=1.5, linestyle=":", label="$F_{grad}$", **mk
    )
    axes[1].set_xlabel("Time", fontsize=12)
    axes[1].set_ylabel("Free energy  (a.u.)", fontsize=12)
    axes[1].set_title("Free energy evolution", fontsize=12)
    axes[1].legend(fontsize=10, frameon=False)
    axes[1].grid(False)

    plt.tight_layout()

    if output_dir is not None:
        p = Path(output_dir) / "result_summary.png"
        plt.savefig(p, dpi=200, bbox_inches="tight")
        _logger.info("Saved → %s", p)

    if show:
        plt.show()
    plt.close()


def _save_binary_frame(
    c: np.ndarray,
    step: int,
    t_phys: float,
    output_dir: Path,
) -> None:
    """Save the field at one snapshot as ``frame_<step>.png`` in ``output_dir``."""
    fig, ax = plt.subplots(figsize=(5, 5))
    vmin, vmax = c.min(), c.max()
    im = ax.imshow(c, cmap="PuOr", origin="lower", vmin=vmin, vmax=vmax)
    plt.colorbar(im, ax=ax, label="$x_V$  (mol fraction V)", shrink=0.25, aspect=10, pad=0.02)
    ax.set_title(f"$t$ = {t_phys:.4g} s  (step {step})", fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    plt.tight_layout()
    plt.savefig(output_dir / f"frame_{step:06d}.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_free_energy_history(
    result,
    output_dir: str | Path | None = None,
    show: bool = False,
) -> None:
    """Plot the energy contributions and the composition range against time.

    The left panel shows |F_total|, |F_bulk|, |F_grad| and, when recorded, |ΔG_mix| on a
    log scale. The right panel shows the minimum and maximum of the field at each
    snapshot, labelled as x_V for the Nb-V system.

    Args:
        result: Output of ``solve_cahn_hilliard_binary``.
        output_dir: If given, the figure is saved there as ``free_energy_history.png``.
        show: Call ``plt.show()`` before closing the figure.
    """
    t = result.time_history
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    F_tot = np.abs(result.free_energy_history)
    F_bulk = np.abs(result.bulk_energy_history)
    F_grad = np.abs(result.grad_energy_history)
    ax.plot(
        t,
        F_tot,
        color="#1f77b4",
        lw=2,
        marker="o",
        markersize=5,
        markerfacecolor="white",
        markeredgewidth=1.5,
        label="|$F_{total}$|",
    )
    ax.plot(
        t,
        F_bulk,
        color="#d62728",
        lw=1.8,
        ls="--",
        marker="s",
        markersize=5,
        markerfacecolor="white",
        markeredgewidth=1.5,
        label="|$F_{bulk}$|",
    )
    ax.plot(
        t,
        F_grad,
        color="#2ca02c",
        lw=1.8,
        ls=":",
        marker="^",
        markersize=6,
        markerfacecolor="white",
        markeredgewidth=1.5,
        label="|$F_{grad}$|  (interfacial)",
    )
    if result.mix_energy_history:
        dG = np.array(result.mix_energy_history)
        ax.plot(
            t,
            np.abs(dG),
            color="#9467bd",
            lw=1.8,
            ls="-.",
            marker="D",
            markersize=5,
            markerfacecolor="white",
            markeredgewidth=1.5,
            label="|$\\Delta G_{mix}$|",
        )
    ax.set_yscale("log")
    ax.set_xlabel("Time  (s)", fontsize=12)
    ax.set_ylabel("|Free energy|  (J/mol)  [log scale]", fontsize=12)
    ax.set_title("Free energy contributions", fontsize=12)
    ax.legend(fontsize=11, frameon=False)
    ax.grid(False)

    ax2 = axes[1]
    if result.composition_history:
        c_min = [c.min() for c in result.composition_history]
        c_max = [c.max() for c in result.composition_history]
        ax2.plot(
            t,
            c_min,
            color="#d62728",
            lw=1.8,
            marker="s",
            markersize=5,
            markerfacecolor="white",
            markeredgewidth=1.5,
            label="$x_V^{min}$  (Nb-rich)",
        )
        ax2.plot(
            t,
            c_max,
            color="#1f77b4",
            lw=1.8,
            marker="o",
            markersize=5,
            markerfacecolor="white",
            markeredgewidth=1.5,
            label="$x_V^{max}$  (V-rich)",
        )
        ax2.set_xlabel("Time  (s)", fontsize=12)
        ax2.set_ylabel("$x_V$  (mol fraction V)", fontsize=12)
        ax2.set_title("Phase composition evolution", fontsize=12)
        ax2.legend(fontsize=11, frameon=False)
    else:
        ax2.text(
            0.5,
            0.5,
            "Composition history\nnot available",
            ha="center",
            va="center",
            transform=ax2.transAxes,
            fontsize=12,
            color="gray",
        )
        ax2.set_title("Phase composition evolution", fontsize=12)
    ax2.grid(False)

    plt.tight_layout()

    if output_dir is not None:
        p = Path(output_dir) / "free_energy_history.png"
        plt.savefig(p, dpi=150, bbox_inches="tight")
        _logger.info("  Saved → %s", p)
    if show:
        plt.show()
    plt.close()


def plot_gm_with_pfm(
    result: CHResult,
    calphad_fit: dict,
    temperature_K: float,
    output_dir: str | Path | None = None,
    show: bool = False,
) -> None:
    """Plot the G_m curve next to the final phase-field composition field.

    The left panel shows G_m(x) from the CALPHAD spline with the line between G_m(0) and
    G_m(1), shades where ΔG_mix < 0 and where d²G/dx² < 0, and marks the mean initial
    composition and the minimum and maximum of the final field. The right panel shows the
    final field. Labels and title assume the Nb-V BCC system.

    Args:
        result: Output of ``solve_cahn_hilliard_binary``.
        calphad_fit: Output of ``build_gm_spline_surrogate``, with ``spline_gm`` and
            ``spline_dfdx``.
        temperature_K: Temperature in K, used to pick the fit and label the plot.
        output_dir: If given, the figure is saved there as ``gm_pfm_combined.png``.
        show: Call ``plt.show()`` before closing the figure.

    Raises:
        ValueError: If the fit at that temperature has no ``spline_gm`` or ``spline_dfdx``.
    """
    stored_T = np.array(calphad_fit["temperatures_K"])
    T_K = stored_T[int(np.argmin(np.abs(stored_T - temperature_K)))]
    el_key = list(calphad_fit["fits"][T_K]["models"].keys())[0]
    mdl = calphad_fit["fits"][T_K]["models"][el_key]

    if "spline_gm" not in mdl or "spline_dfdx" not in mdl:
        raise ValueError("plot_gm_with_pfm() requires calphad_gm_spline surrogate (spline_gm and spline_dfdx keys required).")

    spl_gm = mdl["spline_gm"]
    spl_dfdx = mdl["spline_dfdx"]

    x = np.linspace(0.001, 0.999, 600)
    gm = spl_gm(x)
    d2G = spl_dfdx.derivative()(x)

    gm_0 = float(spl_gm(0.0))
    gm_1 = float(spl_gm(1.0))
    linear_ref = gm_0 * (1.0 - x) + gm_1 * x

    c_final = result.composition
    c0_mean = result.composition_history[0].mean()
    c_min = float(c_final.min())
    c_max = float(c_final.max())
    t_final = result.time_history[-1]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    ax.plot(x, gm, color="#1f77b4", lw=2.2, label="$G_m(x)$  CALPHAD spline")
    ax.plot(x, linear_ref, color="gray", lw=1.2, ls="--", label="Mechanical mixture (ref)")

    dGmix = gm - linear_ref
    ax.fill_between(
        x, gm, linear_ref, where=(dGmix < 0), alpha=0.15, color="#d62728", label="$\\Delta G_{mix} < 0$  (driving force)"
    )

    spinodal_mask = d2G < 0
    if spinodal_mask.any():
        ax.fill_between(
            x, gm.min() - 500, gm, where=spinodal_mask, alpha=0.08, color="#9467bd", label="Spinodal region ($d^2G/dx^2 < 0$)"
        )

    ax.axvline(c0_mean, color="#2ca02c", lw=1.5, ls=":", label=f"$c_0$ = {c0_mean:.3f}")
    ax.axvline(c_min, color="#d62728", lw=1.5, ls="-.", label=f"$c_{{min}}$ = {c_min:.3f}  (Nb-rich)")
    ax.axvline(c_max, color="#1f77b4", lw=1.5, ls="-.", label=f"$c_{{max}}$ = {c_max:.3f}  (V-rich)")

    ax.set_xlabel("$x_V$  (mol fraction V)", fontsize=12)
    ax.set_ylabel("$G_m$  (J/mol)", fontsize=12)
    ax.set_title(
        f"$G_m$ landscape — {temperature_K - 273.15:.0f} °C\nNb-V BCC",
        fontsize=11,
    )
    ax.legend(fontsize=8, frameon=True, loc="lower right")
    ax.grid(False)

    ax2 = axes[1]
    im = ax2.imshow(
        c_final,
        cmap="RdBu_r",
        origin="lower",
        vmin=c_min,
        vmax=c_max,
    )
    plt.colorbar(im, ax=ax2, label="$x_V$  (mol fraction V)", shrink=0.85)
    ax2.set_title(
        f"PFM composition field\n$t$ = {t_final:.3g} s  ($c_0$ = {c0_mean:.2f})",
        fontsize=11,
    )
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.grid(False)

    plt.tight_layout()

    if output_dir is not None:
        p = Path(output_dir) / "gm_pfm_combined.png"
        plt.savefig(p, dpi=150, bbox_inches="tight")
        _logger.info("  Saved → %s", p)
    if show:
        plt.show()
    plt.close()


def _save_frame(
    c: np.ndarray,
    step: int,
    out_dir: Path,
    elements: list[str],
    reference_element: str | None = None,
) -> None:
    """Save a PNG frame of every composition field, including the reference element.

    The file is ``frame_<step>.png`` with the step zero-padded to 7 digits. For 3-D fields
    the frame shows the mid-plane slice along the last axis. Each panel has its own colour
    range.

    Args:
        c: Composition fields, shape ``(n_ind, *grid)``.
        step: Step number, used in the title and the file name.
        out_dir: Output directory.
        elements: Independent elements, in the order of ``c``.
        reference_element: Reference element, shown as 1 − Σ cᵢ. ``None`` leaves it out.
    """
    N_ind = c.shape[0]
    ndim = c.ndim - 1

    fields = [(c[i], elements[i]) for i in range(N_ind)]
    if reference_element is not None:
        c_ref = 1.0 - c.sum(axis=0)
        fields.append((c_ref, reference_element))

    slice_label = ""
    if ndim == 3:
        mid = c.shape[1] // 2
        fields = [(data[..., mid], lbl) for data, lbl in fields]
        slice_label = f"  [z={mid}]"

    n_panels = len(fields)
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 4.5), squeeze=False)
    for ax, (data, label) in zip(axes[0], fields, strict=True):
        im = ax.imshow(data, cmap=_CMAP_FIELD, vmin=data.min(), vmax=data.max(), origin="lower", interpolation="nearest")
        ax.set_title(
            f"$x_{{\\rm {label.capitalize()}}}${slice_label}  (step {step:,})",
            fontsize=11,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_dir / f"frame_{step:07d}.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def _plot_energy(
    out_dir: Path,
    t_hist: list,
    F_hist: list,
    Fb_hist: list,
    Fg_hist: list,
    elements: list[str] | None = None,
    cmin_hist: dict | None = None,
    cmean_hist: dict | None = None,
    cmax_hist: dict | None = None,
) -> None:
    """Plot the free-energy history and, if statistics are given, the composition history.

    Writes ``free_energy_history.png`` (|F_total|, |F_bulk| and F_grad on a log scale) and,
    when ``elements`` and ``cmin_hist`` are given, ``composition_evolution.png`` with the
    minimum, mean and maximum of each element over time.
    """
    t = np.array(t_hist)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    F_arr = np.array(F_hist)
    Fb_arr = np.array(Fb_hist)
    Fg_arr = np.array(Fg_hist)
    ax.plot(t, np.abs(F_arr), label="|$F_{\\rm total}$|", linewidth=2, color="#1f77b4")
    ax.plot(t, np.abs(Fb_arr), label="|$F_{\\rm bulk}$|", linewidth=1.5, linestyle="--", color="#ff7f0e")
    ax.plot(t, np.abs(Fg_arr), label="$F_{\\rm gradient}$", linewidth=1.5, linestyle=":", color="#2ca02c")
    ax.set_yscale("log")
    ax.set_xlabel("Time [s]", fontsize=12)
    ax.set_ylabel("|Free energy|  [J/mol]  (log scale)", fontsize=12)
    ax.set_title("Free energy evolution", fontsize=12)
    ax.legend(fontsize=10, frameon=False)
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(out_dir / "free_energy_history.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    if elements is None or cmin_hist is None or cmean_hist is None or cmax_hist is None:
        return
    n_el = len(elements)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    fig, axes = plt.subplots(1, n_el, figsize=(5 * n_el, 4.5), squeeze=False)
    for p, el in enumerate(elements):
        ax = axes[0, p]
        c_lo = np.array(cmin_hist[el])
        c_mean = np.array(cmean_hist[el])
        c_hi = np.array(cmax_hist[el])
        col = colors[p % len(colors)]

        ax.fill_between(t, c_lo, c_hi, alpha=0.15, color=col, label="min–max range")
        ax.plot(
            t,
            c_lo,
            color=col,
            lw=1.5,
            ls="--",
            marker="s",
            markersize=4,
            markerfacecolor="white",
            label=f"$x_{{{el.capitalize()}}}^{{\\rm min}}$",
        )
        ax.plot(
            t,
            c_hi,
            color=col,
            lw=1.5,
            ls="-.",
            marker="o",
            markersize=4,
            markerfacecolor="white",
            label=f"$x_{{{el.capitalize()}}}^{{\\rm max}}$",
        )
        ax.plot(t, c_mean, color=col, lw=2.2, ls="-", label=f"$\\langle x_{{{el.capitalize()}}} \\rangle$")

        ax.set_xlabel("Time [s]", fontsize=12)
        ax.set_ylabel(f"$x_{{{el.capitalize()}}}$  [mol fraction]", fontsize=12)
        ax.set_title(f"{el.capitalize()} — composition evolution", fontsize=12)
        ax.legend(fontsize=9, frameon=False)
        ax.grid(False)
    fig.tight_layout()
    fig.savefig(out_dir / "composition_evolution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
