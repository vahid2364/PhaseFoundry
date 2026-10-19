"""Plots of the thermodynamic tables written by ``generate_table``.

The functions plot G_m or chemical potentials against composition for one or more
temperatures, plot spinodal and binodal boundaries against temperature, draw a ternary
phase map with the spinodal boundary, and draw an alloy-space map for three or more
components with the AS-UMAP method of Vela et al. (arXiv:2408.07681). AS-UMAP embeds the
whole composition simplex in 2-D with UMAP (t-SNE from scikit-learn is the fallback) and
colours the CALPHAD points by phase region.
"""

from __future__ import annotations

import logging
import warnings
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

from phasefoundry.thermodynamics.storage import load_table

if TYPE_CHECKING:
    from pathlib import Path

_logger = logging.getLogger(__name__)


def plot_free_energy_vs_composition(
    table_paths: list[str | Path],
    x_element: str,
    quantity: str = "GM",
    temperatures_celsius: list[float] | None = None,
    ax: plt.Axes | None = None,
    figsize: tuple[float, float] = (8, 5),
    save_path: str | Path | None = None,
    show: bool = True,
) -> plt.Axes:
    """Plot G_m or a chemical potential against one mole fraction, one curve per table.

    Args:
        table_paths: HDF5 tables from ``generate_table``, usually one per temperature.
            Curves are drawn in this order and coloured along the plasma colormap.
        x_element: Element whose mole fraction is on the x-axis, as stored in the
            table, e.g. ``"V"``.
        quantity: ``"GM"`` for the molar Gibbs energy, or ``"MU/<element>"`` (e.g.
            ``"MU/HF"``) for a chemical potential. Both are in J/mol.
        temperatures_celsius: Legend label of each curve in °C. ``None`` reads the
            temperature from each table's metadata.
        ax: Axes to draw on. ``None`` creates a new figure.
        figsize: Figure size used when ``ax`` is ``None``.
        save_path: File to save the current figure to, at 150 dpi.
        show: Call ``plt.show()`` at the end.

    Returns:
        The axes that were drawn on.

    Raises:
        ValueError: If ``x_element`` is not in a table, or ``quantity`` has another form.
        KeyError: If a table does not store the requested quantity.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    cmap = plt.cm.plasma
    n = len(table_paths)

    for i, path in enumerate(table_paths):
        data = load_table(path)
        elements = data["elements"]

        if x_element not in elements:
            raise ValueError(f"'{x_element}' not in elements {elements} for file {path}")

        col_idx = elements.index(x_element)
        x = data["compositions"][:, col_idx]

        if quantity == "GM":
            if "GM" not in data:
                raise KeyError(f"'GM' not stored in {path}. Re-generate with quantities=('GM',...)")
            y = data["GM"]
            ylabel = "Molar Gibbs energy  $G_m$  [J/mol]"
        elif quantity.startswith("MU/"):
            el = quantity.split("/", 1)[1]
            if "MU" not in data or el not in data["MU"]:
                raise KeyError(f"MU/{el} not stored in {path}")
            y = data["MU"][el]
            ylabel = f"Chemical potential  $\\mu_{{\\rm {el.title()}}}$  [J/mol]"
        else:
            raise ValueError(f"Unknown quantity '{quantity}'. Use 'GM' or 'MU/<element>'.")

        order = np.argsort(x)
        x_sorted, y_sorted = x[order], y[order]

        if temperatures_celsius is not None:
            label = f"{temperatures_celsius[i]:.0f} °C"
        else:
            T_K = data["metadata"].get("temperature", 0)
            label = f"{T_K - 273.15:.0f} °C"

        color = cmap(i / max(n - 1, 1))
        ax.plot(x_sorted, y_sorted, color=color, label=label, linewidth=1.8)

    ax.set_xlabel(f"$x_{{\\rm {x_element.title()}}}$  (mol fraction)", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)

    if temperatures_celsius is not None and len(temperatures_celsius) == 1:
        title = f"{quantity} vs $x_{{\\rm {x_element.title()}}}$ at {temperatures_celsius[0]:.0f} °C"
    else:
        title = f"{quantity} vs $x_{{\\rm {x_element.title()}}}$"
    ax.set_title(title, fontsize=13)
    ax.legend(
        title="Temperature",
        fontsize=9,
        title_fontsize=9,
        loc="best",
        ncol=2 if n > 6 else 1,
    )
    ax.grid(False)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        _logger.info("Saved figure to %s", save_path)

    if show:
        plt.show()

    return ax


def plot_chemical_potential_vs_composition(
    table_paths: list[str | Path],
    x_element: str,
    mu_elements: list[str] | None = None,
    temperatures_celsius: list[float] | None = None,
    figsize: tuple[float, float] = (12, 4),
    save_path: str | Path | None = None,
    show: bool = True,
) -> list[plt.Axes]:
    """Plot chemical potentials against one mole fraction, one subplot per element.

    Each subplot is a ``plot_free_energy_vs_composition`` call with
    ``quantity="MU/<element>"``, so every table gives one curve per subplot.

    Args:
        table_paths: HDF5 tables from ``generate_table``, usually one per temperature.
        x_element: Element whose mole fraction is on the x-axis, as stored in the table.
        mu_elements: Elements whose chemical potentials are plotted. ``None`` uses every
            element of the first table.
        temperatures_celsius: Legend label of each curve in °C. ``None`` reads the
            temperature from each table's metadata.
        figsize: Figure size.
        save_path: File to save the figure to, at 150 dpi.
        show: Call ``plt.show()`` at the end.

    Returns:
        The subplot axes, in the order of ``mu_elements``.
    """
    sample = load_table(table_paths[0])
    elements = sample["elements"]

    if mu_elements is None:
        mu_elements = elements

    fig, axes = plt.subplots(1, len(mu_elements), figsize=figsize, sharey=False)
    if len(mu_elements) == 1:
        axes = [axes]

    for j, el in enumerate(mu_elements):
        plot_free_energy_vs_composition(
            table_paths=table_paths,
            x_element=x_element,
            quantity=f"MU/{el}",
            temperatures_celsius=temperatures_celsius,
            ax=axes[j],
            show=False,
        )
        axes[j].set_title(f"$\\mu_{{\\rm {el.title()}}}$", fontsize=12)
        if j > 0:
            axes[j].set_ylabel("")

    plt.suptitle(
        f"Chemical potentials vs $x_{{\\rm {x_element.title()}}}$",
        fontsize=13,
        y=1.02,
    )
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        _logger.info("Saved figure to %s", save_path)

    if show:
        plt.show()

    return axes


def plot_ternary_spinodal(
    table_path: str | Path,
    independent_elements: list[str],
    corner_labels: list[str] | None = None,
    sim_composition: dict[str, float] | None = None,
    ax: plt.Axes | None = None,
    figsize: tuple[float, float] = (5.5, 5.0),
    marker_size: float = 8,
    save_path: str | Path | None = None,
    show: bool = True,
) -> plt.Axes:
    """Draw a ternary phase map with the spinodal boundary for one phase's CALPHAD table.

    Single-, two- and three-phase points come from the lower convex hull of
    G_m(x_1, x_2) on the fine grid of ``compute_ternary_phase_regions`` and are drawn as
    coloured markers. Tie-triangles are outlined with their vertices marked. The spinodal
    boundary, where the smallest Hessian eigenvalue λ_min is zero, is a dashed black line;
    the spinodal region itself is not filled.

    Args:
        table_path: HDF5 table from ``generate_table`` with ``GM`` and ``dMU``.
        independent_elements: The two independent elements, e.g. ``["NB", "HF"]`` for
            Nb-Hf-V. The remaining element of the table is the reference.
        corner_labels: Labels of the right, top and left corners (python-ternary order),
            which are the first independent element, the second, and the reference.
            ``None`` uses the element names in title case.
        sim_composition: Simulated composition keyed by element, e.g.
            ``{"NB": 0.50, "HF": 0.25}``, drawn as a black star.
        ax: Axes to draw on. ``None`` creates a new figure.
        figsize: Figure size used when ``ax`` is ``None``.
        marker_size: Marker size (``s``) of the region points; tie-triangle vertices are
            twice as large.
        save_path: File to save the figure to, at 300 dpi.
        show: Call ``plt.show()`` at the end.

    Returns:
        The axes that hold the ternary plot.
    """
    import ternary
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from scipy.interpolate import LinearNDInterpolator

    from phasefoundry.thermodynamics.spinodal import compute_ternary_phase_regions, spinodal_boundary_multi

    pr = compute_ternary_phase_regions(table_path, independent_elements)

    comps = pr["compositions"]
    elements = pr["elements"]
    tie_triangles = pr["tie_triangles"]
    T_C = pr["temperature_C"]

    idx = [elements.index(el) for el in independent_elements]
    ref_el = [e for e in elements if e not in independent_elements][0]

    sm = spinodal_boundary_multi([table_path], independent_elements, reference_element=ref_el)
    lam_min_raw = sm["min_eigenvalue"][0]

    x0_raw = comps[:, idx[0]]
    x1_raw = comps[:, idx[1]]

    fg = pr["fine_grid"]
    fine_pts = np.column_stack([fg["x0"], fg["x1"]])
    region_fine = fg["region"]

    from scipy.interpolate import NearestNDInterpolator

    lam_interp_lin = LinearNDInterpolator(list(zip(x0_raw, x1_raw, strict=True)), lam_min_raw)
    lam_interp_nn = NearestNDInterpolator(list(zip(x0_raw, x1_raw, strict=True)), lam_min_raw)
    lam_fine = lam_interp_lin(fine_pts[:, 0], fine_pts[:, 1])
    nan_mask = np.isnan(lam_fine)
    if nan_mask.any():
        # Edge NaNs get a stable sentinel instead of nearest-neighbour values, so the λ = 0 contour ends at the axes.
        x0_nan = fine_pts[nan_mask, 0]
        x1_nan = fine_pts[nan_mask, 1]
        xref_nan = 1.0 - x0_nan - x1_nan
        on_axis = (x0_nan < 0.02) | (x1_nan < 0.02) | (xref_nan < 0.02)

        nan_idx = np.where(nan_mask)[0]
        interior_nan_idx = nan_idx[~on_axis]
        boundary_nan_idx = nan_idx[on_axis]

        if len(interior_nan_idx) > 0:
            lam_fine[interior_nan_idx] = lam_interp_nn(fine_pts[interior_nan_idx, 0], fine_pts[interior_nan_idx, 1])
        if len(boundary_nan_idx) > 0:
            lam_fine[boundary_nan_idx] = float(np.nanmax(lam_min_raw)) + 1.0

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    _, tax = ternary.figure(ax=ax, scale=1)
    tax.boundary(linewidth=1.2)
    tax.gridlines(multiple=0.2, color="gray", linewidth=0.4, alpha=0.4)

    def _tern(xa, xb):
        """Return the python-ternary point (right, top, left) = (x_el0, x_el1, x_ref)."""
        xref = 1.0 - xa - xb
        return (xa, xb, xref)

    from phasefoundry.thermodynamics.spinodal import _points_in_triangle

    in_triangle = np.zeros(len(fine_pts), dtype=bool)
    for tt in tie_triangles:
        in_triangle |= _points_in_triangle(fine_pts[:, 0], fine_pts[:, 1], tt)

    # Wong (2011) colour-blind safe palette
    COLORS = {
        "single": ("#0072B2", 0.70),
        "two": ("#E69F00", 0.80),
        "three": ("#009E73", 0.80),
        "spinodal": ("#D55E00", 0.85),
    }

    for zi, (label, (color, alpha)) in enumerate(COLORS.items()):
        if label == "spinodal":
            continue  # spinodal is drawn as a contour line
        elif label == "single":
            mask = (region_fine == "single") & ~in_triangle
        elif label == "two":
            mask = (region_fine == "two") & ~in_triangle
        else:
            mask = in_triangle

        pts_sel = fine_pts[mask]
        if len(pts_sel) == 0:
            continue
        tern_pts = [_tern(xa, xb) for xa, xb in pts_sel]
        tax.scatter(tern_pts, marker="o", s=marker_size, color=color, alpha=alpha, zorder=3 + zi)

    # Edge anchors let contours exit through the axes instead of closing inside the domain.
    import matplotlib.tri as mtri

    n_edge = 80
    t_edge = np.linspace(0.0, 1.0, n_edge + 1)
    edge_x0 = np.concatenate([t_edge, np.zeros(n_edge + 1), t_edge[::-1]])
    edge_x1 = np.concatenate([np.zeros(n_edge + 1), t_edge, 1.0 - t_edge[::-1]])

    def _augmented_contour(x0_pts, x1_pts, values, edge_val_lam, edge_val_bin, field, level):
        """Return the ``level`` contour of ``values`` as python-ternary paths, with edge anchors.

        Anchor points along the three ternary edges get ``edge_val_lam`` when ``field`` is
        ``"lam"`` and ``edge_val_bin`` otherwise: a large positive λ (stable) or 0.0
        (single-phase), the values at the pure-element limits. Triangles with an edge
        longer than 0.05 are masked before contouring.
        """
        x0_aug = np.concatenate([x0_pts, edge_x0])
        x1_aug = np.concatenate([x1_pts, edge_x1])
        fill = edge_val_lam * np.ones(len(edge_x0)) if field == "lam" else edge_val_bin * np.ones(len(edge_x0))
        vals_aug = np.concatenate([values, fill])

        triang = mtri.Triangulation(x0_aug, x1_aug)
        # Drop triangles that bridge the data cloud and the boundary ring.
        tri_x = x0_aug[triang.triangles]
        tri_y = x1_aug[triang.triangles]

        def _sq(i, j):
            return (tri_x[:, i] - tri_x[:, j]) ** 2 + (tri_y[:, i] - tri_y[:, j]) ** 2

        max_sq = np.maximum(_sq(0, 1), np.maximum(_sq(1, 2), _sq(0, 2)))
        triang.set_mask(max_sq > 0.05**2)

        fig_tmp, ax_tmp = plt.subplots()
        cs = ax_tmp.tricontour(triang, vals_aug, levels=[level])
        paths = []
        for path in cs.get_paths():
            tern = [_tern(v[0], v[1]) for v in np.asarray(path.vertices)]
            paths.append(tern)
        plt.close(fig_tmp)
        return paths

    def _merge_paths(paths, tol=1e-6):
        """Join path segments whose endpoints coincide (within ``tol``) into longer paths.

        This avoids overlapping strokes where contour segments meet.
        """
        if not paths:
            return paths
        segs = [np.array(p) for p in paths]
        merged = []
        used = [False] * len(segs)
        for i in range(len(segs)):
            if used[i]:
                continue
            chain = list(segs[i])
            used[i] = True
            extended = True
            while extended:
                extended = False
                for j in range(len(segs)):
                    if used[j]:
                        continue
                    s = segs[j]
                    if np.linalg.norm(np.array(chain[-1]) - s[0]) < tol:
                        chain.extend(s[1:].tolist())
                        used[j] = True
                        extended = True
                    elif np.linalg.norm(np.array(chain[-1]) - s[-1]) < tol:
                        chain.extend(s[-2::-1].tolist())
                        used[j] = True
                        extended = True
                    elif np.linalg.norm(np.array(chain[0]) - s[-1]) < tol:
                        chain = s[:-1].tolist() + chain
                        used[j] = True
                        extended = True
                    elif np.linalg.norm(np.array(chain[0]) - s[0]) < tol:
                        chain = s[::-1].tolist()[:-1] + chain
                        used[j] = True
                        extended = True
            merged.append(chain)
        return merged

    def _split_at_boundary(path, tol=0.01):
        """Split a path wherever it reaches a ternary edge (any coordinate below ``tol``).

        Each piece runs through the interior and ends at the first point on an edge, so
        boundary lines end at an axis instead of running along it.
        """
        segments = []
        seg = []
        for pt in path:
            on_boundary = any(c < tol for c in pt)
            if on_boundary:
                if seg:
                    seg.append(pt)
                    segments.append(seg)
                    seg = []
            else:
                seg.append(pt)
        if len(seg) > 1:
            segments.append(seg)
        return segments

    def _plot_contour(paths, **kwargs):
        for path in _merge_paths(paths):
            for seg in _split_at_boundary(path):
                if len(seg) > 1:
                    tax.plot(seg, zorder=9, **kwargs)

    # No edge anchors here: the spinodal region reaches the edges, so anchors would add spurious contours.
    def _simple_contour(x0_pts, x1_pts, values, level, max_edge=0.05):
        triang = mtri.Triangulation(x0_pts, x1_pts)
        tri_x = x0_pts[triang.triangles]
        tri_y = x1_pts[triang.triangles]

        def _sq(i, j):
            return (tri_x[:, i] - tri_x[:, j]) ** 2 + (tri_y[:, i] - tri_y[:, j]) ** 2

        max_sq = np.maximum(_sq(0, 1), np.maximum(_sq(1, 2), _sq(0, 2)))
        triang.set_mask(max_sq > max_edge**2)
        fig_tmp, ax_tmp = plt.subplots()
        cs = ax_tmp.tricontour(triang, values, levels=[level])
        paths = []
        for path in cs.get_paths():
            tern = [_tern(v[0], v[1]) for v in np.asarray(path.vertices)]
            paths.append(tern)
        plt.close(fig_tmp)
        return paths

    lam_valid = np.isfinite(lam_fine)
    sp_paths = _simple_contour(fine_pts[lam_valid, 0], fine_pts[lam_valid, 1], lam_fine[lam_valid], level=0.0)
    _plot_contour(sp_paths, color="k", linewidth=1.5, linestyle="--")

    for tt in tie_triangles:
        tern_verts = [_tern(tt[i, 0], tt[i, 1]) for i in range(3)]
        tax.plot(tern_verts + [tern_verts[0]], color="#006d50", linewidth=1.8, zorder=7, linestyle="-")
        tax.scatter(tern_verts, marker="D", s=marker_size * 2, color="#006d50", zorder=8)

    if sim_composition is not None:
        sc_x0 = sim_composition.get(independent_elements[0], 0.0)
        sc_x1 = sim_composition.get(independent_elements[1], 0.0)
        tax.scatter([_tern(sc_x0, sc_x1)], marker="*", s=50, color="k", zorder=10)

    if corner_labels is None:
        corner_labels = [
            independent_elements[0].title(),
            independent_elements[1].title(),
            ref_el.title(),
        ]
    tax.right_corner_label(corner_labels[0], fontsize=9, offset=0.12)
    tax.top_corner_label(corner_labels[1], fontsize=9, offset=0.20)
    tax.left_corner_label(corner_labels[2], fontsize=9, offset=0.18)

    tax.set_title(f"Ternary phase map — {T_C:.0f} °C", fontsize=9, pad=30)
    tax.ticks(axis="lbr", linewidth=0.6, multiple=0.2, tick_formats="%.1f", fontsize=6, offset=0.025)
    tax.clear_matplotlib_ticks()
    ax.axis("off")

    handles = [
        Patch(facecolor=COLORS["single"][0], alpha=COLORS["single"][1], label="single-phase"),
        Patch(facecolor=COLORS["two"][0], alpha=COLORS["two"][1], label="two-phase"),
        Patch(facecolor=COLORS["three"][0], alpha=COLORS["three"][1], label="three-phase"),
        Line2D([0], [0], color="k", linewidth=1.4, linestyle="--", label="spinodal boundary"),
    ]
    if sim_composition is not None:
        handles.append(Line2D([0], [0], marker="*", color="k", markersize=6, ls="none", label="simulated composition"))
    ax.legend(
        handles=handles,
        fontsize=6.5,
        frameon=True,
        loc="upper center",
        ncol=2,
        handletextpad=0.3,
        borderpad=0.4,
        columnspacing=0.7,
        bbox_to_anchor=(0.5, 0.02),
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        _logger.info("Saved ternary phase map to %s", save_path)

    if show:
        plt.show()

    return ax


def plot_as_umap(
    table_path: str | Path,
    independent_elements: list[str],
    reference_element: str,
    corner_labels: list[str] | None = None,
    sim_composition: dict[str, float] | None = None,
    umap_scale: int | None = None,
    figsize: tuple[float, float] = (7, 6),
    save_path: str | Path | None = None,
    show: bool = True,
) -> plt.Figure:
    """Draw an alloy-space (AS-UMAP) map for a system with three or more components.

    This is the method of Vela et al. A regular lattice over the whole composition simplex,
    the CALPHAD points and the simulated composition are embedded together in 2-D (UMAP,
    or t-SNE if UMAP is missing or fails). The lattice is drawn as a grey background with
    its outline, and the CALPHAD points are coloured by the phase region from
    ``compute_phase_regions_multi``, with spinodal points shown as their own class. The
    embedding is rotated so the pure elements sit at a fixed orientation, and each pure
    element's position is labelled.

    Args:
        table_path: HDF5 table from ``generate_table``.
        independent_elements: All elements except the reference element.
        reference_element: Dependent element.
        corner_labels: Labels of the pure-element positions, in the table's element
            order. ``None`` uses the element names.
        sim_composition: Simulated composition keyed by element, drawn as a gold star.
            Missing elements count as 0.
        umap_scale: Number of lattice steps along each simplex edge. Larger values give a
            denser background and a slower embedding. ``None`` uses
            ``max(10, 60 // n_el)``.
        figsize: Figure size.
        save_path: File to save the figure to, at 200 dpi.
        show: Call ``plt.show()`` at the end.

    Returns:
        The figure.

    References:
        Vela et al., arXiv:2408.07681.
    """
    from phasefoundry.thermodynamics.spinodal import compute_phase_regions_multi

    data = load_table(table_path)
    elements = data["elements"]
    n_el = len(elements)
    T_C = data["metadata"].get("temperature", 273.15) - 273.15

    result = compute_phase_regions_multi(table_path, independent_elements, reference_element)
    comps = result["compositions"]
    region = result["region"].copy()
    is_spinodal = result["spinodal"]
    region[is_spinodal] = "spinodal"

    scale = umap_scale or max(10, 60 // n_el)
    simplex_pts = _full_simplex_lattice(n_el, scale)
    n_simplex = len(simplex_pts)
    n_data = len(comps)

    extra_pts = []
    sim_idx = None
    if sim_composition is not None:
        sim_vec = np.array([sim_composition.get(el, 0.0) for el in elements])
        extra_pts.append(sim_vec)
        sim_idx = n_simplex + n_data

    all_pts = np.vstack([simplex_pts, comps] + extra_pts)

    method_name, embedding = _fit_dim_reduction(all_pts)
    simplex_emb = embedding[:n_simplex]
    data_emb = embedding[n_simplex : n_simplex + n_data]
    sim_emb = embedding[sim_idx] if sim_idx is not None else None

    pure_embs = {}
    for i, el in enumerate(elements):
        pure_comp = np.zeros(n_el)
        pure_comp[i] = 1.0
        dists = np.linalg.norm(simplex_pts - pure_comp, axis=1)
        pure_embs[el] = simplex_emb[int(np.argmin(dists))]

    rot_angle, centroid = _canonical_rotation(pure_embs, elements, n_el)
    simplex_emb = _rotate_embedding(simplex_emb, rot_angle, centroid)
    data_emb = _rotate_embedding(data_emb, rot_angle, centroid)
    if sim_emb is not None:
        sim_emb = _rotate_embedding(sim_emb[np.newaxis], rot_angle, centroid)[0]
    for el in elements:
        pure_embs[el] = _rotate_embedding(pure_embs[el][np.newaxis], rot_angle, centroid)[0]

    fig, ax = plt.subplots(figsize=figsize)

    ax.scatter(simplex_emb[:, 0], simplex_emb[:, 1], c="#d8d8d8", s=4, alpha=0.6, zorder=1, rasterized=True, linewidths=0)

    _draw_simplex_boundary(ax, simplex_emb, alpha_frac=0.6)

    _COLORS = {
        "single": ("#2ca02c", 0.80),
        "two": ("#ff7f0e", 0.80),
        "three": ("#1f77b4", 0.80),
        "four": ("#9467bd", 0.80),
        "multi": ("#8c564b", 0.80),
        "spinodal": ("#d62728", 0.90),
    }
    _LABELS = {
        "single": "Single phase",
        "two": "Two phase",
        "three": "Three phase",
        "four": "Four phase",
        "multi": "Multi phase",
        "spinodal": "Spinodal",
    }
    for reg in ("single", "two", "three", "four", "multi", "spinodal"):
        mask = region == reg
        if not mask.any():
            continue
        color, alpha = _COLORS[reg]
        ax.scatter(data_emb[mask, 0], data_emb[mask, 1], c=color, s=22, alpha=alpha, zorder=3, linewidths=0, label=_LABELS[reg])

    if sim_emb is not None:
        ax.scatter(
            [sim_emb[0]],
            [sim_emb[1]],
            marker="*",
            s=160,
            c="gold",
            edgecolors="k",
            linewidths=0.8,
            zorder=5,
            label="Simulated composition",
        )

    lab_els = corner_labels if corner_labels is not None else elements
    for el, lab in zip(elements, lab_els, strict=True):
        ex, ey = pure_embs[el]
        ax.annotate(
            lab,
            (ex, ey),
            fontsize=9,
            fontweight="bold",
            ha="center",
            bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "0.7", "alpha": 0.85, "lw": 0.6},
        )

    el_str = "–".join(e.title() for e in elements)
    ax.set_title(f"Alloy space ({el_str})  @  {T_C:.0f} °C", fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_frame_on(False)
    ax.legend(fontsize=8, loc="best", frameon=True, markerscale=1.2, handletextpad=0.4)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        _logger.info("Saved AS-%s figure: %s", method_name, save_path)

    if show:
        plt.show()

    return fig


def _full_simplex_lattice(n_el: int, scale: int) -> np.ndarray:
    """Return every composition of ``n_el`` elements whose fractions are multiples of ``1 / scale``.

    The points are enumerated by stars and bars, so there are
    ``C(scale + n_el - 1, n_el - 1)`` rows of shape ``(n_el,)``.
    """
    from itertools import combinations

    rows = []
    for c in combinations(range(scale + n_el - 1), n_el - 1):
        ext = (-1,) + c + (scale + n_el - 1,)
        coords = tuple(ext[k + 1] - ext[k] - 1 for k in range(n_el))
        rows.append([v / scale for v in coords])
    return np.array(rows, dtype=float)


def _draw_simplex_boundary(
    ax: plt.Axes, emb: np.ndarray, color: str = "#888888", lw: float = 1.2, alpha_frac: float = 0.25
) -> None:
    """Outline a 2-D point cloud on ``ax`` with the edges of its convex hull.

    The edges are those that belong to exactly one triangle of the Delaunay
    triangulation, which are the convex hull edges. Edges longer than
    ``2 * alpha_frac * span`` are left out, where ``span`` is the larger extent of the
    cloud along x or y; if that leaves nothing, the full hull is drawn. Concave parts of
    the cloud are therefore not traced. Nothing is drawn for fewer than four points.
    """
    from matplotlib.collections import LineCollection
    from scipy.spatial import Delaunay

    if len(emb) < 4:
        return

    tri = Delaunay(emb)
    span = max(np.ptp(emb[:, 0]), np.ptp(emb[:, 1]))
    alpha = alpha_frac * span

    boundary_edges: set[tuple[int, int]] = set()
    for simplex in tri.simplices:
        for i in range(3):
            e = tuple(sorted([simplex[i], simplex[(i + 1) % 3]]))
            boundary_edges ^= {e}

    filtered: list[tuple] = []
    for e in boundary_edges:
        pts_e = emb[list(e)]
        r = 0.5 * np.linalg.norm(pts_e[0] - pts_e[1])
        if r <= alpha:
            filtered.append(e)

    if not filtered:
        from scipy.spatial import ConvexHull

        hull = ConvexHull(emb)
        for s in hull.simplices:
            filtered.append(tuple(s))

    segs = [(emb[e[0]], emb[e[1]]) for e in filtered]
    lc = LineCollection(segs, colors=color, linewidths=lw, alpha=0.75, zorder=2)
    ax.add_collection(lc)


def _fit_dim_reduction(pts: np.ndarray) -> tuple[str, np.ndarray]:
    """Embed points in 2-D with UMAP, or with t-SNE if UMAP cannot be imported or fails.

    The tuning constants are set at the top of the function:

    - ``UMAP_N_NEIGHBORS`` (30): neighbours in the UMAP graph. Larger values (50 to 100)
      give a smoother, more globally consistent embedding with sharper hypocycloid cusps;
      smaller values (5 to 15) keep more local detail but can break the simplex apart.
      About 20 or more is needed to span the whole simplex.
    - ``UMAP_MIN_DIST`` (0.6): minimum distance between embedded points. Values near 0
      give tighter clusters and deeper concavities; 0.3 to 0.5 spread the points toward a
      convex blob without cusps.
    - ``UMAP_RANDOM_STATE`` (42): random seed of both methods. A different seed gives a
      different orientation, which ``_canonical_rotation`` then corrects.
    - ``TSNE_MAX_ITER`` (1000): t-SNE iterations. The t-SNE perplexity is
      ``min(30, max(5, n_pts / 4 - 1))``.

    Args:
        pts: Points to embed, shape ``(n_pts, n_features)``.

    Returns:
        ``("UMAP", embedding)`` or ``("t-SNE", embedding)``, with the embedding of shape
        ``(n_pts, 2)``.
    """
    UMAP_N_NEIGHBORS = 30
    UMAP_MIN_DIST = 0.6
    UMAP_RANDOM_STATE = 42
    TSNE_MAX_ITER = 1000

    try:
        import umap as _umap

        reducer = _umap.UMAP(n_components=2, n_neighbors=UMAP_N_NEIGHBORS, min_dist=UMAP_MIN_DIST, random_state=UMAP_RANDOM_STATE)
        return "UMAP", reducer.fit_transform(pts)
    except Exception:
        pass

    from sklearn.manifold import TSNE

    perp = min(30.0, max(5.0, len(pts) / 4.0 - 1.0))
    reducer = TSNE(n_components=2, perplexity=perp, max_iter=TSNE_MAX_ITER, random_state=UMAP_RANDOM_STATE)
    return "t-SNE", reducer.fit_transform(pts)


def _canonical_rotation(
    pure_embs: dict,
    elements: list[str],
    n_el: int,
) -> tuple[float, np.ndarray]:
    """Find the rotation that turns the pure-element positions toward a fixed orientation.

    The target angles around the centroid are those of a regular n-gon with one vertex at
    the top (90°), or for n = 4 a square with flat sides (vertices at 135°, 45°, −45° and
    −135°). The positions, sorted by angle, are matched to the targets under every cyclic
    shift; the shift with the smallest squared angular error gives the rotation as the
    mean angular offset.

    Args:
        pure_embs: Embedded 2-D position of each pure element, keyed by element.
        elements: Elements, in the order used to read ``pure_embs``.
        n_el: Number of elements.

    Returns:
        ``(angle, centroid)``: the rotation angle in radians and the centroid of the
        pure-element positions, for ``_rotate_embedding``.
    """
    positions = np.array([pure_embs[el] for el in elements])
    centroid = positions.mean(axis=0)
    centered = positions - centroid

    current_angles = np.arctan2(centered[:, 1], centered[:, 0])

    order = np.argsort(current_angles)
    sorted_angles = current_angles[order]

    canonical_start = 3.0 * np.pi / 4.0 if n_el == 4 else np.pi / 2.0

    canonical_angles = canonical_start - 2.0 * np.pi * np.arange(n_el) / n_el

    best_rotation = 0.0
    best_error = np.inf
    for shift in range(n_el):
        target = np.roll(canonical_angles, shift)
        diffs = target - sorted_angles
        diffs = (diffs + np.pi) % (2.0 * np.pi) - np.pi
        error = float(np.sum(diffs**2))
        if error < best_error:
            best_error = error
            best_rotation = float(diffs.mean())

    return best_rotation, centroid


def _rotate_embedding(emb: np.ndarray, angle: float, centroid: np.ndarray) -> np.ndarray:
    """Rotate 2-D points of shape ``(n, 2)`` by ``angle`` radians around ``centroid``."""
    c, s = np.cos(angle), np.sin(angle)
    R = np.array([[c, -s], [s, c]])
    return (emb - centroid) @ R.T + centroid


def plot_spinodal(
    spinodal_data: dict,
    x_element: str,
    ax: plt.Axes | None = None,
    figsize: tuple[float, float] = (7, 5),
    fill_color: str = "#d62728",
    fill_alpha: float = 0.15,
    save_path: str | Path | None = None,
    show: bool = True,
) -> plt.Axes:
    """Plot the spinodal region in composition-temperature space.

    For ``spinodal_boundary_binary`` output the region between the two limits is filled and
    the limits are drawn. Temperatures without both limits are skipped, and if none has
    both, the function warns and returns the axes without labels. For
    ``spinodal_boundary_multi`` output the unstable points are scattered, with the first
    composition column on the x-axis whatever ``x_element`` is. The title always names the
    Nb-V BCC_A2 system.

    Args:
        spinodal_data: Output of ``spinodal_boundary_binary`` or ``spinodal_boundary_multi``.
        x_element: Element named in the x-axis label.
        ax: Axes to draw on. A new figure is created if ``None``.
        figsize: Size of the new figure, if one is created.
        fill_color: Colour of the region and its limits.
        fill_alpha: Opacity of the region fill.
        save_path: If given, the figure is saved there at 150 dpi.
        show: Call ``plt.show()`` at the end.

    Returns:
        The axes.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    temps = np.array(spinodal_data["temperatures_celsius"])

    if "x_spinodal_low" in spinodal_data:
        x_lo = spinodal_data["x_spinodal_low"]
        x_hi = spinodal_data["x_spinodal_high"]

        valid = [(T, lo, hi) for T, lo, hi in zip(temps, x_lo, x_hi, strict=True) if lo is not None and hi is not None]

        if not valid:
            warnings.warn("No spinodal region found across the given temperatures.", stacklevel=2)
            return ax

        T_v, lo_v, hi_v = zip(*valid, strict=True)
        T_v = np.array(T_v)
        lo_v = np.array(lo_v)
        hi_v = np.array(hi_v)

        ax.fill_betweenx(T_v, lo_v, hi_v, color=fill_color, alpha=fill_alpha, label="Spinodal region")
        ax.plot(lo_v, T_v, color=fill_color, linewidth=2, label="Spinodal boundary")
        ax.plot(hi_v, T_v, color=fill_color, linewidth=2)

        ax.scatter(lo_v, T_v, color=fill_color, s=30, zorder=4)
        ax.scatter(hi_v, T_v, color=fill_color, s=30, zorder=4)

    elif "min_eigenvalue" in spinodal_data:
        for T, comps, is_sp in zip(
            spinodal_data["temperatures_celsius"],
            spinodal_data["compositions"],
            spinodal_data["is_spinodal"],
            strict=True,
        ):
            sp_comps = comps[is_sp]
            if len(sp_comps) > 0:
                ax.scatter(
                    sp_comps[:, 0],
                    np.full(len(sp_comps), T),
                    color=fill_color,
                    alpha=0.4,
                    s=15,
                )

    ax.set_xlabel(f"$x_{{\\rm {x_element.title()}}}$  (mol fraction)", fontsize=12)
    ax.set_ylabel("Temperature  (°C)", fontsize=12)
    ax.set_title("Spinodal region — Nb-V  BCC_A2", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(False)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        _logger.info("Saved figure to %s", save_path)

    if show:
        plt.show()

    return ax


def plot_phase_diagram(
    spinodal_data: dict,
    binodal_data: dict,
    x_element: str,
    system_name: str = "Nb-V  BCC_A2",
    ax: plt.Axes | None = None,
    figsize: tuple[float, float] = (7, 5),
    save_path: str | Path | None = None,
    show: bool = True,
) -> plt.Axes:
    """Plot the binodal and spinodal boundaries together on a temperature-composition diagram.

    The binodal (solid blue) bounds the two-phase region, where the two phases have equal
    chemical potentials (common tangent). The spinodal (dashed red) lies inside it;
    compositions within the spinodal decompose without a nucleation barrier. Both are drawn
    as interpolating splines through the per-temperature points, and the midpoint of the
    highest-temperature spinodal pair is marked as the critical point.

    Args:
        spinodal_data: Output of ``spinodal_boundary_binary``, expressed in mole fractions
            of ``x_element`` (``workflows/02_fit_surrogate/run.py`` does this remapping).
        binodal_data: Output of ``binodal_boundary_binary``, also in ``x_element`` fractions.
        x_element: Element on the x-axis, e.g. ``"V"``.
        system_name: System name shown in the title.
        ax: Axes to draw on. A new figure is created if ``None``.
        figsize: Size of the new figure, if one is created.
        save_path: If given, the figure is saved there at 150 dpi.
        show: Call ``plt.show()`` at the end.

    Returns:
        The axes.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    temps_b = np.array(binodal_data["temperatures_celsius"])
    b_lo = binodal_data["x_binodal_low"]
    b_hi = binodal_data["x_binodal_high"]

    valid_b = sorted(
        [(T, lo, hi) for T, lo, hi in zip(temps_b, b_lo, b_hi, strict=True) if lo is not None and hi is not None],
        key=lambda v: v[0],
    )
    if valid_b:
        T_b, lo_b, hi_b = zip(*valid_b, strict=True)
        T_b = np.array(T_b)
        lo_b = np.array(lo_b)
        hi_b = np.array(hi_b)

        # make_interp_spline needs strictly increasing x
        _, unique_idx = np.unique(T_b, return_index=True)
        T_b = T_b[unique_idx]
        lo_b = lo_b[unique_idx]
        hi_b = hi_b[unique_idx]

        from scipy.interpolate import make_interp_spline

        T_fine = np.linspace(T_b.min(), T_b.max(), 400)
        lo_b_sm = make_interp_spline(T_b, lo_b, k=min(3, len(T_b) - 1))(T_fine)
        hi_b_sm = make_interp_spline(T_b, hi_b, k=min(3, len(T_b) - 1))(T_fine)

        ax.fill_betweenx(T_fine, lo_b_sm, hi_b_sm, color="#1f77b4", alpha=0.08, label="Two-phase region (binodal)")
        ax.plot(lo_b_sm, T_fine, color="#1f77b4", linewidth=2.2, label="Binodal")
        ax.plot(hi_b_sm, T_fine, color="#1f77b4", linewidth=2.2)
        ax.scatter(lo_b, T_b, color="#1f77b4", s=30, zorder=5)
        ax.scatter(hi_b, T_b, color="#1f77b4", s=30, zorder=5)

    temps_s = np.array(spinodal_data["temperatures_celsius"])
    s_lo = spinodal_data["x_spinodal_low"]
    s_hi = spinodal_data["x_spinodal_high"]

    valid_s = sorted(
        [(T, lo, hi) for T, lo, hi in zip(temps_s, s_lo, s_hi, strict=True) if lo is not None and hi is not None],
        key=lambda v: v[0],
    )
    if valid_s:
        T_s, lo_s, hi_s = zip(*valid_s, strict=True)
        T_s = np.array(T_s)
        lo_s = np.array(lo_s)
        hi_s = np.array(hi_s)

        # make_interp_spline needs strictly increasing x
        _, unique_idx = np.unique(T_s, return_index=True)
        T_s = T_s[unique_idx]
        lo_s = lo_s[unique_idx]
        hi_s = hi_s[unique_idx]

        T_fine_s = np.linspace(T_s.min(), T_s.max(), 400)
        lo_s_sm = make_interp_spline(T_s, lo_s, k=min(3, len(T_s) - 1))(T_fine_s)
        hi_s_sm = make_interp_spline(T_s, hi_s, k=min(3, len(T_s) - 1))(T_fine_s)

        ax.fill_betweenx(T_fine_s, lo_s_sm, hi_s_sm, color="#d62728", alpha=0.12, label="Spinodal region")
        ax.plot(lo_s_sm, T_fine_s, color="#d62728", linewidth=2.0, linestyle="--", label="Spinodal")
        ax.plot(hi_s_sm, T_fine_s, color="#d62728", linewidth=2.0, linestyle="--")
        ax.scatter(lo_s, T_s, color="#d62728", s=30, zorder=5)
        ax.scatter(hi_s, T_s, color="#d62728", s=30, zorder=5)

    if valid_s:
        T_s_arr = T_s
        lo_s_arr = lo_s
        hi_s_arr = hi_s
        idx_crit = int(np.argmax(T_s_arr))
        T_crit = float(T_s_arr[idx_crit])
        x_crit = float((lo_s_arr[idx_crit] + hi_s_arr[idx_crit]) / 2.0)
        ax.scatter([x_crit], [T_crit], color="#d62728", s=80, zorder=6, marker="^", label=f"Critical point  (~{T_crit:.0f} °C)")
        ax.annotate(
            f"$T_c$ ≈ {T_crit:.0f} °C",
            xy=(x_crit, T_crit),
            xytext=(x_crit + 0.08, T_crit + 100),
            fontsize=9,
            color="#d62728",
            arrowprops={"arrowstyle": "->", "color": "#d62728", "lw": 1.2},
        )

    ax.set_xlabel(f"$x_{{\\rm {x_element.title()}}}$  (mol fraction)", fontsize=12)
    ax.set_ylabel("Temperature  (°C)", fontsize=12)
    ax.set_title(f"Phase diagram — {system_name}", fontsize=13)
    ax.set_xlim(0, 1)
    all_temps = list(temps_b) + list(temps_s)
    if all_temps:
        ax.set_ylim(bottom=None, top=max(all_temps) + 200)
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(False)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        _logger.info("Saved figure to %s", save_path)

    if show:
        plt.show()

    return ax
