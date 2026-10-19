"""Plots of the thermodynamic tables written by ``generate_table``.

The functions plot G_m or chemical potentials against composition for one or more
temperatures, draw a ternary phase map with the spinodal boundary (needs
``python-ternary``), and draw an alloy-space map for three or more components with the
AS-UMAP method of Vela et al. (arXiv:2408.07681). AS-UMAP embeds the whole composition
simplex in 2-D with UMAP (needs ``umap-learn``; t-SNE from scikit-learn is the fallback)
and colours the CALPHAD points by phase region.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .storage import load_table


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

        # Sort by x for a clean line
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
        print(f"Saved figure to {save_path}")

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
        print(f"Saved figure to {save_path}")

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

    Raises:
        ImportError: If ``python-ternary`` is not installed.
    """
    try:
        import ternary
    except ImportError as exc:
        raise ImportError(
            "python-ternary is required for plot_ternary_spinodal(). Install with: pip install python-ternary"
        ) from exc

    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from scipy.interpolate import LinearNDInterpolator

    from .spinodal import compute_ternary_phase_regions, spinodal_boundary_multi

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

    # Use the fine grid from compute_ternary_phase_regions as the single
    # source of truth for all scatter and contour plots.
    fg = pr["fine_grid"]
    fine_pts = np.column_stack([fg["x0"], fg["x1"]])
    region_fine = fg["region"]
    df_fine = fg["driving_force"]

    # Interpolate λ_min from the raw CALPHAD points onto the same fine grid.
    # Use LinearNDInterpolator for interior points; fill NaN (outside convex
    # hull of raw data) with NearestNDInterpolator away from the ternary edges.
    from scipy.interpolate import NearestNDInterpolator

    lam_interp_lin = LinearNDInterpolator(list(zip(x0_raw, x1_raw)), lam_min_raw)
    lam_interp_nn = NearestNDInterpolator(list(zip(x0_raw, x1_raw)), lam_min_raw)
    lam_fine = lam_interp_lin(fine_pts[:, 0], fine_pts[:, 1])
    nan_mask = np.isnan(lam_fine)
    if nan_mask.any():
        # Split NaN into ternary-edge points vs. true interior holes.
        # Edge NaN must NOT be filled with NearestND (which copies spinodal values
        # from adjacent interior points), because that causes the λ=0 contour to
        # run along the ternary axis instead of terminating at it.  Assign a
        # large positive (stable) sentinel so the contour stops before the edge.
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

    is_spinodal_fine = lam_fine < 0

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    _, tax = ternary.figure(ax=ax, scale=1)
    tax.boundary(linewidth=1.2)
    tax.gridlines(multiple=0.2, color="gray", linewidth=0.4, alpha=0.4)

    # python-ternary convention: (right, top, left)
    # Verified: project_point((1,0,0))→bottom-right, (0,1,0)→top, (0,0,1)→bottom-left
    # right_corner = el0 (NB), top_corner = el1 (HF), left_corner = ref (V)
    def _tern(xa, xb):
        """Return the python-ternary point (right, top, left) = (x_el0, x_el1, x_ref)."""
        xref = 1.0 - xa - xb
        return (xa, xb, xref)

    from .spinodal import _points_in_triangle

    in_triangle = np.zeros(len(fine_pts), dtype=bool)
    for tt in tie_triangles:
        in_triangle |= _points_in_triangle(fine_pts[:, 0], fine_pts[:, 1], tt)

    # Colorblind-friendly palette (Wong 2011):
    COLORS = {
        "single": ("#0072B2", 0.70),  # blue
        "two": ("#E69F00", 0.80),  # orange
        "three": ("#009E73", 0.80),  # bluish-green
        "spinodal": ("#D55E00", 0.85),  # vermillion
    }

    for zi, (label, (color, alpha)) in enumerate(COLORS.items()):
        if label == "spinodal":
            continue  # spinodal boundary drawn as contour line; no separate scatter
        elif label == "single":
            mask = (region_fine == "single") & ~in_triangle
        elif label == "two":
            mask = (region_fine == "two") & ~in_triangle
        else:  # three — everything inside the tie-triangle
            mask = in_triangle

        pts_sel = fine_pts[mask]
        if len(pts_sel) == 0:
            continue
        tern_pts = [_tern(xa, xb) for xa, xb in pts_sel]
        tax.scatter(tern_pts, marker="o", s=marker_size, color=color, alpha=alpha, zorder=3 + zi)

    # The fine grid stops at ~0.01 from the ternary edges (NaN filtered).
    # Without boundary anchors, tricontour is forced to close every contour
    # inside the domain producing unreasonable closed loops.
    # Fix: append a ring of points along the three ternary edges (x0+x1=1,
    # x0=0, x1=0) with known boundary values so contours exit to the axes.
    import matplotlib.tri as mtri

    n_edge = 80
    t_edge = np.linspace(0.0, 1.0, n_edge + 1)
    # Three edges: bottom (x1=0), left (x0=0), hypotenuse (x0+x1=1)
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
        if field == "lam":
            fill = edge_val_lam * np.ones(len(edge_x0))
        else:
            fill = edge_val_bin * np.ones(len(edge_x0))
        vals_aug = np.concatenate([values, fill])

        triang = mtri.Triangulation(x0_aug, x1_aug)
        # Mask triangles that span across the gap between the data cloud and
        # the boundary ring (long edges in composition space).
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
            tern = [_tern(v[0], v[1]) for v in path.vertices]
            paths.append(tern)
        plt.close(fig_tmp)
        return paths

    def _merge_paths(paths, tol=1e-6):
        """Join path segments whose endpoints coincide (within ``tol``) into longer paths.

        This avoids overlapping strokes where contour segments meet.
        """
        if not paths:
            return paths
        # Work with numpy arrays for fast endpoint comparison
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
                # Include this boundary point as the terminal of the current
                # segment, then start fresh (skip running along the edge).
                if seg:
                    seg.append(pt)
                    segments.append(seg)
                    seg = []
                # Do NOT start a new segment here — wait for a point that
                # moves back into the interior.
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

    lam_edge_val = float(np.nanmax(lam_fine)) + 1.0  # well above 0 → stable

    # Spinodal line: λ_min = 0 contour drawn on the fine grid alone (no edge
    # anchors).  The λ=0 boundary is a closed loop inside the diagram —
    # edge anchors would create spurious contours at the data domain boundary
    # because the spinodal region (λ<0) extends close to the ternary edges.
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
            tern = [_tern(v[0], v[1]) for v in path.vertices]
            paths.append(tern)
        plt.close(fig_tmp)
        return paths

    lam_valid = np.isfinite(lam_fine)
    sp_paths = _simple_contour(fine_pts[lam_valid, 0], fine_pts[lam_valid, 1], lam_fine[lam_valid], level=0.0)
    _plot_contour(sp_paths, color="k", linewidth=1.5, linestyle="--")

    # Binodal line: temporarily disabled
    # is_multiphase = (region_fine != "single").astype(float)
    # bin_paths = _augmented_contour(
    #     fine_pts[:, 0], fine_pts[:, 1], is_multiphase,
    #     edge_val_lam=lam_edge_val, edge_val_bin=0.0,
    #     field="bin", level=0.5
    # )
    # _plot_contour(bin_paths, color="k", linewidth=1.5, linestyle="-")

    for tt in tie_triangles:
        tern_verts = [_tern(tt[i, 0], tt[i, 1]) for i in range(3)]
        tax.plot(tern_verts + [tern_verts[0]], color="#006d50", linewidth=1.8, zorder=7, linestyle="-")
        tax.scatter(tern_verts, marker="D", s=marker_size * 2, color="#006d50", zorder=8)

    if sim_composition is not None:
        sc_x0 = sim_composition.get(independent_elements[0], 0.0)
        sc_x1 = sim_composition.get(independent_elements[1], 0.0)
        tax.scatter([_tern(sc_x0, sc_x1)], marker="*", s=50, color="k", zorder=10)

    if corner_labels is None:
        # right = el0, top = el1, left = ref
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
        # Line2D([0], [0], color="k", linewidth=1.4, linestyle="-",
        #        label="binodal boundary"),  # temporarily disabled
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
        print(f"Saved ternary phase map to {save_path}")

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
    from .spinodal import compute_phase_regions_multi

    data = load_table(table_path)
    elements = data["elements"]
    n_el = len(elements)
    T_C = data["metadata"].get("temperature", 273.15) - 273.15

    result = compute_phase_regions_multi(table_path, independent_elements, reference_element)
    comps = result["compositions"]
    region = result["region"].copy()
    is_spinodal = result["spinodal"]
    region[is_spinodal] = "spinodal"

    # scale=15 gives 816 background pts for quaternary — dense enough for a
    # clear hypocycloid boundary without being slow.
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

    # n=3 → equilateral triangle, one vertex at top (△)
    # n=4 → square with flat sides, one vertex at top-left (■)
    # n≥5 → regular polygon with one vertex at top
    rot_angle, centroid = _canonical_rotation(pure_embs, elements, n_el)
    simplex_emb = _rotate_embedding(simplex_emb, rot_angle, centroid)
    data_emb = _rotate_embedding(data_emb, rot_angle, centroid)
    if sim_emb is not None:
        sim_emb = _rotate_embedding(sim_emb[np.newaxis], rot_angle, centroid)[0]
    for el in elements:
        pure_embs[el] = _rotate_embedding(pure_embs[el][np.newaxis], rot_angle, centroid)[0]

    fig, ax = plt.subplots(figsize=figsize)

    ax.scatter(simplex_emb[:, 0], simplex_emb[:, 1], c="#d8d8d8", s=4, alpha=0.6, zorder=1, rasterized=True, linewidths=0)

    # Draw the hypocycloid boundary: concave hull of the background simplex.
    # Use alpha-shape via Delaunay triangulation — keeps only boundary edges
    # whose circumradius is below the alpha threshold.
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
    for el, lab in zip(elements, lab_els):
        ex, ey = pure_embs[el]
        ax.annotate(
            lab,
            (ex, ey),
            fontsize=9,
            fontweight="bold",
            ha="center",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.7", alpha=0.85, lw=0.6),
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
        print(f"Saved AS-{method_name} figure: {save_path}")

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
    alpha = alpha_frac * span  # circumradius threshold

    boundary_edges: set[tuple[int, int]] = set()
    for simplex in tri.simplices:
        for i in range(3):
            e = tuple(sorted([simplex[i], simplex[(i + 1) % 3]]))
            boundary_edges ^= {e}  # XOR: keep edges belonging to exactly 1 triangle

    # Also filter by circumradius to get the concave (alpha-shape) boundary
    filtered: list[tuple] = []
    for e in boundary_edges:
        pts_e = emb[list(e)]
        # Circumradius of the degenerate 1-simplex (edge): half its length
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

    # Sort elements by current angle so we can match them cyclically to the
    # canonical n-gon vertices.
    order = np.argsort(current_angles)
    sorted_angles = current_angles[order]

    # Canonical starting angle and step:
    # n=4 → ■ orientation: vertices at 135°, 45°, -45°, -135°  (step = -90°)
    # all other n → vertex at top (90°), step = 360°/n clockwise
    if n_el == 4:
        canonical_start = 3.0 * np.pi / 4.0  # 135°
    else:
        canonical_start = np.pi / 2.0  # 90°

    canonical_angles = canonical_start - 2.0 * np.pi * np.arange(n_el) / n_el

    # Try all n cyclic shifts and pick the one that minimises total angular error.
    best_rotation = 0.0
    best_error = np.inf
    for shift in range(n_el):
        target = np.roll(canonical_angles, shift)
        diffs = target - sorted_angles
        diffs = (diffs + np.pi) % (2.0 * np.pi) - np.pi  # wrap to (-π, π]
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
