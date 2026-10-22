"""
Plotting utilities for thermodynamic tables.

Public functions
────────────────
plot_free_energy_vs_composition(table_paths, x_element, ...)
    Plot a thermodynamic quantity (Gm or a chemical potential) vs mole fraction
    for one element across multiple temperatures.  One curve per HDF5 table file.

plot_chemical_potential_vs_composition(table_paths, x_element, ...)
    Convenience wrapper around plot_free_energy_vs_composition that produces one
    subplot per element showing all chemical potentials μᵢ vs composition.

plot_ternary_spinodal(table_path, independent_elements, ...)
    Ternary phase diagram (python-ternary) showing single-phase, two-phase,
    three-phase, and spinodal regions derived from the lower convex hull of Gm.
    Draws the spinodal boundary (λ_min = 0 contour) as a dashed line.
    For n-component systems (n ≥ 4) call once per element pair.

plot_as_umap(table_path, independent_elements, reference_element, ...)
    Alloy-Space dimensionality reduction plot for any number of components.
    Implements the AS-UMAP method (Vela et al., arXiv:2408.07681): samples the
    full composition simplex, embeds it in 2D with UMAP (or t-SNE as fallback),
    and overlays CALPHAD data colored by phase region and spinodal label.
    Produces deltoid (n=3), astroid (n=4), or regular polygon (n≥5) shapes.

Private helpers
───────────────
_full_simplex_lattice(n_el, scale)
    Generate a stars-and-bars lattice covering the full n-component simplex.
    Used to build the background point cloud for the AS-UMAP embedding.

_draw_simplex_boundary(ax, emb, ...)
    Draw the concave boundary (alpha-shape) of a 2D point cloud via Delaunay
    triangulation.  Produces the hypocycloid outline in AS-UMAP plots.

_fit_dim_reduction(pts)
    Fit a 2D non-linear embedding on pts.  Tries UMAP first (n_neighbors=30,
    min_dist=0.1); falls back to t-SNE if umap-learn is not installed.
    Key UMAP parameters are named constants at the top of the function body.

_canonical_rotation(pure_embs, elements, n_el)
    Compute the rotation angle that aligns the pure-element embedding positions
    to the canonical n-gon orientation (△ for n=3, ■ for n=4, vertex-at-top
    for n≥5).  Returns (angle_rad, centroid).

_rotate_embedding(emb, angle, centroid)
    Apply a 2D rotation of *angle* radians around *centroid* to all points in emb.
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
    """
    Plot a thermodynamic quantity vs mole fraction of one element
    for multiple temperatures (one curve per table file).

    Parameters
    ----------
    table_paths : list of str or Path
        HDF5 table files produced by generate_table(), one per temperature.
        They are plotted in the order given (typically sorted by T).
    x_element : str
        Element whose mole fraction goes on the x-axis, e.g. 'V'.
    quantity : str
        Which quantity to plot: 'GM' (default) or a chemical potential
        string like 'MU/HF'.  For 'GM' pass 'GM'; for a chemical
        potential pass 'MU' and the element via ``mu_element``.
    temperatures_celsius : list of float, optional
        Labels for each curve.  If None, the temperature is read from
        each file's metadata (stored in K, converted to °C for display).
    ax : matplotlib Axes, optional
        Axes to draw on.  A new figure is created when None.
    figsize : tuple
        Figure size when a new figure is created.
    save_path : str or Path, optional
        If given, save the figure to this path.
    show : bool
        Call plt.show() at the end.

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    cmap = plt.cm.plasma
    n = len(table_paths)

    for i, path in enumerate(table_paths):
        data = load_table(path)
        elements = data["elements"]

        if x_element not in elements:
            raise ValueError(
                f"'{x_element}' not in elements {elements} for file {path}"
            )

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

        # Temperature label
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
        title="Temperature", fontsize=9, title_fontsize=9,
        loc="best", ncol=2 if n > 6 else 1,
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
    """
    Plot chemical potentials μᵢ vs mole fraction for multiple temperatures.
    One subplot per element in mu_elements.
    """
    sample = load_table(table_paths[0])
    elements = sample["elements"]

    if mu_elements is None:
        mu_elements = elements  # all elements

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
        fontsize=13, y=1.02,
    )
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved figure to {save_path}")

    if show:
        plt.show()

    return axes


# ── ternary spinodal + phase-region diagram ───────────────────────────────────

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
    """
    Ternary diagram showing single-phase, two-phase, three-phase, and spinodal
    regions for a single-phase CALPHAD dataset.

    Phase regions are identified from the lower convex hull of G_m(x_1, x_2)
    (see ``compute_ternary_phase_regions``).  The spinodal boundary (λ_min < 0)
    is overlaid as a separate marker class.

    Requires ``python-ternary`` (``pip install python-ternary``).

    Parameters
    ----------
    table_path : str or Path
        HDF5 table produced by ``generate_table()``.  Must contain GM and dMU.
    independent_elements : list[str]
        The two independent elements, e.g. ['NB', 'HF'] for Nb-Hf-V.
    corner_labels : list[str], optional
        Labels for the (top, left, right) corners in python-ternary convention.
        Defaults to the element names derived from the table.
    sim_composition : dict[str, float], optional
        Composition of the simulated alloy, e.g. {'NB': 0.50, 'HF': 0.25}.
        Plotted as a star marker if provided.
    ax : matplotlib.axes.Axes, optional
    figsize : tuple
    marker_size : float
        Size of the scatter markers (s parameter).
    save_path : str or Path, optional
    show : bool

    Returns
    -------
    matplotlib.axes.Axes
    """
    try:
        import ternary
    except ImportError as exc:
        raise ImportError(
            "python-ternary is required for plot_ternary_spinodal(). "
            "Install with: pip install python-ternary"
        ) from exc

    from scipy.interpolate import LinearNDInterpolator
    from .spinodal import compute_ternary_phase_regions, spinodal_boundary_multi
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    # ── compute phase regions and spinodal map ────────────────────────────
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
    fine_pts  = np.column_stack([fg["x0"], fg["x1"]])
    region_fine = fg["region"]
    df_fine     = fg["driving_force"]

    # Interpolate λ_min from the raw CALPHAD points onto the same fine grid.
    # Use LinearNDInterpolator for interior points; fill NaN (outside convex
    # hull of raw data) with NearestNDInterpolator so the contour can exit
    # to the binary axes naturally — the spinodal boundary terminates at the
    # ternary edges, it does not close as a loop inside the diagram.
    from scipy.interpolate import NearestNDInterpolator
    lam_interp_lin = LinearNDInterpolator(list(zip(x0_raw, x1_raw)), lam_min_raw)
    lam_interp_nn  = NearestNDInterpolator(list(zip(x0_raw, x1_raw)), lam_min_raw)
    lam_fine = lam_interp_lin(fine_pts[:, 0], fine_pts[:, 1])
    nan_mask = np.isnan(lam_fine)
    if nan_mask.any():
        # Split NaN into ternary-edge points vs. true interior holes.
        # Edge NaN must NOT be filled with NearestND (which copies spinodal values
        # from adjacent interior points), because that causes the λ=0 contour to
        # run along the ternary axis instead of terminating at it.  Assign a
        # large positive (stable) sentinel so the contour stops before the edge.
        x0_nan  = fine_pts[nan_mask, 0]
        x1_nan  = fine_pts[nan_mask, 1]
        xref_nan = 1.0 - x0_nan - x1_nan
        on_axis = (x0_nan < 0.02) | (x1_nan < 0.02) | (xref_nan < 0.02)

        nan_idx = np.where(nan_mask)[0]
        interior_nan_idx = nan_idx[~on_axis]
        boundary_nan_idx = nan_idx[on_axis]

        if len(interior_nan_idx) > 0:
            lam_fine[interior_nan_idx] = lam_interp_nn(
                fine_pts[interior_nan_idx, 0], fine_pts[interior_nan_idx, 1]
            )
        if len(boundary_nan_idx) > 0:
            lam_fine[boundary_nan_idx] = float(np.nanmax(lam_min_raw)) + 1.0

    is_spinodal_fine = lam_fine < 0

    # ── figure ────────────────────────────────────────────────────────────
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
        """Convert (x_el0, x_el1) → python-ternary (right=el0, top=el1, left=ref)."""
        xref = 1.0 - xa - xb
        return (xa, xb, xref)  # (right=el0, top=el1, left=ref)

    from .spinodal import _points_in_triangle

    # Build mask for points inside any tie-triangle
    in_triangle = np.zeros(len(fine_pts), dtype=bool)
    for tt in tie_triangles:
        in_triangle |= _points_in_triangle(fine_pts[:, 0], fine_pts[:, 1], tt)

    # ── scatter markers by region ─────────────────────────────────────────
    # Colorblind-friendly palette (Wong 2011):
    # blue, orange, green, vermillion
    COLORS = {
        "single":   ("#0072B2", 0.70),  # blue
        "two":      ("#E69F00", 0.80),  # orange
        "three":    ("#009E73", 0.80),  # bluish-green
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
        tax.scatter(tern_pts, marker="o", s=marker_size, color=color,
                    alpha=alpha, zorder=3 + zi)

    # ── spinodal and binodal contour lines ───────────────────────────────
    # The fine grid stops at ~0.01 from the ternary edges (NaN filtered).
    # Without boundary anchors, tricontour is forced to close every contour
    # inside the domain producing unreasonable closed loops.
    # Fix: append a ring of points along the three ternary edges (x0+x1=1,
    # x0=0, x1=0) with known boundary values so contours exit to the axes.
    import matplotlib.tri as mtri

    n_edge = 80
    t_edge = np.linspace(0.0, 1.0, n_edge + 1)
    # Three edges: bottom (x1=0), left (x0=0), hypotenuse (x0+x1=1)
    edge_x0 = np.concatenate([t_edge,           np.zeros(n_edge + 1), t_edge[::-1]])
    edge_x1 = np.concatenate([np.zeros(n_edge + 1), t_edge,           1.0 - t_edge[::-1]])

    def _augmented_contour(x0_pts, x1_pts, values,
                           edge_val_lam, edge_val_bin, field, level):
        """
        Add ternary-edge anchor points then draw the contour at *level*.

        edge_val_* are the scalar values assigned to every edge anchor point.
        Physically correct anchors:
          - lam field:  large positive (stable at pure-element limits)
          - bin field:  0.0 (single-phase at pure-element limits)
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
            return (tri_x[:, i]-tri_x[:, j])**2 + (tri_y[:, i]-tri_y[:, j])**2
        max_sq = np.maximum(_sq(0,1), np.maximum(_sq(1,2), _sq(0,2)))
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
        """
        Chain disconnected path segments that share endpoints into longer
        continuous paths, eliminating overlapping strokes at junctions.
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
        """
        Split a contour path into sub-segments that each run through the
        interior, stopping the moment a point lands on or beyond a ternary
        edge.  This ensures boundary lines only *terminate* at an axis —
        they never run *along* one.
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
            return (tri_x[:,i]-tri_x[:,j])**2 + (tri_y[:,i]-tri_y[:,j])**2
        max_sq = np.maximum(_sq(0,1), np.maximum(_sq(1,2), _sq(0,2)))
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
    sp_paths = _simple_contour(
        fine_pts[lam_valid, 0], fine_pts[lam_valid, 1], lam_fine[lam_valid], level=0.0
    )
    _plot_contour(sp_paths, color="k", linewidth=1.5, linestyle="--")

    # Binodal line: temporarily disabled
    # is_multiphase = (region_fine != "single").astype(float)
    # bin_paths = _augmented_contour(
    #     fine_pts[:, 0], fine_pts[:, 1], is_multiphase,
    #     edge_val_lam=lam_edge_val, edge_val_bin=0.0,
    #     field="bin", level=0.5
    # )
    # _plot_contour(bin_paths, color="k", linewidth=1.5, linestyle="-")

    # ── three-phase tie-triangle boundary ────────────────────────────────
    for tt in tie_triangles:
        tern_verts = [_tern(tt[i, 0], tt[i, 1]) for i in range(3)]
        tax.plot(tern_verts + [tern_verts[0]],
                 color="#006d50", linewidth=1.8, zorder=7, linestyle="-")
        tax.scatter(tern_verts, marker="D", s=marker_size * 2,
                    color="#006d50", zorder=8)

    # ── simulation star ───────────────────────────────────────────────────
    if sim_composition is not None:
        sc_x0 = sim_composition.get(independent_elements[0], 0.0)
        sc_x1 = sim_composition.get(independent_elements[1], 0.0)
        tax.scatter([_tern(sc_x0, sc_x1)], marker="*", s=50,
                    color="k", zorder=10)

    # ── corner labels ─────────────────────────────────────────────────────
    # Font sizes tuned for publication figures (readable at reduced scale
    # when placed as a sub-panel).
    CORNER_LABEL_FONTSIZE = 17
    TITLE_FONTSIZE        = 13
    TICK_FONTSIZE         = 11
    LEGEND_FONTSIZE       = 9

    if corner_labels is None:
        # right = el0, top = el1, left = ref
        corner_labels = [
            independent_elements[0].title(),
            independent_elements[1].title(),
            ref_el.title(),
        ]
    tax.right_corner_label(corner_labels[0], fontsize=CORNER_LABEL_FONTSIZE, offset=0.12)
    tax.top_corner_label(corner_labels[1],   fontsize=CORNER_LABEL_FONTSIZE, offset=0.20)
    tax.left_corner_label(corner_labels[2],  fontsize=CORNER_LABEL_FONTSIZE, offset=0.18)

    tax.set_title(
        f"Ternary phase map — {T_C:.0f} °C", fontsize=TITLE_FONTSIZE, pad=30
    )
    tax.ticks(axis="lbr", linewidth=0.6, multiple=0.2,
              tick_formats="%.1f", fontsize=TICK_FONTSIZE, offset=0.025)
    tax.clear_matplotlib_ticks()
    ax.axis("off")

    # ── legend ────────────────────────────────────────────────────────────
    handles = [
        Patch(facecolor=COLORS["single"][0],   alpha=COLORS["single"][1],   label="single-phase"),
        Patch(facecolor=COLORS["two"][0],       alpha=COLORS["two"][1],      label="two-phase"),
        Patch(facecolor=COLORS["three"][0],     alpha=COLORS["three"][1],    label="three-phase"),
        Line2D([0], [0], color="k", linewidth=1.4, linestyle="--",
               label="spinodal boundary"),
    ]
    if sim_composition is not None:
        handles.append(
            Line2D([0], [0], marker="*", color="k", markersize=6,
                   ls="none", label="simulated composition")
        )
    ax.legend(handles=handles, fontsize=LEGEND_FONTSIZE, frameon=True,
              loc="upper center", ncol=2, handletextpad=0.3,
              borderpad=0.4, columnspacing=0.7,
              bbox_to_anchor=(0.5, 0.02))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved ternary phase map to {save_path}")

    if show:
        plt.show()

    return ax


# ── AS-UMAP: alloy-space dimensionality reduction ─────────────────────────────

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
    """
    Alloy-Space dimensionality reduction plot for quaternary and higher systems.

    Implements the AS-UMAP method (Vela et al., arXiv:2408.07681): grid-samples
    the full n-component composition simplex, fits a non-linear dimensionality
    reduction (UMAP when available, t-SNE otherwise) on that full space, then
    overlays CALPHAD data points colored by phase region and spinodal label.

    The simplex background reveals the full topology of the alloy space so
    data coverage and phase-region boundaries can be read against a consistent
    compositional reference frame.

    Parameters
    ----------
    table_path : str or Path
        HDF5 table from generate_table().
    independent_elements : list[str]
        All elements except the reference element.
    reference_element : str
        Reference (dependent) element.
    corner_labels : list[str], optional
        Labels for each element corner annotation (same order as elements
        stored in the table).  Defaults to element symbols.
    sim_composition : dict[str, float], optional
        Simulated composition {element: x} — plotted as a star.
    umap_scale : int, optional
        Simplex lattice resolution for the background grid.  Larger gives a
        denser background but slower embedding.  Defaults to max(8, 24 // n_el).
    figsize : tuple
    save_path : str or Path, optional
    show : bool

    Returns
    -------
    matplotlib.figure.Figure
    """
    from .spinodal import compute_phase_regions_multi

    data     = load_table(table_path)
    elements = data["elements"]
    n_el     = len(elements)
    T_C      = data["metadata"].get("temperature", 273.15) - 273.15

    # ── phase-region classification ───────────────────────────────────────
    result      = compute_phase_regions_multi(
        table_path, independent_elements, reference_element
    )
    comps       = result["compositions"]
    region      = result["region"].copy()
    is_spinodal = result["spinodal"]
    region[is_spinodal] = "spinodal"

    # ── full simplex background for AS embedding ──────────────────────────
    # Minimum lattice scale of 22 (vs. the lower bound used for n_el<=2)
    # keeps the hypocycloid boundary dense enough to read clearly even for
    # quaternary+ systems, where 60 // n_el alone would be too coarse.
    MIN_SIMPLEX_SCALE = 22
    scale       = umap_scale or max(MIN_SIMPLEX_SCALE, 60 // n_el)
    simplex_pts = _full_simplex_lattice(n_el, scale)
    n_simplex   = len(simplex_pts)
    n_data      = len(comps)

    extra_pts = []
    sim_idx   = None
    if sim_composition is not None:
        sim_vec = np.array([sim_composition.get(el, 0.0) for el in elements])
        extra_pts.append(sim_vec)
        sim_idx = n_simplex + n_data

    all_pts = np.vstack([simplex_pts, comps] + extra_pts)

    # ── dimensionality reduction ──────────────────────────────────────────
    method_name, embedding = _fit_dim_reduction(all_pts)
    simplex_emb = embedding[:n_simplex]
    data_emb    = embedding[n_simplex:n_simplex + n_data]
    sim_emb     = embedding[sim_idx] if sim_idx is not None else None

    # ── locate pure-element corners in the background ─────────────────────
    pure_embs = {}
    for i, el in enumerate(elements):
        pure_comp     = np.zeros(n_el); pure_comp[i] = 1.0
        dists         = np.linalg.norm(simplex_pts - pure_comp, axis=1)
        pure_embs[el] = simplex_emb[int(np.argmin(dists))]

    # ── rotate embedding to canonical n-gon orientation ───────────────────
    # n=3 → equilateral triangle, one vertex at top (△)
    # n=4 → square with flat sides, one vertex at top-left (■)
    # n≥5 → regular polygon with one vertex at top
    rot_angle, centroid = _canonical_rotation(pure_embs, elements, n_el)
    simplex_emb = _rotate_embedding(simplex_emb, rot_angle, centroid)
    data_emb    = _rotate_embedding(data_emb,    rot_angle, centroid)
    if sim_emb is not None:
        sim_emb = _rotate_embedding(sim_emb[np.newaxis], rot_angle, centroid)[0]
    for el in elements:
        pure_embs[el] = _rotate_embedding(
            pure_embs[el][np.newaxis], rot_angle, centroid)[0]

    # ── figure ────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=figsize)

    ax.scatter(simplex_emb[:, 0], simplex_emb[:, 1],
               c="#d8d8d8", s=4, alpha=0.6, zorder=1,
               rasterized=True, linewidths=0)

    # Draw the hypocycloid boundary: concave hull of the background simplex.
    # Use alpha-shape via Delaunay triangulation — keeps only boundary edges
    # whose circumradius is below the alpha threshold.
    _draw_simplex_boundary(ax, simplex_emb, alpha_frac=0.6)

    _COLORS = {
        "single":   ("#2ca02c", 0.80),
        "two":      ("#ff7f0e", 0.80),
        "three":    ("#1f77b4", 0.80),
        "four":     ("#9467bd", 0.80),
        "multi":    ("#8c564b", 0.80),
        "spinodal": ("#d62728", 0.90),
    }
    _LABELS = {
        "single":   "Single phase",
        "two":      "Two phase",
        "three":    "Three phase",
        "four":     "Four phase",
        "multi":    "Multi phase",
        "spinodal": "Spinodal",
    }
    for reg in ("single", "two", "three", "four", "multi", "spinodal"):
        mask = region == reg
        if not mask.any():
            continue
        color, alpha = _COLORS[reg]
        ax.scatter(data_emb[mask, 0], data_emb[mask, 1],
                   c=color, s=22, alpha=alpha, zorder=3,
                   linewidths=0, label=_LABELS[reg])

    if sim_emb is not None:
        ax.scatter([sim_emb[0]], [sim_emb[1]], marker="*", s=160,
                   c="gold", edgecolors="k", linewidths=0.8,
                   zorder=5, label="Simulated composition")

    CORNER_ANNOTATION_FONTSIZE = 13

    lab_els = corner_labels if corner_labels is not None else elements
    for el, lab in zip(elements, lab_els):
        ex, ey = pure_embs[el]
        ax.annotate(
            lab, (ex, ey), fontsize=CORNER_ANNOTATION_FONTSIZE,
            fontweight="bold", ha="center",
            bbox=dict(boxstyle="round,pad=0.25", fc="white",
                      ec="0.7", alpha=0.85, lw=0.6),
        )

    el_str = "–".join(e.title() for e in elements)
    ax.set_title(f"Alloy space ({el_str})  @  {T_C:.0f} °C", fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_frame_on(False)
    ax.legend(fontsize=8, loc="best", frameon=True,
              markerscale=1.2, handletextpad=0.4)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved AS-{method_name} figure: {save_path}")

    if show:
        plt.show()

    return fig


# ── private helpers ───────────────────────────────────────────────────────────

def _full_simplex_lattice(n_el: int, scale: int) -> np.ndarray:
    """Stars-and-bars lattice covering the full n-el composition simplex."""
    from itertools import combinations
    rows = []
    for c in combinations(range(scale + n_el - 1), n_el - 1):
        ext    = (-1,) + c + (scale + n_el - 1,)
        coords = tuple(ext[k + 1] - ext[k] - 1 for k in range(n_el))
        rows.append([v / scale for v in coords])
    return np.array(rows, dtype=float)


def _draw_simplex_boundary(ax: plt.Axes, emb: np.ndarray,
                           color: str = "#888888",
                           lw: float = 1.2,
                           alpha_frac: float = 0.25) -> None:
    """
    Draw the concave boundary (alpha-shape) of a 2-D point cloud on ax.

    Uses Delaunay triangulation: a boundary edge belongs to exactly one
    triangle.  The alpha parameter is set to alpha_frac × the span of the
    embedding, which empirically gives the hypocycloid outline.
    """
    from scipy.spatial import Delaunay
    from matplotlib.collections import LineCollection

    if len(emb) < 4:
        return

    tri   = Delaunay(emb)
    span  = max(np.ptp(emb[:, 0]), np.ptp(emb[:, 1]))
    alpha = alpha_frac * span         # circumradius threshold

    boundary_edges: set[tuple[int, int]] = set()
    for simplex in tri.simplices:
        for i in range(3):
            e = tuple(sorted([simplex[i], simplex[(i + 1) % 3]]))
            boundary_edges ^= {e}     # XOR: keep edges belonging to exactly 1 triangle

    # Also filter by circumradius to get the concave (alpha-shape) boundary
    filtered: list[tuple] = []
    for e in boundary_edges:
        pts_e = emb[list(e)]
        # Circumradius of the degenerate 1-simplex (edge): half its length
        r = 0.5 * np.linalg.norm(pts_e[0] - pts_e[1])
        if r <= alpha:
            filtered.append(e)

    if not filtered:
        # Fall back to convex hull
        from scipy.spatial import ConvexHull
        hull = ConvexHull(emb)
        for s in hull.simplices:
            filtered.append(tuple(s))

    segs = [(emb[e[0]], emb[e[1]]) for e in filtered]
    lc   = LineCollection(segs, colors=color, linewidths=lw,
                           alpha=0.75, zorder=2)
    ax.add_collection(lc)


def _fit_dim_reduction(pts: np.ndarray) -> tuple[str, np.ndarray]:
    """
    Fit a 2-D non-linear embedding.  Tries UMAP first; falls back to t-SNE.
    Returns (method_name, embedding) where embedding has shape (n_pts, 2).

    ── UMAP tuning parameters ───────────────────────────────────────────────
    UMAP_N_NEIGHBORS : int  (default 30)
        Number of nearest neighbors used to build the local manifold graph.
        Controls the balance between local and global structure.
        • Increase (e.g. 50–100) → smoother, more globally coherent embedding;
          hypocycloid cusps become more distinct.
        • Decrease (e.g. 5–15)  → finer local detail but cusps may fragment
          and the simplex topology can break apart.
        Must be large enough to "see" across the full simplex (≥ 20 is safe).

    UMAP_MIN_DIST : float  (default 0.1)
        Minimum distance between points in the 2-D embedding.
        • Decrease (e.g. 0.0–0.05) → tighter clusters, deeper concavities,
          more pronounced hypocycloid shape.
        • Increase (e.g. 0.3–0.5)  → points spread out; shape inflates toward
          a convex blob and the cusps disappear.

    UMAP_RANDOM_STATE : int  (default 42)
        Random seed for reproducibility.  Change to get a different (equally
        valid) orientation before the canonical-rotation step corrects it.

    ── t-SNE fallback parameters (used only when umap-learn is not installed) ─
    TSNE_MAX_ITER : int  (default 1000)
        Number of optimisation iterations.  Increase for smoother layouts on
        large datasets.
    ─────────────────────────────────────────────────────────────────────────
    """
    UMAP_N_NEIGHBORS  = 25
    UMAP_MIN_DIST     = 0.4
    UMAP_RANDOM_STATE = 42
    TSNE_MAX_ITER     = 1000

    try:
        import umap as _umap
        reducer = _umap.UMAP(n_components=2,
                              n_neighbors=UMAP_N_NEIGHBORS,
                              min_dist=UMAP_MIN_DIST,
                              random_state=UMAP_RANDOM_STATE)
        return "UMAP", reducer.fit_transform(pts)
    except Exception:
        pass

    from sklearn.manifold import TSNE
    perp    = min(30.0, max(5.0, len(pts) / 4.0 - 1.0))
    reducer = TSNE(n_components=2, perplexity=perp,
                   max_iter=TSNE_MAX_ITER, random_state=UMAP_RANDOM_STATE)
    return "t-SNE", reducer.fit_transform(pts)


def _canonical_rotation(
    pure_embs: dict,
    elements: list[str],
    n_el: int,
) -> tuple[float, np.ndarray]:
    """Return (rotation_angle_rad, centroid) to align the n-gon of pure-element
    embedding positions to the canonical orientation:

    - n=3 (ternary)   : equilateral triangle, one vertex at top (90°)  →  △
    - n=4 (quaternary): square with flat sides, vertex at top-left (135°) → ■
    - n≥5             : regular polygon, first vertex at top (90°)
    """
    positions = np.array([pure_embs[el] for el in elements])
    centroid  = positions.mean(axis=0)
    centered  = positions - centroid

    current_angles = np.arctan2(centered[:, 1], centered[:, 0])

    # Sort elements by current angle so we can match them cyclically to the
    # canonical n-gon vertices.
    order = np.argsort(current_angles)
    sorted_angles = current_angles[order]

    # Canonical starting angle and step:
    # n=4 → ■ orientation: vertices at 135°, 45°, -45°, -135°  (step = -90°)
    # all other n → vertex at top (90°), step = 360°/n clockwise
    if n_el == 4:
        canonical_start = 3.0 * np.pi / 4.0   # 135°
    else:
        canonical_start = np.pi / 2.0          # 90°

    canonical_angles = canonical_start - 2.0 * np.pi * np.arange(n_el) / n_el

    # Try all n cyclic shifts and pick the one that minimises total angular error.
    best_rotation = 0.0
    best_error    = np.inf
    for shift in range(n_el):
        target = np.roll(canonical_angles, shift)
        diffs  = target - sorted_angles
        diffs  = (diffs + np.pi) % (2.0 * np.pi) - np.pi   # wrap to (-π, π]
        error  = float(np.sum(diffs ** 2))
        if error < best_error:
            best_error    = error
            best_rotation = float(diffs.mean())

    return best_rotation, centroid


def _rotate_embedding(emb: np.ndarray, angle: float,
                      centroid: np.ndarray) -> np.ndarray:
    """Apply a 2-D rotation of *angle* radians around *centroid*."""
    c, s = np.cos(angle), np.sin(angle)
    R    = np.array([[c, -s], [s, c]])
    return (emb - centroid) @ R.T + centroid
