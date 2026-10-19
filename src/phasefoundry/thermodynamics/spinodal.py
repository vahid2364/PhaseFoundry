"""Spinodal and binodal boundaries, phase-region maps and phase-field parameter estimates.

A composition is inside the spinodal when the Gibbs energy is concave in composition. For
a binary A-B system with B as the reference element the condition is d²G/dx_A² < 0, with
d²G/dx_A² = dMU[A_A] − dMU[B_A] from the stored Jansson derivatives. For a multicomponent
system the condition is that the Hessian H_ij = dMU[i_j] − dMU[ref_j] = ∂²G/∂c_i∂c_j has
at least one negative eigenvalue.

The binodal follows from equal chemical potentials in the two phases, and the phase-region
maps from the lower convex hull of G_m. All functions read HDF5 tables written by
``generate_table``; the tables store temperatures in K, and the results report them in °C.
"""

from __future__ import annotations

import logging
import warnings
from typing import TYPE_CHECKING

import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import fsolve

from phasefoundry.thermodynamics.storage import load_table

if TYPE_CHECKING:
    from pathlib import Path

_logger = logging.getLogger(__name__)


def binodal_boundary_binary(
    table_paths: list[str | Path],
    independent_element: str,
    temperatures_celsius: list[float] | None = None,
    residual_tol: float = 500.0,
    min_separation: float = 0.15,
) -> dict:
    """Compute the binodal (solvus) compositions of a binary system at each temperature.

    Two phases α and β are in equilibrium when μ_A(x_α) = μ_A(x_β) and
    μ_B(x_α) = μ_B(x_β). The function solves these with ``scipy.optimize.fsolve`` on cubic
    interpolants of the tabulated chemical potentials, trying a fixed list of starting pairs
    on both sides of the two-phase region. The first solution is kept for which ``fsolve``
    converges, the residual norm is below ``residual_tol``, the two compositions are at
    least ``min_separation`` apart, and both lie in (0.001, 0.999).

    Args:
        table_paths: HDF5 tables, one per temperature, with ``MU`` data.
        independent_element: Element whose mole fraction is the composition axis, e.g.
            ``"NB"``. The other element is the first table element that differs from it.
        temperatures_celsius: Temperature labels in °C, one per table. Defaults to the table
            temperatures.
        residual_tol: Largest accepted norm of the two residuals, in J/mol.
        min_separation: Smallest accepted gap between the two compositions, in mole fraction.

    Returns:
        A dict with ``"temperatures_celsius"``, ``"x_binodal_low"`` and
        ``"x_binodal_high"``, each with one entry per table. The compositions are mole
        fractions of ``independent_element``, or ``None`` where no solution was accepted.

    Raises:
        KeyError: If a table has no ``MU`` data.
    """
    _INIT_GUESSES = [
        (0.05, 0.95),
        (0.10, 0.90),
        (0.15, 0.85),
        (0.05, 0.85),
        (0.10, 0.95),
        (0.20, 0.80),
        (0.03, 0.97),
        (0.07, 0.93),
    ]

    temps, x_lo, x_hi = [], [], []

    for i, path in enumerate(table_paths):
        data = load_table(path)
        elements = data["elements"]

        if "MU" not in data:
            raise KeyError(f"MU not found in {path}. Re-generate with quantities=(..., 'MU').")

        T_C = (
            temperatures_celsius[i] if temperatures_celsius is not None else data["metadata"].get("temperature", 273.15) - 273.15
        )
        temps.append(T_C)

        col = elements.index(independent_element)
        x = data["compositions"][:, col]
        mu_A = data["MU"][independent_element]
        ref_el = [e for e in elements if e != independent_element][0]
        mu_B = data["MU"][ref_el]

        order = np.argsort(x)
        x, mu_A, mu_B = x[order], mu_A[order], mu_B[order]

        f_A = interp1d(x, mu_A, kind="cubic", fill_value="extrapolate")
        f_B = interp1d(x, mu_B, kind="cubic", fill_value="extrapolate")

        def residuals(xy):
            xa, xb = xy
            return [f_A(xa) - f_A(xb), f_B(xa) - f_B(xb)]  # noqa: B023

        found_lo = found_hi = None

        for x0_lo, x0_hi in _INIT_GUESSES:
            try:
                sol, _, ier, _ = fsolve(residuals, [x0_lo, x0_hi], full_output=True)
                xa, xb = float(sol[0]), float(sol[1])
                res = residuals([xa, xb])
                resid = (res[0] ** 2 + res[1] ** 2) ** 0.5

                if (
                    ier == 1
                    and resid < residual_tol
                    and xa < xb
                    and xb - xa >= min_separation
                    and 0.001 < xa < 0.999
                    and 0.001 < xb < 0.999
                ):
                    found_lo = xa
                    found_hi = xb
                    break
            except Exception:
                continue

        x_lo.append(found_lo)
        x_hi.append(found_hi)

    return {
        "temperatures_celsius": temps,
        "x_binodal_low": x_lo,
        "x_binodal_high": x_hi,
    }


def spinodal_boundary_binary(
    table_paths: list[str | Path],
    independent_element: str,
    temperatures_celsius: list[float] | None = None,
    reference_element: str | None = None,
) -> dict:
    """Compute the spinodal compositions of a binary system at each temperature.

    The spinodal limits are the zero crossings of d²G/dx², found by linear interpolation
    between neighbouring table points. Compositions between the two limits are unstable to
    spinodal decomposition. With more than two crossings the outermost two are used; with
    exactly one, the upper limit is ``None`` and a warning is issued.

    Args:
        table_paths: HDF5 tables, one per temperature, with ``dMU`` data.
        independent_element: Element whose mole fraction is the composition axis, e.g.
            ``"NB"`` for Nb-V.
        temperatures_celsius: Temperature labels in °C, one per table. Defaults to the table
            temperatures.
        reference_element: The other element of the binary. When given, and its ``dMU`` key
            exists, d²G/dx² = dMU[A_A] − dMU[B_A]. Without it the function uses dMU[A_A]
            alone, which by Gibbs-Duhem is (1 − x_A) d²G/dx²: the zero crossings are the
            same, but the values in ``"d2G_dx2"`` differ.

    Returns:
        A dict with one entry per table under each key:

        - ``"temperatures_celsius"``: temperature in °C.
        - ``"x_spinodal_low"``, ``"x_spinodal_high"``: spinodal limits in mole fraction of
          ``independent_element``, or ``None`` where none was found.
        - ``"x_compositions"``: the table compositions, sorted.
        - ``"d2G_dx2"``: d²G/dx² in J/mol at those compositions.

    Raises:
        KeyError: If a table lacks ``dMU`` or its ``<A>_<A>`` key.
    """
    key = f"{independent_element}_{independent_element}"

    temps, x_lo, x_hi, all_x, all_d2g = [], [], [], [], []

    for i, path in enumerate(table_paths):
        data = load_table(path)
        elements = data["elements"]

        if "dMU" not in data or key not in data["dMU"]:
            raise KeyError(f"dMU/{key} not found in {path}. Re-generate the table with quantities=(..., 'dMU').")

        col = elements.index(independent_element)
        x = data["compositions"][:, col]

        d2g = data["dMU"][key]
        if reference_element is not None:
            ref_key = f"{reference_element}_{independent_element}"
            if ref_key in data["dMU"]:
                d2g = d2g - data["dMU"][ref_key]

        order = np.argsort(x)
        x, d2g = x[order], d2g[order]

        T_C = (
            temperatures_celsius[i] if temperatures_celsius is not None else data["metadata"].get("temperature", 273.15) - 273.15
        )
        temps.append(T_C)
        all_x.append(x)
        all_d2g.append(d2g)

        sign = np.sign(d2g)
        crossings = np.where(np.diff(sign) != 0)[0]

        boundaries = []
        for idx in crossings:
            x0, x1 = x[idx], x[idx + 1]
            f0, f1 = d2g[idx], d2g[idx + 1]
            if f1 != f0:
                x_zero = x0 - f0 * (x1 - x0) / (f1 - f0)
                boundaries.append(float(x_zero))

        if len(boundaries) >= 2:
            x_lo.append(min(boundaries))
            x_hi.append(max(boundaries))
        elif len(boundaries) == 1:
            x_lo.append(boundaries[0])
            x_hi.append(None)
            warnings.warn(f"Only one spinodal boundary found at {T_C} °C.", stacklevel=2)
        else:
            x_lo.append(None)
            x_hi.append(None)

    return {
        "temperatures_celsius": temps,
        "x_spinodal_low": x_lo,
        "x_spinodal_high": x_hi,
        "x_compositions": all_x,
        "d2G_dx2": all_d2g,
    }


def spinodal_boundary_multi(
    table_paths: list[str | Path],
    independent_elements: list[str],
    temperatures_celsius: list[float] | None = None,
    reference_element: str | None = None,
) -> dict:
    """Find the spinodally unstable table points of a multicomponent system.

    At each table point the Hessian H_ij = dMU[i_j] − dMU[ref_j] is assembled from the
    Jansson derivatives, and the point is unstable when the smallest eigenvalue is negative.

    Args:
        table_paths: HDF5 tables, one per temperature.
        independent_elements: The N - 1 independent elements.
        temperatures_celsius: Temperature labels in °C, one per table. Defaults to the table
            temperatures.
        reference_element: Dependent element. When given, dMU[ref_j] is subtracted
            (Gibbs-Duhem); without it H_ij = dMU[i_j].

    Returns:
        A dict with one entry per table under each key:

        - ``"temperatures_celsius"``: temperature in °C.
        - ``"compositions"``: table compositions, shape ``(n_pts, n_elements)``.
        - ``"min_eigenvalue"``: smallest Hessian eigenvalue in J/mol, shape ``(n_pts,)``;
          NaN where ``dMU`` data is missing or NaN.
        - ``"is_spinodal"``: ``min_eigenvalue < 0``, shape ``(n_pts,)``.
    """
    n = len(independent_elements)
    results = {
        "temperatures_celsius": [],
        "compositions": [],
        "min_eigenvalue": [],
        "is_spinodal": [],
    }

    for i, path in enumerate(table_paths):
        data = load_table(path)

        T_C = (
            temperatures_celsius[i] if temperatures_celsius is not None else data["metadata"].get("temperature", 273.15) - 273.15
        )
        results["temperatures_celsius"].append(T_C)
        results["compositions"].append(data["compositions"])

        n_pts = len(data["compositions"])
        min_eigs = np.full(n_pts, np.nan)

        for pt in range(n_pts):
            H = np.zeros((n, n))
            valid = True
            for ri, el1 in enumerate(independent_elements):
                for ci, el2 in enumerate(independent_elements):
                    key = f"{el1}_{el2}"
                    if "dMU" not in data or key not in data["dMU"]:
                        valid = False
                        break
                    val = data["dMU"][key][pt]
                    if reference_element is not None:
                        ref_key = f"{reference_element}_{el2}"
                        if ref_key in data["dMU"]:
                            val -= data["dMU"][ref_key][pt]
                    H[ri, ci] = val
                if not valid:
                    break

            if valid and not np.any(np.isnan(H)):
                eigvals = np.linalg.eigvalsh(H)
                min_eigs[pt] = eigvals.min()

        results["min_eigenvalue"].append(min_eigs)
        results["is_spinodal"].append(min_eigs < 0)

    return results


def compute_ternary_phase_regions(
    table_path: str | Path,
    independent_elements: list[str],
    n_fine: int = 150,
    driving_force_tol_frac: float = 5e-4,
    min_triangle_area: float = 0.05,
) -> dict:
    """Classify ternary compositions as single-, two- or three-phase with the lower convex hull of G_m.

    G_m is interpolated linearly from the table onto a triangular grid with ``n_fine`` steps
    per axis, with nearest-neighbour values outside the hull of the data, so the map covers
    the whole simplex. The lower convex hull of (x_0, x_1, G_m) on that grid then gives:

    - single-phase: points whose driving force (G_m minus the hull) is below
      ``driving_force_tol_frac`` times the G_m range, with the range taken as at least
      1 J/mol;
    - three-phase: points inside a lower-hull facet whose area in the (x_0, x_1) plane is
      at least ``min_triangle_area`` (a tie-triangle);
    - two-phase: all other points.

    Each table point takes the region of the nearest grid point, and its driving force is
    measured against the hull planes.

    Args:
        table_path: HDF5 table with ``GM`` data.
        independent_elements: The two independent elements, e.g. ``["NB", "HF"]`` for Nb-Hf-V.
        n_fine: Grid steps per axis. The default of 150 gives about 11 500 grid points.
        driving_force_tol_frac: Single-phase threshold as a fraction of the G_m range.
        min_triangle_area: Smallest facet area, in squared mole fraction, counted as a
            tie-triangle. Smaller facets are discretization artifacts.

    Returns:
        A dict with:

        - ``"compositions"``: table compositions, shape ``(n_pts, n_el)``.
        - ``"gm"``: G_m in J/mol, shape ``(n_pts,)``.
        - ``"driving_force"``: G_m minus the hull in J/mol, shape ``(n_pts,)``.
        - ``"region"``: ``"single"``, ``"two"`` or ``"three"`` for each table point.
        - ``"tie_triangles"``: vertices of each tie-triangle, arrays of shape ``(3, 2)``
          with columns ``(x_0, x_1)``.
        - ``"elements"``, ``"independent_elements"``: element lists.
        - ``"temperature_C"``: temperature in °C.
        - ``"fine_grid"``: dict with ``"x0"``, ``"x1"``, ``"gm"``, ``"driving_force"`` and
          ``"region"`` on the interpolation grid.
    """
    from scipy.interpolate import LinearNDInterpolator
    from scipy.spatial import ConvexHull

    data = load_table(table_path)
    elements = data["elements"]
    comps = data["compositions"]
    gm = data["GM"]
    T_C = data["metadata"].get("temperature", 273.15) - 273.15

    idx = [elements.index(el) for el in independent_elements]
    x0_raw = comps[:, idx[0]]
    x1_raw = comps[:, idx[1]]

    # LinearNDInterpolator is NaN outside the data hull; nearest-neighbour fills the rest of the simplex
    from scipy.interpolate import NearestNDInterpolator

    raw_pts = list(zip(x0_raw, x1_raw, strict=True))
    gm_interp_lin = LinearNDInterpolator(raw_pts, gm)
    gm_interp_nn = NearestNDInterpolator(raw_pts, gm)

    fine_pts = []
    for i in range(n_fine + 1):
        for j in range(n_fine + 1 - i):
            xa = i / n_fine
            xb = j / n_fine
            if xa + xb <= 1.0:
                fine_pts.append((xa, xb))
    fine_pts = np.array(fine_pts)
    gm_fine = gm_interp_lin(fine_pts[:, 0], fine_pts[:, 1])

    nan_mask = np.isnan(gm_fine)
    if nan_mask.any():
        gm_fine[nan_mask] = gm_interp_nn(fine_pts[nan_mask, 0], fine_pts[nan_mask, 1])

    valid = ~np.isnan(gm_fine)
    fine_pts = fine_pts[valid]
    gm_fine = gm_fine[valid]
    x0_fine = fine_pts[:, 0]
    x1_fine = fine_pts[:, 1]

    pts_3d = np.column_stack([x0_fine, x1_fine, gm_fine])
    hull = ConvexHull(pts_3d)

    lower_mask = hull.equations[:, 2] < 0
    lower_eqs = hull.equations[lower_mask]
    lower_simplices = hull.simplices[lower_mask]

    gm_hull_fine = np.full(len(x0_fine), -np.inf)
    for eq in lower_eqs:
        nx, ny, nz, d = eq
        z_plane = -(nx * x0_fine + ny * x1_fine + d) / nz
        gm_hull_fine = np.maximum(gm_hull_fine, z_plane)

    gm_range = float(np.nanmax(gm_fine) - np.nanmin(gm_fine))
    df_tol = driving_force_tol_frac * max(abs(gm_range), 1.0)
    df_fine = gm_fine - gm_hull_fine

    # Three-phase tie-triangles are the large lower-hull facets; small ones are discretization artifacts
    tie_triangles = []
    for s in lower_simplices:
        v0 = x0_fine[s]
        v1 = x1_fine[s]
        area = 0.5 * abs((v0[1] - v0[0]) * (v1[2] - v1[0]) - (v0[2] - v0[0]) * (v1[1] - v1[0]))
        if area >= min_triangle_area:
            tie_triangles.append(np.column_stack([v0, v1]))

    region_fine = np.where(df_fine < df_tol, "single", "two")
    for tt in tie_triangles:
        inside = _points_in_triangle(x0_fine, x1_fine, tt)
        region_fine[inside] = "three"

    # NearestNDInterpolator needs numeric values
    _enc = {"single": 0, "two": 1, "three": 2}
    _dec = {0: "single", 1: "two", 2: "three"}
    region_enc = np.array([_enc[r] for r in region_fine])

    from scipy.interpolate import NearestNDInterpolator

    region_nn = NearestNDInterpolator(list(zip(x0_fine, x1_fine, strict=True)), region_enc)

    gm_hull_raw = np.full(len(x0_raw), -np.inf)
    for eq in lower_eqs:
        nx, ny, nz, d = eq
        z_plane = -(nx * x0_raw + ny * x1_raw + d) / nz
        gm_hull_raw = np.maximum(gm_hull_raw, z_plane)
    df_raw = gm - gm_hull_raw

    region_raw = np.array([_dec[int(v)] for v in region_nn(x0_raw, x1_raw)])

    return {
        "compositions": comps,
        "gm": gm,
        "driving_force": df_raw,
        "region": region_raw,
        "tie_triangles": tie_triangles,
        "elements": elements,
        "independent_elements": independent_elements,
        "temperature_C": T_C,
        "fine_grid": {
            "x0": x0_fine,
            "x1": x1_fine,
            "gm": gm_fine,
            "driving_force": df_fine,
            "region": region_fine,
        },
    }


def compute_phase_regions_multi(
    table_path: str | Path,
    independent_elements: list[str],
    reference_element: str | None = None,
    driving_force_tol_frac: float = 5e-4,
    min_simplex_vol: float = 0.001,
) -> dict:
    """Classify the compositions of a system with three or more components into phase regions.

    The lower convex hull of (x_1, ..., x_n_ind, G_m) is built directly from the table
    points, without an interpolation grid. Points whose driving force is below
    ``driving_force_tol_frac`` times the G_m range are ``"single"``. Points inside a
    lower-hull facet whose volume in composition space is at least ``min_simplex_vol`` are
    labelled by the facet's vertex count: ``"three"`` for a ternary, ``"four"`` for a
    quaternary and ``"multi"`` beyond. All other points are ``"two"``. Separately, a point
    is flagged as spinodal when its Hessian dMU[i_j] − dMU[ref_j] has a negative eigenvalue.

    If the hull cannot be built, every point is returned as ``"single"`` with zero driving
    force and no spinodal points.

    Args:
        table_path: HDF5 table with ``GM`` data and, for the spinodal flag, ``dMU``.
        independent_elements: All elements except the reference element.
        reference_element: Dependent element. When given, dMU[ref_j] is subtracted in the
            Hessian (Gibbs-Duhem).
        driving_force_tol_frac: Single-phase threshold as a fraction of the G_m range.
        min_simplex_vol: Smallest facet volume in composition space counted as a
            multi-phase region.

    Returns:
        A dict with ``"compositions"`` (shape ``(n_pts, n_el)``), ``"gm"`` and
        ``"driving_force"`` (J/mol, shape ``(n_pts,)``), ``"region"`` (object array of
        labels), ``"spinodal"`` (bool array, all ``False`` without ``dMU``),
        ``"elements"``, ``"independent_elements"`` and ``"temperature_C"``.
    """
    from math import factorial

    from scipy.spatial import ConvexHull

    data = load_table(table_path)
    elements = data["elements"]
    comps = data["compositions"]
    gm = data["GM"]
    T_C = data["metadata"].get("temperature", 273.15) - 273.15

    n_ind = len(independent_elements)
    idx = [elements.index(el) for el in independent_elements]
    x_ind = comps[:, idx]

    pts_nd = np.column_stack([x_ind, gm])

    _fallback = {
        "compositions": comps,
        "gm": gm,
        "driving_force": np.zeros(len(gm)),
        "region": np.full(len(gm), "single"),
        "spinodal": np.zeros(len(gm), dtype=bool),
        "elements": elements,
        "independent_elements": independent_elements,
        "temperature_C": T_C,
    }
    try:
        hull = ConvexHull(pts_nd)
    except Exception:
        return _fallback

    # Lower facets have an outward normal pointing down the G_m axis
    lower_mask = hull.equations[:, n_ind] < 0
    lower_eqs = hull.equations[lower_mask]
    lower_simps = hull.simplices[lower_mask]

    gm_hull = np.full(len(gm), -np.inf)
    for eq in lower_eqs:
        n_x = eq[:n_ind]
        n_g = eq[n_ind]
        d = eq[-1]
        z_plane = -(x_ind @ n_x + d) / n_g
        gm_hull = np.maximum(gm_hull, z_plane)

    gm_range = float(np.nanmax(gm) - np.nanmin(gm))
    df_tol = driving_force_tol_frac * max(abs(gm_range), 1.0)
    df = gm - gm_hull
    region = np.where(df < df_tol, "single", "two").astype(object)

    for s in lower_simps:
        verts = x_ind[s]
        k = len(verts) - 1
        if k < 2:
            continue
        edges = verts[1:] - verts[0]
        G = edges @ edges.T
        vol = float(np.sqrt(max(np.linalg.det(G), 0.0))) / factorial(k)
        if vol >= min_simplex_vol:
            inside = _points_in_simplex_nd(x_ind, verts)
            label = {2: "two", 3: "three", 4: "four"}.get(k + 1, "multi")
            region[inside] = label

    n_pts = len(gm)
    is_spinodal = np.zeros(n_pts, dtype=bool)
    if "dMU" in data:
        for pt in range(n_pts):
            H, valid = np.zeros((n_ind, n_ind)), True
            for ri, el1 in enumerate(independent_elements):
                for ci, el2 in enumerate(independent_elements):
                    key = f"{el1}_{el2}"
                    if key not in data["dMU"]:
                        valid = False
                        break
                    val = data["dMU"][key][pt]
                    if reference_element is not None:
                        ref_key = f"{reference_element}_{el2}"
                        if ref_key in data["dMU"]:
                            val -= data["dMU"][ref_key][pt]
                    H[ri, ci] = val
                if not valid:
                    break
            if valid and not np.any(np.isnan(H)):
                is_spinodal[pt] = np.linalg.eigvalsh(H).min() < 0

    return {
        "compositions": comps,
        "gm": gm,
        "driving_force": df,
        "region": region,
        "spinodal": is_spinodal,
        "elements": elements,
        "independent_elements": independent_elements,
        "temperature_C": T_C,
    }


def _points_in_simplex_nd(
    points: np.ndarray,
    simplex_verts: np.ndarray,
) -> np.ndarray:
    """Return a mask of the points that lie inside a simplex.

    The barycentric coordinates come from a least-squares solve, so this works in any
    dimension. Points on the boundary count as inside.

    Args:
        points: Query points, shape ``(n_pts, n_dim)``.
        simplex_verts: Simplex vertices, shape ``(k + 1, n_dim)``.

    Returns:
        Boolean mask, shape ``(n_pts,)``; all ``False`` if the solve fails.
    """
    n_pts = len(points)
    T = (simplex_verts[1:] - simplex_verts[0]).T
    rhs = (points - simplex_verts[0]).T
    try:
        bary, _, _, _ = np.linalg.lstsq(T, rhs, rcond=None)
    except Exception:
        return np.zeros(n_pts, dtype=bool)
    bary_last = 1.0 - bary.sum(axis=0)
    all_bary = np.vstack([bary, bary_last])
    return np.all(all_bary >= -1e-9, axis=0)


def _points_in_triangle(
    px: np.ndarray,
    py: np.ndarray,
    triangle: np.ndarray,
) -> np.ndarray:
    """Return a mask of the points ``(px, py)`` that lie inside ``triangle``.

    Uses barycentric coordinates. Points on the boundary count as inside, and a degenerate
    triangle contains no points.

    Args:
        px: x coordinates, shape ``(n,)``.
        py: y coordinates, shape ``(n,)``.
        triangle: Vertices as rows of ``(x, y)``, shape ``(3, 2)``.

    Returns:
        Boolean mask, shape ``(n,)``.
    """
    ax, ay = triangle[0]
    bx, by = triangle[1]
    cx, cy = triangle[2]

    denom = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
    if abs(denom) < 1e-15:
        return np.zeros(len(px), dtype=bool)

    w1 = ((by - cy) * (px - cx) + (cx - bx) * (py - cy)) / denom
    w2 = ((cy - ay) * (px - cx) + (ax - cx) * (py - cy)) / denom
    w3 = 1.0 - w1 - w2

    return (w1 >= -1e-9) & (w2 >= -1e-9) & (w3 >= -1e-9)


def recommend_pfm_params(
    calphad_fit: dict,
    temperature_K: float,
    c0: float,
    n_interfaces: int = 10,
    delta_over_dx: float = 5.0,
    safety: float = 0.4,
    M: float | None = None,
) -> dict:
    """Suggest phase-field parameters from the G_m spline surrogate.

    The estimates use the Cahn-Hilliard length and time scales, with f'' = d²G/dx² sampled
    on x in [0.01, 0.99]:

        interface width        δ = sqrt(κ / |f''|_max)
        fastest-growing mode   λ* = 4π sqrt(κ / |f''(c0)|)
        critical wavelength    λ_c = 2π sqrt(2κ / |f''(c0)|)
        stable time step       dt = safety dx² / (M_eff,max |f''|_max)

    where M_eff,max = M / 4 is the peak of the degenerate mobility M c(1 − c). κ is chosen
    by a fixed-point iteration so that the interface spans ``delta_over_dx`` grid points
    on a domain of ``n_interfaces`` wavelengths λ*.

    Args:
        calphad_fit: Output of ``build_gm_spline_surrogate``. The stored temperature
            closest to ``temperature_K`` and its first model are used.
        temperature_K: Simulation temperature in K.
        c0: Mean initial composition, as mole fraction of the reference element (the
            spline's variable).
        n_interfaces: Number of wavelengths λ* the domain should hold.
        delta_over_dx: Grid points per interface width.
        safety: Safety factor for ``dt``.
        M: Mobility in m²·mol/(J·s), consistent with G in J/mol. Needed for ``dt_stable``.

    Returns:
        A dict with ``kappa_min`` (J·m²/mol); ``lambda_star``, ``lambda_crit``, ``delta``,
        ``Lx_min`` and ``dx`` (m); ``N_min`` (grid points, a power of 2); ``dt_stable`` (s,
        or ``None`` without ``M``); ``d2G_max`` and ``d2G_c0`` (J/mol); the spinodal limits
        ``spinodal_lo`` and ``spinodal_hi`` (``None`` unless at least two zero crossings
        are found); and ``c0_in_spinodal``.

    Raises:
        ValueError: If the fit has no ``spline_dfdx``, i.e. it was not built by
            ``build_gm_spline_surrogate``.
    """
    stored_T = np.array(calphad_fit["temperatures_K"])
    T_K = stored_T[int(np.argmin(np.abs(stored_T - temperature_K)))]
    el_key = list(calphad_fit["fits"][T_K]["models"].keys())[0]
    mdl = calphad_fit["fits"][T_K]["models"][el_key]

    if "spline_dfdx" not in mdl:
        raise ValueError(
            "recommend_pfm_params() requires a calphad_gm_spline surrogate "
            "(spline_dfdx key missing). Use build_gm_spline_surrogate()."
        )

    spl_dfdx = mdl["spline_dfdx"]
    x = np.linspace(0.01, 0.99, 2000)
    d2G = spl_dfdx.derivative()(x)

    d2G_max = float(np.max(np.abs(d2G)))
    d2G_c0 = float(abs(float(spl_dfdx.derivative()(c0))))

    sign = np.sign(d2G)
    crossings = np.where(np.diff(sign) != 0)[0]
    boundaries = []
    for idx in crossings:
        x0, x1 = x[idx], x[idx + 1]
        f0, f1 = d2G[idx], d2G[idx + 1]
        if f1 != f0:
            boundaries.append(float(x0 - f0 * (x1 - x0) / (f1 - f0)))
    sp_lo = min(boundaries) if len(boundaries) >= 2 else None
    sp_hi = max(boundaries) if len(boundaries) >= 2 else None
    c0_in_spinodal = sp_lo is not None and sp_hi is not None and sp_lo < c0 < sp_hi

    d2G_c0_safe = max(d2G_c0, 1e-10)
    d2G_max_safe = max(d2G_max, 1e-10)

    lambda_star = 4.0 * np.pi * np.sqrt(1.0 / d2G_c0_safe)
    lambda_crit = 2.0 * np.pi * np.sqrt(2.0 / d2G_c0_safe)

    N_guess = 256
    kappa_min = 1.0
    for _ in range(20):
        Lx_guess = n_interfaces * 4.0 * np.pi * np.sqrt(kappa_min / d2G_c0_safe)
        dx_guess = Lx_guess / N_guess
        delta_guess = delta_over_dx * dx_guess
        kappa_new = delta_guess**2 * d2G_max_safe
        lambda_star = 4.0 * np.pi * np.sqrt(kappa_new / d2G_c0_safe)
        lambda_crit = 2.0 * np.pi * np.sqrt(2.0 * kappa_new / d2G_c0_safe)
        Lx_min = n_interfaces * lambda_star
        dx = Lx_min / N_guess
        N_raw = Lx_min / dx
        N_min = int(2 ** np.ceil(np.log2(N_raw)))
        if abs(kappa_new - kappa_min) / max(abs(kappa_min), 1e-30) < 1e-6:
            kappa_min = kappa_new
            break
        kappa_min = kappa_new
        N_guess = N_min

    dx = Lx_min / N_min
    delta = delta_over_dx * dx

    dt_stable = None
    if M is not None:
        M_eff_max = M * 0.25  # c(1 - c) peaks at 0.25
        dt_stable = safety * dx**2 / (M_eff_max * d2G_max_safe)

    return {
        "kappa_min": kappa_min,
        "lambda_star": lambda_star,
        "lambda_crit": lambda_crit,
        "delta": delta,
        "Lx_min": Lx_min,
        "N_min": N_min,
        "dx": dx,
        "dt_stable": dt_stable,
        "d2G_max": d2G_max,
        "d2G_c0": d2G_c0,
        "spinodal_lo": sp_lo,
        "spinodal_hi": sp_hi,
        "c0_in_spinodal": c0_in_spinodal,
    }
