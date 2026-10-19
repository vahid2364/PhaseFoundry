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

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import fsolve

from .storage import load_table


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
    # Initial guesses: pairs (x_lo, x_hi) spanning the two-phase region
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

        # Cubic-spline interpolants for smooth root-finding
        f_A = interp1d(x, mu_A, kind="cubic", fill_value="extrapolate")
        f_B = interp1d(x, mu_B, kind="cubic", fill_value="extrapolate")

        def residuals(xy):
            xa, xb = xy
            return [f_A(xa) - f_A(xb), f_B(xa) - f_B(xb)]

        found_lo = found_hi = None

        for x0_lo, x0_hi in _INIT_GUESSES:
            try:
                sol, _, ier, _ = fsolve(residuals, [x0_lo, x0_hi], full_output=True)
                xa, xb = float(sol[0]), float(sol[1])
                res = residuals([xa, xb])
                resid = (res[0] ** 2 + res[1] ** 2) ** 0.5

                if (
                    ier == 1  # solver converged
                    and resid < residual_tol  # accurate equilibrium
                    and xa < xb  # ordered pair
                    and xb - xa >= min_separation  # two distinct phases
                    and 0.001 < xa < 0.999  # inside composition range
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

        # d²G/dx_i² = ∂(μ_i − μ_ref)/∂x_i = dMU[i_i] − dMU[ref_i]  (Gibbs-Duhem)
        d2g = data["dMU"][key]
        if reference_element is not None:
            ref_key = f"{reference_element}_{independent_element}"
            if ref_key in data["dMU"]:
                d2g = d2g - data["dMU"][ref_key]

        # Sort by composition for clean interpolation
        order = np.argsort(x)
        x, d2g = x[order], d2g[order]

        T_C = (
            temperatures_celsius[i] if temperatures_celsius is not None else data["metadata"].get("temperature", 273.15) - 273.15
        )
        temps.append(T_C)
        all_x.append(x)
        all_d2g.append(d2g)

        # Find zero-crossings (sign changes) of d²G/dx²
        sign = np.sign(d2g)
        crossings = np.where(np.diff(sign) != 0)[0]

        boundaries = []
        for idx in crossings:
            # Linear interpolation between idx and idx+1
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
            warnings.warn(f"Only one spinodal boundary found at {T_C} °C.")
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
                    # H_ij = ∂²G/∂c_i∂c_j = dMU[i_j] − dMU[ref_j]  (Gibbs-Duhem)
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

    # LinearNDInterpolator covers only the convex hull of the raw data.
    # NearestNDInterpolator fills the rest so the full simplex (0→1 for every
    # element) is always covered, even when composition_ranges don't reach 0 or 1.
    from scipy.interpolate import NearestNDInterpolator

    raw_pts = list(zip(x0_raw, x1_raw))
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

    # Fill NaN (outside data hull) with nearest-neighbor extrapolation so the
    # phase map extends to all simplex edges and corners.
    nan_mask = np.isnan(gm_fine)
    if nan_mask.any():
        gm_fine[nan_mask] = gm_interp_nn(fine_pts[nan_mask, 0], fine_pts[nan_mask, 1])

    valid = ~np.isnan(gm_fine)  # should now be all True
    fine_pts = fine_pts[valid]
    gm_fine = gm_fine[valid]
    x0_fine = fine_pts[:, 0]
    x1_fine = fine_pts[:, 1]

    pts_3d = np.column_stack([x0_fine, x1_fine, gm_fine])
    hull = ConvexHull(pts_3d)

    lower_mask = hull.equations[:, 2] < 0
    lower_eqs = hull.equations[lower_mask]
    lower_simplices = hull.simplices[lower_mask]

    # Hull G_m at each fine-grid point
    gm_hull_fine = np.full(len(x0_fine), -np.inf)
    for eq in lower_eqs:
        nx, ny, nz, d = eq
        z_plane = -(nx * x0_fine + ny * x1_fine + d) / nz
        gm_hull_fine = np.maximum(gm_hull_fine, z_plane)

    gm_range = float(np.nanmax(gm_fine) - np.nanmin(gm_fine))
    df_tol = driving_force_tol_frac * max(abs(gm_range), 1.0)
    df_fine = gm_fine - gm_hull_fine

    # In a ternary single-phase miscibility gap the three-phase tie-triangle
    # appears as the largest flat facet(s) of the lower convex hull.
    # Small facets (area < min_triangle_area) are discretization artifacts.
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

    # Encode region as int for NearestNDInterpolator (strings not supported)
    _enc = {"single": 0, "two": 1, "three": 2}
    _dec = {0: "single", 1: "two", 2: "three"}
    region_enc = np.array([_enc[r] for r in region_fine])

    from scipy.interpolate import NearestNDInterpolator

    region_nn = NearestNDInterpolator(list(zip(x0_fine, x1_fine)), region_enc)

    # Also compute driving force at raw data locations via hull planes
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
    x_ind = comps[:, idx]  # (n_pts, n_ind)

    pts_nd = np.column_stack([x_ind, gm])

    _fallback = dict(
        compositions=comps,
        gm=gm,
        driving_force=np.zeros(len(gm)),
        region=np.full(len(gm), "single"),
        spinodal=np.zeros(len(gm), dtype=bool),
        elements=elements,
        independent_elements=independent_elements,
        temperature_C=T_C,
    )
    try:
        hull = ConvexHull(pts_nd)
    except Exception:
        return _fallback

    # Lower facets: normal component in the Gm axis (index n_ind, i.e. -2) < 0
    lower_mask = hull.equations[:, n_ind] < 0
    lower_eqs = hull.equations[lower_mask]
    lower_simps = hull.simplices[lower_mask]

    # Hull G_m at each composition point
    gm_hull = np.full(len(gm), -np.inf)
    for eq in lower_eqs:
        n_x = eq[:n_ind]  # composition-axis normals
        n_g = eq[n_ind]  # G_m-axis normal
        d = eq[-1]  # offset
        z_plane = -(x_ind @ n_x + d) / n_g
        gm_hull = np.maximum(gm_hull, z_plane)

    gm_range = float(np.nanmax(gm) - np.nanmin(gm))
    df_tol = driving_force_tol_frac * max(abs(gm_range), 1.0)
    df = gm - gm_hull
    region = np.where(df < df_tol, "single", "two").astype(object)

    # Identify large hull simplices = multi-phase tie-regions
    for s in lower_simps:
        verts = x_ind[s]  # (k+1, n_ind) in composition space
        k = len(verts) - 1
        if k < 2:
            continue
        edges = verts[1:] - verts[0]  # (k, n_ind)
        G = edges @ edges.T  # (k, k) Gram matrix
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

    return dict(
        compositions=comps,
        gm=gm,
        driving_force=df,
        region=region,
        spinodal=is_spinodal,
        elements=elements,
        independent_elements=independent_elements,
        temperature_C=T_C,
    )


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
    T = (simplex_verts[1:] - simplex_verts[0]).T  # (n_dim, k)
    rhs = (points - simplex_verts[0]).T  # (n_dim, n_pts)
    try:
        bary, _, _, _ = np.linalg.lstsq(T, rhs, rcond=None)  # (k, n_pts)
    except Exception:
        return np.zeros(n_pts, dtype=bool)
    bary_last = 1.0 - bary.sum(axis=0)
    all_bary = np.vstack([bary, bary_last])  # (k+1, n_pts)
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

    # Spinodal boundaries: zero-crossings of d²G/dx²
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

    # κ_min: interface spans delta_over_dx grid points

    # We want: δ = delta_over_dx × dx  and  Lx = n_interfaces × λ*
    # From δ = sqrt(κ/d2G_max): κ = δ² × d2G_max
    # From λ* = 4π sqrt(κ/d2G_c0): λ* = 4π × δ × sqrt(d2G_max/d2G_c0)
    # dx = δ / delta_over_dx
    # N = Lx / dx = n_interfaces × λ* / dx

    d2G_c0_safe = max(d2G_c0, 1e-10)
    d2G_max_safe = max(d2G_max, 1e-10)

    lambda_star = 4.0 * np.pi * np.sqrt(1.0 / d2G_c0_safe)  # at κ=1 (scale below)
    lambda_crit = 2.0 * np.pi * np.sqrt(2.0 / d2G_c0_safe)

    # Choose κ_min such that δ = delta_over_dx × (Lx/N) and N is power-of-2.
    # Self-consistent iteration: κ depends on δ=dx*delta_over_dx, dx=Lx/N, and
    # Lx itself depends on κ via λ* = 4π√(κ/d2G_c0). Seed with κ=1, iterate to
    # convergence (typically 3–5 steps).
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
        # Round up to next power of 2
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
        M_eff_max = M * 0.25  # degenerate mobility peak at c=0.5
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

    # binary path
    if "x_spinodal_low" in spinodal_data:
        x_lo = spinodal_data["x_spinodal_low"]
        x_hi = spinodal_data["x_spinodal_high"]

        valid = [(T, lo, hi) for T, lo, hi in zip(temps, x_lo, x_hi) if lo is not None and hi is not None]

        if not valid:
            warnings.warn("No spinodal region found across the given temperatures.")
            return ax

        T_v, lo_v, hi_v = zip(*valid)
        T_v = np.array(T_v)
        lo_v = np.array(lo_v)
        hi_v = np.array(hi_v)

        ax.fill_betweenx(T_v, lo_v, hi_v, color=fill_color, alpha=fill_alpha, label="Spinodal region")
        ax.plot(lo_v, T_v, color=fill_color, linewidth=2, label="Spinodal boundary")
        ax.plot(hi_v, T_v, color=fill_color, linewidth=2)

        ax.scatter(lo_v, T_v, color=fill_color, s=30, zorder=4)
        ax.scatter(hi_v, T_v, color=fill_color, s=30, zorder=4)

    # multicomponent path (project onto one element)
    elif "min_eigenvalue" in spinodal_data:
        elements = None
        for T, comps, is_sp in zip(
            spinodal_data["temperatures_celsius"],
            spinodal_data["compositions"],
            spinodal_data["is_spinodal"],
        ):
            sp_comps = comps[is_sp]
            if len(sp_comps) > 0:
                ax.scatter(
                    sp_comps[:, 0],  # first element — caller should pass correct x_element
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
        print(f"Saved figure to {save_path}")

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
        [(T, lo, hi) for T, lo, hi in zip(temps_b, b_lo, b_hi) if lo is not None and hi is not None],
        key=lambda v: v[0],  # sort by temperature
    )
    if valid_b:
        T_b, lo_b, hi_b = zip(*valid_b)
        T_b = np.array(T_b)
        lo_b = np.array(lo_b)
        hi_b = np.array(hi_b)

        # Remove duplicate temperatures before spline fitting
        _, unique_idx = np.unique(T_b, return_index=True)
        T_b = T_b[unique_idx]
        lo_b = lo_b[unique_idx]
        hi_b = hi_b[unique_idx]

        # Smooth curves through data points
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
        [(T, lo, hi) for T, lo, hi in zip(temps_s, s_lo, s_hi) if lo is not None and hi is not None],
        key=lambda v: v[0],  # sort by temperature
    )
    if valid_s:
        T_s, lo_s, hi_s = zip(*valid_s)
        T_s = np.array(T_s)
        lo_s = np.array(lo_s)
        hi_s = np.array(hi_s)

        # Remove duplicate temperatures before spline fitting
        _, unique_idx = np.unique(T_s, return_index=True)
        T_s = T_s[unique_idx]
        lo_s = lo_s[unique_idx]
        hi_s = hi_s[unique_idx]

        # Smooth curves through data points
        T_fine_s = np.linspace(T_s.min(), T_s.max(), 400)
        lo_s_sm = make_interp_spline(T_s, lo_s, k=min(3, len(T_s) - 1))(T_fine_s)
        hi_s_sm = make_interp_spline(T_s, hi_s, k=min(3, len(T_s) - 1))(T_fine_s)

        ax.fill_betweenx(T_fine_s, lo_s_sm, hi_s_sm, color="#d62728", alpha=0.12, label="Spinodal region")
        ax.plot(lo_s_sm, T_fine_s, color="#d62728", linewidth=2.0, linestyle="--", label="Spinodal")
        ax.plot(hi_s_sm, T_fine_s, color="#d62728", linewidth=2.0, linestyle="--")
        ax.scatter(lo_s, T_s, color="#d62728", s=30, zorder=5)
        ax.scatter(hi_s, T_s, color="#d62728", s=30, zorder=5)

    # critical point: apex of spinodal (highest T where spinodal exists)
    if valid_s:
        T_s_arr = T_s  # already np.array, sorted
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
            arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.2),
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
        print(f"Saved figure to {save_path}")

    if show:
        plt.show()

    return ax
