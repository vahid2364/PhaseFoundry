"""Interpolant of the molar Gibbs energy G_m on the composition simplex.

``build_interpolant`` reads a precomputed HDF5 thermodynamic table and returns a
``GmInterpolant`` of G_m(c_1, ..., c_{N-1}) over the N - 1 independent mole fractions. The
backend depends on the number of independent fractions:

- 1 (binary): a 1-D ``CubicSpline``.
- 2 (ternary): ``CloughTocher2DInterpolator`` (C1 smooth), with a nearest-neighbour
  fallback outside the convex hull of the data.
- 3 or more: a thin-plate-spline ``RBFInterpolator``, sampled once on a regular grid and
  then evaluated with a linear ``RegularGridInterpolator``.

The interpolant gives G_m (``gm``), the chemical potentials ∂G/∂c_i (``mu``) and the
Hessian H_ij = ∂²G/∂c_i∂c_j (``d2g``), all in J/mol. ``derivative_method`` chooses how the
derivatives are obtained:

- ``"fd"`` (default): central finite differences of the G_m interpolant with step
  ``fd_eps``. Only G_m is needed, and the truncation error is of order fd_eps² G_m'''.
- ``"jansson"``: interpolants of the CALPHAD Jansson derivatives stored in the table,
  μ_i = MU[i] − MU[ref] and H_ij = dMU[i_j] − dMU[ref_j]. The table must contain ``MU``
  and ``dMU`` (step 1 with ``quantities=("GM", "MU", "dMU")``). This has no
  finite-difference truncation error and no finite-difference noise near the simplex
  boundary.
- ``"hybrid"``: Jansson interpolants for μ and finite differences of G_m for the
  Hessian, for tables whose ``dMU`` is missing or noisy.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal

import numpy as np
from scipy.interpolate import (
    CloughTocher2DInterpolator,
    CubicSpline,
    NearestNDInterpolator,
    RBFInterpolator,
    RegularGridInterpolator,
)

from phasefoundry.thermodynamics.storage import load_table

if TYPE_CHECKING:
    from pathlib import Path

_logger = logging.getLogger(__name__)

DerivativeMethod = Literal["fd", "jansson", "hybrid"]
"""Name of the derivative backend: ``"fd"``, ``"jansson"`` or ``"hybrid"``."""


def build_interpolant(
    table_paths: list[str | Path],
    elements: list[str],
    reference_element: str,
    temperature_K: float,
    fd_eps: float = 1e-4,
    derivative_method: DerivativeMethod = "fd",
) -> GmInterpolant:
    """Build a G_m interpolant from HDF5 thermodynamic tables.

    Every table is read, and the one whose temperature is closest to ``temperature_K`` is
    used.

    Args:
        table_paths: HDF5 tables, one per temperature or a single one.
        elements: Ordered element list, e.g. ``["NB", "HF", "V"]``.
        reference_element: Dependent element, e.g. ``"V"``. Its fraction follows from
            Σ x_i = 1.
        temperature_K: Target temperature in K.
        fd_eps: Finite-difference step in mole fraction. It is used for μ with ``"fd"``
            and for the Hessian with ``"fd"`` and ``"hybrid"``.
        derivative_method: ``"fd"``, ``"jansson"`` or ``"hybrid"``; see the module docstring.

    Returns:
        The interpolant. Its ``temperature_K`` is the temperature of the chosen table.

    Raises:
        ValueError: If ``table_paths`` is empty.
        KeyError: If ``"jansson"`` or ``"hybrid"`` is requested and the table has no ``MU``,
            or ``"jansson"`` is requested and the table lacks ``dMU`` or one of its keys.
    """
    best_path, best_T = None, np.inf
    for p in table_paths:
        data = load_table(p)
        T = float(data["metadata"]["temperature"])
        if abs(T - temperature_K) < abs(best_T - temperature_K):
            best_path, best_T = p, T

    if best_path is None:
        raise ValueError("No table paths provided.")

    data = load_table(best_path)
    comps = data["compositions"]
    GM = data["GM"]

    ind_elements = [e for e in elements if e != reference_element]
    N_ind = len(ind_elements)
    el_list = list(data["elements"]) if "elements" in data else elements

    col_idx = [el_list.index(e) for e in ind_elements]

    gm_interp = _build_1d(comps, GM, col_idx[0]) if N_ind == 1 else _build_nd(comps, GM, col_idx, N_ind)

    if N_ind == 1:
        x_col = comps[:, col_idx[0]]
        bounds = [(float(x_col.min()), float(x_col.max()))]
    else:
        bounds = getattr(gm_interp, "_grid_bounds", None)
        if bounds is None:
            bounds = [(float(comps[:, c].min()), float(comps[:, c].max())) for c in col_idx]

    mu_interps = None
    dmu_interps = None

    if derivative_method in ("jansson", "hybrid"):
        mu_interps = _build_mu_interps(
            data,
            comps,
            ind_elements,
            reference_element,
            col_idx,
            N_ind,
            el_list,
        )

    if derivative_method == "jansson":
        dmu_interps = _build_dmu_interps(
            data,
            comps,
            ind_elements,
            reference_element,
            col_idx,
            N_ind,
            el_list,
        )

    return GmInterpolant(
        interp=gm_interp,
        ind_elements=ind_elements,
        reference_element=reference_element,
        N_ind=N_ind,
        temperature_K=best_T,
        fd_eps=fd_eps,
        col_idx=col_idx,
        grid_bounds=bounds,
        derivative_method=derivative_method,
        mu_interps=mu_interps,
        dmu_interps=dmu_interps,
    )


def _build_mu_interps(
    data: dict,
    comps: np.ndarray,
    ind_elements: list[str],
    reference_element: str,
    col_idx: list[int],
    N_ind: int,
    el_list: list[str],
) -> list:
    """Build one interpolant per independent element for μ_i = MU[i] − MU[ref].

    Under the simplex constraint ∂G/∂c_i = μ_i − μ_ref, which is the Cahn-Hilliard driving
    force. ``el_list`` is not used.

    Raises:
        KeyError: If the table has no ``MU`` data.
    """
    if "MU" not in data:
        raise KeyError(
            "HDF5 table does not contain 'MU' data. "
            "Re-run step 1 with quantities=('GM','MU','dMU') to generate it, "
            "or use derivative_method='fd'."
        )

    mu_ref = data["MU"][reference_element]
    interps = []

    _logger.info("  Building Jansson μ interpolants for %s …", ind_elements)
    for ind_el in ind_elements:
        y = data["MU"][ind_el] - mu_ref
        interp = _build_scalar_interp(comps, y, col_idx, N_ind, label=f"μ_{ind_el}")
        interps.append(interp)

    return interps


def _build_dmu_interps(
    data: dict,
    comps: np.ndarray,
    ind_elements: list[str],
    reference_element: str,
    col_idx: list[int],
    N_ind: int,
    el_list: list[str],
) -> dict:
    """Build one interpolant per Hessian entry H_ij = dMU[i_j] − dMU[ref_j].

    Under the simplex constraint this equals ∂(μ_i − μ_ref)/∂x_j = ∂²G/∂c_i∂c_j. The
    interpolants are keyed by the index pair ``(i, j)``. ``el_list`` is not used.

    Raises:
        KeyError: If the table has no ``dMU`` data or lacks one of the needed keys.
    """
    if "dMU" not in data:
        raise KeyError(
            "HDF5 table does not contain 'dMU' data. "
            "Re-run step 1 with quantities=('GM','MU','dMU') to generate it, "
            "or use derivative_method='fd' or 'hybrid'."
        )

    dmu_data = data["dMU"]
    interps = {}
    n_elements = len(ind_elements)

    _logger.info("  Building Jansson ∂μ/∂x interpolants (%s×%s matrix) …", n_elements, n_elements)

    for i, el_i in enumerate(ind_elements):
        for j, el_j in enumerate(ind_elements):
            key_ij = f"{el_i}_{el_j}"
            key_rj = f"{reference_element}_{el_j}"

            if key_ij not in dmu_data:
                raise KeyError(f"dMU key '{key_ij}' not found in HDF5 table. Available keys: {list(dmu_data.keys())}")
            if key_rj not in dmu_data:
                raise KeyError(f"dMU key '{key_rj}' not found in HDF5 table. Available keys: {list(dmu_data.keys())}")

            y = dmu_data[key_ij] - dmu_data[key_rj]
            label = f"∂μ_{el_i}/∂x_{el_j}"
            interps[(i, j)] = _build_scalar_interp(comps, y, col_idx, N_ind, label=label)

    return interps


def _build_scalar_interp(
    comps: np.ndarray,
    y: np.ndarray,
    col_idx: list[int],
    N_ind: int,
    label: str = "",
):
    """Interpolate a scalar field over composition with the same backend as G_m.

    Using the same backend keeps the μ and Hessian interpolants as smooth as the G_m one.
    """
    if N_ind == 1:
        return _build_1d(comps, y, col_idx[0])
    else:
        pts = comps[:, col_idx]
        if N_ind == 2:
            return _build_ct2d(pts, y)
        else:
            return _build_rbf(pts, y, label=label)


def _build_1d(comps: np.ndarray, GM: np.ndarray, col: int) -> CubicSpline:
    """Return a ``CubicSpline`` of ``GM`` over composition column ``col``, extrapolating beyond the data."""
    x = comps[:, col]
    idx = np.argsort(x)
    return CubicSpline(x[idx], GM[idx], extrapolate=True)


def _build_nd(
    comps: np.ndarray,
    GM: np.ndarray,
    col_idx: list[int],
    N_ind: int,
):
    """Build the G_m interpolant for two or more independent fractions.

    Two fractions (ternary) use ``CloughTocher2DInterpolator``, which is C1 smooth and
    exact at the data points, with a nearest-neighbour fallback. Three or more use the
    RBF-on-a-grid interpolant from ``_build_rbf``.
    """
    pts = comps[:, col_idx]

    if N_ind == 2:
        return _build_ct2d(pts, GM)
    else:
        return _build_rbf(pts, GM)


class _CT2DWithFallback:
    """``CloughTocher2DInterpolator`` with a ``NearestNDInterpolator`` fallback.

    The Clough-Tocher interpolant is C1 smooth inside the convex hull of the data. Points
    just outside it, for example finite-difference perturbations near a simplex edge, get
    the value of the nearest data point instead of NaN.
    """

    def __init__(self, pts: np.ndarray, GM: np.ndarray):
        """Build both interpolants from points of shape ``(n, 2)`` and values of shape ``(n,)``."""
        self._ct = CloughTocher2DInterpolator(pts, GM, fill_value=np.nan)
        self._nearest = NearestNDInterpolator(pts, GM)
        self._grid_bounds = [
            (float(pts[:, 0].min()), float(pts[:, 0].max())),
            (float(pts[:, 1].min()), float(pts[:, 1].max())),
        ]

    def __call__(self, pts: np.ndarray) -> np.ndarray:
        """Evaluate at points of shape ``(n_pts, 2)``."""
        result = self._ct(pts)
        nan_mask = np.isnan(result)
        if nan_mask.any():
            result[nan_mask] = self._nearest(pts[nan_mask])
        return result


def _build_ct2d(pts: np.ndarray, GM: np.ndarray) -> _CT2DWithFallback:
    """Return a ``_CT2DWithFallback`` for the given points and values."""
    return _CT2DWithFallback(pts, GM)


def _build_rbf(pts: np.ndarray, GM: np.ndarray, n_grid: int = 20, label: str = "G_m"):
    """Build a smooth, fast interpolant for three or more independent fractions.

    The scattered data is fitted once with a thin-plate-spline ``RBFInterpolator``, with the
    values scaled by their mean magnitude, ``smoothing=1e-3`` and a linear polynomial term,
    so the fit does not pass exactly through the data. The RBF is then evaluated on a
    regular grid with ``n_grid`` points per axis spanning the data range. Grid points with
    Σ x_i ≥ 0.999 lie outside the simplex and take the value of the nearest valid grid
    point. A linear ``RegularGridInterpolator`` on that grid, which extrapolates beyond it,
    answers all later queries, so each evaluation during time stepping is a grid lookup
    instead of an RBF sum over all data points.

    Args:
        pts: Scattered compositions, shape ``(n_valid, n_ind)``.
        GM: Field values at those points, shape ``(n_valid,)``: G_m, μ_i or H_ij.
        n_grid: Grid points per axis.
        label: Name used in progress messages.

    Returns:
        The ``RegularGridInterpolator``, with a ``_grid_bounds`` attribute that holds the
        ``(min, max)`` of each axis.
    """
    N_ind = pts.shape[1]
    _logger.info("    Building RBF (TPS) for %r from %s points …", label, len(pts))

    y_scale = float(np.abs(GM).mean()) or 1.0
    rbf = RBFInterpolator(
        pts,
        GM / y_scale,
        kernel="thin_plate_spline",
        smoothing=1e-3,
        degree=1,
    )

    axes = [np.linspace(float(pts[:, i].min()), float(pts[:, i].max()), n_grid) for i in range(N_ind)]
    grids = np.meshgrid(*axes, indexing="ij")
    all_pts = np.stack([g.ravel() for g in grids], axis=1)

    simplex_valid = all_pts.sum(axis=1) < 0.999

    y_grid_flat = np.full(n_grid**N_ind, np.nan)
    if simplex_valid.any():
        y_grid_flat[simplex_valid] = rbf(all_pts[simplex_valid]) * y_scale

    y_grid = y_grid_flat.reshape([n_grid] * N_ind)

    valid_mask = ~np.isnan(y_grid_flat)
    if valid_mask.any():
        nn = NearestNDInterpolator(all_pts[valid_mask], y_grid_flat[valid_mask])
        nan_idx = ~valid_mask
        y_grid_flat[nan_idx] = nn(all_pts[nan_idx])
        y_grid = y_grid_flat.reshape([n_grid] * N_ind)

    rgi = RegularGridInterpolator(
        axes,
        y_grid,
        method="linear",
        bounds_error=False,
        fill_value=None,  # None extrapolates instead of returning NaN
    )
    rgi._grid_bounds = [(float(a.min()), float(a.max())) for a in axes]  # ty: ignore[unresolved-attribute]

    _logger.info("    RBF→RGI built for %r (%s-D, %s pts/axis).", label, N_ind, n_grid)
    return rgi


class GmInterpolant:
    """Interpolant of G_m and its derivatives for an N-component system.

    Every method takes compositions ``c`` of shape ``(n_ind, *spatial)``: the first axis
    indexes the independent elements and the others are spatial. For a binary, a field of
    shape ``(*spatial,)`` is also accepted, as long as its first axis does not have length 1.

    ``mu`` uses the Jansson μ interpolants when they were built (``"jansson"`` and
    ``"hybrid"``) and finite differences of G_m otherwise. ``d2g`` uses the Jansson Hessian
    interpolants when they were built (``"jansson"``) and finite differences otherwise.

    Attributes:
        ind_elements: Independent elements, in the order of the first axis of ``c``.
        reference_element: Dependent element.
        N_ind: Number of independent elements.
        temperature_K: Temperature of the table the interpolant was built from, in K.
        grid_bounds: ``(min, max)`` of each independent fraction in the data, or ``None``.
        derivative_method: ``"fd"``, ``"jansson"`` or ``"hybrid"``.
    """

    def __init__(
        self,
        interp,
        ind_elements: list[str],
        reference_element: str,
        N_ind: int,
        temperature_K: float,
        fd_eps: float,
        col_idx: list[int],
        grid_bounds: list[tuple[float, float]] | None = None,
        derivative_method: DerivativeMethod = "fd",
        mu_interps: list | None = None,
        dmu_interps: dict | None = None,
    ) -> None:
        """Store the interpolants. ``build_interpolant`` is the usual way to create one.

        Args:
            interp: G_m interpolant. It takes shape ``(n_pts,)`` for a binary and
                ``(n_pts, n_ind)`` otherwise.
            ind_elements: Independent elements.
            reference_element: Dependent element.
            N_ind: Number of independent elements.
            temperature_K: Table temperature in K.
            fd_eps: Finite-difference step in mole fraction.
            col_idx: Column of each independent element in the table's composition array.
            grid_bounds: ``(min, max)`` of each independent fraction in the data.
            derivative_method: ``"fd"``, ``"jansson"`` or ``"hybrid"``.
            mu_interps: Jansson μ interpolants, one per independent element, or ``None``.
            dmu_interps: Jansson Hessian interpolants keyed by ``(i, j)``, or ``None``.
        """
        self._interp = interp
        self.ind_elements = ind_elements
        self.reference_element = reference_element
        self.N_ind = N_ind
        self.temperature_K = temperature_K
        self._eps = fd_eps
        self._col_idx = col_idx
        self.grid_bounds: list[tuple[float, float]] | None = grid_bounds
        self.derivative_method = derivative_method
        self._mu_interps = mu_interps
        self._dmu_interps = dmu_interps

    def _eval_flat(self, c_flat: np.ndarray) -> np.ndarray:
        """Evaluate G_m at compositions of shape ``(n_pts, n_ind)``; returns shape ``(n_pts,)``."""
        if self.N_ind == 1:
            return self._interp(c_flat[:, 0])
        else:
            return self._interp(c_flat)

    def _apply_interp(self, interp, c_flat: np.ndarray) -> np.ndarray:
        """Evaluate a scalar interpolant at compositions of shape ``(n_pts, n_ind)``.

        A binary interpolant receives the single column, shape ``(n_pts,)``. Returns shape
        ``(n_pts,)``.
        """
        if self.N_ind == 1:
            return interp(c_flat[:, 0])
        else:
            return interp(c_flat)

    def _unpack(self, c: np.ndarray):
        """Flatten compositions to shape ``(n_pts, n_ind)`` and return them with the spatial shape.

        Binary input without the leading axis gets one; see the class docstring.
        """
        if self.N_ind == 1 and (c.ndim == 1 or (c.ndim >= 2 and c.shape[0] != 1)):
            c = c[np.newaxis]
        spatial_shape = c.shape[1:]
        n_pts = int(np.prod(spatial_shape))
        c_flat = c.reshape(self.N_ind, n_pts).T.copy()
        return c_flat, spatial_shape

    def gm(self, c: np.ndarray) -> np.ndarray:
        """Return G_m in J/mol at each point.

        Args:
            c: Compositions, shape ``(n_ind, *spatial)``, or ``(*spatial,)`` for a binary.

        Returns:
            G_m, shape ``(*spatial,)``.
        """
        c_flat, shape = self._unpack(c)
        return self._eval_flat(c_flat).reshape(shape)

    def mu(self, c: np.ndarray) -> np.ndarray:
        """Return the chemical potentials ∂G/∂c_i in J/mol.

        With Jansson μ interpolants (``"jansson"`` and ``"hybrid"``) this evaluates
        MU[i] − MU[ref] directly. Otherwise (``"fd"``) it takes central differences of G_m
        with step ``fd_eps``.

        Args:
            c: Compositions, shape ``(n_ind, *spatial)``, or ``(*spatial,)`` for a binary.

        Returns:
            Chemical potentials, shape ``(n_ind, *spatial)``.
        """
        c_flat, shape = self._unpack(c)

        if self._mu_interps is not None:
            n_pts = c_flat.shape[0]
            mu_out = np.empty((self.N_ind, n_pts))
            for i, interp_i in enumerate(self._mu_interps):
                mu_out[i] = self._apply_interp(interp_i, c_flat)
            return mu_out.reshape((self.N_ind,) + shape)

        else:
            n_pts = c_flat.shape[0]
            eps = self._eps
            mu_out = np.empty((self.N_ind, n_pts))
            for i in range(self.N_ind):
                cp = c_flat.copy()
                cp[:, i] += eps
                cm = c_flat.copy()
                cm[:, i] -= eps
                mu_out[i] = (self._eval_flat(cp) - self._eval_flat(cm)) / (2.0 * eps)
            return mu_out.reshape((self.N_ind,) + shape)

    def d2g(self, c: np.ndarray) -> np.ndarray:
        """Return the Hessian H_ij = ∂²G/∂c_i∂c_j in J/mol.

        With Jansson Hessian interpolants (``"jansson"``) this evaluates
        dMU[i_j] − dMU[ref_j] directly. Otherwise (``"fd"`` and ``"hybrid"``) it takes
        second-order central differences of G_m with step ``fd_eps``.

        Args:
            c: Compositions, shape ``(n_ind, *spatial)``, or ``(*spatial,)`` for a binary.

        Returns:
            Hessian, shape ``(n_ind, n_ind, *spatial)``.
        """
        c_flat, shape = self._unpack(c)
        n_pts = c_flat.shape[0]

        if self._dmu_interps is not None:
            H = np.empty((self.N_ind, self.N_ind, n_pts))
            for i in range(self.N_ind):
                for j in range(self.N_ind):
                    H[i, j] = self._apply_interp(self._dmu_interps[(i, j)], c_flat)
            return H.reshape((self.N_ind, self.N_ind) + shape)

        else:
            eps = self._eps
            H = np.empty((self.N_ind, self.N_ind, n_pts))
            g0 = self._eval_flat(c_flat)

            for i in range(self.N_ind):
                for j in range(self.N_ind):
                    if i == j:
                        cp = c_flat.copy()
                        cp[:, i] += eps
                        cm = c_flat.copy()
                        cm[:, i] -= eps
                        H[i, j] = (self._eval_flat(cp) - 2.0 * g0 + self._eval_flat(cm)) / eps**2
                    else:
                        cpp = c_flat.copy()
                        cpp[:, i] += eps
                        cpp[:, j] += eps
                        cpm = c_flat.copy()
                        cpm[:, i] += eps
                        cpm[:, j] -= eps
                        cmp = c_flat.copy()
                        cmp[:, i] -= eps
                        cmp[:, j] += eps
                        cmm = c_flat.copy()
                        cmm[:, i] -= eps
                        cmm[:, j] -= eps
                        H[i, j] = (self._eval_flat(cpp) - self._eval_flat(cpm) - self._eval_flat(cmp) + self._eval_flat(cmm)) / (
                            4.0 * eps**2
                        )

            return H.reshape((self.N_ind, self.N_ind) + shape)

    def min_eigenvalue(self, c: np.ndarray) -> np.ndarray | float:
        """Return the smallest eigenvalue of the Hessian at each point.

        A negative value means the composition is spinodally unstable. Points whose Hessian
        contains NaN give NaN. The eigenvalues come from ``numpy.linalg.eigvalsh``, which
        assumes a symmetric matrix and reads only its lower triangle.

        Args:
            c: Compositions, shape ``(n_ind, *spatial)``, or ``(*spatial,)`` for a binary.

        Returns:
            Smallest eigenvalue in J/mol, shape ``(*spatial,)``, or a float if ``c`` has no
            spatial axes.
        """
        H = self.d2g(c)
        shape = H.shape[2:]
        n_pts = int(np.prod(shape)) if shape else 1
        H_flat = H.reshape(self.N_ind, self.N_ind, n_pts)
        lmin = np.array(
            [np.linalg.eigvalsh(H_flat[:, :, k]).min() if not np.any(np.isnan(H_flat[:, :, k])) else np.nan for k in range(n_pts)]
        )
        return lmin.reshape(shape) if shape else float(lmin[0])
