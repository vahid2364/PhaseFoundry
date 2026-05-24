"""
General N-component G_m interpolant
=====================================
Builds a surrogate for the molar Gibbs energy G_m(c₁, …, c_{N-1}) on the
composition simplex from pre-computed HDF5 thermodynamic tables.

Works for any number of components:
  binary     N=2  →  1 independent composition, 1-D CubicSpline
  ternary    N=3  →  2 independent compositions, 2-D interpolant
  quaternary N=4  →  3 independent compositions, 3-D interpolant

The surrogate exposes:
  gm(c)        → G_m values, shape (N_ind, *spatial)  [J/mol]
  mu(c)        → ∂G/∂cᵢ for i=1…N_ind, shape (N_ind, *spatial)  [J/mol]
  d2g(c)       → Hessian Hᵢⱼ = ∂²G/∂cᵢ∂cⱼ, shape (N_ind, N_ind, *spatial)
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
from scipy.interpolate import CubicSpline, RegularGridInterpolator

from .storage import load_table


# ── public builder ────────────────────────────────────────────────────────────

def build_interpolant(
    table_paths: list[str | Path],
    elements: list[str],
    reference_element: str,
    temperature_K: float,
    fd_eps: float = 1e-4,
) -> "GmInterpolant":
    """
    Build a G_m interpolant from HDF5 thermodynamic tables.

    Parameters
    ----------
    table_paths     : list of HDF5 paths (one per temperature, or just one).
    elements        : ordered element list, e.g. ['NB', 'HF', 'V'].
    reference_element : the dependent element (e.g. 'V'); its composition is
                        inferred from the simplex constraint Σxᵢ = 1.
    temperature_K   : target temperature; the closest table is used.
    fd_eps          : finite-difference step for derivative estimation.

    Returns
    -------
    GmInterpolant
    """
    # find closest table
    best_path, best_T = None, np.inf
    for p in table_paths:
        data = load_table(p)
        T    = float(data["metadata"]["temperature"])
        if abs(T - temperature_K) < abs(best_T - temperature_K):
            best_path, best_T = p, T

    if best_path is None:
        raise ValueError("No table paths provided.")

    data = load_table(best_path)
    comps = data["compositions"]          # (n_valid, N_el)
    GM    = data["GM"]                    # (n_valid,)

    ind_elements = [e for e in elements if e != reference_element]
    N_ind        = len(ind_elements)
    el_list      = list(data["elements"]) if "elements" in data else elements

    # Column indices of independent elements in the stored composition array
    col_idx = [el_list.index(e) for e in ind_elements]

    if N_ind == 1:
        interp = _build_1d(comps, GM, col_idx[0])
    else:
        interp = _build_nd(comps, GM, col_idx, N_ind)

    return GmInterpolant(
        interp          = interp,
        ind_elements    = ind_elements,
        reference_element = reference_element,
        N_ind           = N_ind,
        temperature_K   = best_T,
        fd_eps          = fd_eps,
        col_idx         = col_idx,
    )


# ── 1-D (binary) backend ──────────────────────────────────────────────────────

def _build_1d(comps: np.ndarray, GM: np.ndarray, col: int) -> CubicSpline:
    """CubicSpline for the binary (1-D) case."""
    x   = comps[:, col]
    idx = np.argsort(x)
    return CubicSpline(x[idx], GM[idx], extrapolate=True)


# ── N-D (ternary+) backend ────────────────────────────────────────────────────

def _build_nd(
    comps:   np.ndarray,
    GM:      np.ndarray,
    col_idx: list[int],
    N_ind:   int,
) -> RegularGridInterpolator:
    """
    RegularGridInterpolator for N_ind ≥ 2 independent compositions.

    Strategy
    --------
    The composition data lies on a regular grid inside the simplex.  We
    reconstruct that grid by finding the unique coordinate values along each
    axis, build a full rectangular grid, fill with NaN where outside the
    simplex, and pass to RegularGridInterpolator.
    """
    axes_vals = []
    for col in col_idx:
        vals = np.unique(np.round(comps[:, col], 8))
        axes_vals.append(vals)

    grid_shape = tuple(len(v) for v in axes_vals)
    gm_grid    = np.full(grid_shape, np.nan)

    # build lookup: rounded coord tuple → GM value
    lookup = {}
    for k in range(len(comps)):
        key = tuple(round(comps[k, col], 8) for col in col_idx)
        lookup[key] = GM[k]

    # fill grid
    for idx in np.ndindex(grid_shape):
        key = tuple(axes_vals[dim][idx[dim]] for dim in range(N_ind))
        if key in lookup:
            gm_grid[idx] = lookup[key]

    return RegularGridInterpolator(
        axes_vals,
        gm_grid,
        method       = "linear",
        bounds_error = False,
        fill_value   = np.nan,
    )


# ── interpolant object ────────────────────────────────────────────────────────

class GmInterpolant:
    """
    General G_m surrogate for an N-component system.

    The composition argument ``c`` passed to all methods is an array of shape
    ``(N_ind, *spatial)`` where the first axis indexes the independent
    elements and the remaining axes are spatial dimensions.

    For binary (N_ind=1) a 1-D array ``(N, N)`` (a single field) is also
    accepted for convenience.
    """

    def __init__(
        self,
        interp,
        ind_elements:     list[str],
        reference_element: str,
        N_ind:            int,
        temperature_K:    float,
        fd_eps:           float,
        col_idx:          list[int],
    ) -> None:
        self._interp   = interp
        self.ind_elements      = ind_elements
        self.reference_element = reference_element
        self.N_ind     = N_ind
        self.temperature_K = temperature_K
        self._eps      = fd_eps
        self._col_idx  = col_idx

    # ── helpers ───────────────────────────────────────────────────────────────

    def _eval_flat(self, c_flat: np.ndarray) -> np.ndarray:
        """
        Evaluate G_m at composition points.

        Parameters
        ----------
        c_flat : shape (N_pts, N_ind) — each row is one composition point.

        Returns
        -------
        gm_flat : shape (N_pts,)
        """
        if self.N_ind == 1:
            return self._interp(c_flat[:, 0])
        else:
            return self._interp(c_flat)

    def _unpack(self, c: np.ndarray):
        """Return (c_flat, spatial_shape) from a (N_ind, *spatial) array."""
        if self.N_ind == 1 and c.ndim == 2:
            # binary convenience: (N, N) → treat as (1, N, N)
            c = c[np.newaxis]
        spatial_shape = c.shape[1:]
        n_pts         = int(np.prod(spatial_shape))
        # c_flat: (n_pts, N_ind) — columns = independent elements
        c_flat = c.reshape(self.N_ind, n_pts).T.copy()
        return c_flat, spatial_shape

    # ── public API ────────────────────────────────────────────────────────────

    def gm(self, c: np.ndarray) -> np.ndarray:
        """
        G_m evaluated on a spatial composition field.

        Parameters
        ----------
        c : (N_ind, *spatial)  or  (*spatial,) for binary

        Returns
        -------
        ndarray  same spatial shape as c[0]
        """
        c_flat, shape = self._unpack(c)
        return self._eval_flat(c_flat).reshape(shape)

    def mu(self, c: np.ndarray) -> np.ndarray:
        """
        Chemical potentials ∂G/∂cᵢ via central finite differences.

        Returns
        -------
        ndarray  shape (N_ind, *spatial)
        """
        c_flat, shape = self._unpack(c)
        n_pts         = c_flat.shape[0]
        eps           = self._eps
        mu_out        = np.empty((self.N_ind, n_pts))

        for i in range(self.N_ind):
            cp = c_flat.copy(); cp[:, i] += eps
            cm = c_flat.copy(); cm[:, i] -= eps
            mu_out[i] = (self._eval_flat(cp) - self._eval_flat(cm)) / (2.0 * eps)

        return mu_out.reshape((self.N_ind,) + shape)

    def d2g(self, c: np.ndarray) -> np.ndarray:
        """
        Hessian Hᵢⱼ = ∂²G/∂cᵢ∂cⱼ via finite differences.

        Returns
        -------
        ndarray  shape (N_ind, N_ind, *spatial)
        """
        c_flat, shape = self._unpack(c)
        n_pts         = c_flat.shape[0]
        eps           = self._eps
        H             = np.empty((self.N_ind, self.N_ind, n_pts))
        g0            = self._eval_flat(c_flat)

        for i in range(self.N_ind):
            for j in range(self.N_ind):
                if i == j:
                    cp = c_flat.copy(); cp[:, i] += eps
                    cm = c_flat.copy(); cm[:, i] -= eps
                    H[i, j] = (self._eval_flat(cp) - 2.0 * g0 + self._eval_flat(cm)) / eps**2
                else:
                    cpp = c_flat.copy(); cpp[:, i] += eps; cpp[:, j] += eps
                    cpm = c_flat.copy(); cpm[:, i] += eps; cpm[:, j] -= eps
                    cmp = c_flat.copy(); cmp[:, i] -= eps; cmp[:, j] += eps
                    cmm = c_flat.copy(); cmm[:, i] -= eps; cmm[:, j] -= eps
                    H[i, j] = (
                        self._eval_flat(cpp) - self._eval_flat(cpm)
                        - self._eval_flat(cmp) + self._eval_flat(cmm)
                    ) / (4.0 * eps**2)

        return H.reshape((self.N_ind, self.N_ind) + shape)

    def min_eigenvalue(self, c: np.ndarray) -> np.ndarray:
        """
        Minimum eigenvalue of the Hessian at each spatial point.
        Negative → spinodally unstable.

        Returns
        -------
        ndarray  shape (*spatial)
        """
        H      = self.d2g(c)                   # (N_ind, N_ind, *spatial)
        shape  = H.shape[2:]
        n_pts  = int(np.prod(shape)) if shape else 1
        H_flat = H.reshape(self.N_ind, self.N_ind, n_pts)
        lmin   = np.array([
            np.linalg.eigvalsh(H_flat[:, :, k]).min()
            for k in range(n_pts)
        ])
        return lmin.reshape(shape) if shape else float(lmin[0])
