"""Interpolants for C11, C12, C44 and the lattice parameter from an SQS or DFT table.

Workflow step 1b and the elastic models used during a simulation both build their
interpolants here, so they interpolate the table the same way.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd


def _element_col(el: str) -> str:
    """Return the table column that holds an element's mole fraction, e.g. ``x_Nb``."""
    return "x_" + el.capitalize()


def load_and_filter(csv_path: Path, elements: list[str]) -> pd.DataFrame:
    """Load an elastic-constant table and keep the rows for the active elements.

    Rows that contain any other element of the table (fraction of 1e-6 or more) are
    dropped. For ``elements=["NB", "HF", "V"]`` and a Hf-Nb-Ti-V table, only the rows with
    ``x_Ti = 0`` are kept.

    Args:
        csv_path: Table CSV with ``x_<El>`` composition columns.
        elements: Active elements, in any case.

    Returns:
        The matching rows, re-indexed from 0.

    Raises:
        ValueError: If no row matches.
    """
    df = pd.read_csv(csv_path)
    all_csv_els = [c[2:].upper() for c in df.columns if c.startswith("x_")]
    active = {e.upper() for e in elements}
    inactive = [e for e in all_csv_els if e not in active]

    mask = np.ones(len(df), dtype=bool)
    for el in inactive:
        col = _element_col(el)
        if col in df.columns:
            mask &= df[col].abs() < 1e-6

    sub = df[mask].copy().reset_index(drop=True)
    if len(sub) == 0:
        raise ValueError(
            f"No CSV rows found for element subspace {sorted(active)}. Check that the CSV contains data for this system."
        )
    return sub


def build_interpolators(df: pd.DataFrame, ind_elements: list[str]) -> dict:
    """Build one scalar interpolant per property for single-composition queries.

    Each interpolant takes a 1-D vector of independent mole fractions and returns a float.
    Binary systems use linear 1-D interpolation, held constant beyond the data. Systems with
    more components use ``LinearNDInterpolator``; outside the convex hull of the data they
    warn and return the nearest data point instead. Step 1b uses these for point queries.

    Args:
        df: Table rows, e.g. from ``load_and_filter``.
        ind_elements: Independent elements, in the order of the query vector.

    Returns:
        Interpolants keyed by column name: ``C11_GPa``, ``C12_GPa``, ``C44_GPa``,
        ``a_relaxed_A``, ``shear_modulus_vrh_GPa``, ``bulk_modulus_vrh_GPa`` and
        ``poisson_ratio``.
    """
    from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator, interp1d

    props = ["C11_GPa", "C12_GPa", "C44_GPa", "a_relaxed_A", "shear_modulus_vrh_GPa", "bulk_modulus_vrh_GPa", "poisson_ratio"]
    coords = np.column_stack([df[_element_col(el)].values for el in ind_elements])
    interps: dict = {}

    if len(ind_elements) == 1:
        x1d = coords[:, 0]
        order = np.argsort(x1d)
        x1d_s = x1d[order]
        for prop in props:
            vals_s = df[prop].values[order]
            f = interp1d(x1d_s, vals_s, kind="linear", bounds_error=False, fill_value=(vals_s[0], vals_s[-1]))

            def _make(fn):
                def _call(x):
                    return float(fn(np.atleast_1d(x)[0]))

                return _call

            interps[prop] = _make(f)
    else:
        for prop in props:
            vals = df[prop].values
            lin = LinearNDInterpolator(coords, vals)
            near = NearestNDInterpolator(coords, vals)

            def _make(l, n):
                def _call(x):
                    x = np.atleast_1d(np.asarray(x, dtype=float))
                    v = l(x.reshape(1, -1)).item()
                    if np.isnan(v):
                        warnings.warn(
                            f"Composition {x} outside convex hull — falling back to nearest-neighbour.",
                            stacklevel=3,
                        )
                        v = n(x.reshape(1, -1)).item()
                    return float(v)

                return _call

            interps[prop] = _make(lin, near)

    return interps


# batch interpolators (used by LocalVegardModel for field evaluation)
class BatchInterp:
    """Vectorized interpolant of one property over many compositions.

    A call maps independent mole fractions of shape ``(n_pts, n_ind)`` to values of shape
    ``(n_pts,)``. Binary systems use linear 1-D interpolation, held constant beyond the
    data. Systems with more components return the nearest data point outside the convex
    hull, without a warning.
    """

    def __init__(self, coords: np.ndarray, vals: np.ndarray, n_ind: int) -> None:
        """Build the interpolant.

        Args:
            coords: Independent mole fractions of the data points, shape ``(n_data, n_ind)``.
            vals: Property values at the data points, shape ``(n_data,)``.
            n_ind: Number of independent elements.
        """
        from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator, interp1d

        self._n_ind = n_ind
        if n_ind == 1:
            order = np.argsort(coords[:, 0])
            x_s = coords[order, 0]
            v_s = vals[order]
            self._lin = interp1d(x_s, v_s, kind="linear", bounds_error=False, fill_value=(v_s[0], v_s[-1]))
            self._near = None
        else:
            self._lin = LinearNDInterpolator(coords, vals)
            self._near = NearestNDInterpolator(coords, vals)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Evaluate the interpolant.

        Args:
            x: Independent mole fractions, shape ``(n_pts, n_ind)``.

        Returns:
            Interpolated values, shape ``(n_pts,)``.
        """
        x = np.asarray(x, dtype=float)
        if self._n_ind == 1:
            return self._lin(x[:, 0])
        v = self._lin(x)  # (n_pts,) — NaN outside hull
        nan_mask = np.isnan(v)
        if nan_mask.any():
            v[nan_mask] = self._near(x[nan_mask])
        return v


def build_batch_interpolators(df: pd.DataFrame, ind_elements: list[str]) -> dict[str, BatchInterp]:
    """Build a ``BatchInterp`` for C11, C12, C44 and the lattice parameter.

    ``LocalVegardModel`` uses these to evaluate the properties at every grid point of the
    composition field.

    Args:
        df: Table rows, e.g. from ``load_and_filter``.
        ind_elements: Independent elements, in the column order of the query arrays.

    Returns:
        Interpolants keyed by ``C11_GPa``, ``C12_GPa``, ``C44_GPa`` and ``a_relaxed_A``.
    """
    props = ["C11_GPa", "C12_GPa", "C44_GPa", "a_relaxed_A"]
    n_ind = len(ind_elements)
    coords = np.column_stack([df[_element_col(el)].values for el in ind_elements])
    return {p: BatchInterp(coords, df[p].values.astype(float), n_ind) for p in props}


def query_composition(interps: dict, comp: dict, ind_elements: list[str]) -> dict:
    """Evaluate every scalar interpolant at one composition.

    Args:
        interps: Output of ``build_interpolators``.
        comp: Mole fractions keyed by uppercase element symbol. Missing elements count as 0.
        ind_elements: Independent elements, in the order the interpolants expect.

    Returns:
        Property values keyed by column name.
    """
    x_ind = np.array([comp.get(el.upper(), 0.0) for el in ind_elements], dtype=float)
    return {prop: fn(x_ind) for prop, fn in interps.items()}
