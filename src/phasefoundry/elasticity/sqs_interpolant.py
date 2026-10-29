"""
phasefoundry.elasticity.sqs_interpolant
=======================================
Load SQS/DFT elastic constant CSV and build N-D interpolants for
C11, C12, C44, and a_relaxed. Shared between workflow step 1b and
the runtime elastic models so the same interpolant logic is used
in both places.

Public API
----------
load_and_filter(csv_path, elements)       → pd.DataFrame
build_interpolators(df, ind_elements)     → dict[str, callable]
build_batch_interpolators(df, ind_els)    → dict[str, BatchInterp]
query_composition(interps, comp, ind_els) → dict[str, float]
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd


# ── element column helpers ─────────────────────────────────────────────────────

def _element_col(el: str) -> str:
    return "x_" + el.capitalize()


def load_and_filter(csv_path: Path, elements: list[str]) -> pd.DataFrame:
    """
    Load CSV and keep only rows in the active element subspace.

    For NbHfV (elements=[NB, HF, V]), rows where x_Ti=0 are kept.
    """
    df = pd.read_csv(csv_path)
    all_csv_els = [c[2:].upper() for c in df.columns if c.startswith("x_")]
    active   = {e.upper() for e in elements}
    inactive = [e for e in all_csv_els if e not in active]

    mask = np.ones(len(df), dtype=bool)
    for el in inactive:
        col = _element_col(el)
        if col in df.columns:
            mask &= df[col].abs() < 1e-6

    sub = df[mask].copy().reset_index(drop=True)
    if len(sub) == 0:
        raise ValueError(
            f"No CSV rows found for element subspace {sorted(active)}. "
            f"Check that the CSV contains data for this system."
        )
    return sub


# ── scalar interpolators (used by step 1b point queries) ──────────────────────

def build_interpolators(df: pd.DataFrame, ind_elements: list[str]) -> dict:
    """
    Build scalar interpolators — each callable takes a 1-D composition vector
    and returns a single float.  Suitable for point queries (step 1b).

    1-D → interp1d; N-D → LinearNDInterpolator + NearestND fallback.
    """
    from scipy.interpolate import (interp1d, LinearNDInterpolator,
                                   NearestNDInterpolator)

    props  = ["C11_GPa", "C12_GPa", "C44_GPa", "a_relaxed_A",
              "shear_modulus_vrh_GPa", "bulk_modulus_vrh_GPa", "poisson_ratio",
              "biaxial_modulus_M_GPa"]
    coords = np.column_stack([df[_element_col(el)].values for el in ind_elements])
    interps: dict = {}

    if len(ind_elements) == 1:
        x1d = coords[:, 0]
        order = np.argsort(x1d)
        x1d_s = x1d[order]
        for prop in props:
            vals_s = df[prop].values[order]
            f = interp1d(x1d_s, vals_s, kind="linear", bounds_error=False,
                         fill_value=(vals_s[0], vals_s[-1]))
            def _make(fn):
                def _call(x):
                    return float(fn(np.atleast_1d(x)[0]))
                return _call
            interps[prop] = _make(f)
    else:
        for prop in props:
            vals = df[prop].values
            lin  = LinearNDInterpolator(coords, vals)
            near = NearestNDInterpolator(coords, vals)
            def _make(l, n):
                def _call(x):
                    x = np.atleast_1d(np.asarray(x, dtype=float))
                    v = l(x.reshape(1, -1)).item()
                    if np.isnan(v):
                        warnings.warn(
                            f"Composition {x} outside convex hull — "
                            "falling back to nearest-neighbour.",
                            stacklevel=3,
                        )
                        v = n(x.reshape(1, -1)).item()
                    return float(v)
                return _call
            interps[prop] = _make(lin, near)

    return interps


# ── batch interpolators (used by LocalVegardModel for field evaluation) ────────

class BatchInterp:
    """
    Vectorized interpolant: accepts (n_pts, n_ind) query array,
    returns (n_pts,) values.  Outside the convex hull falls back to
    nearest-neighbour.
    """

    def __init__(self, coords: np.ndarray, vals: np.ndarray,
                 n_ind: int) -> None:
        from scipy.interpolate import (interp1d, LinearNDInterpolator,
                                       NearestNDInterpolator)
        self._n_ind = n_ind
        if n_ind == 1:
            order = np.argsort(coords[:, 0])
            x_s   = coords[order, 0]
            v_s   = vals[order]
            self._lin  = interp1d(x_s, v_s, kind="linear",
                                  bounds_error=False,
                                  fill_value=(v_s[0], v_s[-1]))
            self._near = None
        else:
            self._lin  = LinearNDInterpolator(coords, vals)
            self._near = NearestNDInterpolator(coords, vals)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """x : (n_pts, n_ind) → values : (n_pts,)"""
        x = np.asarray(x, dtype=float)
        if self._n_ind == 1:
            return self._lin(x[:, 0])
        v = self._lin(x)            # (n_pts,) — NaN outside hull
        nan_mask = np.isnan(v)
        if nan_mask.any():
            v[nan_mask] = self._near(x[nan_mask])
        return v


def build_batch_interpolators(df: pd.DataFrame,
                               ind_elements: list[str]) -> dict[str, BatchInterp]:
    """
    Build batch interpolants for C11, C12, C44, a_relaxed.
    Each BatchInterp accepts (n_pts, n_ind) and returns (n_pts,).
    Used by LocalVegardModel for O(N²) field evaluation.
    """
    props  = ["C11_GPa", "C12_GPa", "C44_GPa", "a_relaxed_A"]
    n_ind  = len(ind_elements)
    coords = np.column_stack([df[_element_col(el)].values for el in ind_elements])
    return {
        p: BatchInterp(coords, df[p].values.astype(float), n_ind)
        for p in props
    }


# ── composition query helper ───────────────────────────────────────────────────

def query_composition(interps: dict, comp: dict,
                      ind_elements: list[str]) -> dict:
    """Query all scalar interpolants at a given composition dict."""
    x_ind = np.array([comp.get(el.upper(), 0.0) for el in ind_elements],
                     dtype=float)
    return {prop: fn(x_ind) for prop, fn in interps.items()}


# ── Vegard eigenstrain coefficients ─────────────────────────────────────────────

def vegard_eta_coefficients(interps: dict, comp: dict,
                            ind_elements: list[str],
                            delta_fd: float = 0.02) -> dict[str, float]:
    """
    Per-element Vegard eigenstrain coefficients eta_k = (1/a) * da/dx_k at
    a single composition, via central finite difference on the
    ``a_relaxed_A`` interpolant.

    Same method and default step size as the alpha-coefficient calculation
    in ``workflows/01b_lookup_elastic/run.py`` (there, evaluated at the
    midpoint between two endpoint compositions for the simulation's
    Vegard-tensor eigenstrain; here, callable at any single composition,
    e.g. for a coherent-Hessian stability analysis at one point).

    Parameters
    ----------
    interps : dict
        Output of ``build_interpolators()``; must include ``"a_relaxed_A"``.
    comp : dict
        Composition to evaluate at, e.g. ``{"NB": 0.25, "HF": 0.25, "TI": 0.25}``.
    ind_elements : list[str]
        Independent elements, in the order eta should be returned.
    delta_fd : float
        Finite-difference step in mole fraction (default 0.02).

    Returns
    -------
    dict[str, float]
        ``{element: eta_k}`` for each element in ``ind_elements``.
    """
    if "a_relaxed_A" not in interps:
        raise KeyError("vegard_eta_coefficients() requires an 'a_relaxed_A' "
                       "interpolant; build it with build_interpolators().")

    x0 = np.array([comp.get(el.upper(), 0.0) for el in ind_elements], dtype=float)
    a_ref = interps["a_relaxed_A"](x0)

    eta: dict[str, float] = {}
    for k, el in enumerate(ind_elements):
        x_p = x0.copy(); x_p[k] += delta_fd
        x_m = x0.copy(); x_m[k] -= delta_fd
        da_dxk = (interps["a_relaxed_A"](x_p) - interps["a_relaxed_A"](x_m)) / (2 * delta_fd)
        eta[el] = float(da_dxk / a_ref)

    return eta
