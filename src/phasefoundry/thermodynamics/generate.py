"""
Compute thermodynamic quantities over a composition grid using pycalphad.

For each grid point the following are evaluated via IsolatedPhase (which
forces pycalphad to stay on the single-phase free energy surface — required
for phase-field driving force calculations):

  GM              molar Gibbs energy                     [J/mol]
  MU(i)           chemical potential of element i        [J/mol]
  MU(i).X(j)      Jansson derivative d mu_i / d x_j     [J/mol]
"""

from __future__ import annotations

import logging
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Literal

import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)

Quantities = tuple[Literal["GM", "MU", "dMU"], ...]


def _eval_point(
    args: tuple,
) -> tuple[int, dict | None]:
    """
    Evaluate thermodynamic quantities at a single composition point.

    This function is designed to run in a worker process.

    Parameters
    ----------
    args : tuple
        (idx, composition_row, elements, phase, db_path, temperature,
         pressure, quantities)

    Returns
    -------
    (idx, result_dict) or (idx, None) on failure
    """
    idx, comp_row, elements, phase, db_path, temperature, pressure, quantities = args

    from pycalphad import Database, Workspace, variables as v
    from pycalphad.property_framework import IsolatedPhase

    cond = {v.T: temperature, v.P: pressure}
    ref_el = elements[-1]  # last element is always reference for conditions
    # Provide N-1 independent composition conditions (omit last element)
    for el, xi in zip(elements[:-1], comp_row[:-1]):
        cond[v.X(el)] = float(xi)

    try:
        db = Database(db_path)
        wks = Workspace(database=db, components=elements, phases=[phase], conditions=cond)
        ip = IsolatedPhase(phase, wks=wks)

        result: dict = {"idx": idx, "composition": comp_row}

        if "GM" in quantities:
            gm = wks.get(ip("GM")).squeeze()
            if not np.isfinite(gm):
                return idx, None
            result["GM"] = float(gm)

        if "MU" in quantities:
            mus = {}
            for el in elements:
                mu = wks.get(ip(f"MU({el})")).squeeze()
                if not np.isfinite(mu):
                    return idx, None
                mus[el] = float(mu)
            result["MU"] = mus

        if "dMU" in quantities:
            dmus = {}
            for el1 in elements:
                for el2 in elements[:-1]:  # derivatives w.r.t. independent elements only
                    dmu = wks.get(ip(f"MU({el1}).X({el2})")).squeeze()
                    dmus[f"{el1}_{el2}"] = float(dmu) if np.isfinite(dmu) else np.nan
            result["dMU"] = dmus

        return idx, result

    except Exception as exc:
        logger.debug("Skipping point %d %s: %s", idx, comp_row, exc)
        return idx, None


def compute_table(
    db_path: str,
    elements: list[str],
    phase: str,
    grid: np.ndarray,
    temperature: float,
    pressure: float = 101325.0,
    quantities: Quantities = ("GM", "MU", "dMU"),
    n_workers: int = 1,
) -> dict:
    """
    Evaluate thermodynamic quantities over a composition grid.

    Parameters
    ----------
    db_path : str
        Path to the TDB thermodynamic database file.
    elements : list[str]
        All elements in the system. The *last* element is used as the
        reference (its mole fraction is derived from the constraint
        sum = 1 and is not passed as a condition to pycalphad).
    phase : str
        Phase name to evaluate (e.g. 'BCC_A2').
    grid : np.ndarray, shape (n_points, N_elements)
        Composition grid as returned by ``grid.simplex_grid()``.
        Columns must be in the same order as ``elements``.
    temperature : float
        Temperature in Kelvin.
    pressure : float
        Pressure in Pa. Default 101325.
    quantities : tuple of str
        Which quantities to compute. Any combination of
        ``'GM'``, ``'MU'``, ``'dMU'``.
    n_workers : int
        Number of parallel worker processes. Set to 1 for serial execution
        (safer for debugging). Values > 1 use ProcessPoolExecutor.

    Returns
    -------
    dict with keys:
        'compositions'  np.ndarray (n_valid, N_elements)
        'GM'            np.ndarray (n_valid,)           if requested
        'MU'            dict[str, np.ndarray(n_valid,)] if requested
        'dMU'           dict[str, np.ndarray(n_valid,)] if requested
        'n_failed'      int   number of points that failed / returned NaN
    """
    n_pts = len(grid)
    args_list = [
        (i, grid[i], elements, phase, db_path, temperature, pressure, quantities)
        for i in range(n_pts)
    ]

    results_by_idx: dict[int, dict] = {}

    if n_workers == 1:
        for args in tqdm(args_list, desc="Computing thermodynamics", unit="pt"):
            idx, res = _eval_point(args)
            if res is not None:
                results_by_idx[idx] = res
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(_eval_point, a): a[0] for a in args_list}
            for fut in tqdm(
                as_completed(futures), total=n_pts,
                desc="Computing thermodynamics (parallel)", unit="pt"
            ):
                idx, res = fut.result()
                if res is not None:
                    results_by_idx[idx] = res

    n_failed = n_pts - len(results_by_idx)
    if n_failed > 0:
        warnings.warn(f"{n_failed}/{n_pts} grid points failed and were skipped.")

    # Collect into arrays (preserve original grid order)
    valid_indices = sorted(results_by_idx.keys())
    out: dict = {
        "compositions": np.array([results_by_idx[i]["composition"] for i in valid_indices]),
        "n_failed": n_failed,
        "elements": elements,
    }

    if "GM" in quantities:
        out["GM"] = np.array([results_by_idx[i]["GM"] for i in valid_indices])

    if "MU" in quantities:
        out["MU"] = {
            el: np.array([results_by_idx[i]["MU"][el] for i in valid_indices])
            for el in elements
        }

    if "dMU" in quantities:
        sample_dmu = next(iter(results_by_idx.values()))["dMU"]
        out["dMU"] = {
            key: np.array([
                results_by_idx[i]["dMU"].get(key, np.nan) for i in valid_indices
            ])
            for key in sample_dmu
        }

    return out
