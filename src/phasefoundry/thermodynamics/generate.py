"""Thermodynamic quantities over a composition grid, computed with pycalphad.

Each grid point is evaluated with ``IsolatedPhase``, which keeps pycalphad on the
single-phase free-energy surface. The phase-field driving force needs that surface, not
the equilibrium one. The quantities are:

- ``GM``: molar Gibbs energy in J/mol;
- ``MU(i)``: chemical potential of element i in J/mol;
- ``MU(i).X(j)``: the Jansson derivative dμ_i/dx_j in J/mol.
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
"""Subset of ``"GM"``, ``"MU"`` and ``"dMU"`` to compute."""


def _eval_point(
    args: tuple,
) -> tuple[int, dict | None]:
    """Evaluate the requested quantities at one composition.

    The function takes a single tuple so it can run in a worker process. The last element
    is the reference: its fraction is not passed to pycalphad as a condition. A point fails
    if pycalphad raises, or if G_m or any chemical potential is not finite; a non-finite
    derivative is stored as NaN instead.

    Args:
        args: ``(idx, composition_row, elements, phase, db_path, temperature, pressure,
            quantities)``.

    Returns:
        ``(idx, result)``, where ``result`` holds the quantities, or ``(idx, None)`` if the
        point failed.
    """
    idx, comp_row, elements, phase, db_path, temperature, pressure, quantities = args

    from pycalphad import Database, Workspace
    from pycalphad import variables as v
    from pycalphad.property_framework import IsolatedPhase

    cond = {v.T: temperature, v.P: pressure}
    # pycalphad takes N - 1 composition conditions; the last element is dependent
    for el, xi in zip(elements[:-1], comp_row[:-1], strict=True):
        cond[v.X(el)] = float(xi)

    try:
        db = Database(db_path)
        wks = Workspace(database=db, components=elements, phases=[phase], conditions=cond)
        ip = IsolatedPhase(phase, wks=wks)

        result: dict = {"idx": idx, "composition": comp_row}

        if "GM" in quantities:
            gm = wks.get(ip("GM")).squeeze()  # ty: ignore[invalid-argument-type]
            if not np.isfinite(gm):
                return idx, None
            result["GM"] = float(gm)

        if "MU" in quantities:
            mus = {}
            for el in elements:
                mu = wks.get(ip(f"MU({el})")).squeeze()  # ty: ignore[invalid-argument-type]
                if not np.isfinite(mu):
                    return idx, None
                mus[el] = float(mu)
            result["MU"] = mus

        if "dMU" in quantities:
            dmus = {}
            for el1 in elements:
                for el2 in elements[:-1]:
                    dmu = wks.get(ip(f"MU({el1}).X({el2})")).squeeze()  # ty: ignore[invalid-argument-type]
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
    """Evaluate thermodynamic quantities over a composition grid.

    Points that fail are skipped with a warning, and the remaining points keep the grid
    order.

    Args:
        db_path: Path to the TDB database.
        elements: All elements of the system. The last one is the reference: its fraction
            follows from Σ x = 1 and is not passed to pycalphad.
        phase: Phase name, e.g. ``"BCC_A2"``.
        grid: Compositions with columns in the order of ``elements``, shape
            ``(n_points, n_elements)``, e.g. from ``simplex_grid``.
        temperature: Temperature in K.
        pressure: Pressure in Pa.
        quantities: Any combination of ``"GM"``, ``"MU"`` and ``"dMU"``.
        n_workers: Number of worker processes. ``1`` runs serially, which is easier to
            debug; larger values use a ``ProcessPoolExecutor``.

    Returns:
        A dict with ``"compositions"`` (shape ``(n_valid, n_elements)``), ``"elements"``,
        ``"n_failed"`` (number of skipped points) and the requested quantities: ``"GM"``
        (shape ``(n_valid,)``), ``"MU"`` (one array per element) and ``"dMU"`` (one array
        per ``"{EL1}_{EL2}"`` pair, dμ_EL1/dx_EL2 for every element EL1 and independent
        element EL2).
    """
    n_pts = len(grid)
    args_list = [(i, grid[i], elements, phase, db_path, temperature, pressure, quantities) for i in range(n_pts)]

    results_by_idx: dict[int, dict] = {}

    if n_workers == 1:
        for args in tqdm(args_list, desc="Computing thermodynamics", unit="pt"):
            idx, res = _eval_point(args)
            if res is not None:
                results_by_idx[idx] = res
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(_eval_point, a): a[0] for a in args_list}
            for fut in tqdm(as_completed(futures), total=n_pts, desc="Computing thermodynamics (parallel)", unit="pt"):
                idx, res = fut.result()
                if res is not None:
                    results_by_idx[idx] = res

    n_failed = n_pts - len(results_by_idx)
    if n_failed > 0:
        warnings.warn(f"{n_failed}/{n_pts} grid points failed and were skipped.", stacklevel=2)

    # Parallel results arrive out of order; sort to restore the grid order
    valid_indices = sorted(results_by_idx.keys())
    out: dict = {
        "compositions": np.array([results_by_idx[i]["composition"] for i in valid_indices]),
        "n_failed": n_failed,
        "elements": elements,
    }

    if "GM" in quantities:
        out["GM"] = np.array([results_by_idx[i]["GM"] for i in valid_indices])

    if "MU" in quantities:
        out["MU"] = {el: np.array([results_by_idx[i]["MU"][el] for i in valid_indices]) for el in elements}

    if "dMU" in quantities:
        sample_dmu = next(iter(results_by_idx.values()))["dMU"]
        out["dMU"] = {key: np.array([results_by_idx[i]["dMU"].get(key, np.nan) for i in valid_indices]) for key in sample_dmu}

    return out
