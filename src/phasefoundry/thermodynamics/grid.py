"""
Composition grid generation on the N-component simplex.

For N elements, there are N-1 independent mole fractions. We iterate
over the first N-2 elements on regular linspace grids and derive the
reference element's fraction as x_ref = 1 - sum(others).
"""

import itertools
import numpy as np


def simplex_grid(
    elements: list[str],
    composition_ranges: dict[str, tuple[float, float]],
    reference_element: str,
    n_points: int,
) -> np.ndarray:
    """
    Build a regular composition grid on the N-component simplex.

    Parameters
    ----------
    elements : list[str]
        All elements in the system, e.g. ['HF', 'NB', 'TI', 'V'].
    composition_ranges : dict[str, tuple[float, float]]
        Min/max mole fraction for each *independent* element (i.e. all
        except the reference element). Example:
        {'HF': (0.05, 0.60), 'NB': (0.05, 0.60), 'TI': (0.05, 0.60)}
    reference_element : str
        Dependent element whose fraction is computed as
        x_ref = 1 - sum(x_i for i != ref).
    n_points : int
        Number of grid points along each element axis.

    Returns
    -------
    np.ndarray, shape (n_valid, N_elements)
        Valid composition rows in the order given by `elements`.
        Invalid rows (x_ref < 0 or any x_i < 0) are excluded.
    """
    if reference_element not in elements:
        raise ValueError(f"reference_element '{reference_element}' not in elements {elements}")

    independent = [e for e in elements if e != reference_element]

    for el in independent:
        if el not in composition_ranges:
            raise ValueError(f"No composition range specified for element '{el}'")

    axes = [np.linspace(*composition_ranges[el], n_points) for el in independent]

    rows = []
    for values in itertools.product(*axes):
        x_ind = dict(zip(independent, values))
        x_ref = 1.0 - sum(values)

        if x_ref < 0.0 or x_ref > 1.0:
            continue
        if any(v < 0.0 for v in values):
            continue

        row = []
        for el in elements:
            row.append(x_ind[el] if el != reference_element else x_ref)
        rows.append(row)

    if not rows:
        raise ValueError(
            "No valid compositions found. Check composition_ranges and reference_element."
        )

    return np.array(rows, dtype=np.float64)


def grid_summary(grid: np.ndarray, elements: list[str]) -> str:
    """Return a short human-readable summary of a composition grid."""
    lines = [
        f"Grid: {len(grid)} valid points, {len(elements)} elements",
    ]
    for i, el in enumerate(elements):
        col = grid[:, i]
        lines.append(f"  {el}: [{col.min():.4f}, {col.max():.4f}]  mean={col.mean():.4f}")
    return "\n".join(lines)
