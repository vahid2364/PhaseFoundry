"""Composition grids on the N-component simplex.

An N-element system has N − 1 independent mole fractions. The grid takes evenly spaced
values of each independent fraction and sets the reference element's fraction to
x_ref = 1 − Σ x_i.
"""

import itertools

import numpy as np


def simplex_grid(
    elements: list[str],
    composition_ranges: dict[str, tuple[float, float]],
    reference_element: str,
    n_points: int,
) -> np.ndarray:
    """Build a regular composition grid on the N-component simplex.

    Args:
        elements: All elements of the system, e.g. ``["HF", "NB", "TI", "V"]``.
        composition_ranges: ``(min, max)`` mole fraction of each independent element (every
            element except the reference), e.g. ``{"HF": (0.05, 0.60), "NB": (0.05, 0.60)}``.
        reference_element: Dependent element, with x_ref = 1 − Σ x_i.
        n_points: Number of grid points along each independent axis.

    Returns:
        Compositions with columns in the order of ``elements``, shape
        ``(n_valid, n_elements)``. Points where x_ref falls outside [0, 1] or any x_i is
        negative are left out.

    Raises:
        ValueError: If ``reference_element`` is not in ``elements``, an independent element
            has no range, or no valid composition remains.
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
        raise ValueError("No valid compositions found. Check composition_ranges and reference_element.")

    return np.array(rows, dtype=np.float64)


def grid_summary(grid: np.ndarray, elements: list[str]) -> str:
    """Summarize a composition grid: the point count and each element's range and mean.

    Args:
        grid: Compositions, shape ``(n_points, n_elements)``.
        elements: Element names, in column order.

    Returns:
        A multi-line summary.
    """
    lines = [
        f"Grid: {len(grid)} valid points, {len(elements)} elements",
    ]
    for i, el in enumerate(elements):
        col = grid[:, i]
        lines.append(f"  {el}: [{col.min():.4f}, {col.max():.4f}]  mean={col.mean():.4f}")
    return "\n".join(lines)
