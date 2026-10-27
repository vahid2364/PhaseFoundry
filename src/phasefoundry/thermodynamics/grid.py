"""
Composition grid / sample generation on the N-component simplex.

Sampling modes
--------------
grid      -- Cartesian linspace grid per element axis, filtered to the simplex.
             ``n_points`` is the number of grid lines *per axis*; total valid
             points depends on the composition ranges.  Good for binary systems.
sobol     -- Quasi-random low-discrepancy Sobol sequence mapped uniformly onto
             the simplex via stick-breaking.  ``n_points`` is the *total* number
             of valid samples returned.  Requires scipy >= 1.7.
dirichlet -- Pseudo-random uniform samples from Dirichlet(1,...,1), filtered by
             composition ranges.  ``n_points`` is the total number returned.
"""

from __future__ import annotations

import warnings

import numpy as np


# ── public entry point ────────────────────────────────────────────────────────

def simplex_grid(
    elements: list[str],
    composition_ranges: dict[str, tuple[float, float]],
    reference_element: str,
    n_points: int,
    sampling: str = "sobol",
    seed: int = 42,
) -> np.ndarray:
    """
    Return a composition array of shape ``(n_valid, N_elements)`` on the simplex.

    Parameters
    ----------
    elements : list[str]
        All elements including the reference element.
    composition_ranges : dict[str, (float, float)]
        Min/max mole fraction for each independent element (all except reference).
    reference_element : str
        Dependent element; its fraction is computed as 1 − Σ others.
    n_points : int
        For ``sampling="grid"``: grid lines per axis.
        For ``sampling="sobol"`` or ``"dirichlet"``: total valid points to return.
    sampling : {"sobol", "dirichlet", "grid"}
        Sampling strategy.  Default ``"sobol"``.
    seed : int
        Random seed (used by sobol and dirichlet).

    Returns
    -------
    np.ndarray, shape (n_valid, N_elements)
        Rows ordered by `elements`; all rows satisfy x_i ≥ 0, Σ x_i = 1.
    """
    if reference_element not in elements:
        raise ValueError(
            f"reference_element '{reference_element}' not in elements {elements}"
        )
    independent = [e for e in elements if e != reference_element]
    for el in independent:
        if el not in composition_ranges:
            raise ValueError(f"No composition range specified for element '{el}'")

    n_el = len(elements)

    if sampling == "grid" or n_el == 2:
        # Simplex lattice grid: symmetric for all numbers of components.
        # All three (or more) element axes are sampled equally, including
        # the reference element's axis.
        return _simplex_lattice_grid(elements, composition_ranges,
                                     reference_element, independent, n_points)

    if sampling == "sobol":
        return _sobol_sample(elements, composition_ranges, reference_element,
                             independent, n_el, n_points, seed)

    if sampling == "dirichlet":
        return _dirichlet_sample(elements, composition_ranges, reference_element,
                                 independent, n_el, n_points, seed)

    raise ValueError(f"Unknown sampling mode '{sampling}'. "
                     "Choose from 'sobol', 'dirichlet', 'grid'.")


# ── sampling backends ─────────────────────────────────────────────────────────

def _simplex_lattice_grid(elements, composition_ranges, reference_element,
                          independent, n_points):
    """
    Symmetric simplex lattice grid for any number of components.

    Generates all non-negative integer tuples (i_1, …, i_k) with
    sum = scale = n_points - 1, then maps each to a composition
    x_j = i_j / scale.  This is the standard stars-and-bars lattice
    on the (k-1)-simplex and treats every element axis — including the
    reference element — symmetrically.

    Total lattice points before filtering:
        C(scale + n_el - 1, n_el - 1)
    e.g. ternary n=15 → C(17,2)=136, quaternary n=15 → C(18,3)=816,
         pentenary n=10 → C(13,4)=715.

    For a binary system (n_el == 2) this general lattice-then-filter
    approach always discards exactly 2 points (the true x=0 and x=1
    endpoints, which fall outside any composition_ranges that excludes
    the pure elements), returning n_points - 2 valid rows rather than
    n_points. That generality is unnecessary with only one independent
    axis, so the binary case is handled separately below by sampling
    composition_ranges directly: every requested point is used, and the
    returned grid spans the configured range exactly.
    """
    n_el = len(elements)

    if n_el == 2:
        el     = independent[0]
        lo, hi = composition_ranges[el]
        x_ind  = np.linspace(lo, hi, n_points)
        return np.array(
            [[xi, 1.0 - xi] if elements[0] == el else [1.0 - xi, xi]
             for xi in x_ind],
            dtype=np.float64,
        )

    from itertools import combinations

    scale = n_points - 1

    # Warn if the unfiltered lattice is very large
    from math import comb
    n_total = comb(scale + n_el - 1, n_el - 1)
    if n_total > 200_000:
        warnings.warn(
            f"Simplex lattice has {n_total:,} points before filtering "
            f"(n_points={n_points}, {n_el} elements). "
            "Consider reducing n_points or using sampling='sobol'.",
            stacklevel=4,
        )

    rows = []
    # Stars-and-bars: place n_el-1 dividers among scale+n_el-1 positions
    for c in combinations(range(scale + n_el - 1), n_el - 1):
        extended = (-1,) + c + (scale + n_el - 1,)
        coords   = tuple(extended[k + 1] - extended[k] - 1 for k in range(n_el))
        x = {el: coords[j] / scale for j, el in enumerate(elements)}
        # Filter by composition ranges for independent elements only;
        # the reference element is unconstrained (it is whatever remains).
        if all(composition_ranges[el][0] <= x[el] <= composition_ranges[el][1]
               for el in independent):
            rows.append([x[el] for el in elements])

    if not rows:
        raise ValueError(
            "No valid compositions found. Check composition_ranges."
        )
    return np.array(rows, dtype=np.float64)


def _sobol_sample(elements, composition_ranges, reference_element,
                  independent, n_el, n_target, seed):
    """Quasi-random Sobol sequence mapped to the simplex via stick-breaking."""
    try:
        from scipy.stats.qmc import Sobol
    except ImportError:
        warnings.warn(
            "scipy >= 1.7 is required for Sobol sampling; falling back to dirichlet.",
            stacklevel=3,
        )
        return _dirichlet_sample(elements, composition_ranges, reference_element,
                                 independent, n_el, n_target, seed)

    k = n_el - 1          # simplex dimension (number of independent vars)
    rows = []
    batch_size = max(n_target * 8, 512)
    sampler = Sobol(d=k, scramble=True, seed=seed)

    while len(rows) < n_target:
        # Draw Sobol points in [0,1]^k
        u = sampler.random(batch_size)        # shape (batch, k)
        # Stick-breaking: sort each row, take differences → uniform on k-simplex
        u_sorted = np.sort(u, axis=1)
        # Prepend 0 and append 1 to compute differences
        padded = np.concatenate(
            [np.zeros((batch_size, 1)), u_sorted, np.ones((batch_size, 1))],
            axis=1,
        )                                     # shape (batch, k+2)
        x_all = np.diff(padded, axis=1)       # shape (batch, k+1) — simplex coords

        # Map columns to elements: first k → independent, last → reference
        for row in x_all:
            x = dict(zip(independent, row[:k]))
            x[reference_element] = row[k]
            if _in_bounds(x, independent, composition_ranges):
                rows.append([x[el] for el in elements])
                if len(rows) >= n_target:
                    break

        batch_size = min(batch_size * 2, 65536)  # grow batch if hit-rate is low

    return np.array(rows[:n_target], dtype=np.float64)


def _dirichlet_sample(elements, composition_ranges, reference_element,
                      independent, n_el, n_target, seed):
    """Pseudo-random uniform samples via Dirichlet(1,...,1), rejection-filtered."""
    rng = np.random.default_rng(seed)
    rows = []
    batch_size = max(n_target * 8, 512)

    while len(rows) < n_target:
        samples = rng.dirichlet(np.ones(n_el), size=batch_size)
        for s in samples:
            x = dict(zip(elements, s))
            if _in_bounds(x, independent, composition_ranges):
                rows.append([x[el] for el in elements])
                if len(rows) >= n_target:
                    break
        batch_size = min(batch_size * 2, 65536)

    return np.array(rows[:n_target], dtype=np.float64)


# ── helpers ───────────────────────────────────────────────────────────────────

def _in_bounds(x: dict, independent: list[str],
               composition_ranges: dict) -> bool:
    return all(
        composition_ranges[el][0] <= x[el] <= composition_ranges[el][1]
        for el in independent
    )


def grid_summary(grid: np.ndarray, elements: list[str]) -> str:
    """Return a short human-readable summary of a composition grid."""
    lines = [f"Grid: {len(grid)} valid points, {len(elements)} elements"]
    for i, el in enumerate(elements):
        col = grid[:, i]
        lines.append(
            f"  {el}: [{col.min():.4f}, {col.max():.4f}]  mean={col.mean():.4f}"
        )
    return "\n".join(lines)
