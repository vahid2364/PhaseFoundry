"""
thermodynamics — pre-compute and store CALPHAD thermodynamic tables.

Quick start
-----------
    from thermodynamics import generate_table, load_table

    generate_table(
        db_path='PhaseFoundry/HF_NB_TI_V.TDB',
        elements=['HF', 'NB', 'TI', 'V'],
        phase='BCC_A2',
        composition_ranges={'HF': (0.05, 0.60), 'NB': (0.05, 0.60), 'TI': (0.05, 0.60)},
        reference_element='V',
        n_points=20,
        temperature=1000,
        output_path='data/hfnbtiv_1000K.h5',
    )

    data = load_table('data/hfnbtiv_1000K.h5')
    # data['compositions']  shape (n, 4)
    # data['GM']            shape (n,)   J/mol
    # data['MU']['HF']      shape (n,)   J/mol
    # data['dMU']['HF_NB']  shape (n,)   J/mol
"""

from .generate import compute_table
from .grid import simplex_grid, grid_summary
from .storage import load_table, save_table, table_info
from .plot import (
    plot_free_energy_vs_composition,
    plot_chemical_potential_vs_composition,
    plot_ternary_spinodal,
    plot_as_umap,
)
from .spinodal import (
    spinodal_boundary_binary, spinodal_boundary_multi, plot_spinodal,
    binodal_boundary_binary, plot_phase_diagram, recommend_pfm_params,
    compute_ternary_phase_regions, compute_phase_regions_multi,
)
from .fitting import fit_delta_mu, save_fits, load_fits, eval_delta_mu, build_spline_surrogate, build_gm_spline_surrogate
from .interpolant import build_interpolant, GmInterpolant


def generate_table(
    db_path: str,
    elements: list[str],
    phase: str,
    composition_ranges: dict[str, tuple[float, float]],
    reference_element: str,
    n_points: int,
    temperature: float,
    pressure: float = 101325.0,
    output_path: str = "thermodynamic_table.h5",
    n_workers: int = 1,
    quantities: tuple[str, ...] = ("GM", "MU", "dMU"),
    sampling: str = "sobol",
    seed: int = 42,
) -> dict:
    """
    Build a composition grid, evaluate pycalphad, and save to HDF5.

    Parameters
    ----------
    db_path : str
        Path to the TDB thermodynamic database file.
    elements : list[str]
        All elements, e.g. ['HF', 'NB', 'TI', 'V'].
        The reference element should be last or specified via
        ``reference_element``.
    phase : str
        Phase name, e.g. 'BCC_A2'.
    composition_ranges : dict
        Mole-fraction (min, max) for each *independent* element
        (every element except the reference).
        Example: {'HF': (0.05, 0.60), 'NB': (0.05, 0.60), 'TI': (0.05, 0.60)}
    reference_element : str
        Dependent element. Its fraction is computed as
        x_ref = 1 - sum(others).
    n_points : int
        For sampling="grid": grid lines per axis.
        For sampling="sobol"/"dirichlet": total valid points returned.
    temperature : float
        Temperature in Kelvin.
    pressure : float
        Pressure in Pa. Default 101325.
    output_path : str
        Where to write the HDF5 file.
    n_workers : int
        Worker processes for parallel evaluation. Default 1 (serial).
    quantities : tuple[str]
        Subset of ('GM', 'MU', 'dMU') to compute.
    sampling : {"grid", "sobol", "dirichlet"}
        Composition sampling strategy.  "grid" (default) uses a Cartesian
        linspace approach per element axis.  "sobol" gives quasi-random
        uniform coverage of the constrained simplex.  "dirichlet" gives
        pseudo-random uniform simplex samples.
    seed : int
        Random seed for "sobol" and "dirichlet" modes.  Default 42.

    Returns
    -------
    dict — same as load_table(output_path) after saving.
    """
    # 1. Build grid
    grid = simplex_grid(elements, composition_ranges, reference_element,
                        n_points, sampling=sampling, seed=seed)
    print(grid_summary(grid, elements))

    # Ensure reference element is last in the list (required by compute_table)
    ordered = [e for e in elements if e != reference_element] + [reference_element]

    # Reorder grid columns to match ordered
    orig_idx = {el: elements.index(el) for el in elements}
    grid = grid[:, [orig_idx[el] for el in ordered]]

    # 2. Evaluate pycalphad
    data = compute_table(
        db_path=db_path,
        elements=ordered,
        phase=phase,
        grid=grid,
        temperature=temperature,
        pressure=pressure,
        quantities=quantities,
        n_workers=n_workers,
    )

    # 3. Save
    save_table(
        path=output_path,
        data=data,
        db_path=db_path,
        elements=ordered,
        phase=phase,
        temperature=temperature,
        pressure=pressure,
        composition_ranges=composition_ranges,
        n_points=n_points,
    )

    return load_table(output_path)


__all__ = [
    "generate_table",
    "load_table",
    "save_table",
    "table_info",
    "simplex_grid",
    "grid_summary",
    "compute_table",
    "plot_free_energy_vs_composition",
    "plot_chemical_potential_vs_composition",
]
