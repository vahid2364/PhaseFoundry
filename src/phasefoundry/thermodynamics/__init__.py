"""CALPHAD thermodynamic tables: computation, storage, surrogates and phase analysis.

Tables of G_m, chemical potentials and their composition derivatives are computed with
pycalphad over a composition grid and stored in HDF5. The subpackage also fits surrogates
to those tables, builds the free-energy interpolant used by the phase-field solver, finds
spinodal and binodal boundaries, and plots the results.

Examples:
    ```python
    from phasefoundry.thermodynamics import generate_table, load_table

    generate_table(
        db_path="data/tdb/HfNbTiV_BCC_PhaseForge_v3.tdb",
        elements=["HF", "NB", "TI", "V"],
        phase="BCC_A2",
        composition_ranges={"HF": (0.05, 0.60), "NB": (0.05, 0.60), "TI": (0.05, 0.60)},
        reference_element="V",
        n_points=20,
        temperature=1000,
        output_path="results/hfnbtiv_1000K.h5",
    )

    data = load_table("results/hfnbtiv_1000K.h5")
    data["compositions"]  # shape (n, 4), columns HF, NB, TI, V
    data["GM"]  # shape (n,), J/mol
    data["MU"]["HF"]  # shape (n,), J/mol
    data["dMU"]["HF_NB"]  # shape (n,), dμ_HF/dx_NB in J/mol
    ```
"""

from .fitting import build_gm_spline_surrogate, build_spline_surrogate, eval_delta_mu, fit_delta_mu, load_fits, save_fits
from .generate import Quantities, compute_table
from .grid import grid_summary, simplex_grid
from .interpolant import GmInterpolant, build_interpolant
from .plot import (
    plot_as_umap,
    plot_chemical_potential_vs_composition,
    plot_free_energy_vs_composition,
    plot_ternary_spinodal,
)
from .spinodal import (
    binodal_boundary_binary,
    compute_phase_regions_multi,
    compute_ternary_phase_regions,
    plot_phase_diagram,
    plot_spinodal,
    recommend_pfm_params,
    spinodal_boundary_binary,
    spinodal_boundary_multi,
)
from .storage import load_table, save_table, table_info


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
    quantities: Quantities = ("GM", "MU", "dMU"),
) -> dict:
    """Build a composition grid, evaluate it with pycalphad and save the table to HDF5.

    The table stores the elements with the reference element moved to the last column,
    which is the order ``compute_table`` needs.

    Args:
        db_path: Path to the TDB database.
        elements: All elements of the system, e.g. ``["HF", "NB", "TI", "V"]``, in any order.
        phase: Phase name, e.g. ``"BCC_A2"``.
        composition_ranges: ``(min, max)`` mole fraction of each independent element (every
            element except the reference), e.g. ``{"HF": (0.05, 0.60), "NB": (0.05, 0.60)}``.
        reference_element: Dependent element, with x_ref = 1 − Σ x_i.
        n_points: Number of grid points along each independent axis.
        temperature: Temperature in K.
        pressure: Pressure in Pa.
        output_path: Path of the HDF5 file to write.
        n_workers: Number of worker processes. ``1`` runs serially.
        quantities: Subset of ``("GM", "MU", "dMU")`` to compute.

    Returns:
        The saved table, as returned by ``load_table(output_path)``.
    """
    grid = simplex_grid(elements, composition_ranges, reference_element, n_points)
    print(grid_summary(grid, elements))

    # Ensure reference element is last in the list (required by compute_table)
    ordered = [e for e in elements if e != reference_element] + [reference_element]

    orig_idx = {el: elements.index(el) for el in elements}
    grid = grid[:, [orig_idx[el] for el in ordered]]

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
    "GmInterpolant",
    "binodal_boundary_binary",
    "build_gm_spline_surrogate",
    "build_interpolant",
    "build_spline_surrogate",
    "compute_phase_regions_multi",
    "compute_table",
    "compute_ternary_phase_regions",
    "eval_delta_mu",
    "fit_delta_mu",
    "generate_table",
    "grid_summary",
    "load_fits",
    "load_table",
    "plot_as_umap",
    "plot_chemical_potential_vs_composition",
    "plot_free_energy_vs_composition",
    "plot_phase_diagram",
    "plot_spinodal",
    "plot_ternary_spinodal",
    "recommend_pfm_params",
    "save_fits",
    "save_table",
    "simplex_grid",
    "spinodal_boundary_binary",
    "spinodal_boundary_multi",
    "table_info",
]
