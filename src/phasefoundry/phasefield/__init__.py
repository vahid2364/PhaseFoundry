"""Cahn-Hilliard phase-field solvers for multicomponent alloys.

``solve_cahn_hilliard`` solves the Cahn-Hilliard equation for any number of components
on 2-D or 3-D grids, with optional elastic coupling. ``solve_cahn_hilliard_binary`` is a
binary-only solver whose free energy is either a synthetic double well, for benchmarks,
or a CALPHAD surrogate fitted with ``phasefoundry.thermodynamics``. ``visualize`` saves
images of the composition field.

Examples:
    A double-well benchmark, then a CALPHAD run for Nb-V at 500 °C. The fit file is a
    placeholder for the output of ``phasefoundry.thermodynamics.save_fits``.

    ```python
    from phasefoundry.phasefield import plot_result, solve_cahn_hilliard_binary
    from phasefoundry.thermodynamics import load_fits

    result = solve_cahn_hilliard_binary(free_energy="double_well", N=256, Lx=200, kappa=0.5, M=1.0, c0=0.4, dt=0.1, n_steps=2000)
    plot_result(result)

    fit = load_fits("delta_mu_fits.h5")
    result = solve_cahn_hilliard_binary(
        free_energy="calphad",
        N=256,
        Lx=100e-6,
        kappa=40e-19,
        M=1.0,
        c0=0.4,
        dt=5e-7,
        n_steps=2000,
        calphad_fit=fit,
        temperature_K=773.15,
    )
    plot_result(result)
    ```
"""

from .binary import (
    CHResult,
    plot_free_energy_history,
    plot_gm_with_pfm,
    plot_result,
    solve_cahn_hilliard_binary,
    stable_dt,
    write_simulation_params,
)
from .free_energy import calphad_df_dc, calphad_f, double_well_df_dc, double_well_f
from .solver import SimResult, solve_cahn_hilliard
from .spectral import laplacian_operator, wavenumbers
from .visualize import save_pyvista_3d, save_slices_2d

__all__ = [
    # general solver
    "solve_cahn_hilliard",
    "SimResult",
    # visualization
    "save_slices_2d",
    "save_pyvista_3d",
    # binary (legacy, kept for backward compatibility)
    "solve_cahn_hilliard_binary",
    "plot_result",
    "CHResult",
    "stable_dt",
    "double_well_df_dc",
    "double_well_f",
    "calphad_df_dc",
    "laplacian_operator",
    "wavenumbers",
]
