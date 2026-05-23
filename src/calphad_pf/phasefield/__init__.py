"""
phasefield — Phase-field solvers for multicomponent alloy systems.

Solvers
-------
binary    Cahn-Hilliard equation for binary alloys (this module)
ternary   (future)
quaternary (future)

Free energy models
------------------
double_well   Synthetic polynomial double-well (benchmarking)
calphad       CALPHAD-derived Δμ via polynomial surrogate

Quick start — double-well
--------------------------
    from phasefield.binary import solve_cahn_hilliard_binary, plot_result

    result = solve_cahn_hilliard_binary(
        free_energy='double_well',
        N=256, Lx=200, kappa=0.5, M=1.0,
        c0=0.4, dt=0.1, n_steps=2000,
    )
    plot_result(result)

Quick start — CALPHAD (Nb-V)
-----------------------------
    from thermodynamics.fitting import load_fits
    from phasefield.binary import solve_cahn_hilliard_binary, plot_result

    fit = load_fits('pipelines/output/nb_v/delta_mu_fits.h5')

    result = solve_cahn_hilliard_binary(
        free_energy='calphad',
        N=256, Lx=100e-6, kappa=40e-19, M=1.0,
        c0=0.4, dt=5e-7, n_steps=2000,
        calphad_fit=fit, temperature_K=773.15,   # 500 °C
    )
    plot_result(result)
"""

from .binary import solve_cahn_hilliard_binary, plot_result, CHResult, stable_dt, write_simulation_params, plot_free_energy_history, plot_gm_with_pfm
from .free_energy import double_well_df_dc, double_well_f, calphad_df_dc, calphad_f
from .spectral import laplacian_operator, wavenumbers

__all__ = [
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
