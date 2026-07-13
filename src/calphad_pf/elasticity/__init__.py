"""
calphad_pf.elasticity — FFT-based elasticity solver for periodic microstructures
==================================================================================

Implements the displacement-formulation iterative scheme from:
    Gururajan, M.P. & Abinandanan, T.A. (2007).
    "Phase field study of precipitate rafting under a uniaxial stress."
    Acta Materialia, 55(15), 5015-5026.

Public API
----------
solve_elasticity(c, C_mat, C_ppt, eigenstrain, sigma_applied=None, tol=1e-6, max_iter=1000)
    Solve the mechanical equilibrium for a two-phase microstructure described
    by composition field `c` (values in [0,1]).
get_scheme(name) / InterpolationScheme
    Composition-interpolation functions (h, alpha, beta and derivatives) for
    the stiffness and eigenstrain fields. See `interpolation.py`.
elastic_potential(...) / compute_mu_el_from_result(...)
    Elastic contribution mu_el to the chemical potential (Eq. 36).
    Computes ONLY the elastic term -- the chemical potential mu_ch is
    computed separately by GmInterpolant.mu(c); the caller sums them:
    ``mu_total = mu_ch + mu_el``. See `elastic_potential.py`.
"""

from .fft_solver import solve_elasticity, build_isotropic_stiffness, build_cubic_stiffness
from .models import (
    EigenstrainModel,
    FixedVegardModel,
    LocalVegardModel,
    build_eigenstrain_model,
)
from .interpolation import (
    InterpolationScheme,
    get_scheme,
    register_scheme,
)
from .elastic_potential import (
    elastic_strain,
    elastic_free_energy_density,
    elastic_potential,
    compute_mu_el_from_result,
    build_vegard_tensor,
    compute_mu_el_tensorial,
)

__all__ = [
    "solve_elasticity",
    "build_isotropic_stiffness",
    "build_cubic_stiffness",
    "InterpolationScheme",
    "get_scheme",
    "register_scheme",
    "elastic_strain",
    "elastic_free_energy_density",
    "elastic_potential",
    "compute_mu_el_from_result",
    "build_vegard_tensor",
    "compute_mu_el_tensorial",
    "EigenstrainModel",
    "FixedVegardModel",
    "LocalVegardModel",
    "build_eigenstrain_model",
]
