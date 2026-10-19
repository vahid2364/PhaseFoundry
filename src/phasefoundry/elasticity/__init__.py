"""FFT-based elasticity for periodic microstructures.

The solver uses the displacement formulation of Gururajan and Abinandanan (2007), "Phase
field study of precipitate rafting under a uniaxial stress", Acta Materialia 55(15),
5015-5026. ``solve_elasticity`` finds mechanical equilibrium for a composition field,
``interpolation`` holds the composition-interpolation functions for the stiffness and
eigenstrain, and ``elastic_potential`` and ``models`` give the elastic contribution μ_el
to the chemical potential. μ_el is only the elastic term; the caller adds it to the
chemical potential from ``GmInterpolant.mu``.
"""

from phasefoundry.elasticity.bond_model import BondLatticeModel
from phasefoundry.elasticity.elastic_potential import (
    build_vegard_tensor,
    compute_mu_el_from_result,
    compute_mu_el_tensorial,
    elastic_free_energy_density,
    elastic_potential,
    elastic_strain,
)
from phasefoundry.elasticity.fft_solver import build_cubic_stiffness, build_isotropic_stiffness, solve_elasticity
from phasefoundry.elasticity.interpolation import InterpolationScheme, get_scheme, register_scheme
from phasefoundry.elasticity.models import EigenstrainModel, FixedVegardModel, LocalVegardModel, build_eigenstrain_model

__all__ = [
    "BondLatticeModel",
    "EigenstrainModel",
    "FixedVegardModel",
    "InterpolationScheme",
    "LocalVegardModel",
    "build_cubic_stiffness",
    "build_eigenstrain_model",
    "build_isotropic_stiffness",
    "build_vegard_tensor",
    "compute_mu_el_from_result",
    "compute_mu_el_tensorial",
    "elastic_free_energy_density",
    "elastic_potential",
    "elastic_strain",
    "get_scheme",
    "register_scheme",
    "solve_elasticity",
]
