"""phasefoundry.elasticity.elastic_potential
=========================================
Elastic contribution to the chemical potential, μ_el (Eq. 36 in
Gururajan & Abinandanan, 2007).

This module computes ONLY the elastic term.  It knows nothing about the
chemical (thermodynamic) potential μ_ch -- that is computed independently
by ``GmInterpolant.mu(c)`` in ``phasefoundry.thermodynamics.interpolant``.
The two are summed by the caller (the Cahn-Hilliard timestep loop in
``phasefoundry.phasefield.solver``):

    mu_total = mu_ch + mu_el

Governing equation (paper Eq. 36)
-----------------------------------
    N_V * mu_el(x) = 1/2 * alpha'(c) * ΔC_ijkl * eps_el_kl * eps_el_ij
                     - beta'(c) * eps_T * delta_ij * sigma_ij

where eps_el = eps - eps0(c) is the elastic strain, eps0(c) = beta(c) * eigenstrain
is the (composition-dependent) eigenstrain field, and sigma = C(c) : eps_el is
the Cauchy stress -- both produced by ``solve_elasticity()``.

We generalize the second term from the paper's purely-dilatational form
(eps_T * delta_ij * sigma_ij = eps_T * tr(sigma)) to a full tensor
contraction `eigenstrain_ij * sigma_ij`, which reduces exactly to the
paper's expression when eigenstrain = eps_T * I, but also supports
non-dilatational (e.g. tetragonal) misfit strains.
"""

from __future__ import annotations

import numpy as np

from .interpolation import InterpolationScheme, get_scheme


def build_vegard_tensor(alpha: dict, ind_elements: list[str], ndim: int = 2) -> np.ndarray:
    """Build the Vegard eigenstrain coefficient tensor Λ, shape (N_ind, ndim, ndim).

    Λ[k] = α_k × I_d,   where α_k = (a_k − a_ref) / a_ref

    For isotropic Vegard's law the per-element eigenstrain coefficient is a
    scaled identity (no shear component).  The total eigenstrain field at each
    pixel is then::

        eps*(x) = Σ_k  Λ[k] × c_k(x)   →  einsum('kxy,kij->xyij', c, Lambda)

    Parameters
    ----------
    alpha : dict  {ELEMENT_UPPER: float}
        Per-element Vegard coefficient α_k = (a_k − a_ref) / a_ref.
        Computed by step 1b and stored in elastic_params.yaml.
    ind_elements : list[str]
        Independent elements in the same order as the composition fields c[k].
    ndim : int
        Spatial dimension (2 for 2-D, 3 for 3-D).

    Returns:
    -------
    Lambda : ndarray, shape (N_ind, ndim, ndim)
    """
    N_ind = len(ind_elements)
    Lambda = np.zeros((N_ind, ndim, ndim), dtype=np.float64)
    for k, el in enumerate(ind_elements):
        Lambda[k] = float(alpha[el.upper()]) * np.eye(ndim)
    return Lambda


def compute_mu_el_tensorial(sigma: np.ndarray, Lambda: np.ndarray, N_V: float) -> np.ndarray:
    """Elastic chemical potential in the tensorial Vegard formulation.

    μ_el_k(x) = −(1/N_V) × σ(x) : Λ_k
               = −(1/N_V) × σ_ij(x) × Λ_{k,ij}

    This is the variational derivative δF_el/δc_k of the elastic free energy
    with respect to the k-th composition field, for a linear (Vegard) coupling
    between composition and eigenstrain.  The stiffness-contrast term (term 1
    in G&A Eq. 36) is omitted here because it is negligible when the stiffness
    contrast between phases is small relative to the eigenstrain coupling.

    Parameters
    ----------
    sigma  : ndarray, shape (*spatial, d, d)
        Cauchy stress field from ``solve_elasticity()``.
    Lambda : ndarray, shape (N_ind, d, d)
        Vegard eigenstrain coefficient tensor (from ``build_vegard_tensor``).
    N_V    : float
        Molar density [mol/m³] = 1/V_m.  Converts Pa → J/mol so the result
        can be added directly to μ_ch [J/mol].

    Returns:
    -------
    mu_el : ndarray, shape (N_ind, *spatial)
        Elastic chemical potential for each independent element [J/mol].
    """
    return -np.einsum("...ij,kij->k...", sigma, Lambda) / N_V


def elastic_strain(
    strain: np.ndarray, c: np.ndarray, eigenstrain: np.ndarray, scheme: InterpolationScheme | str = "cubic"
) -> np.ndarray:
    """Elastic strain  eps^el(x) = strain(x) - beta(c(x)) * eigenstrain.

    Parameters
    ----------
    strain : ndarray (N, N, 2, 2)
        Total strain field (from solve_elasticity()["strain"]).
    c : ndarray (N, N)
        Composition field.
    eigenstrain : ndarray (2, 2) or (3, 3)
        Stress-free transformation strain tensor; only the 2x2 in-plane
        block is used.
    scheme : InterpolationScheme or str
        Composition-interpolation scheme (default "cubic", matches the paper).

    Returns:
    -------
    eps_el : ndarray (N, N, 2, 2)
    """
    if isinstance(scheme, str):
        scheme = get_scheme(scheme)
    beta = scheme.beta(c)  # (N, N)
    es2 = eigenstrain[:2, :2]  # (2, 2)
    return strain - beta[..., None, None] * es2[None, None]


def elastic_free_energy_density(elastic_strain_field: np.ndarray, stress: np.ndarray) -> np.ndarray:
    """Elastic free-energy density  f_el(x) = 1/2 * sigma_ij * eps^el_ij  (Eq. 8).

    Not used in the Cahn-Hilliard timestep itself -- this exists purely so
    that mu_el can be validated against a finite-difference derivative of
    the total elastic energy  F_el = (1/N_V) * integral(f_el) dOmega.

    Parameters
    ----------
    elastic_strain_field : ndarray (N, N, 2, 2)
    stress : ndarray (N, N, 2, 2)

    Returns:
    -------
    f_el : ndarray (N, N)
    """
    return 0.5 * np.einsum("xyij,xyij->xy", stress, elastic_strain_field)


def elastic_potential(
    c: np.ndarray,
    strain: np.ndarray,
    stress: np.ndarray,
    C_mat: np.ndarray,
    C_ppt: np.ndarray,
    eigenstrain: np.ndarray,
    N_V: float = 1.0,
    scheme: InterpolationScheme | str = "cubic",
) -> np.ndarray:
    """Elastic contribution to the chemical potential, mu_el(x) (Eq. 36).

    mu_el(x) = [ 1/2 * alpha'(c) * (eps^el : ΔC : eps^el)
                 - beta'(c) * (eigenstrain : sigma) ] / N_V

    Parameters
    ----------
    c : ndarray (N, N)
        Composition field.
    strain : ndarray (N, N, 2, 2)
        Total strain field (solve_elasticity()["strain"]).
    stress : ndarray (N, N, 2, 2)
        Cauchy stress field (solve_elasticity()["stress"]).
    C_mat, C_ppt : ndarray (3, 3, 3, 3)
        Matrix / precipitate stiffness tensors.
    eigenstrain : ndarray (2, 2) or (3, 3)
        Stress-free transformation strain tensor.
    N_V : float
        Molar-volume-like normalization constant (matches the convention
        used for F_ch in the rest of the codebase; default 1.0).
    scheme : InterpolationScheme or str
        Composition-interpolation scheme (default "cubic").

    Returns:
    -------
    mu_el : ndarray (N, N)
        Elastic chemical potential, same units as mu_ch, ready to be added
        to it by the caller: ``mu_total = mu_ch + mu_el``.
    """
    if isinstance(scheme, str):
        scheme = get_scheme(scheme)

    eps_el = elastic_strain(strain, c, eigenstrain, scheme)  # (N, N, 2, 2)

    dC = (C_ppt - C_mat)[:2, :2, :2, :2]  # (2,2,2,2)
    es2 = eigenstrain[:2, :2]  # (2,2)

    dalpha = scheme.dalpha(c)  # (N, N)
    dbeta = scheme.dbeta(c)  # (N, N)

    term1 = 0.5 * dalpha * np.einsum("xyij,ijkl,xykl->xy", eps_el, dC, eps_el)  # (N, N)
    term2 = dbeta * np.einsum("ij,xyij->xy", es2, stress)  # (N, N)

    return (term1 - term2) / N_V


def compute_mu_el_from_result(
    c: np.ndarray,
    result: dict,
    C_mat: np.ndarray,
    C_ppt: np.ndarray,
    eigenstrain: np.ndarray,
    N_V: float = 1.0,
    scheme: InterpolationScheme | str = "cubic",
) -> np.ndarray:
    """Compute mu_el directly from a ``solve_elasticity()`` result dict.

    Equivalent to::

        elastic_potential(c, result["strain"], result["stress"], C_mat, C_ppt, eigenstrain, N_V, scheme)

    Parameters
    ----------
    c : ndarray (N, N)
    result : dict
        Output of ``phasefoundry.elasticity.solve_elasticity()``.
    C_mat, C_ppt : ndarray (3, 3, 3, 3)
    eigenstrain : ndarray (2, 2) or (3, 3)
    N_V : float
    scheme : InterpolationScheme or str

    Returns:
    -------
    mu_el : ndarray (N, N)
    """
    return elastic_potential(
        c,
        result["strain"],
        result["stress"],
        C_mat,
        C_ppt,
        eigenstrain,
        N_V=N_V,
        scheme=scheme,
    )
