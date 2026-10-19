"""Elastic contribution μ_el to the chemical potential (Gururajan and Abinandanan 2007, Eq. 36).

This module computes only the elastic term. The chemical part μ_ch comes from
``GmInterpolant.mu`` in ``phasefoundry.thermodynamics.interpolant``, and the
Cahn-Hilliard loop in ``phasefoundry.phasefield.solver`` adds the two:
``mu_total = mu_ch + mu_el``.

The paper's Eq. 36 is

    N_V μ_el(x) = ½ α'(c) ΔC_ijkl ε^el_kl ε^el_ij − β'(c) ε_T δ_ij σ_ij

where ε^el = ε − ε0(c) is the elastic strain, ε0(c) = β(c) ε* is the eigenstrain field,
and σ = C(c) : ε^el is the Cauchy stress, both from ``solve_elasticity``. Here the second
term is the full contraction ε*_ij σ_ij instead of the purely dilatational
ε_T δ_ij σ_ij = ε_T tr(σ). The two agree when ε* = ε_T I, and the tensor form also
handles non-dilatational (for example tetragonal) misfit.

``elastic_strain``, ``elastic_free_energy_density``, ``elastic_potential`` and
``compute_mu_el_from_result`` handle 2-D fields only. ``build_vegard_tensor`` and
``compute_mu_el_tensorial`` work in 2-D and 3-D.
"""

from __future__ import annotations

import numpy as np

from phasefoundry.elasticity.interpolation import InterpolationScheme, get_scheme


def build_vegard_tensor(alpha: dict, ind_elements: list[str], ndim: int = 2) -> np.ndarray:
    """Build the Vegard eigenstrain coefficient tensor Λ.

    Λ[k] = α_k I, with α_k = (a_k − a_ref) / a_ref. Isotropic Vegard's law gives a scaled
    identity for each element, with no shear part. The eigenstrain field is then
    ε*(x) = Σ_k Λ[k] c_k(x), i.e. ``einsum("k...,kij->...ij", c, Lambda)``.

    Args:
        alpha: Vegard coefficient α_k of each element, keyed by uppercase symbol. Step 1b
            computes these and stores them in ``elastic_params.yaml``.
        ind_elements: Independent elements, in the order of the composition fields.
        ndim: Spatial dimension, 2 or 3.

    Returns:
        Λ, shape ``(n_ind, ndim, ndim)``.
    """
    N_ind = len(ind_elements)
    Lambda = np.zeros((N_ind, ndim, ndim), dtype=np.float64)
    for k, el in enumerate(ind_elements):
        Lambda[k] = float(alpha[el.upper()]) * np.eye(ndim)
    return Lambda


def compute_mu_el_tensorial(sigma: np.ndarray, Lambda: np.ndarray, N_V: float) -> np.ndarray:
    """Return the elastic chemical potential in the tensorial Vegard form.

    μ_el,k(x) = −σ(x) : Λ_k / N_V is the variational derivative δF_el/δc_k for a linear
    (Vegard) coupling between composition and eigenstrain. The stiffness-contrast term
    (term 1 of Eq. 36) is neglected, which holds when the stiffness contrast between the
    phases is small compared with the eigenstrain coupling.

    Args:
        sigma: Cauchy stress from ``solve_elasticity``, shape ``(*spatial, d, d)``.
        Lambda: Vegard tensor from ``build_vegard_tensor``, shape ``(n_ind, d, d)``.
        N_V: Molar density in mol/m³ (1 / molar volume). Dividing by it converts Pa to
            J/mol, so the result can be added to μ_ch.

    Returns:
        μ_el in J/mol for each independent element, shape ``(n_ind, *spatial)``.
    """
    return -np.einsum("...ij,kij->k...", sigma, Lambda) / N_V


def elastic_strain(
    strain: np.ndarray, c: np.ndarray, eigenstrain: np.ndarray, scheme: InterpolationScheme | str = "cubic"
) -> np.ndarray:
    """Return the elastic strain ε^el(x) = ε(x) − β(c(x)) ε* of a 2-D field.

    Args:
        strain: Total strain, e.g. ``solve_elasticity(...)["strain"]``, shape ``(N, N, 2, 2)``.
        c: Composition field, shape ``(N, N)``.
        eigenstrain: Eigenstrain tensor ε*, shape ``(2, 2)`` or ``(3, 3)``. Only the
            in-plane 2 × 2 block is used.
        scheme: Interpolation scheme, or its name. Defaults to the paper's ``"cubic"``.

    Returns:
        Elastic strain, shape ``(N, N, 2, 2)``.
    """
    if isinstance(scheme, str):
        scheme = get_scheme(scheme)
    beta = scheme.beta(c)
    es2 = eigenstrain[:2, :2]
    return strain - beta[..., None, None] * es2[None, None]


def elastic_free_energy_density(elastic_strain_field: np.ndarray, stress: np.ndarray) -> np.ndarray:
    """Return the elastic free-energy density f_el(x) = ½ σ_ij ε^el_ij (Eq. 8) of a 2-D field.

    The Cahn-Hilliard step does not use it. It is here to check μ_el against a
    finite-difference derivative of the total elastic energy F_el = (1 / N_V) ∫ f_el dΩ.

    Args:
        elastic_strain_field: Elastic strain, shape ``(N, N, 2, 2)``.
        stress: Cauchy stress, shape ``(N, N, 2, 2)``.

    Returns:
        Energy density in the unit of stress, shape ``(N, N)``.
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
    """Return the elastic contribution μ_el(x) to the chemical potential (Eq. 36) of a 2-D field.

    μ_el(x) = [½ α'(c) (ε^el : ΔC : ε^el) − β'(c) (ε* : σ)] / N_V

    Args:
        c: Composition field, shape ``(N, N)``.
        strain: Total strain, ``solve_elasticity(...)["strain"]``, shape ``(N, N, 2, 2)``.
        stress: Cauchy stress, ``solve_elasticity(...)["stress"]``, shape ``(N, N, 2, 2)``.
        C_mat: Matrix stiffness, shape ``(3, 3, 3, 3)``.
        C_ppt: Precipitate stiffness, shape ``(3, 3, 3, 3)``.
        eigenstrain: Eigenstrain tensor ε*, shape ``(2, 2)`` or ``(3, 3)``.
        N_V: Molar density in mol/m³. The default of 1.0 leaves the result in the unit of
            stress.
        scheme: Interpolation scheme, or its name. Defaults to ``"cubic"``.

    Returns:
        μ_el, shape ``(N, N)``. With ``N_V`` in mol/m³ and stresses in Pa it is in J/mol,
        ready to add to μ_ch.
    """
    if isinstance(scheme, str):
        scheme = get_scheme(scheme)

    eps_el = elastic_strain(strain, c, eigenstrain, scheme)

    dC = (C_ppt - C_mat)[:2, :2, :2, :2]
    es2 = eigenstrain[:2, :2]

    dalpha = scheme.dalpha(c)
    dbeta = scheme.dbeta(c)

    term1 = 0.5 * dalpha * np.einsum("xyij,ijkl,xykl->xy", eps_el, dC, eps_el)
    term2 = dbeta * np.einsum("ij,xyij->xy", es2, stress)

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
    """Compute μ_el of a 2-D field from a ``solve_elasticity`` result.

    This is ``elastic_potential`` with ``result["strain"]`` and ``result["stress"]`` as
    the strain and stress.

    Args:
        c: Composition field, shape ``(N, N)``.
        result: Output of ``solve_elasticity``.
        C_mat: Matrix stiffness, shape ``(3, 3, 3, 3)``.
        C_ppt: Precipitate stiffness, shape ``(3, 3, 3, 3)``.
        eigenstrain: Eigenstrain tensor ε*, shape ``(2, 2)`` or ``(3, 3)``.
        N_V: Molar density in mol/m³.
        scheme: Interpolation scheme, or its name.

    Returns:
        μ_el, shape ``(N, N)``.
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
