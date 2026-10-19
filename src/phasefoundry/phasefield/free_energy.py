"""Free energy models for the binary Cahn-Hilliard equation.

Each model exposes a driving-force function (and a matching f(c)):

    df_dc(c, **params) -> np.ndarray

which returns the bulk chemical potential driving force dF/dc at every
grid point, given the composition field c (shape N×N, values in [0,1]).

Available models
----------------
double_well   : Symmetric polynomial double-well
                f(c) = ρ_s (c − c_a)² (c − c_b)²
                Parameters: rho_s, c_a, c_b

calphad       : CALPHAD-derived Δμ from polynomial surrogate
                Δμ(c) = μ_component − μ_reference
                Parameters: fit (dict from thermodynamics.fitting.load_fits),
                            temperature_K (float)
"""

from __future__ import annotations

import numpy as np


def double_well_df_dc(
    c: np.ndarray,
    rho_s: float = 5.0,
    c_a: float = 0.3,
    c_b: float = 0.7,
) -> np.ndarray:
    """Bulk chemical potential for a symmetric double-well free energy.

    Free energy density:
        f(c) = ρ_s (c − c_a)² (c − c_b)²

    Derivative (analytic):
        df/dc = 2 ρ_s (c − c_a)(c − c_b)(c_a + c_b − 2c)
              = −2 ρ_s (c − c_a)(c − c_b)(2c − c_a − c_b)

    Parameters
    ----------
    c : np.ndarray
        Composition field.
    rho_s : float
        Free energy barrier height.
    c_a, c_b : float
        Equilibrium compositions of the two phases (well minima).

    Returns:
    -------
    np.ndarray  dF/dc  (same shape as c)
    """
    return -2.0 * rho_s * (c - c_a) * (c - c_b) * (c_a + c_b - 2.0 * c)


def double_well_f(
    c: np.ndarray,
    rho_s: float = 5.0,
    c_a: float = 0.3,
    c_b: float = 0.7,
) -> np.ndarray:
    """Free energy density f(c) = ρ_s (c − c_a)² (c − c_b)²."""
    return rho_s * (c - c_a) ** 2 * (c - c_b) ** 2


def calphad_df_dc(
    c: np.ndarray,
    fit: dict,
    temperature_K: float,
    x_element_col: int = 0,
) -> np.ndarray:
    """Bulk chemical potential Δμ from a CALPHAD polynomial surrogate.

    Retrieves Δμ = μ_i − μ_ref at every composition point using the
    polynomial coefficients stored by thermodynamics.fitting.save_fits().
    The surrogate is evaluated without sklearn — only NumPy is required.

    Parameters
    ----------
    c : np.ndarray, shape (N, N)
        Composition field (mole fraction of the independent element).
    fit : dict
        Output of ``thermodynamics.fitting.load_fits()``.
        Must contain the requested temperature.
    temperature_K : float
        Temperature in Kelvin. Must match one of the stored temperatures.
    x_element_col : int
        Column index of the composition variable in the surrogate
        (always 0 for a binary system).

    Returns:
    -------
    np.ndarray  Δμ  [J/mol]  (same shape as c)
    """
    stored_T = np.array(fit["temperatures_K"])
    idx = int(np.argmin(np.abs(stored_T - temperature_K)))
    T_match = stored_T[idx]

    tol = 1.0  # K
    if abs(T_match - temperature_K) > tol:
        raise ValueError(
            f"Temperature {temperature_K} K not found in fit file. "
            f"Available: {sorted(stored_T.tolist())}. "
            f"Closest is {T_match} K (Δ = {abs(T_match - temperature_K):.1f} K > {tol} K)."
        )

    T_fit = fit["fits"][T_match]
    # Binary: only one independent element
    el = list(T_fit["models"].keys())[0]
    model = T_fit["models"][el]
    coeffs = model["coefficients"]  # (n_features,)
    powers = model["feature_powers"]  # (n_features, 1) for binary

    c_flat = c.ravel()[:, np.newaxis]  # (N*N, 1)

    # Evaluate polynomial: Δμ = Σ_k coeff_k · c^power[k,0]
    n_features = len(coeffs)
    terms = np.ones((len(c_flat), n_features))
    for k in range(n_features):
        p = int(powers[k, 0])
        if p > 0:
            terms[:, k] = c_flat[:, 0] ** p

    delta_mu = terms @ coeffs + model["intercept"]
    return delta_mu.reshape(c.shape)


def calphad_df_dc_gm_spline(
    c: np.ndarray,
    gm_spline_fit: dict,
    temperature_K: float,
) -> np.ndarray:
    """Bulk driving force dG_m/dx_V from a CubicSpline fitted directly to G_m.

    Uses the analytic derivative of the G_m spline — no integration error.
    This is the recommended driving force for CALPHAD-PFM coupling because
    the double-well shape of G_m is preserved exactly.

    Parameters
    ----------
    c : np.ndarray, shape (N, N)
        Composition field (mol fraction of reference element, e.g. x_V).
    gm_spline_fit : dict
        Output of ``build_gm_spline_surrogate()``.
    temperature_K : float

    Returns:
    -------
    np.ndarray  dG_m/dx_V  [J/mol]  (same shape as c)
    """
    stored_T = np.array(gm_spline_fit["temperatures_K"])
    idx = int(np.argmin(np.abs(stored_T - temperature_K)))
    T_match = stored_T[idx]

    tol = 1.0
    if abs(T_match - temperature_K) > tol:
        raise ValueError(f"Temperature {temperature_K} K not found. Closest: {T_match} K.")

    T_fit = gm_spline_fit["fits"][T_match]
    el = list(T_fit["models"].keys())[0]
    spl_dfdx = T_fit["models"][el]["spline_dfdx"]

    return spl_dfdx(c.ravel()).reshape(c.shape)


def calphad_f_gm_spline(
    c: np.ndarray,
    gm_spline_fit: dict,
    temperature_K: float,
) -> np.ndarray:
    """Bulk free energy G_m from a CubicSpline fitted directly to pycalphad G_m.

    Parameters
    ----------
    c : np.ndarray
        Composition field (mol fraction of reference element, e.g. x_V).
    gm_spline_fit : dict
        Output of ``build_gm_spline_surrogate()``.
    temperature_K : float

    Returns:
    -------
    np.ndarray  G_m  [J/mol]  (same shape as c)
    """
    stored_T = np.array(gm_spline_fit["temperatures_K"])
    idx = int(np.argmin(np.abs(stored_T - temperature_K)))
    T_match = stored_T[idx]

    tol = 1.0
    if abs(T_match - temperature_K) > tol:
        raise ValueError(f"Temperature {temperature_K} K not found. Closest: {T_match} K.")

    T_fit = gm_spline_fit["fits"][T_match]
    el = list(T_fit["models"].keys())[0]
    spl_gm = T_fit["models"][el]["spline_gm"]

    return spl_gm(c.ravel()).reshape(c.shape)


def calphad_df_dc_spline(
    c: np.ndarray,
    spline_fit: dict,
    temperature_K: float,
) -> np.ndarray:
    """Bulk chemical potential Δμ from a CubicSpline surrogate.

    Drop-in replacement for :func:`calphad_df_dc` using the spline
    surrogate built by ``thermodynamics.fitting.build_spline_surrogate``.

    Parameters
    ----------
    c : np.ndarray, shape (N, N)
        Composition field (mole fraction of the independent element).
    spline_fit : dict
        Output of ``build_spline_surrogate()``.
    temperature_K : float

    Returns:
    -------
    np.ndarray  Δμ  [J/mol]  (same shape as c)
    """
    stored_T = np.array(spline_fit["temperatures_K"])
    idx = int(np.argmin(np.abs(stored_T - temperature_K)))
    T_match = stored_T[idx]

    tol = 1.0
    if abs(T_match - temperature_K) > tol:
        raise ValueError(f"Temperature {temperature_K} K not found in spline fit. Closest is {T_match} K.")

    T_fit = spline_fit["fits"][T_match]
    el = list(T_fit["models"].keys())[0]
    spl = T_fit["models"][el]["spline"]

    return spl(c.ravel()).reshape(c.shape)


def calphad_f_spline(
    c: np.ndarray,
    spline_fit: dict,
    temperature_K: float,
) -> np.ndarray:
    """Bulk free energy density G_m from the CubicSpline surrogate.

    Uses the analytic antiderivative of the spline (PPoly) so G_m is
    exact — no numerical integration required.

    Parameters
    ----------
    c : np.ndarray
        Composition field.
    spline_fit : dict
        Output of ``build_spline_surrogate()``.
    temperature_K : float

    Returns:
    -------
    np.ndarray  G_m  [J/mol]  (same shape as c, up to additive constant)
    """
    stored_T = np.array(spline_fit["temperatures_K"])
    idx = int(np.argmin(np.abs(stored_T - temperature_K)))
    T_match = stored_T[idx]

    tol = 1.0
    if abs(T_match - temperature_K) > tol:
        raise ValueError(f"Temperature {temperature_K} K not found in spline fit. Closest is {T_match} K.")

    T_fit = spline_fit["fits"][T_match]
    el = list(T_fit["models"].keys())[0]
    spl_intg = T_fit["models"][el]["spline_antideriv"]

    return spl_intg(c.ravel()).reshape(c.shape)


def calphad_f(
    c: np.ndarray,
    fit: dict,
    temperature_K: float,
) -> np.ndarray:
    """Bulk free energy density G_m from the CALPHAD polynomial surrogate.

    Uses the thermodynamic identity for a binary system.

    The surrogate polynomial is fitted in x_NB space (x_NB is the
    independent element), so:

        dG_m / dx_NB = μ_NB − μ_V = Δμ

    Therefore:

        G_m(x_NB) = +∫ Δμ(x) dx  (+ arbitrary constant, omitted)

    Since Δμ = Σ_k aₖ · x^{pₖ}, the antiderivative is analytic:

        ∫ Δμ dx = Σ_k aₖ / (pₖ + 1) · x^{pₖ+1}

    The constant of integration is irrelevant for tracking energy changes.

    Parameters
    ----------
    c : np.ndarray, shape (N, N)
        Composition field (mole fraction of NB, the independent element).
    fit : dict
        Output of ``thermodynamics.fitting.load_fits()``.
    temperature_K : float
        Temperature in Kelvin.

    Returns:
    -------
    np.ndarray  G_m  [J/mol]  (same shape as c, up to additive constant)
    """
    stored_T = np.array(fit["temperatures_K"])
    idx = int(np.argmin(np.abs(stored_T - temperature_K)))
    T_match = stored_T[idx]

    tol = 1.0
    if abs(T_match - temperature_K) > tol:
        raise ValueError(f"Temperature {temperature_K} K not found in fit file. Closest is {T_match} K.")

    T_fit = fit["fits"][T_match]
    el = list(T_fit["models"].keys())[0]
    model = T_fit["models"][el]
    coeffs = model["coefficients"]
    powers = model["feature_powers"]

    c_flat = c.ravel()

    # Antiderivative of Δμ(x) = Σ aₖ x^pₖ  →  Σ aₖ/(pₖ+1) · x^{pₖ+1}
    # G_m = +∫ Δμ dx  (dG_m/dx_NB = +Δμ, polynomial fitted in x_NB)
    gm = np.zeros_like(c_flat, dtype=float)
    for k in range(len(coeffs)):
        p = int(powers[k, 0])
        gm += coeffs[k] / (p + 1) * c_flat ** (p + 1)

    # Intercept term: ∫ intercept dx = intercept · x
    gm += model["intercept"] * c_flat

    return gm.reshape(c.shape)
