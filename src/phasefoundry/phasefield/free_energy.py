"""Bulk free energy models for the binary Cahn-Hilliard solver.

Each model has a driving-force function, which returns dF/dc at every point of the
composition field ``c``, and a matching free energy function. Both return arrays of the
shape of ``c``.

- Double well: f(c) = ρ_s (c − c_a)² (c − c_b)², a synthetic model for benchmarks.
- CALPHAD polynomial: Δμ = μ_i − μ_ref from the polynomial fit read by
  ``thermodynamics.fitting.load_fits``.
- CALPHAD splines: Δμ or G_m from the cubic-spline surrogates built by
  ``thermodynamics.fitting.build_spline_surrogate`` and ``build_gm_spline_surrogate``.

The CALPHAD functions use the stored temperature closest to ``temperature_K`` and raise
``ValueError`` if it is more than 1 K away. They read only the first fitted element, so
they are meant for binary systems.
"""

from __future__ import annotations

import numpy as np


def double_well_df_dc(
    c: np.ndarray,
    rho_s: float = 5.0,
    c_a: float = 0.3,
    c_b: float = 0.7,
) -> np.ndarray:
    """Return the driving force df/dc of the double-well free energy.

    For f(c) = ρ_s (c − c_a)² (c − c_b)² the derivative is

        df/dc = 2 ρ_s (c − c_a)(c − c_b)(2c − c_a − c_b)

    Args:
        c: Composition field, any shape.
        rho_s: Energy scale ρ_s. The barrier at c = (c_a + c_b) / 2 is
            ρ_s ((c_b − c_a) / 2)⁴.
        c_a: Composition of the first well minimum.
        c_b: Composition of the second well minimum.

    Returns:
        df/dc, same shape as ``c``.
    """
    return -2.0 * rho_s * (c - c_a) * (c - c_b) * (c_a + c_b - 2.0 * c)


def double_well_f(
    c: np.ndarray,
    rho_s: float = 5.0,
    c_a: float = 0.3,
    c_b: float = 0.7,
) -> np.ndarray:
    """Return the double-well free energy density f(c) = ρ_s (c − c_a)² (c − c_b)².

    Args:
        c: Composition field, any shape.
        rho_s: Energy scale ρ_s.
        c_a: Composition of the first well minimum.
        c_b: Composition of the second well minimum.

    Returns:
        f(c), same shape as ``c``.
    """
    return rho_s * (c - c_a) ** 2 * (c - c_b) ** 2


def calphad_df_dc(
    c: np.ndarray,
    fit: dict,
    temperature_K: float,
    x_element_col: int = 0,
) -> np.ndarray:
    """Return Δμ = μ_i − μ_ref from the CALPHAD polynomial surrogate.

    The polynomial in the independent element's mole fraction is evaluated from the stored
    ``coefficients``, ``feature_powers`` and ``intercept`` with NumPy alone, so scikit-learn
    is not needed.

    Args:
        c: Mole fraction of the independent element, any shape.
        fit: Output of ``thermodynamics.fitting.load_fits``.
        temperature_K: Temperature in K. The closest stored temperature is used.
        x_element_col: Not used. The first column of ``feature_powers`` is always read,
            which is the only column for a binary system.

    Returns:
        Δμ in J/mol, same shape as ``c``.

    Raises:
        ValueError: If no stored temperature is within 1 K of ``temperature_K``.
    """
    stored_T = np.array(fit["temperatures_K"])
    idx = int(np.argmin(np.abs(stored_T - temperature_K)))
    T_match = stored_T[idx]

    tol = 1.0
    if abs(T_match - temperature_K) > tol:
        raise ValueError(
            f"Temperature {temperature_K} K not found in fit file. "
            f"Available: {sorted(stored_T.tolist())}. "
            f"Closest is {T_match} K (Δ = {abs(T_match - temperature_K):.1f} K > {tol} K)."
        )

    T_fit = fit["fits"][T_match]
    el = list(T_fit["models"].keys())[0]
    model = T_fit["models"][el]
    coeffs = model["coefficients"]
    powers = model["feature_powers"]

    c_flat = c.ravel()[:, np.newaxis]

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
    """Return dG_m/dx_ref from a cubic spline fitted to G_m.

    The driving force is the analytic derivative of the G_m spline, so no integration is
    involved, and the spline passes through the tabulated G_m, so its double-well shape
    is kept.

    Args:
        c: Mole fraction of the reference element (e.g. x_V), any shape.
        gm_spline_fit: Output of ``thermodynamics.fitting.build_gm_spline_surrogate``.
        temperature_K: Temperature in K. The closest stored temperature is used.

    Returns:
        dG_m/dx_ref in J/mol, same shape as ``c``.

    Raises:
        ValueError: If no stored temperature is within 1 K of ``temperature_K``.
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
    """Return G_m from a cubic spline fitted to the pycalphad G_m.

    Args:
        c: Mole fraction of the reference element (e.g. x_V), any shape.
        gm_spline_fit: Output of ``thermodynamics.fitting.build_gm_spline_surrogate``.
        temperature_K: Temperature in K. The closest stored temperature is used.

    Returns:
        G_m in J/mol, same shape as ``c``.

    Raises:
        ValueError: If no stored temperature is within 1 K of ``temperature_K``.
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
    """Return Δμ = μ_i − μ_ref from the cubic-spline surrogate.

    This is the spline counterpart of ``calphad_df_dc``.

    Args:
        c: Mole fraction of the independent element, any shape.
        spline_fit: Output of ``thermodynamics.fitting.build_spline_surrogate``.
        temperature_K: Temperature in K. The closest stored temperature is used.

    Returns:
        Δμ in J/mol, same shape as ``c``.

    Raises:
        ValueError: If no stored temperature is within 1 K of ``temperature_K``.
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
    """Return G_m from the cubic-spline surrogate of Δμ.

    G_m is the analytic antiderivative of the Δμ spline (a ``PPoly``), so no numerical
    integration is involved.

    Args:
        c: Mole fraction of the independent element, any shape.
        spline_fit: Output of ``thermodynamics.fitting.build_spline_surrogate``.
        temperature_K: Temperature in K. The closest stored temperature is used.

    Returns:
        G_m in J/mol up to an additive constant, same shape as ``c``.

    Raises:
        ValueError: If no stored temperature is within 1 K of ``temperature_K``.
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
    """Return G_m from the CALPHAD polynomial surrogate.

    The polynomial is fitted in the mole fraction x of the independent element, and in a
    binary system dG_m/dx = μ_i − μ_ref = Δμ. With Δμ = Σ_k a_k x^p_k + b, G_m is the
    analytic antiderivative

        G_m(x) = Σ_k a_k / (p_k + 1) x^(p_k + 1) + b x

    The constant of integration is left out; it does not affect energy changes.

    Args:
        c: Mole fraction of the independent element, any shape.
        fit: Output of ``thermodynamics.fitting.load_fits``.
        temperature_K: Temperature in K. The closest stored temperature is used.

    Returns:
        G_m in J/mol up to an additive constant, same shape as ``c``.

    Raises:
        ValueError: If no stored temperature is within 1 K of ``temperature_K``.
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

    gm = np.zeros_like(c_flat, dtype=float)
    for k in range(len(coeffs)):
        p = int(powers[k, 0])
        gm += coeffs[k] / (p + 1) * c_flat ** (p + 1)

    gm += model["intercept"] * c_flat

    return gm.reshape(c.shape)
