"""Composition-interpolation functions for the elastically inhomogeneous Cahn-Hilliard model.

Following Gururajan and Abinandanan (2007), the stiffness tensor and the eigenstrain are
explicit functions of composition (Eqs. 12-14):

    C_ijkl(c) = C^eff_ijkl + α(c) ΔC_ijkl
    ε0_ij(c) = β(c) ε_T δ_ij

with ΔC = C^precipitate − C^matrix and C^eff = (C^matrix + C^precipitate) / 2. The
elastic chemical potential (Eq. 36) needs their derivatives with respect to c:

    N_V μ_el = ½ α'(c) ΔC_ijkl ε^el_kl ε^el_ij − β'(c) ε_T δ_ij σ_ij

``fft_solver`` and ``elastic_potential`` both take h, α, β and their derivatives from
here. Three schemes are registered:

- ``"cubic"`` (default): the paper's choice (Eqs. 47-48), h(c) = c³(10 − 15c + 6c²),
  with h(0) = 0, h(1) = 1 and h'(0) = h'(1) = 0.
- ``"linear"``: h(c) = c, for ablation studies.
- ``"step"``: a sharp 0/1 indicator at c = 0.5. Its derivative is returned as zero, so
  use it only to build sharp stiffness and eigenstrain fields for static benchmarks such
  as the Eshelby circular-precipitate test, never to compute μ_el during time evolution.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

ArrayFn = Callable[[np.ndarray], np.ndarray]
"""Function from a composition array to an array of the same shape."""


# cubic (G&A 2007, Eq. 47-48)
def cubic_h(c: np.ndarray) -> np.ndarray:
    """Return h(c) = c³(10 − 15c + 6c²), which rises from h(0) = 0 to h(1) = 1."""
    return c**3 * (10.0 - 15.0 * c + 6.0 * c**2)


def cubic_dh(c: np.ndarray) -> np.ndarray:
    """Return h'(c) = 30 c² (1 − c)², which is non-negative and zero at c = 0 and c = 1."""
    return 30.0 * c**2 * (1.0 - c) ** 2


def cubic_alpha(c: np.ndarray) -> np.ndarray:
    """Return the stiffness interpolant α(c) = h(c) − ½ (Eq. 48)."""
    return cubic_h(c) - 0.5


def cubic_dalpha(c: np.ndarray) -> np.ndarray:
    """Return α'(c) = h'(c)."""
    return cubic_dh(c)


def cubic_beta(c: np.ndarray) -> np.ndarray:
    """Return the eigenstrain interpolant β(c) = h(c) (Eq. 12)."""
    return cubic_h(c)


def cubic_dbeta(c: np.ndarray) -> np.ndarray:
    """Return β'(c) = h'(c)."""
    return cubic_dh(c)


# linear (ablation / sanity-check alternative)
def linear_h(c: np.ndarray) -> np.ndarray:
    """Return h(c) = c."""
    return c


def linear_dh(c: np.ndarray) -> np.ndarray:
    """Return h'(c) = 1."""
    return np.ones_like(c)


def linear_alpha(c: np.ndarray) -> np.ndarray:
    """Return α(c) = c − ½."""
    return c - 0.5


def linear_dalpha(c: np.ndarray) -> np.ndarray:
    """Return α'(c) = 1."""
    return np.ones_like(c)


def linear_beta(c: np.ndarray) -> np.ndarray:
    """Return β(c) = c."""
    return c


def linear_dbeta(c: np.ndarray) -> np.ndarray:
    """Return β'(c) = 1."""
    return np.ones_like(c)


# sharp step (for sharp-interface benchmarks only)
def step_h(c: np.ndarray) -> np.ndarray:
    """Return 1 where c ≥ 0.5 and 0 elsewhere, like a binary phase mask."""
    return (c >= 0.5).astype(c.dtype)


def step_dh(c: np.ndarray) -> np.ndarray:
    """Return zeros in place of the step's derivative.

    The true derivative is a delta function, which a grid cannot represent. μ_el computed
    with this scheme is wrong, so do not use it to drive Cahn-Hilliard evolution.
    """
    return np.zeros_like(c)


def step_alpha(c: np.ndarray) -> np.ndarray:
    """Return α(c) = h(c) − ½ for the step h."""
    return step_h(c) - 0.5


def step_dalpha(c: np.ndarray) -> np.ndarray:
    """Return zeros in place of α'(c); see ``step_dh``."""
    return step_dh(c)


def step_beta(c: np.ndarray) -> np.ndarray:
    """Return β(c) = h(c) for the step h."""
    return step_h(c)


def step_dbeta(c: np.ndarray) -> np.ndarray:
    """Return zeros in place of β'(c); see ``step_dh``."""
    return step_dh(c)


@dataclass(frozen=True)
class InterpolationScheme:
    """A named family of interpolants and their derivatives.

    Attributes:
        name: Name that ``get_scheme`` looks up.
        h: Interpolant h(c) for the stiffness blend and the eigenstrain field in
            ``solve_elasticity``.
        dh: Derivative h'(c).
        alpha: Stiffness interpolant α(c).
        dalpha: Derivative α'(c).
        beta: Eigenstrain interpolant β(c).
        dbeta: Derivative β'(c).
    """

    name: str
    h: ArrayFn
    dh: ArrayFn
    alpha: ArrayFn
    dalpha: ArrayFn
    beta: ArrayFn
    dbeta: ArrayFn


_SCHEMES: dict[str, InterpolationScheme] = {
    "cubic": InterpolationScheme(
        name="cubic",
        h=cubic_h,
        dh=cubic_dh,
        alpha=cubic_alpha,
        dalpha=cubic_dalpha,
        beta=cubic_beta,
        dbeta=cubic_dbeta,
    ),
    "linear": InterpolationScheme(
        name="linear",
        h=linear_h,
        dh=linear_dh,
        alpha=linear_alpha,
        dalpha=linear_dalpha,
        beta=linear_beta,
        dbeta=linear_dbeta,
    ),
    "step": InterpolationScheme(
        name="step",
        h=step_h,
        dh=step_dh,
        alpha=step_alpha,
        dalpha=step_dalpha,
        beta=step_beta,
        dbeta=step_dbeta,
    ),
}


def get_scheme(name: str = "cubic") -> InterpolationScheme:
    """Look up an interpolation scheme by name.

    Args:
        name: ``"cubic"`` (the default and the paper's choice), ``"linear"``, ``"step"``, or
            the name of a scheme added with ``register_scheme``.

    Returns:
        The scheme.

    Raises:
        ValueError: If no scheme has that name.
    """
    try:
        return _SCHEMES[name]
    except KeyError:
        raise ValueError(f"Unknown interpolation scheme {name!r}. Available: {sorted(_SCHEMES)}") from None


def register_scheme(scheme: InterpolationScheme) -> None:
    """Add a scheme under ``scheme.name``, replacing any scheme of the same name.

    Args:
        scheme: The scheme to add.
    """
    _SCHEMES[scheme.name] = scheme
