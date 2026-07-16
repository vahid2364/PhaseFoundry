"""
phasefoundry.elasticity.interpolation
====================================
Composition-interpolation functions for the elastically inhomogeneous
Cahn-Hilliard model (Gururajan & Abinandanan, 2007).

The elastic modulus tensor and the eigenstrain are both written as
explicit functions of composition (Eqs. 12-14 in the paper):

    C_ijkl(c) = C^eff_ijkl + alpha(c) * ΔC_ijkl         (Eq. 13-14)
    eps0_ij(c) = beta(c) * eps_T * delta_ij             (Eq. 12)

where ΔC = C^precipitate - C^matrix and C^eff = (C^matrix + C^precipitate)/2.

The elastic chemical potential (Eq. 36) needs the *derivatives* of these
interpolants with respect to c:

    N_V * mu_el = 1/2 * alpha'(c) * ΔC_ijkl * eps_el_kl * eps_el_ij
                  - beta'(c) * eps_T * delta_ij * sigma_ij

This module centralizes every interpolant (and its derivative) so that
``fft_solver.py`` and the (future) elastic chemical-potential module both
read from one place instead of redefining ``h(c)`` inline.

Available schemes
------------------
"cubic" (default) -- the paper's choice (Eq. 47-48), C^1 continuous,
                      h(0)=0, h(1)=1, h'(0)=h'(1)=0.
"linear"           -- simple linear ramp, for ablation studies.
"step"             -- sharp 0/1 indicator; derivative is (formally) a
                      delta function, returned here as zero almost
                      everywhere -- only meaningful for comparing
                      stiffness/eigenstrain fields against a sharp-interface
                      benchmark (e.g. the Eshelby circular-precipitate test),
                      NOT for computing mu_el during time evolution.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

ArrayFn = Callable[[np.ndarray], np.ndarray]


# ── cubic (G&A 2007, Eq. 47-48) ─────────────────────────────────────────────

def cubic_h(c: np.ndarray) -> np.ndarray:
    """Smoothstep interpolant h(c) = c^3 (10 - 15c + 6c^2).  h(0)=0, h(1)=1."""
    return c**3 * (10.0 - 15.0 * c + 6.0 * c**2)


def cubic_dh(c: np.ndarray) -> np.ndarray:
    """dh/dc = 30 c^2 (1 - c)^2  (always >= 0, vanishes at c=0 and c=1)."""
    return 30.0 * c**2 * (1.0 - c)**2


def cubic_alpha(c: np.ndarray) -> np.ndarray:
    """Stiffness interpolant alpha(c) = h(c) - 1/2  (Eq. 48)."""
    return cubic_h(c) - 0.5


def cubic_dalpha(c: np.ndarray) -> np.ndarray:
    """alpha'(c) = h'(c)  (the constant -1/2 offset drops out)."""
    return cubic_dh(c)


def cubic_beta(c: np.ndarray) -> np.ndarray:
    """Eigenstrain interpolant beta(c) = h(c)  (Eq. 12, same function as h)."""
    return cubic_h(c)


def cubic_dbeta(c: np.ndarray) -> np.ndarray:
    """beta'(c) = h'(c)."""
    return cubic_dh(c)


# ── linear (ablation / sanity-check alternative) ────────────────────────────

def linear_h(c: np.ndarray) -> np.ndarray:
    return c


def linear_dh(c: np.ndarray) -> np.ndarray:
    return np.ones_like(c)


def linear_alpha(c: np.ndarray) -> np.ndarray:
    return c - 0.5


def linear_dalpha(c: np.ndarray) -> np.ndarray:
    return np.ones_like(c)


def linear_beta(c: np.ndarray) -> np.ndarray:
    return c


def linear_dbeta(c: np.ndarray) -> np.ndarray:
    return np.ones_like(c)


# ── sharp step (for sharp-interface benchmarks only) ────────────────────────

def step_h(c: np.ndarray) -> np.ndarray:
    """Sharp indicator: 1 where c >= 0.5, else 0. Matches a binary phase mask."""
    return (c >= 0.5).astype(c.dtype)


def step_dh(c: np.ndarray) -> np.ndarray:
    """
    Derivative of a step function is a delta function; not representable on
    a grid. Returned as zero everywhere -- this scheme must NOT be used to
    drive Cahn-Hilliard evolution (mu_el would be silently wrong), only for
    constructing a sharp stiffness/eigenstrain field for static benchmarks.
    """
    return np.zeros_like(c)


def step_alpha(c: np.ndarray) -> np.ndarray:
    return step_h(c) - 0.5


def step_dalpha(c: np.ndarray) -> np.ndarray:
    return step_dh(c)


def step_beta(c: np.ndarray) -> np.ndarray:
    return step_h(c)


def step_dbeta(c: np.ndarray) -> np.ndarray:
    return step_dh(c)


# ── scheme bundle + registry ─────────────────────────────────────────────────

@dataclass(frozen=True)
class InterpolationScheme:
    """Bundles an interpolant family's value/derivative pairs under one name."""
    name: str
    h: ArrayFn
    dh: ArrayFn
    alpha: ArrayFn
    dalpha: ArrayFn
    beta: ArrayFn
    dbeta: ArrayFn


_SCHEMES: dict[str, InterpolationScheme] = {
    "cubic": InterpolationScheme(
        name="cubic", h=cubic_h, dh=cubic_dh,
        alpha=cubic_alpha, dalpha=cubic_dalpha,
        beta=cubic_beta, dbeta=cubic_dbeta,
    ),
    "linear": InterpolationScheme(
        name="linear", h=linear_h, dh=linear_dh,
        alpha=linear_alpha, dalpha=linear_dalpha,
        beta=linear_beta, dbeta=linear_dbeta,
    ),
    "step": InterpolationScheme(
        name="step", h=step_h, dh=step_dh,
        alpha=step_alpha, dalpha=step_dalpha,
        beta=step_beta, dbeta=step_dbeta,
    ),
}


def get_scheme(name: str = "cubic") -> InterpolationScheme:
    """
    Look up an interpolation scheme by name.

    Parameters
    ----------
    name : str
        One of "cubic" (default, paper's choice), "linear", "step".

    Returns
    -------
    InterpolationScheme

    Raises
    ------
    ValueError
        If `name` is not a registered scheme.
    """
    try:
        return _SCHEMES[name]
    except KeyError:
        raise ValueError(
            f"Unknown interpolation scheme {name!r}. "
            f"Available: {sorted(_SCHEMES)}"
        )


def register_scheme(scheme: InterpolationScheme) -> None:
    """Register a custom InterpolationScheme (e.g. for a new test/ablation)."""
    _SCHEMES[scheme.name] = scheme
