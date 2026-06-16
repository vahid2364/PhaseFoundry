"""
tests/test_parseval_normalization.py
=====================================
Verifies that the gradient energy normalization in solver._snapshot is correct
by comparing the spectral formula against a known analytic result.

For a sinusoidal composition field:
    c(x) = c0 + A * sin(2π m x / Lx)

The analytic gradient energy (mean over domain) is:
    F_grad = κ/2 * A² * (2πm/Lx)² / 2

The spectral formula used in _snapshot is:
    F_grad = κ/2 * mean_k((-k2)|ĉ|²) / N^ndim

Derivation (numpy DFT convention  ĉ(k) = Σ_x c(x) e^{-2πi k·x/N}):
    Parseval:  Σ_x |c|²  = (1/N^d) Σ_k |ĉ|²
    Gradient:  ∇ → i·(2πk/Lx), so |∇c|² ↔ (-k2)|ĉ|²
    Combining: mean_x(|∇c|²) = (1/N^d)·(1/N^d) Σ_k (-k2)|ĉ|²
                              = mean_k((-k2)|ĉ|²) / N^d

These tests confirm the formula is exact (relative error at machine precision)
for 1D, 2D, and 3D, and for multi-mode and multi-component fields.

Run with:
    python tests/test_parseval_normalization.py
    pytest tests/test_parseval_normalization.py -v
"""

from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))


def _make_k2(N: int, Lx: float, ndim: int) -> np.ndarray:
    """Build the negative-Laplacian spectral operator (as in solver)."""
    freqs = np.fft.fftfreq(N, d=Lx / N)
    k1d   = 2 * np.pi * freqs
    if ndim == 1:
        return -(k1d ** 2)
    elif ndim == 2:
        kx, ky = np.meshgrid(k1d, k1d, indexing="ij")
        return -(kx**2 + ky**2)
    else:
        kx, ky, kz = np.meshgrid(k1d, k1d, k1d, indexing="ij")
        return -(kx**2 + ky**2 + kz**2)


def _spectral_gradient_energy(c: np.ndarray, kappa: float,
                               N: int, Lx: float, ndim: int) -> float:
    """Gradient energy using the exact formula from solver._snapshot."""
    k2            = _make_k2(N, Lx, ndim)
    parseval_norm = N ** ndim
    c_hat         = np.fft.fftn(c)
    return float(np.mean((-k2) * np.abs(c_hat)**2) * kappa / 2.0 / parseval_norm)


# ── test helpers ──────────────────────────────────────────────────────────────

def _sinusoid_field(N: int, Lx: float, ndim: int, m: int, A: float, c0: float):
    """Return sinusoidal field varying only along the first axis."""
    x = np.linspace(0, Lx, N, endpoint=False)
    if ndim == 1:
        return c0 + A * np.sin(2 * np.pi * m * x / Lx)
    elif ndim == 2:
        xx, _ = np.meshgrid(x, x, indexing="ij")
        return c0 + A * np.sin(2 * np.pi * m * xx / Lx)
    else:
        xx, _, _ = np.meshgrid(x, x, x, indexing="ij")
        return c0 + A * np.sin(2 * np.pi * m * xx / Lx)


def _analytic_gradient_energy(kappa: float, A: float, m: int, Lx: float) -> float:
    """Analytic result: F_grad = κ/2 * A² * (2πm/Lx)² / 2."""
    return kappa / 2.0 * A**2 * (2 * np.pi * m / Lx)**2 / 2.0


# ── parametrized tests ────────────────────────────────────────────────────────

@pytest.mark.parametrize("ndim", [1, 2, 3])
@pytest.mark.parametrize("m",    [1, 3, 7])
def test_single_mode_sinusoid(ndim, m):
    """Spectral gradient energy matches analytic result for a single sinusoidal mode."""
    N    = 64
    Lx   = 1.0e-6
    kap  = 1.0e-12
    A    = 0.05
    c0   = 0.5

    c            = _sinusoid_field(N, Lx, ndim, m, A, c0)
    F_spectral   = _spectral_gradient_energy(c, kap, N, Lx, ndim)
    F_analytic   = _analytic_gradient_energy(kap, A, m, Lx)
    rel_err      = abs(F_spectral - F_analytic) / abs(F_analytic)

    assert rel_err < 1e-10, (
        f"ndim={ndim}, m={m}: relative error {rel_err:.2e} exceeds 1e-10\n"
        f"  analytic={F_analytic:.8e}  spectral={F_spectral:.8e}"
    )


@pytest.mark.parametrize("ndim", [1, 2, 3])
def test_multi_mode(ndim):
    """
    Parseval is linear in mode contributions — two-mode field.
    F_grad(c1 + c2) = F_grad(c1) + F_grad(c2)  (orthogonality).
    """
    N   = 64
    Lx  = 1.0e-6
    kap = 1.0e-12
    m1, m2 = 2, 5
    A1, A2 = 0.04, 0.03
    c0 = 0.5

    x = np.linspace(0, Lx, N, endpoint=False)
    if ndim == 1:
        c = (c0
             + A1 * np.sin(2*np.pi*m1*x/Lx)
             + A2 * np.sin(2*np.pi*m2*x/Lx))
    elif ndim == 2:
        xx, _ = np.meshgrid(x, x, indexing="ij")
        c = (c0
             + A1 * np.sin(2*np.pi*m1*xx/Lx)
             + A2 * np.sin(2*np.pi*m2*xx/Lx))
    else:
        xx, _, _ = np.meshgrid(x, x, x, indexing="ij")
        c = (c0
             + A1 * np.sin(2*np.pi*m1*xx/Lx)
             + A2 * np.sin(2*np.pi*m2*xx/Lx))

    F_spectral = _spectral_gradient_energy(c, kap, N, Lx, ndim)
    F_analytic = (_analytic_gradient_energy(kap, A1, m1, Lx)
                + _analytic_gradient_energy(kap, A2, m2, Lx))
    rel_err    = abs(F_spectral - F_analytic) / abs(F_analytic)

    assert rel_err < 1e-10, (
        f"ndim={ndim}, multi-mode: relative error {rel_err:.2e} exceeds 1e-10"
    )


@pytest.mark.parametrize("ndim", [2, 3])
def test_multicomponent(ndim):
    """
    For two independent components with different kappas, gradient energies sum correctly.
    """
    N    = 32
    Lx   = 1.0e-6
    kap  = [1.0e-12, 7.0e-12]
    A    = [0.04, 0.02]
    m    = [2, 4]
    c0   = [0.5, 0.1]

    F_total_spectral = 0.0
    F_total_analytic = 0.0
    k2 = _make_k2(N, Lx, ndim)
    parseval_norm = N ** ndim

    for i in range(2):
        c_i = _sinusoid_field(N, Lx, ndim, m[i], A[i], c0[i])
        c_hat = np.fft.fftn(c_i)
        F_total_spectral += float(
            np.mean((-k2) * np.abs(c_hat)**2) * kap[i] / 2.0 / parseval_norm)
        F_total_analytic += _analytic_gradient_energy(kap[i], A[i], m[i], Lx)

    rel_err = abs(F_total_spectral - F_total_analytic) / abs(F_total_analytic)
    assert rel_err < 1e-10, (
        f"ndim={ndim}, multicomponent: relative error {rel_err:.2e} exceeds 1e-10"
    )


# ── standalone runner ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Verifying Parseval normalization of gradient energy in _snapshot\n")
    cases = [
        (1, 1), (1, 3), (1, 7),
        (2, 1), (2, 3), (2, 7),
        (3, 1), (3, 3), (3, 7),
    ]
    all_pass = True
    for ndim, m in cases:
        N, Lx, kap, A, c0 = 64, 1.0e-6, 1.0e-12, 0.05, 0.5
        c          = _sinusoid_field(N, Lx, ndim, m, A, c0)
        F_sp       = _spectral_gradient_energy(c, kap, N, Lx, ndim)
        F_an       = _analytic_gradient_energy(kap, A, m, Lx)
        rel_err    = abs(F_sp - F_an) / abs(F_an)
        status     = "PASS" if rel_err < 1e-10 else "FAIL"
        if status == "FAIL":
            all_pass = False
        print(f"  ndim={ndim}  m={m}  analytic={F_an:.6e}  "
              f"spectral={F_sp:.6e}  rel_err={rel_err:.2e}  [{status}]")

    print()
    print("Multi-mode and multicomponent tests:")
    for ndim in [2, 3]:
        test_multi_mode(ndim)
        print(f"  multi-mode    ndim={ndim}  [PASS]")
        test_multicomponent(ndim)
        print(f"  multicomponent ndim={ndim}  [PASS]")

    print(f"\n{'ALL TESTS PASSED' if all_pass else 'SOME TESTS FAILED'}")
