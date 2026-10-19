"""Tests for the Parseval normalization of the gradient energy in ``solver._snapshot``.

For c(x) = c0 + A sin(2π m x / Lx), the mean gradient energy is
F_grad = κ/2 · A² (2π m / Lx)² / 2. The snapshot code computes
F_grad = κ/2 · mean_k(−k2 |ĉ|²) / N^d. With numpy's DFT, ĉ(k) = Σ_x c(x) e^(−2πi k·x / N),
Parseval's theorem Σ_x |c|² = (1 / N^d) Σ_k |ĉ|² and |∇c|² ↔ −k2 |ĉ|² give

    mean_x(|∇c|²) = (1 / N^d) (1 / N^d) Σ_k (−k2) |ĉ|² = mean_k(−k2 |ĉ|²) / N^d

The tests compare the spectral and analytic values to a relative error of 1e-10 in 1-D,
2-D and 3-D, for one mode, two modes and two components. They evaluate a copy of the
formula in ``_snapshot`` and do not call the solver.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))


def _make_k2(N: int, Lx: float, ndim: int) -> np.ndarray:
    """Return the Fourier symbol of the Laplacian, −|k|², built from numpy's ``fftfreq``.

    The solver builds k2 with ``laplacian_operator``, which also sets the Nyquist mode to
    zero. The two agree for the sinusoids tested here, which have no Nyquist component.
    """
    freqs = np.fft.fftfreq(N, d=Lx / N)
    k1d = 2 * np.pi * freqs
    if ndim == 1:
        return -(k1d**2)
    elif ndim == 2:
        kx, ky = np.meshgrid(k1d, k1d, indexing="ij")
        return -(kx**2 + ky**2)
    else:
        kx, ky, kz = np.meshgrid(k1d, k1d, k1d, indexing="ij")
        return -(kx**2 + ky**2 + kz**2)


def _spectral_gradient_energy(c: np.ndarray, kappa: float, N: int, Lx: float, ndim: int) -> float:
    """Return κ/2 · mean_k(−k2 |ĉ|²) / N^d, the formula used in ``solver._snapshot``."""
    k2 = _make_k2(N, Lx, ndim)
    parseval_norm = N**ndim
    c_hat = np.fft.fftn(c)
    return float(np.mean((-k2) * np.abs(c_hat) ** 2) * kappa / 2.0 / parseval_norm)


def _sinusoid_field(N: int, Lx: float, ndim: int, m: int, A: float, c0: float):
    """Return c0 + A sin(2π m x / Lx), varying along the first axis only."""
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
    """Return the analytic mean gradient energy κ/2 · A² (2π m / Lx)² / 2."""
    return kappa / 2.0 * A**2 * (2 * np.pi * m / Lx) ** 2 / 2.0


@pytest.mark.parametrize("ndim", [1, 2, 3])
@pytest.mark.parametrize("m", [1, 3, 7])
def test_single_mode_sinusoid(ndim, m):
    """The spectral gradient energy of one sinusoidal mode matches the analytic value."""
    N = 64
    Lx = 1.0e-6
    kap = 1.0e-12
    A = 0.05
    c0 = 0.5

    c = _sinusoid_field(N, Lx, ndim, m, A, c0)
    F_spectral = _spectral_gradient_energy(c, kap, N, Lx, ndim)
    F_analytic = _analytic_gradient_energy(kap, A, m, Lx)
    rel_err = abs(F_spectral - F_analytic) / abs(F_analytic)

    assert rel_err < 1e-10, (
        f"ndim={ndim}, m={m}: relative error {rel_err:.2e} exceeds 1e-10\n  analytic={F_analytic:.8e}  spectral={F_spectral:.8e}"
    )


@pytest.mark.parametrize("ndim", [1, 2, 3])
def test_multi_mode(ndim):
    """For two modes, the spectral energy equals the sum of the single-mode analytic energies."""
    N = 64
    Lx = 1.0e-6
    kap = 1.0e-12
    m1, m2 = 2, 5
    A1, A2 = 0.04, 0.03
    c0 = 0.5

    x = np.linspace(0, Lx, N, endpoint=False)
    if ndim == 1:
        c = c0 + A1 * np.sin(2 * np.pi * m1 * x / Lx) + A2 * np.sin(2 * np.pi * m2 * x / Lx)
    elif ndim == 2:
        xx, _ = np.meshgrid(x, x, indexing="ij")
        c = c0 + A1 * np.sin(2 * np.pi * m1 * xx / Lx) + A2 * np.sin(2 * np.pi * m2 * xx / Lx)
    else:
        xx, _, _ = np.meshgrid(x, x, x, indexing="ij")
        c = c0 + A1 * np.sin(2 * np.pi * m1 * xx / Lx) + A2 * np.sin(2 * np.pi * m2 * xx / Lx)

    F_spectral = _spectral_gradient_energy(c, kap, N, Lx, ndim)
    F_analytic = _analytic_gradient_energy(kap, A1, m1, Lx) + _analytic_gradient_energy(kap, A2, m2, Lx)
    rel_err = abs(F_spectral - F_analytic) / abs(F_analytic)

    assert rel_err < 1e-10, f"ndim={ndim}, multi-mode: relative error {rel_err:.2e} exceeds 1e-10"


@pytest.mark.parametrize("ndim", [2, 3])
def test_multicomponent(ndim):
    """Two components with different κ give a total energy equal to the sum of their analytic energies."""
    N = 32
    Lx = 1.0e-6
    kap = [1.0e-12, 7.0e-12]
    A = [0.04, 0.02]
    m = [2, 4]
    c0 = [0.5, 0.1]

    F_total_spectral = 0.0
    F_total_analytic = 0.0
    k2 = _make_k2(N, Lx, ndim)
    parseval_norm = N**ndim

    for i in range(2):
        c_i = _sinusoid_field(N, Lx, ndim, m[i], A[i], c0[i])
        c_hat = np.fft.fftn(c_i)
        F_total_spectral += float(np.mean((-k2) * np.abs(c_hat) ** 2) * kap[i] / 2.0 / parseval_norm)
        F_total_analytic += _analytic_gradient_energy(kap[i], A[i], m[i], Lx)

    rel_err = abs(F_total_spectral - F_total_analytic) / abs(F_total_analytic)
    assert rel_err < 1e-10, f"ndim={ndim}, multicomponent: relative error {rel_err:.2e} exceeds 1e-10"


if __name__ == "__main__":
    print("Verifying Parseval normalization of gradient energy in _snapshot\n")
    cases = [
        (1, 1),
        (1, 3),
        (1, 7),
        (2, 1),
        (2, 3),
        (2, 7),
        (3, 1),
        (3, 3),
        (3, 7),
    ]
    all_pass = True
    for ndim, m in cases:
        N, Lx, kap, A, c0 = 64, 1.0e-6, 1.0e-12, 0.05, 0.5
        c = _sinusoid_field(N, Lx, ndim, m, A, c0)
        F_sp = _spectral_gradient_energy(c, kap, N, Lx, ndim)
        F_an = _analytic_gradient_energy(kap, A, m, Lx)
        rel_err = abs(F_sp - F_an) / abs(F_an)
        status = "PASS" if rel_err < 1e-10 else "FAIL"
        if status == "FAIL":
            all_pass = False
        print(f"  ndim={ndim}  m={m}  analytic={F_an:.6e}  spectral={F_sp:.6e}  rel_err={rel_err:.2e}  [{status}]")

    print()
    print("Multi-mode and multicomponent tests:")
    for ndim in [2, 3]:
        test_multi_mode(ndim)
        print(f"  multi-mode    ndim={ndim}  [PASS]")
        test_multicomponent(ndim)
        print(f"  multicomponent ndim={ndim}  [PASS]")

    print(f"\n{'ALL TESTS PASSED' if all_pass else 'SOME TESTS FAILED'}")
