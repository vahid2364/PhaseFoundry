"""
Spectral (Fourier) operators for the pseudo-spectral Cahn-Hilliard solver.

All operators use the imaginary-wavenumber convention so that the
Laplacian operator in Fourier space is real-negative:

    k2 = kx² + ky²  where  kx = i·(2π/Lx)·m

This gives  k2 = −|k|²  (negative definite), which is the eigenvalue of
the Laplacian ∇².  The Bi-Laplacian is then k2² = |k|⁴ (positive).

This convention is consistent with the MATLAB-style frequency vector used
in the original PhaseFoundry notebooks.
"""

from __future__ import annotations

import numpy as np


def wavenumbers(N: int, L: float) -> np.ndarray:
    """
    Fourier wavenumber vector for an N-point grid of length L.

    Follows the MATLAB convention: [0, 1, …, N/2−1, 0, −N/2+1, …, −1],
    where the Nyquist frequency is set to zero to avoid phase artefacts.

    Parameters
    ----------
    N : int
        Number of grid points (should be even).
    L : float
        Physical domain length.

    Returns
    -------
    np.ndarray  complex  shape (N,)
        Imaginary wavenumbers i·(2π/L)·m.
    """
    m = np.concatenate((
        np.arange(0, N // 2),
        [0],                        # Nyquist → 0
        np.arange(-N // 2 + 1, 0),
    ))
    return (2.0 * np.pi / L) * 1j * m


def laplacian_operator(N: int, Lx: float, Ly: float | None = None) -> np.ndarray:
    """
    2-D Laplacian operator k2 = kx² + ky² in Fourier space.

    k2 is real-negative (eigenvalue of ∇²).

    Parameters
    ----------
    N : int
        Number of grid points in each direction.
    Lx, Ly : float
        Domain lengths. If Ly is None, a square domain (Ly = Lx) is assumed.

    Returns
    -------
    np.ndarray  float  shape (N, N)
    """
    Ly   = Ly or Lx
    kx   = wavenumbers(N, Lx)
    ky   = wavenumbers(N, Ly)
    KX, KY = np.meshgrid(kx, ky)
    return np.real(KX ** 2 + KY ** 2)   # imaginary parts cancel → real negative
