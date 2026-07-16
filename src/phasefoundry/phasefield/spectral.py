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


def laplacian_operator(
    N:    int,
    Lx:   float,
    Ly:   float | None = None,
    Lz:   float | None = None,
    ndim: int  = 2,
) -> np.ndarray:
    """
    Laplacian operator k2 = Σ kᵢ² in Fourier space (2-D or 3-D).

    k2 is real-negative (eigenvalue of ∇²).

    Parameters
    ----------
    N : int
        Number of grid points in each direction (isotropic grid assumed).
    Lx, Ly, Lz : float
        Domain lengths.  Missing values default to Lx (cubic / square domain).
    ndim : int
        Spatial dimension — 2 (default) or 3.

    Returns
    -------
    np.ndarray  float
        shape (N, N)    for ndim=2
        shape (N, N, N) for ndim=3
    """
    Ly = Ly or Lx
    Lz = Lz or Lx
    kx = wavenumbers(N, Lx)
    ky = wavenumbers(N, Ly)

    if ndim == 2:
        KX, KY = np.meshgrid(kx, ky, indexing="ij")
        return np.real(KX ** 2 + KY ** 2)          # (N, N)

    if ndim == 3:
        kz = wavenumbers(N, Lz)
        KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing="ij")
        return np.real(KX ** 2 + KY ** 2 + KZ ** 2)  # (N, N, N)

    raise ValueError(f"ndim must be 2 or 3, got {ndim}.")
