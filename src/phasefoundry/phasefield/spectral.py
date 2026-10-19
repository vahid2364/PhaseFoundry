"""Spectral (Fourier) operators for the pseudo-spectral Cahn-Hilliard solver.

The wavenumbers are imaginary, kx = i (2π/Lx) m, so the Laplacian symbol
k2 = kx² + ky² = −|k|² is real and negative: it is the eigenvalue of ∇². The
bi-Laplacian symbol is k2² = |k|⁴. This matches the MATLAB-style frequency vector of the
original PhaseFoundry notebooks.
"""

from __future__ import annotations

import numpy as np


def wavenumbers(N: int, L: float) -> np.ndarray:
    """Return the Fourier wavenumbers of an N-point periodic grid of length L.

    The integer indices m follow the MATLAB order 0, 1, …, N/2 − 1, 0, −N/2 + 1, …, −1,
    with the Nyquist entry set to zero to avoid phase artefacts.

    Args:
        N: Number of grid points. It should be even.
        L: Domain length.

    Returns:
        Imaginary wavenumbers i (2π/L) m, complex, shape ``(N,)``.
    """
    m = np.concatenate(
        (
            np.arange(0, N // 2),
            [0],  # Nyquist → 0
            np.arange(-N // 2 + 1, 0),
        )
    )
    return (2.0 * np.pi / L) * 1j * m


def laplacian_operator(
    N: int,
    Lx: float,
    Ly: float | None = None,
    Lz: float | None = None,
    ndim: int = 2,
) -> np.ndarray:
    """Return the Laplacian symbol k2 = Σ kᵢ² on a 2-D or 3-D periodic grid.

    k2 is real and negative, the eigenvalue of ∇² for each Fourier mode.

    Args:
        N: Grid points along each axis.
        Lx: Domain length along x.
        Ly: Domain length along y. Defaults to ``Lx``.
        Lz: Domain length along z, used in 3-D. Defaults to ``Lx``.
        ndim: Spatial dimension, 2 or 3.

    Returns:
        k2, shape ``(N, N)`` for ``ndim=2`` or ``(N, N, N)`` for ``ndim=3``.

    Raises:
        ValueError: If ``ndim`` is not 2 or 3.
    """
    Ly = Ly or Lx
    Lz = Lz or Lx
    kx = wavenumbers(N, Lx)
    ky = wavenumbers(N, Ly)

    if ndim == 2:
        KX, KY = np.meshgrid(kx, ky, indexing="ij")
        return np.real(KX**2 + KY**2)  # (N, N)

    if ndim == 3:
        kz = wavenumbers(N, Lz)
        KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing="ij")
        return np.real(KX**2 + KY**2 + KZ**2)  # (N, N, N)

    raise ValueError(f"ndim must be 2 or 3, got {ndim}.")
