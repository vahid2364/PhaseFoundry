"""
phasefoundry.elasticity.fft_solver
=================================
FFT-based iterative elasticity solver for periodic microstructures.

Algorithm:  Gururajan & Abinandanan (2007), Acta Materialia 55, 5015-5026.
            Displacement formulation (Section 2.2), Eqs. (14)-(26).

Key steps
---------
1. Choose C^eff = (C^matrix + C^precipitate) / 2  as the homogeneous reference.
2. Build the acoustic (Green's operator) tensor G(k) = [C^eff_ijkl k_j k_k]^{-1}.
   Explicitly set G(k=0) = 0 to remove the singular mode.
3. At each iteration:
     a. Compute the polarisation stress  τ_ij = (C_ij - C^eff_ij) ε_kl + C_ij ε*_kl
     b. FFT(τ), contract with G to get correction u_hat, IFFT → u
     c. Update strain from displacement gradient
     d. Update macroscopic strain E from volume-average equilibrium (Eq. 24-26)
4. Repeat until |Δε| < tol.

Notation follows the paper:
    u_i        — displacement field
    ε_ij       — total strain  = (∂u_i/∂x_j + ∂u_j/∂x_i) / 2
    ε*_ij      — eigenstrain (stress-free transformation strain)
    σ_ij       — stress = C_ijkl (ε_kl - ε*_kl)
    E_ij       — volume-averaged (macroscopic) strain
    σ^A_ij     — applied (far-field) stress (default: zero)

Spatial dimensions
------------------
Both 2D (square N×N grid) and 3D (cubic N×N×N grid) are supported.
The solver detects dimensionality from c.ndim and dispatches accordingly.

Units
-----
Stiffness in Pa, strains dimensionless, stress in Pa.
"""

from __future__ import annotations

import numpy as np
from typing import Optional

from .interpolation import InterpolationScheme, get_scheme


# ── stiffness utilities ────────────────────────────────────────────────────────

def build_isotropic_stiffness(G: float, nu: float) -> np.ndarray:
    """
    Build the 3×3×3×3 isotropic stiffness tensor.

    Parameters
    ----------
    G : float
        Shear modulus [Pa].
    nu : float
        Poisson's ratio (dimensionless).

    Returns
    -------
    C : ndarray, shape (3, 3, 3, 3)
    """
    lam = 2.0 * G * nu / (1.0 - 2.0 * nu)
    C   = np.zeros((3, 3, 3, 3), dtype=np.float64)
    for i in range(3):
        for j in range(3):
            for k in range(3):
                for l in range(3):
                    C[i, j, k, l] = (
                        lam * (i == j) * (k == l)
                        + G * ((i == k) * (j == l) + (i == l) * (j == k))
                    )
    return C


def build_cubic_stiffness(C11: float, C12: float, C44: float) -> np.ndarray:
    """
    Build the 3×3×3×3 stiffness tensor for a cubic-symmetry crystal.

    Zener anisotropy ratio: A_Z = 2 C44 / (C11 - C12)
        A_Z = 1  →  isotropic
        A_Z > 1  →  ⟨100⟩ elastically soft, precipitate squares up along x/y axes
        A_Z < 1  →  ⟨110⟩ elastically soft, precipitate squares up at 45°

    Parameters
    ----------
    C11, C12, C44 : float
        Independent cubic elastic constants [Pa or dimensionless].

    Returns
    -------
    C : ndarray (3, 3, 3, 3)
    """
    C = np.zeros((3, 3, 3, 3), dtype=np.float64)
    for i in range(3):
        C[i, i, i, i] = C11
        for j in range(3):
            if i != j:
                C[i, i, j, j] = C12
                C[i, j, i, j] = C44
                C[i, j, j, i] = C44
    return C


def _interp_stiffness(c: np.ndarray,
                      C_mat: np.ndarray,
                      C_ppt: np.ndarray,
                      scheme: InterpolationScheme) -> np.ndarray:
    """
    Spatially varying stiffness:  C(x) = (1 - h(c)) C^mat + h(c) C^ppt

    Parameters
    ----------
    c      : ndarray (*spatial)      composition field in [0, 1]
    C_mat  : ndarray (3,3,3,3)       matrix stiffness
    C_ppt  : ndarray (3,3,3,3)       precipitate stiffness
    scheme : InterpolationScheme

    Returns
    -------
    C_field : ndarray (*spatial, 3, 3, 3, 3)
    """
    h  = scheme.h(c)                                      # shape (*spatial,)
    dC = C_ppt - C_mat                                    # (3,3,3,3)
    return C_mat + h[..., None, None, None, None] * dC


# ── Green's function ───────────────────────────────────────────────────────────

def _build_green_operator(C_eff: np.ndarray,
                          shape: tuple[int, ...],
                          Lx: float) -> np.ndarray:
    """
    Build the Green's operator (acoustic tensor inverse) in Fourier space.

    G_ik(k) = [A_ik(k)]^{-1},  A_ik = C^eff_ijkl k_j k_l

    Supports 2D (shape=(Nx,Ny)) and 3D (shape=(Nx,Ny,Nz)).
    The k=0 mode is explicitly set to zero (removes singular rigid-body mode).

    Parameters
    ----------
    C_eff : ndarray (3,3,3,3)   effective (reference) stiffness
    shape : tuple               spatial grid shape, e.g. (N,N) or (N,N,N)
    Lx    : float               physical domain size [m] (assumed same in all axes)

    Returns
    -------
    G : ndarray (*shape, d, d)  where d = len(shape)
    """
    d = len(shape)
    # Build wave-vector arrays for each axis
    kv_axes = [2.0 * np.pi * np.fft.fftfreq(shape[ax], d=Lx / shape[ax])
               for ax in range(d)]
    grids = np.meshgrid(*kv_axes, indexing="ij")     # d arrays, each (*shape,)
    k = np.stack(grids, axis=-1)                      # (*shape, d)

    # Acoustic tensor  A_ik = C^eff_{ijkl} k_j k_l   (i,k in 0..d-1)
    C_eff_d = C_eff[:d, :d, :d, :d]                  # (d,d,d,d)
    A = np.einsum("...j,...l,ijkl->...ik", k, k, C_eff_d)  # (*shape, d, d)

    # Invert pointwise via np.linalg.inv (works for any batch shape)
    A_flat = A.reshape(-1, d, d)
    k2 = np.sum(k**2, axis=-1).reshape(-1)            # |k|^2 at each grid point
    G_flat = np.zeros_like(A_flat)
    nonzero = k2 > 1e-30
    if nonzero.any():
        G_flat[nonzero] = np.linalg.inv(A_flat[nonzero])
    # k=0 mode remains zero
    return G_flat.reshape(*shape, d, d)


# ── Voigt helpers ──────────────────────────────────────────────────────────────

def _voigt_setup(d: int):
    """
    Return Voigt index pairs and (tensor→voigt, voigt→tensor) helpers for dim d.

    2D: 3 components  [11, 22, 12]
    3D: 6 components  [11, 22, 33, 23, 13, 12]
    """
    if d == 2:
        pairs = [(0, 0), (1, 1), (0, 1)]
    else:
        pairs = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]

    def tensor_to_voigt(T: np.ndarray) -> np.ndarray:
        return np.array([T[i, j] for i, j in pairs], dtype=np.float64)

    def voigt_to_tensor(v: np.ndarray) -> np.ndarray:
        T = np.zeros((d, d), dtype=np.float64)
        for idx, (i, j) in enumerate(pairs):
            T[i, j] = v[idx]
            T[j, i] = v[idx]
        return T

    return pairs, tensor_to_voigt, voigt_to_tensor


# ── main solver ────────────────────────────────────────────────────────────────

def solve_elasticity(
    c            : np.ndarray,
    C_mat        : np.ndarray,
    C_ppt        : np.ndarray,
    eigenstrain  : Optional[np.ndarray] = None,
    sigma_applied: Optional[np.ndarray] = None,
    Lx           : float = 1.0,
    tol          : float = 1e-7,
    max_iter     : int   = 2000,
    verbose      : bool  = False,
    scheme       : InterpolationScheme | str = "cubic",
    u_init       : Optional[np.ndarray] = None,
    eps_star_field: Optional[np.ndarray] = None,
    C_eff_override : Optional[np.ndarray] = None,
    C_field_override: Optional[np.ndarray] = None,
) -> dict:
    """
    Solve mechanical equilibrium for a 2D or 3D periodic microstructure.

    Parameters
    ----------
    c : ndarray (N, N) or (N, N, N)
        Composition field (0 = matrix, 1 = precipitate).  Values in [0, 1].
    C_mat : ndarray (3, 3, 3, 3)
        Stiffness tensor of the matrix phase.
    C_ppt : ndarray (3, 3, 3, 3)
        Stiffness tensor of the precipitate phase.
    eigenstrain : ndarray (d, d) or None
        Constant stress-free strain tensor ε* applied via the interpolation
        scheme: eps*(x) = h(c(x)) × eigenstrain.  Mutually exclusive with
        ``eps_star_field``; provide exactly one.
    sigma_applied : ndarray (d, d) or None
        Applied (far-field) average stress.  Default: zero (stress-free BC).
    Lx : float
        Physical domain size [m] (same in all spatial axes).
    tol : float
        Convergence criterion on the RMS strain change per iteration.
    max_iter : int
        Maximum number of iterations.
    verbose : bool
        Print convergence info every 100 iterations.
    scheme : InterpolationScheme or str
        Composition-interpolation scheme for the stiffness and eigenstrain
        fields.  Default "cubic" matches the paper.
    u_init : ndarray (*spatial, d) or None
        Warm-start displacement field from the previous solve.  When supplied
        the iteration starts from u_init instead of zero.
    eps_star_field : ndarray (*spatial, d, d) or None
        Precomputed spatially varying eigenstrain field.  When provided it is
        used directly for ε*(x) instead of h(c) × eigenstrain, enabling the
        tensorial Vegard formulation for multi-component systems::

            eps_star_field = einsum('k...,kij->...ij', c_all, Lambda)

        Mutually exclusive with ``eigenstrain``; provide exactly one.
    C_eff_override : ndarray (3,3,3,3) or None
        If provided, used as the homogeneous reference stiffness for the
        Green's operator instead of the default ½(C_mat + C_ppt).
        Pass ``model.C_eff(C_mat, C_ppt)`` when using LocalVegardModel so
        the reference stiffness matches the parent-phase composition.
    C_field_override : ndarray (*spatial, 3,3,3,3) or None
        If provided, used directly as the spatially varying stiffness field
        C(x) instead of the default h(c)-blend between C_mat and C_ppt.
        Pass ``model.C_field(c_all, C_mat, C_ppt)`` for LocalVegardModel.

    Returns
    -------
    result : dict with keys
        "strain"           : ndarray (*spatial, d, d)  total strain ε_ij(x)
        "elastic_strain"   : ndarray (*spatial, d, d)  ε^el = ε - ε*(x)
        "stress"           : ndarray (*spatial, d, d)  Cauchy stress σ_ij(x)
        "displacement"     : ndarray (*spatial, d)     displacement u_i(x)
        "E"                : ndarray (d, d)            macroscopic strain
        "n_iter"           : int                       iterations to convergence
        "residual"         : float                     final RMS residual
        "residual_history" : list[float]               RMS Δε per iteration
    """
    spatial_shape = c.shape          # (N,N) or (N,N,N)
    d             = c.ndim           # 2 or 3
    fft_axes      = tuple(range(d))

    if isinstance(scheme, str):
        scheme = get_scheme(scheme)

    if sigma_applied is None:
        sigma_applied = np.zeros((d, d), dtype=np.float64)

    # Reference (effective) stiffness — override from model or default Voigt average
    C_eff = C_eff_override if C_eff_override is not None else 0.5 * (C_mat + C_ppt)

    # Spatially varying stiffness  C(x)  — override from model or h(c) blend
    if C_field_override is not None:
        C_field = C_field_override
    else:
        C_field = _interp_stiffness(c, C_mat, C_ppt, scheme)
    # Restrict to in-plane/in-volume d×d×d×d block
    C_d     = C_field[..., :d, :d, :d, :d]   # (*spatial, d, d, d, d)
    C_eff_d = C_eff[:d, :d, :d, :d]           # (d, d, d, d)

    # Eigenstrain field ε*(x)
    if eps_star_field is not None:
        eps_star = eps_star_field[..., :d, :d]  # (*spatial, d, d)
    else:
        if eigenstrain is None:
            raise ValueError(
                "solve_elasticity: provide either 'eigenstrain' (d,d) "
                "or 'eps_star_field' (*spatial,d,d)."
            )
        h = scheme.h(c)                                      # (*spatial,)
        es = eigenstrain[:d, :d]                             # (d, d)
        eps_star = h[..., None, None] * es[None, ...]        # (*spatial, d, d)

    # Green's operator in Fourier space — (*spatial, d, d)
    G = _build_green_operator(C_eff_d, spatial_shape, Lx)

    # Wave-vector grid — (*spatial, d)
    kv_axes = [2.0 * np.pi * np.fft.fftfreq(spatial_shape[ax], d=Lx / spatial_shape[ax])
               for ax in range(d)]
    grids = np.meshgrid(*kv_axes, indexing="ij")
    k = np.stack(grids, axis=-1)                             # (*spatial, d)

    # Coordinate grid for macroscopic displacement  u_macro_i = E_ij x_j
    x_axes = [np.linspace(0.0, Lx, spatial_shape[ax], endpoint=False)
              for ax in range(d)]
    x_grids = np.meshgrid(*x_axes, indexing="ij")
    x_coord = np.stack(x_grids, axis=-1)                    # (*spatial, d)

    # Voigt helpers for macroscopic strain update
    _, tensor_to_voigt, voigt_to_tensor = _voigt_setup(d)
    n_voigt = 3 if d == 2 else 6
    _, voigt_pairs, _ = _voigt_setup(d) if False else (None, None, None)
    pairs_list = [(0,0),(1,1),(0,1)] if d == 2 else [(0,0),(1,1),(2,2),(1,2),(0,2),(0,1)]
    C_eff_V = np.array(
        [[C_eff_d[i, j, k_, l] for (k_, l) in pairs_list]
         for (i, j) in pairs_list],
        dtype=np.float64,
    )
    S_eff_V = np.linalg.inv(C_eff_V)

    # ── initialise ──────────────────────────────────────────────────────────────
    residual_history: list[float] = []

    if u_init is not None:
        u_fluct_init  = u_init
        u_hat_init    = np.fft.fftn(u_fluct_init, axes=fft_axes)
        eps_fluct_hat = 0.5j * (
            np.einsum("...j,...i->...ij", k, u_hat_init) +
            np.einsum("...i,...j->...ij", k, u_hat_init)
        )
        eps_fluct = np.real(np.fft.ifftn(eps_fluct_hat, axes=fft_axes))
        eps_el_0  = eps_fluct - eps_star
        sigma_0   = np.einsum("...ijkl,...kl->...ij", C_d, eps_el_0)
        tau_0     = sigma_0 - np.einsum("ijkl,...kl->...ij", C_eff_d, eps_fluct)
        tau_mean_0 = tau_0.mean(axis=tuple(range(d)))
        E   = voigt_to_tensor(S_eff_V @ tensor_to_voigt(sigma_applied - tau_mean_0))
        eps = eps_fluct + E[None, ...]
        u   = u_fluct_init
    else:
        eps = np.zeros((*spatial_shape, d, d), dtype=np.float64)
        u   = np.zeros((*spatial_shape, d),    dtype=np.float64)
        E   = voigt_to_tensor(S_eff_V @ tensor_to_voigt(sigma_applied))

    d_eps = np.inf
    for iteration in range(max_iter):
        eps_old = eps.copy()

        # stress  σ_ij = C_ijkl(x) [ε_kl - ε*_kl(x)]
        eps_el = eps - eps_star
        sigma  = np.einsum("...ijkl,...kl->...ij", C_d, eps_el)

        # polarisation stress  τ_ij = σ_ij - C^eff_ijkl ε_kl
        sigma_eff = np.einsum("ijkl,...kl->...ij", C_eff_d, eps)
        tau = sigma - sigma_eff

        # FFT of τ
        tau_hat = np.fft.fftn(tau, axes=fft_axes)

        # displacement correction in Fourier space
        b_hat     = 1j * np.einsum("...l,...il->...i", k, tau_hat)
        u_hat_corr = np.einsum("...ij,...j->...i", G, b_hat)

        # fluctuation displacement in real space
        u_fluct = np.real(np.fft.ifftn(u_hat_corr, axes=fft_axes))

        # total displacement = fluctuation + macroscopic (E · x)
        u = u_fluct + np.einsum("ij,...j->...i", E, x_coord)

        # strain from fluctuation displacement gradient (spectral)
        u_hat_full    = np.fft.fftn(u_fluct, axes=fft_axes)
        eps_fluct_hat = 0.5j * (
            np.einsum("...j,...i->...ij", k, u_hat_full) +
            np.einsum("...i,...j->...ij", k, u_hat_full)
        )
        eps_fluct = np.real(np.fft.ifftn(eps_fluct_hat, axes=fft_axes))
        eps = eps_fluct + E[None, ...]

        # update macroscopic strain from <σ> = σ^A
        tau_mean = tau.mean(axis=tuple(range(d)))
        E = voigt_to_tensor(S_eff_V @ tensor_to_voigt(sigma_applied - tau_mean))

        # convergence
        d_eps = float(np.sqrt(np.mean((eps - eps_old)**2)))
        residual_history.append(d_eps)
        if verbose and iteration % 100 == 0:
            print(f"  iter {iteration:4d}  Δε_rms = {d_eps:.3e}")
        if d_eps < tol:
            if verbose:
                print(f"  Converged at iter {iteration}  Δε_rms = {d_eps:.3e}")
            break
    else:
        if verbose:
            print(f"  WARNING: did not converge in {max_iter} iterations  "
                  f"Δε_rms = {d_eps:.3e}")

    eps_el = eps - eps_star
    sigma_final = np.einsum("...ijkl,...kl->...ij", C_d, eps_el)

    return {
        "strain"           : eps,
        "elastic_strain"   : eps_el,
        "stress"           : sigma_final,
        "displacement"     : u,
        "E"                : E,
        "n_iter"           : iteration + 1,
        "residual"         : d_eps,
        "residual_history" : residual_history,
    }
