"""
calphad_pf.elasticity.fft_solver
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
    Build the 3×3×3×3 isotropic stiffness tensor for plane-strain 2D.

    Parameters
    ----------
    G : float
        Shear modulus [Pa].
    nu : float
        Poisson's ratio (dimensionless).

    Returns
    -------
    C : ndarray, shape (3, 3, 3, 3)
        Full stiffness tensor.  For plane-strain only the (2D) components
        C[0,0,0,0], C[0,0,1,1], C[1,1,0,0], C[1,1,1,1], C[0,1,0,1] etc.
        are non-zero in-plane; out-of-plane components are not used by the
        2D solver but are included for completeness.
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


def _interp_stiffness(c: np.ndarray,
                      C_mat: np.ndarray,
                      C_ppt: np.ndarray,
                      scheme: InterpolationScheme) -> np.ndarray:
    """
    Spatially varying stiffness:  C(x) = (1 - h(c)) C^mat + h(c) C^ppt
    where h(c) is the interpolant supplied by `scheme` (see
    calphad_pf.elasticity.interpolation; default "cubic" matches the paper).

    Parameters
    ----------
    c      : ndarray (Nx, Ny)   composition field in [0, 1]
    C_mat  : ndarray (3,3,3,3)  matrix stiffness
    C_ppt  : ndarray (3,3,3,3)  precipitate stiffness
    scheme : InterpolationScheme

    Returns
    -------
    C_field : ndarray (Nx, Ny, 3, 3, 3, 3)
    """
    h  = scheme.h(c)                                      # shape (Nx, Ny)
    dC = C_ppt - C_mat                                    # shape (3,3,3,3)
    # broadcast: h[..., None, None, None, None] * dC
    return C_mat[None, None] + h[..., None, None, None, None] * dC[None, None]


# ── Green's function ───────────────────────────────────────────────────────────

def _build_green_operator(C_eff: np.ndarray,
                          N: int, Lx: float) -> np.ndarray:
    """
    Build the Green's operator (acoustic tensor inverse) in Fourier space.

    G_ik(k) = [A_ik(k)]^{-1},  A_ik = C^eff_ijkl k_j k_l

    The k=0 mode is explicitly set to zero (removes singular rigid-body mode).

    Returns
    -------
    G : ndarray, shape (N, N, 2, 2)
        Green's operator for the 2D problem (in-plane components only).
    """
    freqs = np.fft.fftfreq(N, d=Lx / N)
    kv    = 2.0 * np.pi * freqs               # wave-vector components
    kx, ky = np.meshgrid(kv, kv, indexing="ij")  # (N, N)
    k = np.stack([kx, ky], axis=-1)              # (N, N, 2)

    # Acoustic tensor  A_ik = C^eff_{ijkl} k_j k_l   (i,k in 0..1 for 2D)
    A = np.zeros((N, N, 2, 2), dtype=np.float64)
    for i in range(2):
        for kc in range(2):
            for j in range(2):
                for l in range(2):
                    A[..., i, kc] += C_eff[i, j, kc, l] * k[..., j] * k[..., l]

    # Invert pointwise; set k=0 mode to zero
    G = np.zeros_like(A)
    det = A[..., 0, 0] * A[..., 1, 1] - A[..., 0, 1] * A[..., 1, 0]
    nonzero = np.abs(det) > 1e-30
    G[nonzero, 0, 0] =  A[nonzero, 1, 1] / det[nonzero]
    G[nonzero, 1, 1] =  A[nonzero, 0, 0] / det[nonzero]
    G[nonzero, 0, 1] = -A[nonzero, 0, 1] / det[nonzero]
    G[nonzero, 1, 0] = -A[nonzero, 1, 0] / det[nonzero]
    # k=0 mode remains zero (already initialised to 0)
    return G


# ── main solver ────────────────────────────────────────────────────────────────

def solve_elasticity(
    c          : np.ndarray,
    C_mat      : np.ndarray,
    C_ppt      : np.ndarray,
    eigenstrain: np.ndarray,
    sigma_applied: Optional[np.ndarray] = None,
    Lx         : float = 1.0,
    tol        : float = 1e-7,
    max_iter   : int   = 2000,
    verbose    : bool  = False,
    scheme     : InterpolationScheme | str = "cubic",
    u_init     : Optional[np.ndarray] = None,
) -> dict:
    """
    Solve mechanical equilibrium for a 2D periodic two-phase microstructure.

    Parameters
    ----------
    c : ndarray (N, N)
        Composition field (0 = matrix, 1 = precipitate).  Values in [0, 1].
    C_mat : ndarray (3, 3, 3, 3)
        Stiffness tensor of the matrix phase.
    C_ppt : ndarray (3, 3, 3, 3)
        Stiffness tensor of the precipitate phase.
    eigenstrain : ndarray (2, 2)
        Stress-free (transformation) strain tensor ε*.
        For a purely dilatational misfit: eigenstrain = delta * I_2x2.
    sigma_applied : ndarray (2, 2) or None
        Applied (far-field) average stress.  Default: zero (stress-free BC).
    Lx : float
        Physical domain size [m] (same in x and y).
    tol : float
        Convergence criterion on the RMS strain change per iteration.
    max_iter : int
        Maximum number of iterations.
    verbose : bool
        Print convergence info every 100 iterations.
    scheme : InterpolationScheme or str
        Composition-interpolation scheme for the stiffness and eigenstrain
        fields (see calphad_pf.elasticity.interpolation).  Default "cubic"
        matches the paper (Eqs. 47-48).  Pass a string to look up by name,
        or an InterpolationScheme instance directly.
    u_init : ndarray (N, N, 2) or None
        Warm-start displacement field from the previous solve (e.g.
        result["displacement"] from the last CH step).  When supplied the
        iteration starts from u_init instead of zero, which cuts the number
        of iterations needed for small composition changes between steps.
        Default None → cold start from zero.

    Returns
    -------
    result : dict with keys
        "strain"         : ndarray (N, N, 2, 2)  total strain field ε_ij(x)
        "elastic_strain" : ndarray (N, N, 2, 2)  elastic strain ε^el = ε - ε⁰(c)
        "stress"         : ndarray (N, N, 2, 2)  Cauchy stress σ_ij(x)
        "displacement"   : ndarray (N, N, 2)     displacement field u_i(x)
        "E"              : ndarray (2, 2)         macroscopic strain
        "n_iter"         : int                   iterations to convergence
        "residual"       : float                 final RMS residual
        "residual_history" : list[float]       RMS Δε per iteration (for convergence plots)
    """
    N   = c.shape[0]
    assert c.shape == (N, N), "Only square 2D grids supported."

    if isinstance(scheme, str):
        scheme = get_scheme(scheme)

    if sigma_applied is None:
        sigma_applied = np.zeros((2, 2), dtype=np.float64)

    # Reference (effective) stiffness — arithmetic mean (Voigt bound)
    C_eff = 0.5 * (C_mat + C_ppt)

    # Spatially varying stiffness  C(x)
    C_field = _interp_stiffness(c, C_mat, C_ppt, scheme)  # (N, N, 3, 3, 3, 3)
    # We only need the 2D in-plane block (i,j,k,l in 0..1)
    C2 = C_field[..., :2, :2, :2, :2]                # (N, N, 2, 2, 2, 2)
    C_eff2 = C_eff[:2, :2, :2, :2]

    # Eigenstrain field  ε*(x) = eigenstrain * h(c(x))
    h = scheme.h(c)                                   # smooth indicator (N,N)
    es2 = eigenstrain[:2, :2]                         # (2,2) in-plane misfit

    # Green's operator in Fourier space
    G = _build_green_operator(C_eff2, N, Lx)          # (N, N, 2, 2)

    # Wave-vector array (for displacement-to-strain)
    freqs = np.fft.fftfreq(N, d=Lx / N)
    kv    = 2.0 * np.pi * freqs
    kx, ky = np.meshgrid(kv, kv, indexing="ij")
    k = np.stack([kx, ky], axis=-1)                   # (N, N, 2)

    # --- Compliance of C_eff for the macroscopic strain update ---
    # S_eff = C_eff^{-1}  (2×2×2×2 Mandel inverse)
    # We use a small helper: flatten to Voigt (3-component plane strain)
    # Voigt order: [11, 22, 12]  → indices [(0,0),(1,1),(0,1)]
    voigt_idx = [(0,0), (1,1), (0,1)]
    factors   = [1.0, 1.0, 2.0]    # engineering shear factor
    C_eff_V = np.zeros((3, 3), dtype=np.float64)
    for a, (i, j) in enumerate(voigt_idx):
        for b, (k_, l) in enumerate(voigt_idx):
            C_eff_V[a, b] = C_eff2[i, j, k_, l] * factors[a] * factors[b] / 4.0
            # standard Mandel: no extra factor needed for symmetric convention
            # just use raw Cijkl
    # Redo without Mandel factors (plain Voigt):
    C_eff_V = np.array([
        [C_eff2[i, j, k_, l] for b, (k_, l) in enumerate(voigt_idx)]
        for a, (i, j) in enumerate(voigt_idx)
    ], dtype=np.float64)
    S_eff_V = np.linalg.inv(C_eff_V)

    def _voigt_to_tensor(sv):
        """3-vector (Voigt) → 2×2 symmetric tensor."""
        T = np.zeros((2, 2))
        T[0, 0] = sv[0]; T[1, 1] = sv[1]; T[0, 1] = T[1, 0] = sv[2]
        return T

    def _tensor_to_voigt(T):
        """2×2 symmetric tensor → 3-vector (Voigt)."""
        return np.array([T[0, 0], T[1, 1], T[0, 1]])

    # ── initialise ──────────────────────────────────────────────────────────
    residual_history: list[float] = []

    if u_init is not None:
        # Warm start: derive initial strain from the supplied displacement field.
        u_fluct_init = u_init  # treat supplied field as the fluctuation part
        u_hat_init   = np.fft.fftn(u_fluct_init, axes=(0, 1))
        eps_fluct_hat_init = 0.5j * (
            np.einsum("xyj,xyi->xyij", k, u_hat_init) +
            np.einsum("xyi,xyj->xyij", k, u_hat_init)
        )
        eps_fluct_init = np.real(np.fft.ifftn(eps_fluct_hat_init, axes=(0, 1)))
        # Compute E from <σ> = σ^A with the warm-start strain
        eps_el_init = eps_fluct_init - h[..., None, None] * es2[None, None]
        sigma_init  = np.einsum("xyijkl,xykl->xyij", C2, eps_el_init)
        tau_init    = sigma_init - np.einsum("ijkl,xykl->xyij", C_eff2, eps_fluct_init)
        tau_mean_init = tau_init.mean(axis=(0, 1))
        E   = _voigt_to_tensor(S_eff_V @ _tensor_to_voigt(sigma_applied - tau_mean_init))
        eps = eps_fluct_init + E[None, None]
        u   = u_fluct_init
    else:
        eps = np.zeros((N, N, 2, 2), dtype=np.float64)
        u   = np.zeros((N, N, 2),    dtype=np.float64)
        # Initial macroscopic strain from applied stress (Eq. 25)
        E = _voigt_to_tensor(S_eff_V @ _tensor_to_voigt(sigma_applied))

    for iteration in range(max_iter):
        eps_old = eps.copy()

        # --- stress at each point  σ_ij = C_ijkl(x) [ε_kl - ε*_kl h(c)] ---
        eps_el = eps - h[..., None, None] * es2[None, None]  # elastic strain
        sigma  = np.einsum("xyijkl,xykl->xyij", C2, eps_el) # (N,N,2,2)

        # --- polarisation stress  τ_ij = σ_ij - C^eff_ijkl ε_kl (Eq. 16) ---
        sigma_eff = np.einsum("ijkl,xykl->xyij", C_eff2, eps)
        tau = sigma - sigma_eff                               # (N, N, 2, 2)

        # --- FFT of τ ---
        tau_hat = np.fft.fftn(tau, axes=(0, 1))              # (N, N, 2, 2)

        # --- displacement correction in Fourier space ---
        # Equilibrium: A_ik u_k = b_i  where b_i = ik_j τ_ij  (div of polarization)
        # Solution:    u_i = G_ij b_j   (G = A^{-1}, the Green's operator)
        # Note: POSITIVE sign — not negative.
        b_hat = 1j * np.einsum("xyl,xyil->xyi", k, tau_hat)  # b_i = ik_l τ_il
        u_hat_corr = np.einsum("xyij,xyj->xyi", G, b_hat)    # u_i = G_ij b_j
        # k=0 mode already zero in G, so u_hat_corr[0,0] = 0 automatically

        # --- displacement in real space ---
        u_fluct = np.real(np.fft.ifftn(u_hat_corr, axes=(0, 1)))  # (N,N,2)

        # --- total displacement = fluctuation + macroscopic (E · x) ---
        x_arr = np.linspace(0.0, Lx, N, endpoint=False)
        xx, yy = np.meshgrid(x_arr, x_arr, indexing="ij")
        u = u_fluct + np.stack([
            E[0, 0] * xx + E[0, 1] * yy,
            E[1, 0] * xx + E[1, 1] * yy,
        ], axis=-1)

        # --- strain from displacement gradient (spectral, Eq. 11) ---
        u_hat_full = np.fft.fftn(u_fluct, axes=(0, 1))  # (N, N, 2)
        # ε_ij_hat = (i k_j u_hat_i + i k_i u_hat_j) / 2
        eps_fluct_hat = 0.5j * (
            np.einsum("xyj,xyi->xyij", k, u_hat_full) +
            np.einsum("xyi,xyj->xyij", k, u_hat_full)
        )
        eps_fluct = np.real(np.fft.ifftn(eps_fluct_hat, axes=(0, 1)))
        eps = eps_fluct + E[None, None]   # total strain = fluctuation + E

        # --- update macroscopic strain E from <σ> = σ^A (Eq. 24-26) ---
        # <σ> = C_eff E + <τ>  =  σ^A
        # => E = S_eff (σ^A - <τ>)
        tau_mean = tau.mean(axis=(0, 1))      # (2, 2) volume average
        rhs_V    = _tensor_to_voigt(sigma_applied - tau_mean)
        E        = _voigt_to_tensor(S_eff_V @ rhs_V)

        # --- convergence check ---
        d_eps = np.sqrt(np.mean((eps - eps_old)**2))
        residual_history.append(float(d_eps))
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

    # Final stress
    eps_el = eps - h[..., None, None] * es2[None, None]
    sigma_final = np.einsum("xyijkl,xykl->xyij", C2, eps_el)

    return {
        "strain"         : eps,
        "elastic_strain" : eps_el,   # eps - eps0(c);  C2 : elastic_strain = stress
        "stress"         : sigma_final,
        "displacement"   : u,
        "E"              : E,
        "n_iter"         : iteration + 1,
        "residual"       : float(d_eps),
        "residual_history" : residual_history,
    }
