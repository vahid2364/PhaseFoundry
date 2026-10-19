"""FFT-based iterative elasticity solver for periodic microstructures.

The algorithm is the displacement formulation of Gururajan and Abinandanan (2007), Acta
Materialia 55, 5015-5026 (Section 2.2, Eqs. 14-26):

1. Use C^eff = (C^matrix + C^precipitate) / 2 as the homogeneous reference stiffness.
2. Build the Green's operator G(k) = [C^eff_ijkl k_j k_l]^-1 in Fourier space, with
   G(0) = 0 to remove the rigid-body mode.
3. In each iteration, compute the polarisation stress τ = C(x) : (ε − ε*) − C^eff : ε,
   solve for the periodic displacement in Fourier space with G, update the strain from
   its gradient, and update the macroscopic strain E so that the average stress equals
   the applied stress (Eqs. 24-26).
4. Stop when the RMS change in strain between iterations falls below ``tol``.

Notation follows the paper: u_i is the displacement, ε_ij = (∂u_i/∂x_j + ∂u_j/∂x_i) / 2
the total strain, ε*_ij the eigenstrain (stress-free transformation strain),
σ_ij = C_ijkl (ε_kl − ε*_kl) the stress, E_ij the volume-averaged strain, and σ^A_ij the
applied stress, which is zero by default.

The solver works on 2-D and 3-D periodic grids with the same edge length along every
axis, and takes the dimension from ``c.ndim``. Stiffness and stress share one unit (Pa
in this package); strains are dimensionless.
"""

from __future__ import annotations

import numpy as np

from .interpolation import InterpolationScheme, get_scheme


def build_isotropic_stiffness(G: float, nu: float) -> np.ndarray:
    """Build the isotropic stiffness tensor from the shear modulus and Poisson's ratio.

    Args:
        G: Shear modulus. The tensor has the same unit.
        nu: Poisson's ratio.

    Returns:
        Stiffness tensor C_ijkl, shape ``(3, 3, 3, 3)``.
    """
    lam = 2.0 * G * nu / (1.0 - 2.0 * nu)
    C = np.zeros((3, 3, 3, 3), dtype=np.float64)
    for i in range(3):
        for j in range(3):
            for k in range(3):
                for l_ in range(3):
                    C[i, j, k, l_] = lam * (i == j) * (k == l_) + G * ((i == k) * (j == l_) + (i == l_) * (j == k))
    return C


def build_cubic_stiffness(C11: float, C12: float, C44: float) -> np.ndarray:
    """Build the stiffness tensor of a cubic crystal.

    With the Zener ratio A_Z = 2 C44 / (C11 − C12), A_Z = 1 is isotropic. For A_Z > 1 the
    ⟨100⟩ directions are elastically soft and precipitates square up along the x and y
    axes; for A_Z < 1 the ⟨110⟩ directions are soft and precipitates align at 45°.

    Args:
        C11: Elastic constant C11.
        C12: Elastic constant C12.
        C44: Elastic constant C44. All three share one unit, which the tensor keeps.

    Returns:
        Stiffness tensor C_ijkl, shape ``(3, 3, 3, 3)``.
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


def _interp_stiffness(c: np.ndarray, C_mat: np.ndarray, C_ppt: np.ndarray, scheme: InterpolationScheme) -> np.ndarray:
    """Blend the matrix and precipitate stiffness: C(x) = (1 − h(c)) C^mat + h(c) C^ppt.

    Args:
        c: Composition field with values in [0, 1].
        C_mat: Matrix stiffness, shape ``(3, 3, 3, 3)``.
        C_ppt: Precipitate stiffness, shape ``(3, 3, 3, 3)``.
        scheme: Interpolation scheme that supplies h(c).

    Returns:
        Stiffness field, shape ``(*c.shape, 3, 3, 3, 3)``.
    """
    h = scheme.h(c)  # shape (*spatial,)
    dC = C_ppt - C_mat  # (3,3,3,3)
    return C_mat + h[..., None, None, None, None] * dC


def _build_green_operator(C_eff: np.ndarray, shape: tuple[int, ...], Lx: float) -> np.ndarray:
    """Build the Green's operator, the inverse acoustic tensor, in Fourier space.

    G_ik(k) = [A_ik(k)]^-1 with A_ik = C^eff_ijkl k_j k_l. The k = 0 entry is set to zero
    to remove the rigid-body mode.

    Args:
        C_eff: Reference stiffness. Only its leading ``(d, d, d, d)`` block is used.
        shape: Grid shape, ``(N, N)`` or ``(N, N, N)``.
        Lx: Domain edge length, the same along every axis.

    Returns:
        Green's operator, shape ``(*shape, d, d)`` with ``d = len(shape)``.
    """
    d = len(shape)
    # Build wave-vector arrays for each axis
    kv_axes = [2.0 * np.pi * np.fft.fftfreq(shape[ax], d=Lx / shape[ax]) for ax in range(d)]
    grids = np.meshgrid(*kv_axes, indexing="ij")  # d arrays, each (*shape,)
    k = np.stack(grids, axis=-1)  # (*shape, d)

    # Acoustic tensor  A_ik = C^eff_{ijkl} k_j k_l   (i,k in 0..d-1)
    C_eff_d = C_eff[:d, :d, :d, :d]  # (d,d,d,d)
    A = np.einsum("...j,...l,ijkl->...ik", k, k, C_eff_d)  # (*shape, d, d)

    # Invert pointwise via np.linalg.inv (works for any batch shape)
    A_flat = A.reshape(-1, d, d)
    k2 = np.sum(k**2, axis=-1).reshape(-1)  # |k|^2 at each grid point
    G_flat = np.zeros_like(A_flat)
    nonzero = k2 > 1e-30
    if nonzero.any():
        G_flat[nonzero] = np.linalg.inv(A_flat[nonzero])
    # k=0 mode remains zero
    return G_flat.reshape(*shape, d, d)


def _voigt_setup(d: int):
    """Return the Voigt index pairs and conversion functions for dimension ``d``.

    In 2-D the components are ordered 11, 22, 12; in 3-D they are 11, 22, 33, 23, 13, 12.

    Returns:
        ``(pairs, tensor_to_voigt, voigt_to_tensor)``. The two functions convert a
        symmetric ``(d, d)`` tensor to its Voigt vector and back.
    """
    pairs = [(0, 0), (1, 1), (0, 1)] if d == 2 else [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]

    def tensor_to_voigt(T: np.ndarray) -> np.ndarray:
        return np.array([T[i, j] for i, j in pairs], dtype=np.float64)

    def voigt_to_tensor(v: np.ndarray) -> np.ndarray:
        T = np.zeros((d, d), dtype=np.float64)
        for idx, (i, j) in enumerate(pairs):
            T[i, j] = v[idx]
            T[j, i] = v[idx]
        return T

    return pairs, tensor_to_voigt, voigt_to_tensor


def solve_elasticity(
    c: np.ndarray,
    C_mat: np.ndarray,
    C_ppt: np.ndarray,
    eigenstrain: np.ndarray | None = None,
    sigma_applied: np.ndarray | None = None,
    Lx: float = 1.0,
    tol: float = 1e-7,
    max_iter: int = 2000,
    verbose: bool = False,
    scheme: InterpolationScheme | str = "cubic",
    u_init: np.ndarray | None = None,
    eps_star_field: np.ndarray | None = None,
    C_eff_override: np.ndarray | None = None,
    C_field_override: np.ndarray | None = None,
) -> dict:
    """Solve mechanical equilibrium for a 2-D or 3-D periodic microstructure.

    Args:
        c: Composition field, 0 for matrix and 1 for precipitate, shape ``(N, N)`` or
            ``(N, N, N)``.
        C_mat: Matrix stiffness, shape ``(3, 3, 3, 3)``.
        C_ppt: Precipitate stiffness, shape ``(3, 3, 3, 3)``.
        eigenstrain: Eigenstrain tensor ε* of the precipitate; its leading ``(d, d)`` block
            is used. The field is ε*(x) = h(c(x)) ε*. Ignored when ``eps_star_field`` is
            given; one of the two is required.
        sigma_applied: Average applied stress, shape ``(d, d)``. Defaults to zero
            (traction-free).
        Lx: Domain edge length, the same along every axis. The displacement is returned in
            this unit.
        tol: Convergence threshold on the RMS strain change between iterations.
        max_iter: Maximum number of iterations.
        verbose: Print the residual every 100 iterations and at the end.
        scheme: Interpolation scheme, or its name, for the stiffness and eigenstrain fields.
            The default ``"cubic"`` is the paper's choice.
        u_init: Starting displacement, shape ``(*c.shape, d)``. The solver treats it as the
            periodic part of the displacement. Defaults to zero.
        eps_star_field: Precomputed eigenstrain field ε*(x), shape ``(*c.shape, d, d)``,
            used instead of h(c) ε*. The tensorial Vegard form, for example, passes
            ``einsum("k...,kij->...ij", c_all, Lambda)``.
        C_eff_override: Reference stiffness for the Green's operator, replacing
            ``(C_mat + C_ppt) / 2``. Pass ``model.C_eff(C_mat, C_ppt)`` with
            ``LocalVegardModel`` so the reference matches the parent composition.
        C_field_override: Stiffness field C(x), shape ``(*c.shape, 3, 3, 3, 3)``, replacing
            the h(c) blend of ``C_mat`` and ``C_ppt``. Pass
            ``model.C_field(c_all, C_mat, C_ppt)`` with ``LocalVegardModel``.

    Returns:
        A dict with these entries, where ``s`` is ``c.shape``:

        - ``"strain"``: total strain ε(x), shape ``(*s, d, d)``.
        - ``"elastic_strain"``: ε(x) − ε*(x), shape ``(*s, d, d)``.
        - ``"stress"``: Cauchy stress σ(x), shape ``(*s, d, d)``.
        - ``"displacement"``: total displacement, the periodic part plus E · x, shape
          ``(*s, d)``.
        - ``"E"``: macroscopic strain, shape ``(d, d)``.
        - ``"n_iter"``: number of iterations run.
        - ``"residual"``: RMS strain change in the last iteration.
        - ``"residual_history"``: RMS strain change of every iteration.

    Raises:
        ValueError: If neither ``eigenstrain`` nor ``eps_star_field`` is given.
    """
    spatial_shape = c.shape  # (N,N) or (N,N,N)
    d = c.ndim  # 2 or 3
    fft_axes = tuple(range(d))

    if isinstance(scheme, str):
        scheme = get_scheme(scheme)

    if sigma_applied is None:
        sigma_applied = np.zeros((d, d), dtype=np.float64)

    # Reference (effective) stiffness — override from model or default Voigt average
    C_eff = C_eff_override if C_eff_override is not None else 0.5 * (C_mat + C_ppt)

    # Spatially varying stiffness  C(x)  — override from model or h(c) blend
    C_field = C_field_override if C_field_override is not None else _interp_stiffness(c, C_mat, C_ppt, scheme)
    # Restrict to in-plane/in-volume d×d×d×d block
    C_d = C_field[..., :d, :d, :d, :d]  # (*spatial, d, d, d, d)
    C_eff_d = C_eff[:d, :d, :d, :d]  # (d, d, d, d)

    # Eigenstrain field ε*(x)
    if eps_star_field is not None:
        eps_star = eps_star_field[..., :d, :d]  # (*spatial, d, d)
    else:
        if eigenstrain is None:
            raise ValueError("solve_elasticity: provide either 'eigenstrain' (d,d) or 'eps_star_field' (*spatial,d,d).")
        h = scheme.h(c)  # (*spatial,)
        es = eigenstrain[:d, :d]  # (d, d)
        eps_star = h[..., None, None] * es[None, ...]  # (*spatial, d, d)

    # Green's operator in Fourier space — (*spatial, d, d)
    G = _build_green_operator(C_eff_d, spatial_shape, Lx)

    # Wave-vector grid — (*spatial, d)
    kv_axes = [2.0 * np.pi * np.fft.fftfreq(spatial_shape[ax], d=Lx / spatial_shape[ax]) for ax in range(d)]
    grids = np.meshgrid(*kv_axes, indexing="ij")
    k = np.stack(grids, axis=-1)  # (*spatial, d)

    # Coordinate grid for macroscopic displacement  u_macro_i = E_ij x_j
    x_axes = [np.linspace(0.0, Lx, spatial_shape[ax], endpoint=False) for ax in range(d)]
    x_grids = np.meshgrid(*x_axes, indexing="ij")
    x_coord = np.stack(x_grids, axis=-1)  # (*spatial, d)

    # Voigt helpers for macroscopic strain update
    _, tensor_to_voigt, voigt_to_tensor = _voigt_setup(d)
    pairs_list = [(0, 0), (1, 1), (0, 1)] if d == 2 else [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]
    C_eff_V = np.array(
        [[C_eff_d[i, j, k_, l_] for (k_, l_) in pairs_list] for (i, j) in pairs_list],
        dtype=np.float64,
    )
    S_eff_V = np.linalg.inv(C_eff_V)

    residual_history: list[float] = []

    if u_init is not None:
        u_fluct_init = u_init
        u_hat_init = np.fft.fftn(u_fluct_init, axes=fft_axes)
        eps_fluct_hat = 0.5j * (np.einsum("...j,...i->...ij", k, u_hat_init) + np.einsum("...i,...j->...ij", k, u_hat_init))
        eps_fluct = np.real(np.fft.ifftn(eps_fluct_hat, axes=fft_axes))
        eps_el_0 = eps_fluct - eps_star
        sigma_0 = np.einsum("...ijkl,...kl->...ij", C_d, eps_el_0)
        tau_0 = sigma_0 - np.einsum("ijkl,...kl->...ij", C_eff_d, eps_fluct)
        tau_mean_0 = tau_0.mean(axis=tuple(range(d)))
        E = voigt_to_tensor(S_eff_V @ tensor_to_voigt(sigma_applied - tau_mean_0))
        eps = eps_fluct + E[None, ...]
        u = u_fluct_init
    else:
        eps = np.zeros((*spatial_shape, d, d), dtype=np.float64)
        u = np.zeros((*spatial_shape, d), dtype=np.float64)
        E = voigt_to_tensor(S_eff_V @ tensor_to_voigt(sigma_applied))

    d_eps = np.inf
    for iteration in range(max_iter):
        eps_old = eps.copy()

        # stress  σ_ij = C_ijkl(x) [ε_kl - ε*_kl(x)]
        eps_el = eps - eps_star
        sigma = np.einsum("...ijkl,...kl->...ij", C_d, eps_el)

        # polarisation stress  τ_ij = σ_ij - C^eff_ijkl ε_kl
        sigma_eff = np.einsum("ijkl,...kl->...ij", C_eff_d, eps)
        tau = sigma - sigma_eff

        tau_hat = np.fft.fftn(tau, axes=fft_axes)

        # displacement correction in Fourier space
        b_hat = 1j * np.einsum("...l,...il->...i", k, tau_hat)
        u_hat_corr = np.einsum("...ij,...j->...i", G, b_hat)

        u_fluct = np.real(np.fft.ifftn(u_hat_corr, axes=fft_axes))

        # total displacement = fluctuation + macroscopic (E · x)
        u = u_fluct + np.einsum("ij,...j->...i", E, x_coord)

        # strain from fluctuation displacement gradient (spectral)
        u_hat_full = np.fft.fftn(u_fluct, axes=fft_axes)
        eps_fluct_hat = 0.5j * (np.einsum("...j,...i->...ij", k, u_hat_full) + np.einsum("...i,...j->...ij", k, u_hat_full))
        eps_fluct = np.real(np.fft.ifftn(eps_fluct_hat, axes=fft_axes))
        eps = eps_fluct + E[None, ...]

        # update macroscopic strain from <σ> = σ^A
        tau_mean = tau.mean(axis=tuple(range(d)))
        E = voigt_to_tensor(S_eff_V @ tensor_to_voigt(sigma_applied - tau_mean))

        d_eps = float(np.sqrt(np.mean((eps - eps_old) ** 2)))
        residual_history.append(d_eps)
        if verbose and iteration % 100 == 0:
            print(f"  iter {iteration:4d}  Δε_rms = {d_eps:.3e}")
        if d_eps < tol:
            if verbose:
                print(f"  Converged at iter {iteration}  Δε_rms = {d_eps:.3e}")
            break
    else:
        if verbose:
            print(f"  WARNING: did not converge in {max_iter} iterations  Δε_rms = {d_eps:.3e}")

    eps_el = eps - eps_star
    sigma_final = np.einsum("...ijkl,...kl->...ij", C_d, eps_el)

    return {
        "strain": eps,
        "elastic_strain": eps_el,
        "stress": sigma_final,
        "displacement": u,
        "E": E,
        "n_iter": iteration + 1,
        "residual": d_eps,
        "residual_history": residual_history,
    }
