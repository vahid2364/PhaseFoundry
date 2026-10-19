"""General N-component Cahn-Hilliard solver.

Solves the conserved Cahn-Hilliard equation for any number of components with a
pseudo-spectral semi-implicit scheme. For N components there are N − 1 independent
composition fields c₁, …, c_{N−1}, and the reference component is c_ref = 1 − Σ cᵢ.

The governing equations are

    ∂cᵢ/∂t = ∇ · Σⱼ Mᵢⱼ(c) ∇μⱼ,   i = 1 … N − 1
    μⱼ = ∂G/∂cⱼ − κⱼ ∇²cⱼ

with the degenerate mobility Mᵢⱼ(c) = M̊ᵢⱼ c₁ c₂ … c_N, which vanishes in pure phases. M̊
is the constant mobility matrix supplied by the caller. The scheme replaces the
degenerate factor by its spatial average, M̄ᵢⱼ = M̊ᵢⱼ ⟨c₁ c₂ … c_N⟩, treats the diagonal
terms implicitly with an Eyre stabilization constant Aᵢ, and treats the off-diagonal
terms explicitly. With k2 = −|k|² the Fourier symbol of ∇² and k4 = k2²:

    ĉᵢⁿ⁺¹ = [ĉᵢⁿ + Δt k2 (M̄ᵢᵢ (μ̂ᵢ − Aᵢ ĉᵢⁿ) + Σⱼ≠ᵢ M̄ᵢⱼ μ̂ⱼ) − Δt k4 Σⱼ≠ᵢ M̄ᵢⱼ κⱼ ĉⱼⁿ]
            / [1 − Δt k2 M̄ᵢᵢ Aᵢ + Δt k4 M̄ᵢᵢ κᵢ]

Here μ̂ is the transform of the bulk chemical potential ∂G/∂cᵢ, plus the elastic term
μ_el when elasticity is enabled. For a binary system with Aᵢ = 0 the update is the one in
``solve_cahn_hilliard_binary``.

The default time step is

    dt_stable = 0.3 dx² / (M̊_max |H|_max)

where M̊_max is the largest diagonal mobility entry and |H|_max is the largest absolute
Hessian eigenvalue over a coarse composition scan around c0 (see ``stable_dt``).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import fftn as _np_fftn
from numpy.fft import ifftn as _np_ifftn
from tqdm import tqdm

from ..elasticity import (
    build_isotropic_stiffness,
    solve_elasticity,
)
from .spectral import laplacian_operator

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ..thermodynamics.interpolant import GmInterpolant

# Optional GPU backend via CuPy.  Falls back to NumPy transparently.
try:
    import cupy as _cp
    from cupy.fft import fftn as _cp_fftn
    from cupy.fft import ifftn as _cp_ifftn

    _CUPY_AVAILABLE = True
except ImportError:
    _cp = None
    _CUPY_AVAILABLE = False


def _get_fft_backend(use_gpu: bool):
    """Return the array module and FFT functions of NumPy or CuPy.

    Args:
        use_gpu: Use CuPy instead of NumPy.

    Returns:
        ``(xp, fftn, ifftn)``.

    Raises:
        RuntimeError: If ``use_gpu`` is True and CuPy is not installed.
    """
    if use_gpu:
        if not _CUPY_AVAILABLE:
            raise RuntimeError(
                "use_gpu=True requires CuPy.  Install with:\n  pip install cupy-cuda12x   # adjust for your CUDA version"
            )
        return _cp, _cp_fftn, _cp_ifftn
    return np, _np_fftn, _np_ifftn


@dataclass
class SimResult:
    """Result of ``solve_cahn_hilliard``.

    ``grid`` is the spatial grid shape, ``(N, N)`` or ``(N, N, N)``, and ``d`` its
    dimension. The four elastic histories are filled only when ``elastic_enabled`` and
    ``elastic_save_fields`` are both set; they skip the initial snapshot and are stored as
    float32.

    Attributes:
        compositions: Final composition fields of the independent elements, shape
            ``(n_ind, *grid)``.
        compositions_history: Composition snapshots, each of shape ``(n_ind, *grid)``. The
            first one is the initial state.
        time_history: Physical time in s of each snapshot.
        free_energy_history: Domain-averaged free energy F = F_bulk + F_grad in J/mol at
            each snapshot. The elastic energy is not included.
        bulk_energy_history: Bulk part F_bulk, the domain average of G_m, in J/mol.
        grad_energy_history: Gradient part F_grad = Σᵢ κᵢ ⟨|∇cᵢ|²⟩ / 2 in J/mol.
        params: Main run parameters (grid, mobility, kappa, c0, noise, seed, time step and
            step counts, plus the isotropic elastic settings when elasticity is on).
            ``dt`` is the time step at the end of the run, which differs from the initial
            one in adaptive mode.
        elapsed_seconds: Wall-clock time of the time-stepping loop in s.
        ind_elements: Independent elements, in the order of axis 0 of ``compositions``.
        mu_ch_history: Chemical potential snapshots in J/mol, shape ``(n_ind, *grid)``.
        mu_el_history: Elastic chemical potential snapshots, shape ``(n_ind, *grid)``.
        stress_history: Cauchy stress snapshots, shape ``(*grid, d, d)``.
        strain_history: Total strain snapshots, shape ``(*grid, d, d)``.
    """

    compositions: np.ndarray
    compositions_history: list = field(default_factory=list)
    time_history: list = field(default_factory=list)
    free_energy_history: list = field(default_factory=list)
    bulk_energy_history: list = field(default_factory=list)
    grad_energy_history: list = field(default_factory=list)
    params: dict = field(default_factory=dict)
    elapsed_seconds: float = 0.0
    ind_elements: list = field(default_factory=list)
    mu_ch_history: list = field(default_factory=list)
    mu_el_history: list = field(default_factory=list)
    stress_history: list = field(default_factory=list)
    strain_history: list = field(default_factory=list)

    # convenience: min/max across all fields
    @property
    def c_min(self) -> float:
        """Smallest value in the final composition fields."""
        return float(self.compositions.min())

    @property
    def c_max(self) -> float:
        """Largest value in the final composition fields."""
        return float(self.compositions.max())


def _hessian_scan(
    interpolant: GmInterpolant,
    c0: np.ndarray | None = None,
    c0_window: float = 0.3,
    n_scan: int = 20,
) -> np.ndarray:
    """Evaluate the Hessian of G on a coarse composition grid.

    Each axis spans the interpolant's grid bounds (0.01 to 0.99 if it has none), shrunk by
    1e-4 at both ends and, when ``c0`` is given, limited to ``c0 ± c0_window``. Points with
    Σ cᵢ ≥ 0.99 or any cᵢ ≤ 0.001 are dropped; if none remain, the equimolar point is used.
    ``stable_dt`` and the stabilization constants of ``solve_cahn_hilliard`` both use this
    scan.

    Args:
        interpolant: Free-energy interpolant.
        c0: Centre of the scan, shape ``(n_ind,)``. ``None`` scans the whole grid.
        c0_window: Half-width of the scan along each axis, in mole fraction.
        n_scan: Points per axis.

    Returns:
        Hessian at the kept points, shape ``(n_ind, n_ind, n_points)``.
    """
    N_ind = interpolant.N_ind
    axes_1d = []
    for i in range(N_ind):
        if interpolant.grid_bounds is not None:
            lo_grid, hi_grid = interpolant.grid_bounds[i]
        else:
            lo_grid, hi_grid = 0.01, 0.99
        if c0 is not None:
            c0_i = float(np.atleast_1d(c0)[i])
            lo = max(lo_grid + 1e-4, c0_i - c0_window)
            hi = min(hi_grid - 1e-4, c0_i + c0_window)
        else:
            lo, hi = lo_grid + 1e-4, hi_grid - 1e-4
        axes_1d.append(np.linspace(lo, hi, n_scan))

    grids = np.meshgrid(*axes_1d, indexing="ij")
    pts = np.stack([g.ravel() for g in grids], axis=0)  # (N_ind, n_pts)
    c_sum = pts.sum(axis=0)
    valid = (c_sum < 0.99) & np.all(pts > 0.001, axis=0)
    pts = pts[:, valid]
    if pts.shape[1] == 0:
        pts = np.full((N_ind, 1), 1.0 / (N_ind + 1))
    return interpolant.d2g(pts)  # (N_ind, N_ind, n_valid_pts)


def stable_dt(
    M_diag: np.ndarray | float,
    dx: float,
    interpolant: GmInterpolant,
    c0: np.ndarray | None = None,
    c0_window: float = 0.3,
    n_scan: int = 20,
    safety: float = 0.3,
) -> float:
    """Return a stable time step for the semi-implicit scheme.

    dt = safety · dx² / (M_max |H|_max), where M_max is the largest entry of ``M_diag`` and
    |H|_max is the largest absolute Hessian eigenvalue from ``_hessian_scan``. Centring the
    scan on ``c0`` keeps it away from the simplex corners, where |H| is very large but the
    simulation never goes. Points with NaN Hessians are skipped, and |H|_max is set to 1 if
    no eigenvalue is found.

    Args:
        M_diag: Diagonal mobility entries, or a scalar.
        dx: Grid spacing in m.
        interpolant: Free-energy interpolant.
        c0: Mean initial composition, shape ``(n_ind,)``. ``None`` scans the whole grid.
        c0_window: Half-width of the scan along each axis, in mole fraction.
        n_scan: Scan points per axis.
        safety: Safety factor.

    Returns:
        Time step, in s when the inputs use the units of ``solve_cahn_hilliard``.
    """
    M_diag = np.atleast_1d(np.asarray(M_diag, dtype=float))
    M_max = float(M_diag.max())

    H_scan = _hessian_scan(interpolant, c0, c0_window, n_scan)
    n_pts = H_scan.shape[2]
    eig_max = 0.0
    for k in range(n_pts):
        Hk = H_scan[:, :, k]
        if np.any(np.isnan(Hk)):
            continue
        try:
            eigs = np.linalg.eigvalsh(Hk)
            eig_max = max(eig_max, float(np.abs(eigs).max()))
        except np.linalg.LinAlgError:
            continue

    if eig_max == 0.0:
        eig_max = 1.0  # fallback

    return safety * dx**2 / (M_max * eig_max)


def _flush_npz_checkpoint(
    path: Path,
    comp_hist: list,
    time_hist: list,
    F_hist: list,
    ind_elements: list,
    meta: dict,
) -> None:
    """Write a checkpoint NPZ with every snapshot so far, replacing any earlier file.

    The file holds ``x_<el>`` (all snapshots, float32) and ``x_<el>_final`` for each
    independent element in lowercase, ``time``, ``free_energy`` and the entries of
    ``meta``.
    """
    save_dict: dict = {}
    for i, el in enumerate(ind_elements):
        history = np.array([s[i] for s in comp_hist], dtype=np.float32)
        save_dict[f"x_{el.lower()}"] = history
        save_dict[f"x_{el.lower()}_final"] = comp_hist[-1][i].astype(np.float32)
    save_dict["time"] = np.array(time_hist, dtype=np.float64)
    save_dict["free_energy"] = np.array(F_hist, dtype=np.float64)
    for k, v in meta.items():
        save_dict[k] = v
    np.savez_compressed(path, **save_dict)


def solve_cahn_hilliard(
    interpolant: GmInterpolant,
    M: float | Sequence | np.ndarray,
    kappa: float | Sequence | np.ndarray,
    c0: float | Sequence | np.ndarray,
    N: int = 256,
    Lx: float = 1.0e-6,
    noise: float = 0.01,
    seed: int = 42,
    dt: float | None = None,
    n_steps: int = 30000,
    save_every: int = 5000,
    output_dir: str | Path | None = None,
    save_frames: bool = True,
    stabilization: float | None = None,
    snapshot_callback=None,
    adaptive_dt: bool = False,
    dt_min: float | None = None,
    dt_max: float | None = None,
    dc_tol_lo: float = 1e-4,
    dc_tol_hi: float = 2e-2,
    ndim: int = 2,
    use_gpu: bool = False,
    elastic_enabled: bool = False,
    # Model-based interface: supply a pre-built EigenstrainModel.
    # Controls how eps_star, mu_el, C_eff, and C_field are computed.
    # When None, falls back to FixedVegardModel built from Lambda/C_mat/C_ppt.
    elastic_model: object | None = None,
    # pre-built stiffness tensors (from run.py)
    elastic_C_mat: np.ndarray | None = None,
    elastic_C_ppt: np.ndarray | None = None,
    # isotropic fallback (used only when elastic_C_mat/C_ppt are None)
    elastic_G_matrix: float | None = None,
    elastic_delta: float = 1.0,
    elastic_nu: float = 0.3,
    # Vegard tensor Λ, shape (N_ind, ndim, ndim) — built by run.py from alpha values
    # Replaces scalar elastic_eps_T: eps*(x) = einsum('kxy,kij->xyij', c, Lambda)
    elastic_Lambda: np.ndarray | None = None,
    elastic_sigma_applied: np.ndarray | None = None,
    elastic_solve_every: int = 1,
    elastic_warm_start: bool = True,
    elastic_tol: float = 1e-8,
    elastic_max_iter: int = 500,
    elastic_N_V: float = 1.0,
    elastic_save_fields: bool = False,
    incremental_npz: bool = False,
    npz_meta: dict | None = None,
) -> SimResult:
    """Run a Cahn-Hilliard simulation for any number of components.

    The initial field is ``c0`` plus uniform noise. Snapshots are recorded at the start,
    every ``save_every`` steps and at the end. With ``output_dir`` set, the run also writes
    PNG frames, ``free_energy_log.txt``, ``free_energy_history.png``,
    ``composition_evolution.png`` and ``sim_params.txt`` there.

    Args:
        interpolant: Free-energy interpolant from ``build_interpolant``, providing ``mu``,
            ``gm`` and ``d2g``.
        M: Constant mobility matrix M̊ in m²·mol/(J·s), consistent with μ in J/mol and κ in
            J·m²/mol. A scalar gives a diagonal matrix with that value, a length-``n_ind``
            vector gives a diagonal matrix, and an ``(n_ind, n_ind)`` array is used as is.
        kappa: Gradient energy coefficient of each independent element in J·m²/mol. A
            scalar applies to all of them.
        c0: Mean initial mole fraction of each independent element. A scalar applies to
            all of them.
        N: Grid points along each axis.
        Lx: Domain edge length in m.
        noise: Width of the initial perturbation. Each grid point gets ``c0`` plus a
            uniform random value in [−noise/2, noise/2), clipped to [1e-6, 1 − 1e-6].
        seed: Seed of the random number generator for the initial noise.
        dt: Time step in s. ``None`` uses ``stable_dt`` with the scan centred on ``c0``.
        n_steps: Number of time steps. In adaptive mode the run instead lasts
            ``n_steps × dt`` of physical time, with ``dt`` the initial step.
        save_every: Steps between snapshots. In adaptive mode the ``n_steps // save_every``
            snapshots are instead spaced evenly in physical time.
        output_dir: Directory for frames, logs and plots. It is created if needed, and any
            ``frame_*.png`` already in it is deleted. ``None`` writes nothing.
        save_frames: Write a PNG frame at each snapshot when ``output_dir`` is set.
        stabilization: Eyre stabilization constant Aᵢ in J/mol, applied to every component.
            ``None`` sets Aᵢ = max(0, −H_ii) over a Hessian scan around ``c0``, separately
            for each component.
        snapshot_callback: Function called as ``snapshot_callback(c, t)`` after every
            snapshot except the initial one, with the composition array and the physical
            time in s.
        adaptive_dt: Adapt the time step to dc_max, the largest change of any composition
            value in a step. Below ``dc_tol_lo`` the step doubles and above ``dc_tol_hi``
            it halves, within ``[dt_min, dt_max]``. Every step is accepted and the new
            value applies from the next step. The last step is shortened to end at the
            final time.
        dt_min: Smallest adaptive time step in s. Defaults to the initial ``dt`` / 100.
        dt_max: Largest adaptive time step in s. Defaults to the initial ``dt`` × 20.
        dc_tol_lo: Value of dc_max below which the time step doubles.
        dc_tol_hi: Value of dc_max above which the time step halves.
        ndim: Spatial dimension, 2 (grid ``N × N``) or 3 (grid ``N × N × N``). In 3-D the
            PNG frames show the mid-plane slice along the last axis.
        use_gpu: Compute the FFTs with CuPy instead of NumPy.
        elastic_enabled: Add the elastic contribution μ_el (Gururajan and Abinandanan 2007,
            Eq. 36) to the chemical potential. When False, the ``elastic_*`` arguments are
            ignored and ``solve_elasticity`` is never called. Works in 2-D and 3-D.
        elastic_model: Eigenstrain model, e.g. from ``build_eigenstrain_model``, that
            supplies ε*, μ_el and the stiffness fields. ``None`` uses a
            ``FixedVegardModel`` built from ``elastic_Lambda`` and the stiffness tensors.
        elastic_C_mat: Matrix stiffness tensor in Pa, shape ``(3, 3, 3, 3)``. When this
            and ``elastic_C_ppt`` are both given, the isotropic settings are not used.
        elastic_C_ppt: Precipitate stiffness tensor in Pa, shape ``(3, 3, 3, 3)``.
        elastic_G_matrix: Matrix shear modulus in Pa for isotropic stiffness tensors, used
            when ``elastic_C_mat`` or ``elastic_C_ppt`` is missing.
        elastic_delta: Ratio of precipitate to matrix shear modulus for the isotropic
            tensors. 1.0 is elastically homogeneous.
        elastic_nu: Poisson's ratio of both phases for the isotropic tensors.
        elastic_Lambda: Vegard tensor from ``build_vegard_tensor``, shape
            ``(n_ind, ndim, ndim)``. Required when ``elastic_enabled`` is set, even with
            ``elastic_model``.
        elastic_sigma_applied: Average applied stress, shape ``(ndim, ndim)``. ``None``
            means zero (traction-free).
        elastic_solve_every: Solve mechanical equilibrium every this many steps and reuse
            μ_el in between (quasi-static approximation). The first step always solves.
        elastic_warm_start: Start each elastic solve from the previous displacement instead
            of zero.
        elastic_tol: Convergence tolerance of each elastic solve.
        elastic_max_iter: Iteration limit of each elastic solve.
        elastic_N_V: Molar density in mol/m³ that converts μ_el from Pa to J/mol. The
            default of 1.0 leaves μ_el in Pa.
        elastic_save_fields: Store μ_ch, μ_el, stress and strain at each snapshot in the
            result.
        incremental_npz: Rewrite ``output_dir/fields_checkpoint.npz`` with all snapshots at
            every snapshot, so an interrupted run keeps its data. Needs ``output_dir``.
        npz_meta: Extra entries for the checkpoint file.

    Returns:
        The final fields, snapshot histories and run parameters.

    Raises:
        ValueError: If ``ndim`` is not 2 or 3, if ``M``, ``kappa`` or ``c0`` has the wrong
            size, or if ``elastic_enabled`` is set without stiffness tensors (or
            ``elastic_G_matrix``) or without ``elastic_Lambda``.
        RuntimeError: If ``use_gpu`` is set and CuPy is not installed, or if μ_el contains
            NaN or Inf because the elastic solve diverged.
    """
    if ndim not in (2, 3):
        raise ValueError(f"ndim must be 2 or 3, got {ndim}.")

    xp, _fftn, _ifftn = _get_fft_backend(use_gpu)
    if use_gpu:
        print(f"  FFT backend: CuPy (GPU)  device = {_cp.cuda.Device().id}")
    else:
        print("  FFT backend: NumPy (CPU)")

    N_ind = interpolant.N_ind
    dx = Lx / N

    if elastic_enabled:
        if elastic_C_mat is not None and elastic_C_ppt is not None:
            _C_mat = elastic_C_mat
            _C_ppt = elastic_C_ppt
        elif elastic_G_matrix is not None:
            _C_mat = build_isotropic_stiffness(elastic_G_matrix, elastic_nu)
            _C_ppt = build_isotropic_stiffness(elastic_delta * elastic_G_matrix, elastic_nu)
        else:
            raise ValueError("elastic_enabled=True requires elastic_C_mat/elastic_C_ppt or elastic_G_matrix.")
        if elastic_Lambda is None:
            raise ValueError(
                "elastic_enabled=True requires elastic_Lambda — the Vegard "
                "eigenstrain coefficient tensor of shape (N_ind, ndim, ndim). "
                "Build it with build_vegard_tensor(alpha, ind_elements, ndim) "
                "using alpha values from elastic_params.yaml (step 1b output)."
            )
        _Lambda = elastic_Lambda  # (N_ind, ndim, ndim)

        if elastic_model is not None:
            _model = elastic_model
            print(
                f"  Elastic coupling ENABLED ({_model.__class__.__name__}):  "
                f"N_ind={N_ind}  solve_every={elastic_solve_every}  "
                f"N_V={elastic_N_V:.4e}"
            )
        else:
            from ..elasticity.models import FixedVegardModel

            _model = FixedVegardModel(_Lambda, _C_mat, _C_ppt)
            print(
                f"  Elastic coupling ENABLED (FixedVegardModel):  "
                f"N_ind={N_ind}  solve_every={elastic_solve_every}  "
                f"N_V={elastic_N_V:.4e}"
            )
            for k, el in enumerate(interpolant.ind_elements):
                print(f"    α_{el} = {_Lambda[k, 0, 0]:.5f}")
    else:
        _C_mat = _C_ppt = _Lambda = None
    _mu_el_cache = None  # (N_ind, N, N), reused across elastic_solve_every steps
    _elastic_u_prev = None  # warm-start displacement from last solve
    _elastic_result = None  # last full elastic solve result (for save_fields)

    M = _to_matrix(M, N_ind)  # (N_ind, N_ind)
    kappa = _to_vector(kappa, N_ind)  # (N_ind,)
    c0 = _to_vector(c0, N_ind)  # (N_ind,)
    M_diag = np.diag(M)  # (N_ind,)

    dt_stable = stable_dt(M_diag, dx, interpolant, c0=c0)
    if dt is None:
        dt = dt_stable
        print(f"  auto dt = {dt:.4e} s  (stability criterion, scan ±0.3 around c0)")
    else:
        print(f"  dt_stable = {dt_stable:.4e} s  |  dt used = {dt:.4e} s")

    dt_lo = dt_min if dt_min is not None else dt / 100.0
    dt_hi = dt_max if dt_max is not None else dt * 20.0
    if adaptive_dt:
        print(f"  Adaptive dt ENABLED  dc_tol = [{dc_tol_lo:.1e}, {dc_tol_hi:.1e}]  dt bounds = [{dt_lo:.3e}, {dt_hi:.3e}] s")
    # track total physical end-time so adaptive mode still reaches t_final
    t_final = dt * n_steps

    # per-component stabilization A_i (Eyre convex-splitting)
    # Each field i gets its own stabilization constant A_i ≥ |H_ii(c0)|.
    # This is tighter than the global scalar (which uses λ_min of the full
    # Hessian) and gives each component exactly the damping it needs:
    #   - Strongly unstable elements (large |H_ii|) get large A_i → heavy damping
    #   - Weakly unstable elements get small A_i → less artificial drag
    # Scheme per field i:
    #   c_i_new·(1 + dt·k²·M_ii·(A_i + κ_i·k²)) = c_i_old + dt·k²·M_ii·(A_i·c_i_old - μ_i_old)
    if stabilization is None:
        # Scan the same composition neighbourhood used by stable_dt so that
        # A_i = max_{c in scan} max(0, -H_ii(c)).
        # Using only c0 is insufficient: H_ii can be significantly more
        # negative at compositions the simulation will visit during
        # decomposition (e.g. in NbV at 500C, H_NB reaches -19840 J/mol
        # at x_NB~0.75 vs -7239 J/mol at x_NB=0.50).
        try:
            H_scan = _hessian_scan(interpolant, c0, c0_window=0.3, n_scan=10)
            A_stab = np.array([max(0.0, float(np.nanmax(-H_scan[i, i, :]))) for i in range(N_ind)])
        except Exception:
            # fallback: single-point evaluation at c0
            c0_field = c0[:, None, None] * np.ones((N_ind, 1, 1))
            try:
                H = interpolant.d2g(c0_field)
                H2d = H[:, :, 0, 0]
                A_stab = np.array([max(0.0, -H2d[i, i]) for i in range(N_ind)])
            except Exception:
                A_stab = np.zeros(N_ind)
    else:
        # scalar broadcast to every component
        A_stab = np.broadcast_to(np.atleast_1d(float(stabilization)), (N_ind,)).copy()

    for i, el in enumerate(interpolant.ind_elements):
        if A_stab[i] > 0:
            print(f"  Stabilization A_{el} = {A_stab[i]:.4e} J/mol  (Eyre, component-specific)")
        else:
            print(f"  Stabilization A_{el} = 0  (convex at c0)")

    grid_shape = (N,) * ndim
    k2 = laplacian_operator(N, Lx, ndim=ndim)  # (N,N) or (N,N,N), real-negative
    k4 = k2**2  # real-positive

    # Parseval normalization factor: 1/N^ndim
    parseval_norm = N**ndim

    print(
        f"  Spatial dimension  ndim = {ndim}  "
        f"({'N×N' if ndim == 2 else 'N×N×N'} = {N}{'×' + str(N) if ndim == 3 else ''}{'×' + str(N) if ndim == 3 else ''})"
    )

    rng = np.random.default_rng(seed)
    c = np.empty((N_ind,) + grid_shape)
    for i in range(N_ind):
        field = c0[i] + noise * (rng.random(grid_shape) - 0.5)
        c[i] = np.clip(field, 1e-6, 1.0 - 1e-6)

    if use_gpu:
        c = xp.asarray(c)

    if output_dir is not None:
        output_dir = Path(output_dir)
        if output_dir.exists():
            old = list(output_dir.glob("frame_*.png"))
            for f in old:
                f.unlink()
            if old:
                print(f"  Cleared {len(old)} old file(s) from {output_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)

    comp_hist = []
    time_hist = []
    F_hist = []
    Fb_hist = []
    Fg_hist = []
    # per-element histories of c_min, c_mean, c_max (one dict each)
    all_elements = interpolant.ind_elements + [interpolant.reference_element]
    cmin_hist = {el: [] for el in all_elements}
    cmean_hist = {el: [] for el in all_elements}
    cmax_hist = {el: [] for el in all_elements}
    t_phys = 0.0

    # elastic field snapshots (populated only when elastic_save_fields=True)
    mu_ch_hist: list[np.ndarray] = []
    mu_el_hist: list[np.ndarray] = []
    stress_hist: list[np.ndarray] = []
    strain_hist: list[np.ndarray] = []

    def _snapshot(c_arr: np.ndarray, step: int, mu_ch_snap=None, mu_el_snap=None, stress_snap=None, strain_snap=None) -> None:
        """Record the composition, time, energies and composition statistics.

        Also stores the elastic fields when they are given, and writes the PNG frame and
        the NPZ checkpoint if those are enabled.
        """
        nonlocal t_phys
        comp_hist.append(c_arr.copy())
        time_hist.append(t_phys)

        Fb = float(np.mean(interpolant.gm(c_arr)))
        # Gradient energy — Parseval derivation (verified against analytic sinusoid):
        #
        #   F_grad = κ/2 · mean_x(|∇c|²)
        #
        # numpy DFT convention:  ĉ(k) = Σ_x c(x) e^{-2πi k·x/N}
        # Parseval:               Σ_x |c|²  = (1/N^d) Σ_k |ĉ|²
        # Gradient in k-space:   ∇ → i·(2πk/Lx), so |∇c|² ↔ (-k2)|ĉ|²
        # Combining:
        #   mean_x(|∇c|²) = (1/N^d) Σ_x |∇c|²
        #                  = (1/N^d)·(1/N^d) Σ_k (-k2)|ĉ|²
        #                  = mean_k((-k2)|ĉ|²) / N^d
        #
        # Therefore:
        #   F_grad = κ/2 · mean_k((-k2)|ĉ|²) / parseval_norm    ✓
        Fg = 0.0
        for i in range(N_ind):
            c_hat = _fftn(c_arr[i])
            Fg += float(xp.mean((-k2) * xp.abs(c_hat) ** 2) * kappa[i] / 2.0 / parseval_norm)
        F_hist.append(Fb + Fg)
        Fb_hist.append(Fb)
        Fg_hist.append(Fg)

        # per-element composition statistics
        c_ref_arr = 1.0 - c_arr.sum(axis=0)
        for i, el in enumerate(interpolant.ind_elements):
            cmin_hist[el].append(float(c_arr[i].min()))
            cmean_hist[el].append(float(c_arr[i].mean()))
            cmax_hist[el].append(float(c_arr[i].max()))
        ref_el = interpolant.reference_element
        cmin_hist[ref_el].append(float(c_ref_arr.min()))
        cmean_hist[ref_el].append(float(c_ref_arr.mean()))
        cmax_hist[ref_el].append(float(c_ref_arr.max()))

        if elastic_save_fields and mu_ch_snap is not None:
            mu_ch_hist.append(np.array(mu_ch_snap, dtype=np.float32))
            mu_el_hist.append(np.array(mu_el_snap, dtype=np.float32))
            stress_hist.append(np.array(stress_snap, dtype=np.float32))
            strain_hist.append(np.array(strain_snap, dtype=np.float32))

        if output_dir is not None and save_frames:
            _save_frame(c_arr, step, output_dir, interpolant.ind_elements, reference_element=interpolant.reference_element)

        if incremental_npz and output_dir is not None:
            _flush_npz_checkpoint(
                output_dir / "fields_checkpoint.npz",
                comp_hist,
                time_hist,
                F_hist,
                interpolant.ind_elements,
                npz_meta or {},
            )

    _snapshot(c, 0)
    t_start = time.perf_counter()

    step = 0
    dt_change_count = 0  # number of adaptive dt adjustments (for logging)
    snap_interval = t_final / max(1, n_steps // save_every)  # adaptive snap cadence
    next_snap_t = snap_interval
    pbar = tqdm(total=n_steps, desc="C-H (general)", unit="step")

    while True:
        if adaptive_dt:
            if t_phys >= t_final:
                break
            # shrink last step so we land exactly on t_final
            dt = min(dt, t_final - t_phys)
        elif step >= n_steps:
            break

        step += 1

        # degenerate mobility scalar: product of ALL compositions
        c_ref = 1.0 - c.sum(axis=0)  # (*spatial,)
        deg = np.prod(c, axis=0) * np.clip(c_ref, 1e-8, 1)  # (*spatial,)
        deg_mean = float(np.mean(deg))

        # spatially-averaged effective mobility matrix
        M_eff = M * deg_mean  # (N_ind, N_ind)
        M_eff_diag = np.diag(M_eff)  # (N_ind,)

        # chemical potentials from interpolant
        # Clamp only for interpolant evaluation to avoid NaN from out-of-bounds
        # queries when any pixel overshoots the table boundary.  The stored
        # composition field c is never modified, so mass is exactly conserved.
        eps = 1e-4
        c_clamp = np.clip(c, eps, 1.0 - eps)
        # also enforce simplex: last component must be positive
        c_ref_clamp = 1.0 - c_clamp.sum(axis=0)
        scale = np.where(c_ref_clamp < eps, (1.0 - eps * (interpolant.N_ind + 1)) / np.maximum(c_clamp.sum(axis=0), 1e-10), 1.0)
        c_eval = c_clamp * scale[np.newaxis]
        mu_bulk = interpolant.mu(c_eval)  # (N_ind, *spatial)

        # Clamp composition to the physical simplex before passing to the
        # elastic model — the SQS interpolant (local_vegard) falls back to
        # nearest-neighbour outside [0,1], which can amplify overshoot.
        # Mass conservation is unaffected: the stored field c is never modified.
        if elastic_enabled and (step % elastic_solve_every == 0 or _mu_el_cache is None):
            c_el = np.clip(c, 0.0, 1.0)
            # re-normalise so rows still sum to ≤ 1 on the simplex
            row_sum = c_el.sum(axis=0, keepdims=True)
            c_el = np.where(row_sum > 1.0, c_el / row_sum, c_el)
            eps_star = _model.eps_star_field(c_el, ndim)
            C_eff_ov = _model.C_eff(_C_mat, _C_ppt)
            C_field_ov = _model.C_field(c_el, _C_mat, _C_ppt)
            c_ref = np.clip(1.0 - c.sum(axis=0), 0.0, 1.0)
            _elastic_result = solve_elasticity(
                c_ref,
                _C_mat,
                _C_ppt,
                eps_star_field=eps_star,
                C_eff_override=C_eff_ov,
                C_field_override=C_field_ov,
                sigma_applied=elastic_sigma_applied,
                Lx=Lx,
                tol=elastic_tol,
                max_iter=elastic_max_iter,
                verbose=False,
                u_init=(_elastic_u_prev if elastic_warm_start else None),
            )
            _elastic_u_prev = _elastic_result["displacement"]
            _mu_el_new = _model.mu_el(_elastic_result["stress"], c_el, elastic_N_V)
            if np.any(~np.isfinite(_mu_el_new)):
                raise RuntimeError(
                    f"Step {step}: elastic mu_el contains NaN/Inf — "
                    "the FFT elasticity solver diverged (likely unconverged at a "
                    "sharp interface). Try: increase tol, increase max_iter, "
                    "increase solve_every, or reduce dt."
                )
            _mu_el_cache = _mu_el_new  # shape (N_ind, *spatial)
        if elastic_enabled:
            mu_bulk = mu_bulk + _mu_el_cache  # both (N_ind, *spatial)

        c_hat = [_fftn(c[i]) for i in range(N_ind)]
        mu_hat = [_fftn(mu_bulk[i]) for i in range(N_ind)]

        # update each field (stabilized semi-implicit / Eyre convex-splitting)
        c_new = xp.empty_like(c)
        for i in range(N_ind):
            # explicit cross-terms from off-diagonal mobility
            cross_bulk = sum(M_eff[i, j] * mu_hat[j] for j in range(N_ind) if j != i)
            cross_grad = sum(M_eff[i, j] * kappa[j] * k4 * c_hat[j] for j in range(N_ind) if j != i)

            # per-component stabilization: A_i treated implicitly
            # denom gains +dt·(-k²)·M_ii·A_i; numerator subtracts A_i·c_hat_i
            Ai = A_stab[i]
            denom = 1.0 + M_eff_diag[i] * dt * (-k2) * Ai + M_eff_diag[i] * kappa[i] * dt * k4

            numerator = c_hat[i] + dt * k2 * (M_eff_diag[i] * (mu_hat[i] - Ai * c_hat[i]) + cross_bulk) - dt * cross_grad
            c_new[i] = xp.real(_ifftn(numerator / denom))

        if adaptive_dt:
            dc_max = float(xp.max(xp.abs(c_new - c)))
            if dc_max < dc_tol_lo and dt < dt_hi:
                dt = min(dt * 2.0, dt_hi)
                dt_change_count += 1
            elif dc_max > dc_tol_hi and dt > dt_lo:
                dt = max(dt / 2.0, dt_lo)
                dt_change_count += 1

        c = c_new
        t_phys += dt
        pbar.update(1)

        if adaptive_dt:
            should_snap = (t_phys >= next_snap_t) or (t_phys >= t_final)
            if should_snap:
                next_snap_t += snap_interval
        else:
            should_snap = (step % save_every == 0) or (step == n_steps)

        if should_snap:
            if elastic_enabled and elastic_save_fields and _elastic_result is not None:
                _snapshot(
                    c,
                    step,
                    mu_ch_snap=mu_bulk - _mu_el_cache,
                    mu_el_snap=_mu_el_cache,
                    stress_snap=_elastic_result["stress"],
                    strain_snap=_elastic_result["strain"],
                )
            else:
                _snapshot(c, step)
            if snapshot_callback is not None:
                snapshot_callback(c, t_phys)

    pbar.close()
    if adaptive_dt:
        print(f"  Adaptive dt: {dt_change_count} adjustment(s), final dt = {dt:.4e} s, total steps = {step}")

    elapsed = time.perf_counter() - t_start

    if output_dir is not None:
        _save_log(
            output_dir, time_hist, F_hist, Fb_hist, Fg_hist, comp_hist, save_every, all_elements, cmin_hist, cmean_hist, cmax_hist
        )
        _plot_energy(output_dir, time_hist, F_hist, Fb_hist, Fg_hist, all_elements, cmin_hist, cmean_hist, cmax_hist)
        _write_params(
            output_dir,
            N,
            Lx,
            dx,
            kappa,
            M,
            c0,
            noise,
            seed,
            dt,
            dt_stable,
            n_steps,
            save_every,
            elapsed,
            interpolant,
            adaptive_dt=adaptive_dt,
            dt_lo=dt_lo,
            dt_hi=dt_hi,
            dc_tol_lo=dc_tol_lo,
            dc_tol_hi=dc_tol_hi,
            dt_changes=dt_change_count,
            steps_actual=step,
            ndim=ndim,
        )

    if use_gpu:
        c = xp.asnumpy(c)
        comp_hist = [xp.asnumpy(s) for s in comp_hist]

    return SimResult(
        compositions=c,
        compositions_history=comp_hist,
        time_history=time_hist,
        free_energy_history=F_hist,
        bulk_energy_history=Fb_hist,
        grad_energy_history=Fg_hist,
        params=dict(
            N=N,
            Lx=Lx,
            kappa=kappa.tolist(),
            M=M.tolist(),
            c0=c0.tolist(),
            noise=noise,
            seed=seed,
            dt=dt,
            n_steps=n_steps,
            save_every=save_every,
            elastic_enabled=elastic_enabled,
            **(
                {
                    "elastic_G_matrix": elastic_G_matrix,
                    "elastic_delta": elastic_delta,
                    "elastic_nu": elastic_nu,
                    "elastic_solve_every": elastic_solve_every,
                    "elastic_warm_start": elastic_warm_start,
                }
                if elastic_enabled
                else {}
            ),
        ),
        elapsed_seconds=elapsed,
        ind_elements=interpolant.ind_elements,
        mu_ch_history=mu_ch_hist,
        mu_el_history=mu_el_hist,
        stress_history=stress_hist,
        strain_history=strain_hist,
    )


_CMAP_FIELD = "RdBu_r"
"""Colormap of the composition frames."""


def _save_frame(
    c: np.ndarray,
    step: int,
    out_dir: Path,
    elements: list[str],
    reference_element: str | None = None,
) -> None:
    """Save a PNG frame of every composition field, including the reference element.

    The file is ``frame_<step>.png`` with the step zero-padded to 7 digits. For 3-D fields
    the frame shows the mid-plane slice along the last axis. Each panel has its own colour
    range.

    Args:
        c: Composition fields, shape ``(n_ind, *grid)``.
        step: Step number, used in the title and the file name.
        out_dir: Output directory.
        elements: Independent elements, in the order of ``c``.
        reference_element: Reference element, shown as 1 − Σ cᵢ. ``None`` leaves it out.
    """
    N_ind = c.shape[0]
    ndim = c.ndim - 1  # spatial dims = total dims minus element axis

    # build list of (data, label) — independent fields + reference if available
    fields = [(c[i], elements[i]) for i in range(N_ind)]
    if reference_element is not None:
        c_ref = 1.0 - c.sum(axis=0)
        fields.append((c_ref, reference_element))

    # for 3-D take mid-plane z-slice; label the slice in the title
    slice_label = ""
    if ndim == 3:
        mid = c.shape[1] // 2
        fields = [(data[..., mid], lbl) for data, lbl in fields]
        slice_label = f"  [z={mid}]"

    n_panels = len(fields)
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 4.5), squeeze=False)
    for ax, (data, label) in zip(axes[0], fields, strict=True):
        im = ax.imshow(data, cmap=_CMAP_FIELD, vmin=data.min(), vmax=data.max(), origin="lower", interpolation="nearest")
        ax.set_title(
            f"$x_{{\\rm {label.capitalize()}}}${slice_label}  (step {step:,})",
            fontsize=11,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_dir / f"frame_{step:07d}.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def _save_log(
    out_dir: Path,
    t_hist: list,
    F_hist: list,
    Fb_hist: list,
    Fg_hist: list,
    c_hist: list,
    save_every: int,
    elements: list[str],
    cmin_hist: dict,
    cmean_hist: dict,
    cmax_hist: dict,
) -> None:
    """Write ``free_energy_log.txt`` with one row per snapshot.

    The columns are the step, the time in s, F_total, F_bulk, F_grad, and the minimum, mean
    and maximum of every element. The step is the snapshot index times ``save_every``,
    which is the true step only for fixed-step snapshots on multiples of ``save_every``.
    """
    path = out_dir / "free_energy_log.txt"
    # header: fixed columns + per-element triplet (min, mean, max)
    el_cols = "  ".join(f"{'x' + el + '_min':>12}  {'x' + el + '_mean':>12}  {'x' + el + '_max':>12}" for el in elements)
    header = f"{'step':>10}  {'time_s':>14}  {'F_total':>16}  {'F_bulk':>16}  {'F_grad':>16}  {el_cols}"
    with path.open("w") as fh:
        fh.write(header + "\n")
        for k, (t, F, Fb, Fg) in enumerate(zip(t_hist, F_hist, Fb_hist, Fg_hist, strict=True)):
            step = k * save_every
            el_vals = "  ".join(
                f"{cmin_hist[el][k]:>12.5f}  {cmean_hist[el][k]:>12.5f}  {cmax_hist[el][k]:>12.5f}" for el in elements
            )
            row = f"{step:>10d}  {t:>14.6e}  {F:>16.6e}  {Fb:>16.6e}  {Fg:>16.6e}  {el_vals}"
            fh.write(row + "\n")


def _plot_energy(
    out_dir: Path,
    t_hist: list,
    F_hist: list,
    Fb_hist: list,
    Fg_hist: list,
    elements: list[str] | None = None,
    cmin_hist: dict | None = None,
    cmean_hist: dict | None = None,
    cmax_hist: dict | None = None,
) -> None:
    """Plot the free-energy history and, if statistics are given, the composition history.

    Writes ``free_energy_history.png`` (|F_total|, |F_bulk| and F_grad on a log scale) and,
    when ``elements`` and ``cmin_hist`` are given, ``composition_evolution.png`` with the
    minimum, mean and maximum of each element over time.
    """
    t = np.array(t_hist)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    F_arr = np.array(F_hist)
    Fb_arr = np.array(Fb_hist)
    Fg_arr = np.array(Fg_hist)
    # use absolute values; gradient energy is always positive
    ax.plot(t, np.abs(F_arr), label="|$F_{\\rm total}$|", linewidth=2, color="#1f77b4")
    ax.plot(t, np.abs(Fb_arr), label="|$F_{\\rm bulk}$|", linewidth=1.5, linestyle="--", color="#ff7f0e")
    ax.plot(t, np.abs(Fg_arr), label="$F_{\\rm gradient}$", linewidth=1.5, linestyle=":", color="#2ca02c")
    ax.set_yscale("log")
    ax.set_xlabel("Time [s]", fontsize=12)
    ax.set_ylabel("|Free energy|  [J/mol]  (log scale)", fontsize=12)
    ax.set_title("Free energy evolution", fontsize=12)
    ax.legend(fontsize=10, frameon=False)
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(out_dir / "free_energy_history.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    if elements is None or cmin_hist is None:
        return
    n_el = len(elements)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    fig, axes = plt.subplots(1, n_el, figsize=(5 * n_el, 4.5), squeeze=False)
    for p, el in enumerate(elements):
        ax = axes[0, p]
        c_lo = np.array(cmin_hist[el])
        c_mean = np.array(cmean_hist[el])
        c_hi = np.array(cmax_hist[el])
        col = colors[p % len(colors)]

        ax.fill_between(t, c_lo, c_hi, alpha=0.15, color=col, label="min–max range")
        ax.plot(
            t,
            c_lo,
            color=col,
            lw=1.5,
            ls="--",
            marker="s",
            markersize=4,
            markerfacecolor="white",
            label=f"$x_{{{el.capitalize()}}}^{{\\rm min}}$",
        )
        ax.plot(
            t,
            c_hi,
            color=col,
            lw=1.5,
            ls="-.",
            marker="o",
            markersize=4,
            markerfacecolor="white",
            label=f"$x_{{{el.capitalize()}}}^{{\\rm max}}$",
        )
        ax.plot(t, c_mean, color=col, lw=2.2, ls="-", label=f"$\\langle x_{{{el.capitalize()}}} \\rangle$")

        ax.set_xlabel("Time [s]", fontsize=12)
        ax.set_ylabel(f"$x_{{{el.capitalize()}}}$  [mol fraction]", fontsize=12)
        ax.set_title(f"{el.capitalize()} — composition evolution", fontsize=12)
        ax.legend(fontsize=9, frameon=False)
        ax.grid(False)
    fig.tight_layout()
    fig.savefig(out_dir / "composition_evolution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _write_params(
    out_dir: Path,
    N: int,
    Lx: float,
    dx: float,
    kappa: np.ndarray,
    M: np.ndarray,
    c0: np.ndarray,
    noise: float,
    seed: int,
    dt: float,
    dt_stable: float,
    n_steps: int,
    save_every: int,
    elapsed: float,
    interp: GmInterpolant,
    adaptive_dt: bool = False,
    dt_lo: float | None = None,
    dt_hi: float | None = None,
    dc_tol_lo: float = 1e-4,
    dc_tol_hi: float = 2e-2,
    dt_changes: int = 0,
    steps_actual: int = 0,
    ndim: int = 2,
) -> None:
    """Write ``sim_params.txt``, a text summary of the run.

    The summary lists the domain and physical parameters, the Hessian at ``c0``, length
    scales derived from κ and the most negative Hessian eigenvalue (interface width,
    fastest-growing and critical wavelengths), the time integration settings, and the
    stability ratio M_ii |H|_max dt / (4 dx²) of each field.
    """
    N_ind = interp.N_ind
    t_total = dt * n_steps

    c0_col = c0.reshape(N_ind, 1)
    H_c0 = interp.d2g(c0_col)[:, :, 0]  # (N_ind, N_ind)
    eigs = np.linalg.eigvalsh(H_c0)  # ascending
    lam_min = float(eigs[0])  # most negative
    lam_max_abs = float(np.abs(eigs).max())

    # Hessian diagonal elements (per-field self-interaction)
    H_diag = np.array([H_c0[i, i] for i in range(N_ind)])

    # Use |λ_min| (most negative eigenvalue) as the relevant driving force
    # for interface width and spinodal wavelengths; this is the worst case.
    f2_eff = max(abs(lam_min), 1e-30)  # effective |d²G|

    delta = [float(np.sqrt(kappa[i] / f2_eff)) for i in range(N_ind)]
    lambda_star = [4.0 * np.pi * np.sqrt(kappa[i] / f2_eff) for i in range(N_ind)]
    lambda_crit = [2.0 * np.pi * np.sqrt(2.0 * kappa[i] / f2_eff) for i in range(N_ind)]
    kappa_min = [f2_eff * (5.0 * dx) ** 2 for _ in range(N_ind)]  # for δ = 5·dx

    # Effective M peaks at c=0.5 for each field (degenerate mobility)
    stab = [float(M[i, i] * 0.25 * lam_max_abs * dt / dx**2) for i in range(N_ind)]

    lines = [
        "=" * 60,
        "  Cahn-Hilliard Simulation Parameters (general solver)",
        "=" * 60,
        f"  Spatial dimension  ndim     = {ndim}",
        f"  Components         N_el     = {N_ind + 1}",
        f"  Independent fields N_ind    = {N_ind}",
        f"  Elements           ind      = {interp.ind_elements}",
        f"  Reference element           = {interp.reference_element}",
        f"  Temperature        T        = {interp.temperature_K:.2f} K  ({interp.temperature_K - 273.15:.1f} °C)",
        f"  Derivative method           = {interp.derivative_method}",
        "",
        "  ── Domain ──────────────────────────────────────────",
        f"  Grid points        N        = {N}",
        f"  Domain size        Lx       = {Lx:.3e} m",
        f"  Grid spacing       dx       = {dx:.3e} m",
        "",
        "  ── Physical parameters ─────────────────────────────",
    ]
    for i in range(N_ind):
        lines.append(f"  κ_{interp.ind_elements[i]:<4}            kappa[{i}] = {kappa[i]:.3e} J·m²/mol")
    lines.append("  Mobility matrix    M  =")
    for i in range(N_ind):
        row_str = "  " + "  ".join(f"{M[i, j]:.3e}" for j in range(N_ind))
        lines.append(row_str)
    lines.append(f"  Mean composition   c0       = {c0.tolist()}")

    if interp.grid_bounds is not None:
        lines += ["", "  ── Interpolant grid bounds ─────────────────────────"]
        for i, (lo, hi) in enumerate(interp.grid_bounds):
            lines.append(f"  x_{interp.ind_elements[i]:<4}  grid range         = [{lo:.4f}, {hi:.4f}]")

    lines += [
        "",
        "  ── Hessian at c0 (J/mol) ───────────────────────────",
        f"  Eigenvalues        λ        = {[f'{e:.1f}' for e in eigs]}",
        f"  |λ|_max (driving)  |λ|_max  = {lam_max_abs:.3e} J/mol",
        f"  λ_min (spinodal)   λ_min    = {lam_min:.3e} J/mol  ({'UNSTABLE' if lam_min < 0 else 'stable'})",
    ]
    for i in range(N_ind):
        lines.append(f"  H_{interp.ind_elements[i]}{interp.ind_elements[i]}  (diagonal)         = {H_diag[i]:.3e} J/mol")

    lines += ["", "  ── Derived length scales (per field, using |λ_min|) ──"]
    for i in range(N_ind):
        el = interp.ind_elements[i]
        warn_d = "  ⚠  δ < dx — UNRESOLVED!" if delta[i] < dx else ""
        warn_l = "  ⚠  λ* < dx — UNRESOLVED!" if lambda_star[i] < dx else ""
        warn_k = f"\n    ⚠  Suggested κ_{el} for δ=5·dx: κ_min ≈ {kappa_min[i]:.3e} J·m²/mol" if delta[i] < dx else ""
        lines += [
            f"  [{el}]",
            f"    Interface width    δ_{el:<4}  = {delta[i]:.3e} m  ({delta[i] / dx:.2f} × dx){warn_d}",
            f"    Most unstable λ*   λ*_{el:<4} = {lambda_star[i]:.3e} m  ({lambda_star[i] / dx:.1f} × dx){warn_l}",
            f"    Critical λ_c       λ_c_{el:<4}= {lambda_crit[i]:.3e} m  ({lambda_crit[i] / dx:.1f} × dx)",
            f"    Domain / λ*                 = {Lx / lambda_star[i]:.1f}",
        ]
        if warn_k:
            lines.append(warn_k)

    lines += [
        "",
        "  ── Time integration ────────────────────────────────",
        f"  Stable dt (safety=0.4) dt_stable= {dt_stable:.4e} s",
        f"  Time step used       dt       = {dt:.4e} s",
        f"  Steps                n_steps  = {n_steps}  ({'actual: ' + str(steps_actual) if adaptive_dt else 'fixed'})",
        f"  Save interval        save_every= {save_every}  (every {save_every * dt:.4e} s)",
        f"  Total physical time  t_total  = {t_total:.4e} s",
    ]
    if adaptive_dt:
        lines += [
            "",
            "  ── Adaptive time-stepping ──────────────────────────",
            "  Enabled              adaptive  = True",
            f"  dt bounds            [dt_lo, dt_hi] = [{dt_lo:.3e}, {dt_hi:.3e}] s",
            f"  Field-change tols    [lo, hi]  = [{dc_tol_lo:.1e}, {dc_tol_hi:.1e}]",
            f"  Number of dt changes           = {dt_changes}",
            f"  Actual steps taken             = {steps_actual}",
        ]
    for i in range(N_ind):
        el = interp.ind_elements[i]
        warn_s = "  ⚠  > 0.5 — may be unstable!" if stab[i] > 0.5 else ""
        lines.append(f"  Stability ratio [{el}]  M·|λ|·dt/dx² = {stab[i]:.4f}{warn_s}")
    lines += [
        f"  Wall time            elapsed  = {elapsed:.1f} s",
        "=" * 60,
    ]

    (out_dir / "sim_params.txt").write_text("\n".join(lines) + "\n")


def _to_vector(x, N: int) -> np.ndarray:
    """Return ``x`` as a length-``N`` float array, repeating a scalar.

    Raises:
        ValueError: If ``x`` has more than one value but not ``N``.
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    if x.size == 1:
        return np.full(N, x.item())
    if x.size != N:
        raise ValueError(f"Expected scalar or length-{N} array, got {x.size}.")
    return x.copy()


def _to_matrix(M, N: int) -> np.ndarray:
    """Return ``M`` as an ``(N, N)`` mobility matrix.

    A scalar or single value gives M I, a length-``N`` vector gives diag(M), and an
    ``(N, N)`` array is copied.

    Raises:
        ValueError: If ``M`` has any other shape.
    """
    M = np.asarray(M, dtype=float)
    if M.ndim == 0:
        return M.item() * np.eye(N)
    if M.ndim == 1:
        if M.size == 1:
            return M.item() * np.eye(N)
        if M.size == N:
            return np.diag(M)
        raise ValueError(f"1-D M must have length {N}, got {M.size}.")
    if M.shape == (N, N):
        return M.copy()
    raise ValueError(f"M must be scalar, length-{N} vector, or ({N},{N}) matrix.")
