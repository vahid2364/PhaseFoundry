"""
General N-component Cahn-Hilliard Solver
==========================================
Solves the conserved Cahn-Hilliard equation for systems with any number of
components (binary, ternary, quaternary, …) using a pseudo-spectral
semi-implicit scheme.

For N components there are N−1 independent composition fields c₁, …, c_{N-1}.
The reference component satisfies c_ref = 1 − Σcᵢ.

Governing equations
-------------------
    ∂cᵢ/∂t = ∇ · Σⱼ Mᵢⱼ(c) ∇μⱼ        i = 1 … N−1

    μⱼ = ∂G/∂cⱼ − κⱼ ∇²cⱼ             (variational derivative)

Mobility
--------
    Mᵢⱼ(c) = M̊ᵢⱼ · c₁ · c₂ · … · c_N   (degenerate — vanishes in pure phases)

where M̊ is the constant mobility matrix supplied by the user.

Numerical method (pseudo-spectral semi-implicit)
-------------------------------------------------
Diagonal mobility terms are treated implicitly; off-diagonal (cross) terms
are treated explicitly:

    ĉᵢⁿ⁺¹ = [ ĉᵢⁿ + Δt k² Σⱼ M̄ᵢⱼ μ̂ⱼ_bulk − Δt k⁴ Σⱼ≠ᵢ M̄ᵢⱼ κⱼ ĉⱼⁿ ]
             / [ 1 + Δt k⁴ M̄ᵢᵢ κᵢ ]

where M̄ᵢⱼ = M̊ᵢⱼ · ⟨c₁ · c₂ · … · c_N⟩  (spatially averaged).

For binary (N=2, diagonal M) this reduces exactly to the existing
solve_cahn_hilliard_binary() scheme.

Stability dt
------------
    dt_stable = 0.4 · dx² / ( M̊_max · |f''|_max )

where M̊_max is the largest diagonal entry and |f''|_max is the maximum
absolute Hessian eigenvalue over a coarse composition scan.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import fft2, ifft2
from tqdm import tqdm

from .spectral import laplacian_operator
from ..thermodynamics.interpolant import GmInterpolant


# ── result container ──────────────────────────────────────────────────────────

@dataclass
class SimResult:
    """
    Output of ``solve_cahn_hilliard``.

    Attributes
    ----------
    compositions : ndarray, shape (N_ind, N, N)
        Final composition fields for each independent element.
    compositions_history : list of ndarray, shape (N_ind, N, N)
        Snapshots saved every ``save_every`` steps.
    time_history : list[float]
        Physical time [s] at each snapshot.
    free_energy_history : list[float]
        Total free energy F = F_bulk + F_grad at each snapshot [J/mol].
    bulk_energy_history : list[float]
        Bulk contribution F_bulk [J/mol].
    grad_energy_history : list[float]
        Gradient contribution F_grad [J/mol].
    params : dict
        Full parameter set for reproducibility.
    elapsed_seconds : float
    ind_elements : list[str]
        Names of the independent composition fields (same order as axis 0
        of ``compositions``).
    """
    compositions:         np.ndarray
    compositions_history: list = field(default_factory=list)
    time_history:         list = field(default_factory=list)
    free_energy_history:  list = field(default_factory=list)
    bulk_energy_history:  list = field(default_factory=list)
    grad_energy_history:  list = field(default_factory=list)
    params:               dict = field(default_factory=dict)
    elapsed_seconds:      float = 0.0
    ind_elements:         list = field(default_factory=list)

    # convenience: min/max across all fields
    @property
    def c_min(self) -> float:
        return float(self.compositions.min())

    @property
    def c_max(self) -> float:
        return float(self.compositions.max())


# ── stable time-step ──────────────────────────────────────────────────────────

def stable_dt(
    M_diag:       np.ndarray | float,
    dx:           float,
    interpolant:  GmInterpolant,
    n_scan:       int   = 40,
    safety:       float = 0.4,
) -> float:
    """
    CFL-style stable time step for the semi-implicit CH scheme.

        dt = safety · dx² / ( M_max · |H_max| )

    where |H_max| is the max absolute Hessian eigenvalue over a coarse scan.
    """
    M_diag  = np.atleast_1d(np.asarray(M_diag, dtype=float))
    M_max   = float(M_diag.max())
    N_ind   = interpolant.N_ind

    # coarse scan on the interior of the simplex
    axis    = np.linspace(0.05, 0.95 / N_ind, n_scan)
    grids   = np.meshgrid(*[axis] * N_ind, indexing="ij")
    pts     = np.stack([g.ravel() for g in grids], axis=0)   # (N_ind, n_pts)
    # keep only simplex-interior points
    c_sum   = pts.sum(axis=0)
    valid   = (c_sum < 0.95) & np.all(pts > 0.01, axis=0)
    pts     = pts[:, valid]

    if pts.shape[1] == 0:
        # fallback: single midpoint
        pts = np.full((N_ind, 1), 1.0 / (N_ind + 1))

    H       = interpolant.d2g(pts)     # (N_ind, N_ind, n_pts)
    n_pts   = pts.shape[1]
    eig_max = 0.0
    for k in range(n_pts):
        eigs    = np.linalg.eigvalsh(H[:, :, k])
        eig_max = max(eig_max, float(np.abs(eigs).max()))

    if eig_max == 0.0:
        eig_max = 1.0   # fallback

    return safety * dx**2 / (M_max * eig_max)


# ── main solver ───────────────────────────────────────────────────────────────

def solve_cahn_hilliard(
    interpolant:  GmInterpolant,
    M:            float | Sequence | np.ndarray,
    kappa:        float | Sequence | np.ndarray,
    c0:           float | Sequence | np.ndarray,
    N:            int   = 256,
    Lx:           float = 1.0e-6,
    noise:        float = 0.01,
    seed:         int   = 42,
    dt:           float | None = None,
    n_steps:      int   = 30000,
    save_every:   int   = 5000,
    output_dir:   str | Path | None = None,
    save_frames:  bool  = True,
) -> SimResult:
    """
    General Cahn-Hilliard phase-field solver.

    Parameters
    ----------
    interpolant : GmInterpolant
        Built by ``build_interpolant()``.  Provides ``mu(c)`` and ``gm(c)``.
    M : float or (N_ind, N_ind) array
        Constant mobility matrix [m⁵/(J·s)].  A scalar is broadcast to a
        diagonal matrix; a 1-D array of length N_ind is treated as diagonal.
    kappa : float or length-N_ind array
        Gradient energy coefficient(s) [J·m²/mol].  A scalar is broadcast.
    c0 : float or length-N_ind array
        Mean initial composition(s) (mol fraction of each independent element).
    N : int
        Number of grid points per side (domain is N × N).
    Lx : float
        Physical domain size [m].
    noise : float
        Amplitude of the initial random perturbation.
    seed : int
        Random seed.
    dt : float or None
        Time step [s].  If None, computed from ``stable_dt()``.
    n_steps : int
        Number of time steps.
    save_every : int
        Save a snapshot every this many steps.
    output_dir : path or None
        Directory for PNG frames and log files.  Created if needed.
    save_frames : bool
        Whether to write PNG frames.

    Returns
    -------
    SimResult
    """
    N_ind = interpolant.N_ind
    dx    = Lx / N

    # ── normalise M, kappa, c0 ────────────────────────────────────────────────
    M     = _to_matrix(M,     N_ind)    # (N_ind, N_ind)
    kappa = _to_vector(kappa, N_ind)    # (N_ind,)
    c0    = _to_vector(c0,    N_ind)    # (N_ind,)
    M_diag = np.diag(M)                 # (N_ind,)

    # ── time step ─────────────────────────────────────────────────────────────
    if dt is None:
        dt = stable_dt(M_diag, dx, interpolant)
        print(f"  auto dt = {dt:.4e} s  (stability criterion)")

    # ── spectral operators ────────────────────────────────────────────────────
    k2 = laplacian_operator(N, Lx)   # (N, N), real-negative
    k4 = k2 ** 2                     # (N, N), real-positive

    # ── initial condition ─────────────────────────────────────────────────────
    rng = np.random.default_rng(seed)
    c   = np.empty((N_ind, N, N))
    for i in range(N_ind):
        field = c0[i] + noise * (rng.random((N, N)) - 0.5)
        c[i]  = np.clip(field, 1e-6, 1.0 - 1e-6)

    # ── output directory ──────────────────────────────────────────────────────
    if output_dir is not None:
        output_dir = Path(output_dir)
        # clear old files from previous runs
        if output_dir.exists():
            old = list(output_dir.glob("frame_*.png"))
            for f in old:
                f.unlink()
            if old:
                print(f"  Cleared {len(old)} old file(s) from {output_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)

    # ── history storage ───────────────────────────────────────────────────────
    comp_hist  = []
    time_hist  = []
    F_hist     = []
    Fb_hist    = []
    Fg_hist    = []
    t_phys     = 0.0

    def _snapshot(c_arr: np.ndarray, step: int) -> None:
        """Save a snapshot of the current state."""
        nonlocal t_phys
        comp_hist.append(c_arr.copy())
        time_hist.append(t_phys)

        # bulk free energy
        Fb = float(np.mean(interpolant.gm(c_arr)))
        # gradient energy: sum over components
        Fg = 0.0
        for i in range(N_ind):
            c_hat = fft2(c_arr[i])
            Fg   += float(kappa[i] / 2.0 * np.mean((-k2) * np.abs(c_hat)**2) / N**2)
        F_hist.append(Fb + Fg)
        Fb_hist.append(Fb)
        Fg_hist.append(Fg)

        if output_dir is not None and save_frames:
            _save_frame(c_arr, step, output_dir, interpolant.ind_elements)

    # ── time integration ──────────────────────────────────────────────────────
    _snapshot(c, 0)
    t_start = time.perf_counter()

    for step in tqdm(range(1, n_steps + 1), desc="C-H (general)", unit="step"):
        # degenerate mobility scalar: product of ALL compositions
        c_ref = 1.0 - c.sum(axis=0)                          # (N, N)
        deg   = np.prod(c, axis=0) * np.clip(c_ref, 1e-8, 1) # (N, N)
        deg_mean = float(np.mean(deg))

        # spatially-averaged effective mobility matrix
        M_eff = M * deg_mean                                  # (N_ind, N_ind)
        M_eff_diag = np.diag(M_eff)                          # (N_ind,)

        # chemical potentials from interpolant
        mu_bulk = interpolant.mu(c)                           # (N_ind, N, N)

        # FFT of each field and its bulk chemical potential
        c_hat   = [fft2(c[i])        for i in range(N_ind)]
        mu_hat  = [fft2(mu_bulk[i])  for i in range(N_ind)]

        # update each field
        for i in range(N_ind):
            # explicit cross-terms from off-diagonal mobility
            cross_bulk = sum(
                M_eff[i, j] * mu_hat[j]
                for j in range(N_ind) if j != i
            )
            cross_grad = sum(
                M_eff[i, j] * kappa[j] * k4 * c_hat[j]
                for j in range(N_ind) if j != i
            )

            denom = 1.0 + M_eff_diag[i] * kappa[i] * dt * k4

            numerator = (
                c_hat[i]
                + dt * k2 * (M_eff_diag[i] * mu_hat[i] + cross_bulk)
                - dt * cross_grad
            )
            c[i] = np.real(ifft2(numerator / denom))

        t_phys += dt

        if step % save_every == 0 or step == n_steps:
            _snapshot(c, step)

    elapsed = time.perf_counter() - t_start

    # ── save log ──────────────────────────────────────────────────────────────
    if output_dir is not None:
        _save_log(output_dir, time_hist, F_hist, Fb_hist, Fg_hist, comp_hist, save_every)
        _plot_energy(output_dir, time_hist, F_hist, Fb_hist, Fg_hist)
        _write_params(output_dir, N, Lx, dx, kappa, M, c0, noise, seed, dt,
                      n_steps, save_every, elapsed, interpolant)

    return SimResult(
        compositions         = c,
        compositions_history = comp_hist,
        time_history         = time_hist,
        free_energy_history  = F_hist,
        bulk_energy_history  = Fb_hist,
        grad_energy_history  = Fg_hist,
        params               = dict(N=N, Lx=Lx, kappa=kappa.tolist(),
                                    M=M.tolist(), c0=c0.tolist(),
                                    noise=noise, seed=seed, dt=dt,
                                    n_steps=n_steps, save_every=save_every),
        elapsed_seconds      = elapsed,
        ind_elements         = interpolant.ind_elements,
    )


# ── I/O helpers ───────────────────────────────────────────────────────────────

_CMAP_FIELD = "RdBu_r"

def _save_frame(
    c:         np.ndarray,
    step:      int,
    out_dir:   Path,
    elements:  list[str],
) -> None:
    """Save a PNG frame showing all composition fields side by side."""
    N_ind = c.shape[0]
    fig, axes = plt.subplots(1, N_ind, figsize=(5 * N_ind, 4.5),
                             squeeze=False)
    for i, ax in enumerate(axes[0]):
        im = ax.imshow(c[i], cmap=_CMAP_FIELD, vmin=0, vmax=1,
                       origin="lower", interpolation="nearest")
        ax.set_title(f"$x_{{\\rm {elements[i].capitalize()}}}$  (step {step:,})",
                     fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_dir / f"frame_{step:07d}.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def _save_log(
    out_dir:   Path,
    t_hist:    list,
    F_hist:    list,
    Fb_hist:   list,
    Fg_hist:   list,
    c_hist:    list,
    save_every: int,
) -> None:
    path = out_dir / "free_energy_log.txt"
    header = (f"{'step':>10}  {'time_s':>14}  {'F_total':>16}  "
              f"{'F_bulk':>16}  {'F_grad':>16}  {'c_min':>10}  {'c_max':>10}")
    with open(path, "w") as fh:
        fh.write(header + "\n")
        for k, (t, F, Fb, Fg, c_snap) in enumerate(
                zip(t_hist, F_hist, Fb_hist, Fg_hist, c_hist)):
            step = k * save_every
            row  = (f"{step:>10d}  {t:>14.6e}  {F:>16.6e}  "
                    f"{Fb:>16.6e}  {Fg:>16.6e}  "
                    f"{c_snap.min():>10.5f}  {c_snap.max():>10.5f}")
            fh.write(row + "\n")


def _plot_energy(
    out_dir: Path,
    t_hist:  list,
    F_hist:  list,
    Fb_hist: list,
    Fg_hist: list,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    t = np.array(t_hist)
    ax.plot(t, F_hist,  label="$F_{\\rm total}$",    linewidth=2)
    ax.plot(t, Fb_hist, label="$F_{\\rm bulk}$",     linestyle="--")
    ax.plot(t, Fg_hist, label="$F_{\\rm gradient}$", linestyle=":")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Free energy [J/mol]")
    ax.legend()
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(out_dir / "free_energy_history.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _write_params(
    out_dir:     Path,
    N:           int,
    Lx:          float,
    dx:          float,
    kappa:       np.ndarray,
    M:           np.ndarray,
    c0:          np.ndarray,
    noise:       float,
    seed:        int,
    dt:          float,
    n_steps:     int,
    save_every:  int,
    elapsed:     float,
    interp:      GmInterpolant,
) -> None:
    N_ind    = interp.N_ind
    t_total  = dt * n_steps

    lines = [
        "=" * 60,
        "  Cahn-Hilliard Simulation Parameters (general solver)",
        "=" * 60,
        f"  Components         N_el     = {N_ind + 1}",
        f"  Independent fields N_ind    = {N_ind}",
        f"  Elements           ind      = {interp.ind_elements}",
        f"  Reference element           = {interp.reference_element}",
        f"  Temperature        T        = {interp.temperature_K:.2f} K  "
        f"({interp.temperature_K - 273.15:.1f} °C)",
        "",
        "  ── Domain ──────────────────────────────────────────",
        f"  Grid points        N        = {N}",
        f"  Domain size        Lx       = {Lx:.3e} m",
        f"  Grid spacing       dx       = {dx:.3e} m",
        "",
        "  ── Physical parameters ─────────────────────────────",
    ]
    for i in range(N_ind):
        lines.append(f"  κ_{interp.ind_elements[i]:4s}             kappa[{i}] = {kappa[i]:.3e} J·m²/mol")
    lines.append(f"  Mobility matrix    M  =")
    for i in range(N_ind):
        row_str = "  " + "  ".join(f"{M[i,j]:.3e}" for j in range(N_ind))
        lines.append(row_str)
    lines.append(f"  Mean composition   c0       = {c0.tolist()}")
    lines += [
        "",
        "  ── Time integration ────────────────────────────────",
        f"  Time step          dt       = {dt:.4e} s",
        f"  Steps              n_steps  = {n_steps}",
        f"  Save interval      save_every= {save_every}",
        f"  Total phys. time   t_total  = {t_total:.4e} s",
        f"  Wall time          elapsed  = {elapsed:.1f} s",
        "=" * 60,
    ]
    (out_dir / "sim_params.txt").write_text("\n".join(lines) + "\n")


# ── normalisation utilities ───────────────────────────────────────────────────

def _to_vector(x, N: int) -> np.ndarray:
    """Broadcast scalar or sequence to a 1-D array of length N."""
    x = np.atleast_1d(np.asarray(x, dtype=float))
    if x.size == 1:
        return np.full(N, x.item())
    if x.size != N:
        raise ValueError(f"Expected scalar or length-{N} array, got {x.size}.")
    return x.copy()


def _to_matrix(M, N: int) -> np.ndarray:
    """
    Broadcast M to an (N, N) array.
      scalar    → M * I
      1-D len N → diag(M)
      2-D (N,N) → as-is
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
