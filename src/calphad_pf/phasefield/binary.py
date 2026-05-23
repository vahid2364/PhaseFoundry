"""
Binary Cahn-Hilliard Phase-Field Solver
========================================

Solves the conserved Cahn-Hilliard (C-H) equation for a binary system:

    ∂c/∂t = ∇ · ( M ∇μ )

where the chemical potential is the variational derivative of the
Ginzburg-Landau free energy functional:

    μ = dF/dc − κ ∇²c

The bulk driving force dF/dc is provided by one of two models:

    'double_well'   Synthetic symmetric double-well (for benchmarking and
                    pedagogical use). Does not require thermodynamic data.

    'calphad'       CALPHAD-derived exchange chemical potential
                    Δμ = μ_i − μ_ref, evaluated via a pre-fitted polynomial
                    surrogate produced by thermodynamics.fitting.

Numerical method
----------------
Pseudo-spectral semi-implicit scheme [1]:

    ĉⁿ⁺¹ = ( ĉⁿ + M Δt k² μ̂_bulk ) / ( 1 + M κ Δt k⁴ )

where k² is the Laplacian operator in Fourier space (real-negative) and
k⁴ = (k²)² the Bi-Laplacian. The gradient energy term is treated
implicitly, giving unconditional stability with respect to the
interfacial terms. The bulk free energy term is explicit.

References
----------
[1] Chen, L.-Q. (2002). Phase-field models for microstructure evolution.
    Annual Review of Materials Research, 32(1), 113–140.
[2] Provatas, N., & Elder, K. (2010). Phase-Field Methods in Materials
    Science and Engineering. Wiley-VCH.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Literal

import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import fft2, ifft2
from tqdm import tqdm

from .free_energy import (
    double_well_df_dc, double_well_f,
    calphad_df_dc, calphad_f,
    calphad_df_dc_spline, calphad_f_spline,
    calphad_df_dc_gm_spline, calphad_f_gm_spline,
)
from .spectral import laplacian_operator


# ── stability helpers ─────────────────────────────────────────────────────────

def stable_dt(
    M: float,
    dx: float,
    calphad_fit: dict,
    temperature_K: float,
    safety: float = 0.5,
) -> float:
    """
    Compute the largest time step that satisfies the explicit bulk-term
    stability criterion for the semi-implicit Cahn-Hilliard solver:

        M · |d²G/dx²|_max · Δt / Δx²  <  safety

    The second derivative is estimated from the surrogate by sampling
    1000 points over the full composition range [0, 1].

    Parameters
    ----------
    M : float
        Interfacial mobility [m⁵/(J·s)].
    dx : float
        Grid spacing [m].
    calphad_fit : dict
        Output of ``build_gm_spline_surrogate``, ``build_spline_surrogate``,
        or ``load_fits``.  All three store a spline or polynomial that can
        be numerically differentiated.
    temperature_K : float
        Temperature in Kelvin (must be present in ``calphad_fit``).
    safety : float
        Safety factor applied to the criterion (default 0.5, i.e. right
        at the stability boundary).  Use 0.4 or lower for extra margin.

    Returns
    -------
    float
        Maximum stable time step in seconds.
    """
    stored_T = np.array(calphad_fit["temperatures_K"])
    idx      = int(np.argmin(np.abs(stored_T - temperature_K)))
    T_K      = stored_T[idx]

    # Sample d²G/dx² over [0,1] using the available surrogate
    x   = np.linspace(0.01, 0.99, 1000)
    el  = list(calphad_fit["fits"][T_K]["models"].keys())[0]
    mdl = calphad_fit["fits"][T_K]["models"][el]

    if "spline_dfdx" in mdl:          # G_m spline: analytic 2nd derivative
        d2G = mdl["spline_dfdx"].derivative()(x)
    elif "spline" in mdl:             # Δμ spline: analytic derivative = d²G/dx²
        d2G = mdl["spline"].derivative()(x)
    else:                             # polynomial: finite-difference estimate
        h   = 1e-5
        mu  = lambda v: np.polyval(mdl["coefficients"][::-1], v)   # placeholder
        d2G = (mu(x + h) - 2 * mu(x) + mu(x - h)) / h ** 2

    d2G_max = float(np.max(np.abs(d2G)))
    # With degenerate mobility M(c) = M₀·c·(1−c), the effective mobility at
    # the mean composition c0 gives the tightest stability bound.
    # We use c0 = 0.5 (worst case) so the estimate is conservative.
    M_eff_max = M * 0.25          # M·c·(1-c) peaks at c=0.5 with value M/4
    dt_max = safety * dx ** 2 / (M_eff_max * d2G_max)
    return dt_max


def write_simulation_params(
    output_dir: str | Path,
    N: int,
    Lx: float,
    kappa: float,
    M: float,
    dt: float,
    n_steps: int,
    save_every: int,
    calphad_fit: dict,
    temperature_K: float,
    dt_stable: float | None = None,
) -> Path:
    """
    Write a human-readable parameter summary to ``sim_params.txt`` in
    ``output_dir``.

    Computed quantities
    -------------------
    dx          Grid spacing  Δx = Lx / N
    delta       Diffuse interface width  δ ≈ sqrt(κ / |d²G/dx²|_max)
    lambda_star Most unstable spinodal wavelength
                λ* = 4π · sqrt(κ / |d²G/dx²|_c0)  (evaluated at mean c)
    lambda_crit Critical (onset) wavelength
                λ_c = 2π · sqrt(2κ / |d²G/dx²|_c0)
    sigma       Approximate interface energy  σ ≈ |d²G/dx²|_max · δ² [J/m²]
    t_total     Total simulated physical time  n_steps × dt

    Parameters
    ----------
    output_dir : path
        Directory where ``sim_params.txt`` is written.
    N, Lx, kappa, M, dt, n_steps : simulation parameters
    calphad_fit : dict
        Surrogate fit (needed to evaluate d²G/dx²).
    temperature_K : float
    dt_stable : float, optional
        The auto-computed stable dt (for reference); printed alongside
        the actual dt used.

    Returns
    -------
    Path to the written file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dx = Lx / N

    # ── retrieve d²G/dx² from surrogate ──────────────────────────────────────
    stored_T = np.array(calphad_fit["temperatures_K"])
    T_K      = stored_T[int(np.argmin(np.abs(stored_T - temperature_K)))]
    el       = list(calphad_fit["fits"][T_K]["models"].keys())[0]
    mdl      = calphad_fit["fits"][T_K]["models"][el]

    x = np.linspace(0.01, 0.99, 2000)
    if "spline_dfdx" in mdl:
        gm_arr  = mdl["spline_gm"](x)
        d2G_arr = mdl["spline_dfdx"].derivative()(x)
    elif "spline" in mdl:
        gm_arr  = mdl["spline_antideriv"](x) if "spline_antideriv" in mdl else None
        d2G_arr = mdl["spline"].derivative()(x)
    else:
        h = 1e-5
        mu      = lambda v: np.polyval(mdl["coefficients"][::-1], v)
        d2G_arr = (mu(x + h) - 2 * mu(x) + mu(x - h)) / h ** 2
        gm_arr  = None

    d2G_max = float(np.max(np.abs(d2G_arr)))
    # most negative d²G → most unstable spinodal point
    d2G_c0  = float(np.abs(d2G_arr[np.argmin(d2G_arr)]))

    # ── interface energy: σ = ∫ sqrt(2κ · ΔG_mix) dc  (common-tangent reference) ──
    sigma = float('nan')
    if gm_arr is not None:
        try:
            # find spinodal region: where d²G < 0
            spinodal_mask = d2G_arr < 0
            if spinodal_mask.any():
                # common-tangent approximation: linear fit between spinodal edges
                idx_lo = int(np.argmax(spinodal_mask))
                idx_hi = int(len(spinodal_mask) - 1 - np.argmax(spinodal_mask[::-1]))
                x_lo, x_hi   = x[idx_lo], x[idx_hi]
                gm_lo, gm_hi = gm_arr[idx_lo], gm_arr[idx_hi]
                tangent       = gm_lo + (gm_hi - gm_lo) / (x_hi - x_lo) * (x - x_lo)
                dG_mix        = gm_arr - tangent        # excess above common tangent
                # integrate only where dG_mix >= 0 (inside spinodal)
                mask   = (x >= x_lo) & (x <= x_hi) & (dG_mix >= 0)
                if mask.sum() > 2:
                    from scipy.integrate import trapezoid
                    sigma = float(trapezoid(
                        np.sqrt(2.0 * kappa * dG_mix[mask]), x[mask]
                    ))
        except Exception:
            sigma = float('nan')

    # ── derived length scales ─────────────────────────────────────────────────
    delta        = float(np.sqrt(kappa / d2G_max))
    lambda_star  = 4.0 * np.pi * np.sqrt(kappa / max(d2G_c0, 1e-30))
    lambda_crit  = 2.0 * np.pi * np.sqrt(2.0 * kappa / max(d2G_c0, 1e-30))
    t_total      = dt * n_steps
    # Stability ratio with degenerate mobility: effective M peaks at c=0.5 → M/4
    M_eff_max       = M * 0.25
    stability_ratio = M_eff_max * d2G_max * dt / dx ** 2

    # ── warnings ──────────────────────────────────────────────────────────────
    warn_delta  = "  ⚠  δ < dx — interface UNRESOLVED on grid!" if delta < dx else ""
    warn_lambda = "  ⚠  λ* < dx — spinodal wavelength UNRESOLVED!" if lambda_star < dx else ""
    warn_kappa  = (f"  ⚠  Suggested κ for δ = 5·dx:  κ_min ≈ {d2G_max * (5*dx)**2:.3e} J·m²/mol"
                   if delta < dx else "")

    sigma_str = f"{sigma:.3e} J/m²" if not np.isnan(sigma) else "n/a (spinodal not found)"

    # ── format and write ──────────────────────────────────────────────────────
    lines = [
        "=" * 60,
        "  Cahn-Hilliard Simulation Parameters",
        "=" * 60,
        "",
        f"  Temperature          T        = {temperature_K:.2f} K  ({temperature_K - 273.15:.1f} °C)",
        "",
        "  ── Domain ──────────────────────────────────────────",
        f"  Grid points          N        = {N}",
        f"  Domain size          Lx       = {Lx:.3e} m",
        f"  Grid spacing         dx       = {dx:.3e} m",
        "",
        "  ── Physical parameters ─────────────────────────────",
        f"  Gradient energy      κ        = {kappa:.3e} J·m²/mol",
        f"  Mobility             M        = {M:.3e} m⁵/(J·s)",
        "",
        "  ── Derived length scales ───────────────────────────",
        f"  |d²G/dx²|_max        f''_max  = {d2G_max:.3e} J/mol",
        f"  Interface width      δ        = {delta:.3e} m  ({delta/dx:.2f} × dx){('  ' + warn_delta) if warn_delta else ''}",
        f"  Interface energy     σ        = {sigma_str}",
        f"  Most unstable λ*     λ*       = {lambda_star:.3e} m  ({lambda_star/dx:.1f} × dx){('  ' + warn_lambda) if warn_lambda else ''}",
        f"  Critical wavelength  λ_c      = {lambda_crit:.3e} m  ({lambda_crit/dx:.1f} × dx)",
        f"  Domain / λ*                   = {Lx/lambda_star:.1f}",
    ]
    if warn_kappa:
        lines.append(f"  {warn_kappa}")
    lines += [
        "",
        "  ── Time integration ────────────────────────────────",
    ]
    if dt_stable is not None:
        lines.append(f"  Stable dt (safety=0.4) dt_stable= {dt_stable:.4e} s")
    lines += [
        f"  Time step used       dt       = {dt:.4e} s",
        f"  Steps                n_steps  = {n_steps}",
        f"  Save interval        save_every= {save_every}  (every {save_every * dt:.4e} s)",
        f"  Total physical time  t_total  = {t_total:.4e} s",
        f"  Stability ratio  M·f''·dt/dx² = {stability_ratio:.4f}  (must be < 0.5)",
        "",
        "=" * 60,
    ]

    out_path = output_dir / "sim_params.txt"
    with open(out_path, "w") as fh:
        fh.write("\n".join(lines) + "\n")

    print(f"  Parameters written → {out_path}")
    return out_path


# ── result container ──────────────────────────────────────────────────────────

@dataclass
class CHResult:
    """
    Output of :func:`solve_cahn_hilliard_binary`.

    Attributes
    ----------
    composition : np.ndarray, shape (N, N)
        Final composition field c(x,y).
    composition_history : list of np.ndarray
        Composition snapshots saved at ``save_every`` intervals.
    time_history : list of float
        Physical times corresponding to each snapshot.
    free_energy_history : list of float
        Total free energy F = ∫(f_bulk + κ/2 |∇c|²) dV at each snapshot.
    bulk_energy_history : list of float
        Bulk contribution F_bulk = ∫f_bulk dV at each snapshot.
    grad_energy_history : list of float
        Gradient contribution F_grad = κ/2 ∫|∇c|² dV at each snapshot.
    mix_energy_history : list of float
        Mixing free energy ΔG_mix = ∫(G_m - linear_ref) dV, where
        linear_ref is the mechanical mixture line between the pure endpoints.
        Only populated for CALPHAD G_m spline models; empty list otherwise.
    params : dict
        All input parameters (for reproducibility).
    elapsed_seconds : float
        Wall-clock runtime.
    """
    composition:         np.ndarray
    composition_history: list
    time_history:        list
    free_energy_history: list
    bulk_energy_history: list
    grad_energy_history: list
    mix_energy_history:  list
    params:              dict
    elapsed_seconds:     float


# ── solver ────────────────────────────────────────────────────────────────────

def solve_cahn_hilliard_binary(
    # ── free energy model ─────────────────────────────────────────────────
    free_energy: Literal["double_well", "calphad", "calphad_spline", "calphad_gm_spline"],

    # ── domain ────────────────────────────────────────────────────────────
    N:  int   = 256,
    Lx: float = 200.0,

    # ── physical parameters ───────────────────────────────────────────────
    kappa: float = 0.5,
    M:     float = 1.0,

    # ── initial composition ───────────────────────────────────────────────
    c0:    float = 0.4,
    noise: float = 0.01,
    seed:  int   = 0,

    # ── time integration ──────────────────────────────────────────────────
    dt:       float | None = None,   # None → auto-select from stability criterion
    dt_scale: float        = 1.0,    # multiply auto dt by this factor (e.g. 1e-3 for testing)
    n_steps:  int          = 2000,

    # ── double-well parameters (used when free_energy='double_well') ──────
    rho_s: float = 5.0,
    c_a:   float = 0.3,
    c_b:   float = 0.7,

    # ── CALPHAD parameters (used when free_energy='calphad') ─────────────
    calphad_fit:   dict | None  = None,
    temperature_K: float | None = None,

    # ── output ────────────────────────────────────────────────────────────
    save_every:  int            = 100,
    output_dir:  str | Path | None = None,
    save_frames: bool           = False,
) -> CHResult:
    """
    Solve the binary Cahn-Hilliard equation on a 2-D periodic domain.

    Parameters
    ----------
    free_energy : {'double_well', 'calphad'}
        Free energy model to use:

        * ``'double_well'`` — synthetic polynomial double-well, suitable
          for benchmarking and parameter studies. No thermodynamic data
          required.
        * ``'calphad'`` — CALPHAD-derived exchange chemical potential
          Δμ = μ_i − μ_ref via a pre-fitted polynomial surrogate. Requires
          ``calphad_fit`` (output of ``thermodynamics.fitting.load_fits``)
          and ``temperature_K``.

    N : int
        Number of grid points along each spatial dimension.
    Lx : float
        Physical domain length [m for CALPHAD, arbitrary units for
        double-well].
    kappa : float
        Gradient energy coefficient κ [J·m²/mol for CALPHAD;
        arbitrary units for double-well].
    M : float
        Interfacial mobility [m⁵/(J·s) for CALPHAD; arbitrary for
        double-well].
    c0 : float
        Mean initial composition (mole fraction).
    noise : float
        Amplitude of random initial perturbation around c0.
    seed : int
        Random seed for reproducibility.
    dt : float
        Time step [s for CALPHAD; arbitrary for double-well].
    n_steps : int
        Total number of time steps.
    rho_s : float
        Free energy barrier height for the double-well model.
    c_a, c_b : float
        Equilibrium compositions (well minima) for the double-well model.
    calphad_fit : dict
        Polynomial surrogate loaded by ``thermodynamics.fitting.load_fits``.
        Required when ``free_energy='calphad'``.
    temperature_K : float
        Simulation temperature in Kelvin.
        Required when ``free_energy='calphad'``.
    save_every : int
        Save a composition snapshot every this many steps.
    output_dir : str or Path, optional
        If given, save PNG frames and the final field to this directory.
    save_frames : bool
        If True and output_dir is set, save a PNG at each snapshot step.

    Returns
    -------
    CHResult
        Dataclass containing the final field, full history, and metadata.

    Raises
    ------
    ValueError
        If ``free_energy='calphad'`` but ``calphad_fit`` or
        ``temperature_K`` are not provided.

    Examples
    --------
    Double-well (no thermodynamic data required):

    >>> result = solve_cahn_hilliard_binary(
    ...     free_energy='double_well',
    ...     N=256, Lx=200, kappa=0.5, M=1.0,
    ...     c0=0.4, dt=0.1, n_steps=2000,
    ... )

    CALPHAD-coupled (Nb-V at 500 °C):

    >>> from thermodynamics.fitting import load_fits
    >>> fit = load_fits('pipelines/output/nb_v/delta_mu_fits.h5')
    >>> result = solve_cahn_hilliard_binary(
    ...     free_energy='calphad',
    ...     N=256, Lx=100e-6, kappa=40e-19, M=1.0,
    ...     c0=0.4, dt=5e-7, n_steps=2000,
    ...     calphad_fit=fit, temperature_K=773.15,
    ... )
    """
    # ── validation ────────────────────────────────────────────────────────
    if free_energy not in ("double_well", "calphad", "calphad_spline", "calphad_gm_spline"):
        raise ValueError(
            f"free_energy must be 'double_well', 'calphad', or 'calphad_spline', "
            f"got '{free_energy}'."
        )
    if free_energy in ("calphad", "calphad_spline", "calphad_gm_spline"):
        if calphad_fit is None:
            raise ValueError(
                f"calphad_fit is required when free_energy='{free_energy}'. "
                "Load it with thermodynamics.fitting.load_fits() or "
                "thermodynamics.fitting.build_spline_surrogate()."
            )
        if temperature_K is None:
            raise ValueError(
                f"temperature_K (in Kelvin) is required when free_energy='{free_energy}'."
            )

    # ── output directory — clear old results before new run ──────────────
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        # Remove previous frames and parameter file
        removed = 0
        for f in output_dir.iterdir():
            if f.suffix in (".png", ".txt", ".npy") or f.name == "sim_params.txt":
                f.unlink()
                removed += 1
        if removed:
            print(f"  Cleared {removed} old file(s) from {output_dir}")

    # ── auto time-step selection ──────────────────────────────────────────
    dx            = Lx / N
    dt_stable_val = None
    if dt is None:
        if free_energy in ("calphad", "calphad_spline", "calphad_gm_spline"):
            dt_stable_val = stable_dt(M=M, dx=dx, calphad_fit=calphad_fit,
                                      temperature_K=temperature_K, safety=0.4)
            dt = dt_stable_val * dt_scale
            if dt_scale == 1.0:
                print(f"  auto dt = {dt:.4e} s  (stability criterion, safety=0.4)")
            else:
                print(f"  auto dt = {dt_stable_val:.4e} s  ×  dt_scale={dt_scale:.3g}"
                      f"  →  dt = {dt:.4e} s")
        else:
            dt = 0.1 * dt_scale
            print(f"  auto dt = {dt:.4e} s  (double-well default × dt_scale={dt_scale:.3g})")
    else:
        # Warn if user-supplied dt violates stability for CALPHAD models
        if free_energy in ("calphad", "calphad_spline", "calphad_gm_spline"):
            dt_stable_val = stable_dt(M=M, dx=dx, calphad_fit=calphad_fit,
                                      temperature_K=temperature_K, safety=0.5)
            if dt > dt_stable_val:
                import warnings
                warnings.warn(
                    f"dt={dt:.3e} s exceeds the stability limit dt_max={dt_stable_val:.3e} s "
                    f"({dt/dt_stable_val:.2f}×). "
                    "Consider setting dt=None for automatic selection.",
                    UserWarning, stacklevel=2,
                )

    # ── write parameter file ──────────────────────────────────────────────
    if output_dir is not None and free_energy in ("calphad", "calphad_spline", "calphad_gm_spline"):
        write_simulation_params(
            output_dir=output_dir, N=N, Lx=Lx, kappa=kappa, M=M,
            dt=dt, n_steps=n_steps, save_every=save_every,
            calphad_fit=calphad_fit, temperature_K=temperature_K,
            dt_stable=dt_stable_val,
        )

    # ── spectral operators ────────────────────────────────────────────────
    k2 = laplacian_operator(N, Lx)    # ∇²  in Fourier space (N×N, negative)
    k4 = k2 ** 2                      # ∇⁴  in Fourier space (positive)
    # denom is recomputed each step when degenerate mobility is used

    # ── initial condition ─────────────────────────────────────────────────
    rng = np.random.default_rng(seed)
    c   = c0 + noise * (rng.random((N, N)) - 0.5)
    c   = np.clip(c, 0.0, 1.0)

    # ── degenerate mobility M(c) = M₀ · c · (1 − c) ─────────────────────
    # Ensures diffusion vanishes in pure phases (c→0 or c→1), preserving
    # mass conservation and preventing unphysical fluxes at boundaries.
    # The effective mobility field is recomputed each step from current c.
    def mobility(c: np.ndarray) -> float:
        """Spatially averaged degenerate mobility M(c) = M₀·c·(1−c)."""
        return float(np.mean(M * c * (1.0 - c)))

    # ── select bulk driving force function ────────────────────────────────
    if free_energy == "double_well":
        def bulk_mu(c: np.ndarray) -> np.ndarray:
            return double_well_df_dc(c, rho_s=rho_s, c_a=c_a, c_b=c_b)

        def bulk_f(c: np.ndarray) -> float:
            return float(np.mean(double_well_f(c, rho_s=rho_s, c_a=c_a, c_b=c_b)))

        bulk_gm = None  # ΔG_mix not available for double-well

    elif free_energy == "calphad":
        def bulk_mu(c: np.ndarray) -> np.ndarray:
            return calphad_df_dc(c, fit=calphad_fit, temperature_K=temperature_K)

        def bulk_f(c: np.ndarray) -> float:
            return float(np.mean(calphad_f(c, fit=calphad_fit, temperature_K=temperature_K)))

        bulk_gm = None

    elif free_energy == "calphad_spline":
        def bulk_mu(c: np.ndarray) -> np.ndarray:
            return calphad_df_dc_spline(c, spline_fit=calphad_fit, temperature_K=temperature_K)

        def bulk_f(c: np.ndarray) -> float:
            return float(np.mean(calphad_f_spline(c, spline_fit=calphad_fit, temperature_K=temperature_K)))

        bulk_gm = None

    else:  # calphad_gm_spline
        def bulk_mu(c: np.ndarray) -> np.ndarray:
            return calphad_df_dc_gm_spline(c, gm_spline_fit=calphad_fit, temperature_K=temperature_K)

        def bulk_f(c: np.ndarray) -> float:
            return float(np.mean(calphad_f_gm_spline(c, gm_spline_fit=calphad_fit, temperature_K=temperature_K)))

        # G_m spline — build mechanical mixture (linear) reference for ΔG_mix
        stored_T  = np.array(calphad_fit["temperatures_K"])
        T_K_idx   = int(np.argmin(np.abs(stored_T - temperature_K)))
        T_K_key   = stored_T[T_K_idx]
        el_key    = list(calphad_fit["fits"][T_K_key]["models"].keys())[0]
        spl_gm    = calphad_fit["fits"][T_K_key]["models"][el_key]["spline_gm"]
        gm_at_0   = float(spl_gm(0.0))
        gm_at_1   = float(spl_gm(1.0))

        def bulk_gm(c: np.ndarray) -> np.ndarray:
            """G_m(c) evaluated on the composition field."""
            return spl_gm(c)

        def linear_ref(c: np.ndarray) -> np.ndarray:
            """Mechanical mixture line: G_m(0)*(1-c) + G_m(1)*c."""
            return gm_at_0 * (1.0 - c) + gm_at_1 * c

    # ── storage ───────────────────────────────────────────────────────────
    composition_history  = []
    time_history         = []
    free_energy_history  = []
    bulk_energy_history  = []
    grad_energy_history  = []
    mix_energy_history   = []

    def _snapshot(c: np.ndarray, step: int) -> None:
        t_phys = step * dt
        F_bulk = bulk_f(c)

        # Gradient energy: F_grad = κ/2 ∫|∇c|²dV  (via Parseval's theorem)
        c_hat  = fft2(c)
        F_grad = float(kappa / 2.0 * np.mean((-k2) * np.abs(c_hat) ** 2) / (N ** 2))
        F_tot  = F_bulk + F_grad

        composition_history.append(c.copy())
        time_history.append(t_phys)
        free_energy_history.append(F_tot)
        bulk_energy_history.append(F_bulk)
        grad_energy_history.append(F_grad)

        # ΔG_mix = mean(G_m(c) − linear_ref(c))  [only for G_m spline model]
        if bulk_gm is not None:
            dG = float(np.mean(bulk_gm(c) - linear_ref(c)))
            mix_energy_history.append(dG)

        if output_dir is not None and save_frames:
            _save_frame(c, step, t_phys, output_dir)

    # ── time integration ──────────────────────────────────────────────────
    _snapshot(c, 0)
    t_start = time.perf_counter()

    for step in tqdm(range(1, n_steps + 1), desc=f"C-H ({free_energy})", unit="step"):
        # Degenerate mobility: M_eff(c) = M₀·⟨c·(1−c)⟩  (spatially averaged)
        # Vanishes in pure phases → no artificial mass injection at c=0 or c=1.
        # Recomputed every step so it tracks the evolving composition field.
        M_eff = float(np.mean(M * c * (1.0 - c)))
        denom = 1.0 + M_eff * kappa * dt * k4  # implicit denominator (≥ 0)

        mu_bulk = bulk_mu(c)
        mu_hat  = fft2(mu_bulk)
        c_hat   = fft2(c)

        # Semi-implicit update: gradient energy implicit, bulk explicit
        #   ĉⁿ⁺¹ = (ĉⁿ + M_eff Δt k² μ̂_bulk) / (1 + M_eff κ Δt k⁴)
        c_hat_new = (c_hat + M_eff * dt * k2 * mu_hat) / denom
        c = np.real(ifft2(c_hat_new))
        # No hard clip — degenerate mobility naturally suppresses fluxes in
        # pure phases; clipping would inject/remove mass and break conservation.

        if step % save_every == 0 or step == n_steps:
            _snapshot(c, step)

    elapsed = time.perf_counter() - t_start

    # ── save energy log and plot ──────────────────────────────────────────
    if output_dir is not None:
        _save_energy_log(
            output_dir=output_dir,
            time_history=time_history,
            free_energy_history=free_energy_history,
            bulk_energy_history=bulk_energy_history,
            grad_energy_history=grad_energy_history,
            mix_energy_history=mix_energy_history,
            composition_history=composition_history,
            save_every=save_every,
            dt=dt,
        )
        # plot will be generated after CHResult is constructed — call separately

    # ── collect parameters ────────────────────────────────────────────────
    params = dict(
        free_energy   = free_energy,
        N=N, Lx=Lx, kappa=kappa, M=M,
        c0=c0, noise=noise, seed=seed,
        dt=dt, n_steps=n_steps, save_every=save_every,
    )
    if free_energy == "double_well":
        params.update(rho_s=rho_s, c_a=c_a, c_b=c_b)
    else:
        meta = calphad_fit.get("metadata", {})
        params["temperature_K"] = temperature_K
        if "reference_element" in meta:
            params["reference_element"] = meta["reference_element"]
        if "x_variables" in meta:
            params["x_variables"] = meta["x_variables"]
        if free_energy in ("calphad", "calphad_spline") and "poly_degree" in meta:
            pd = meta["poly_degree"]
            params["poly_degree"] = pd if isinstance(pd, str) else int(pd)

    result = CHResult(
        composition         = c,
        composition_history = composition_history,
        time_history        = time_history,
        free_energy_history = free_energy_history,
        bulk_energy_history = bulk_energy_history,
        grad_energy_history = grad_energy_history,
        mix_energy_history  = mix_energy_history,
        params              = params,
        elapsed_seconds     = elapsed,
    )

    if output_dir is not None:
        plot_free_energy_history(result, output_dir=output_dir)

    return result


# ── plotting helpers ──────────────────────────────────────────────────────────

def plot_result(
    result: CHResult,
    output_dir: str | Path | None = None,
    show: bool = True,
) -> None:
    """
    Plot the final composition field and free energy evolution.

    Parameters
    ----------
    result : CHResult
    output_dir : str or Path, optional
        Directory to save figures.
    show : bool
        Call plt.show().
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Composition field
    im = axes[0].imshow(
        result.composition, cmap="RdBu_r", origin="lower",
    )
    plt.colorbar(im, ax=axes[0], label="Composition  (mol fraction)")
    axes[0].set_title(
        f"Composition field  —  $t$ = {result.time_history[-1]:.3g}\n"
        f"[{result.params['free_energy']} model]",
        fontsize=12,
    )
    axes[0].set_xlabel("")
    axes[0].set_ylabel("")
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    axes[0].grid(False)

    # Free energy — total, bulk, gradient
    mk = dict(marker="o", markersize=4, markerfacecolor="white", markeredgewidth=1.2)
    axes[1].plot(result.time_history, result.free_energy_history,
                 color="#1f77b4", linewidth=1.8, label="$F_{total}$", **mk)
    axes[1].plot(result.time_history, result.bulk_energy_history,
                 color="#d62728", linewidth=1.5, linestyle="--", label="$F_{bulk}$", **mk)
    axes[1].plot(result.time_history, result.grad_energy_history,
                 color="#2ca02c", linewidth=1.5, linestyle=":", label="$F_{grad}$", **mk)
    axes[1].set_xlabel("Time", fontsize=12)
    axes[1].set_ylabel("Free energy  (a.u.)", fontsize=12)
    axes[1].set_title("Free energy evolution", fontsize=12)
    axes[1].legend(fontsize=10, frameon=False)
    axes[1].grid(False)

    plt.tight_layout()

    if output_dir is not None:
        p = Path(output_dir) / "result_summary.png"
        plt.savefig(p, dpi=200, bbox_inches="tight")
        print(f"Saved → {p}")

    if show:
        plt.show()
    plt.close()


def _save_frame(
    c: np.ndarray,
    step: int,
    t_phys: float,
    output_dir: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(5, 5))
    vmin, vmax = c.min(), c.max()
    im = ax.imshow(c, cmap="PuOr", origin="lower", vmin=vmin, vmax=vmax)
    plt.colorbar(im, ax=ax, label="$x_V$  (mol fraction V)",
                 shrink=0.25, aspect=10, pad=0.02)
    ax.set_title(f"$t$ = {t_phys:.4g} s  (step {step})", fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    plt.tight_layout()
    plt.savefig(output_dir / f"frame_{step:06d}.png", dpi=150, bbox_inches="tight")
    plt.close()


def _save_energy_log(
    output_dir: Path,
    time_history: list,
    free_energy_history: list,
    bulk_energy_history: list,
    grad_energy_history: list,
    mix_energy_history: list,
    composition_history: list,
    save_every: int,
    dt: float,
) -> Path:
    """Write free energy log to ``free_energy_log.txt`` in output_dir."""
    path = output_dir / "free_energy_log.txt"
    has_mix = len(mix_energy_history) > 0
    header = (
        f"{'step':>10}  {'time_s':>14}  {'F_total':>16}  "
        f"{'F_bulk':>16}  {'F_grad':>16}  {'c_min':>10}  {'c_max':>10}"
    )
    if has_mix:
        header += f"  {'dG_mix':>16}"
    with open(path, "w") as fh:
        fh.write(header + "\n")
        for i, (t, F, Fb, Fg, c) in enumerate(zip(
            time_history, free_energy_history,
            bulk_energy_history, grad_energy_history,
            composition_history,
        )):
            step = i * save_every
            row = (
                f"{step:>10d}  {t:>14.6e}  {F:>16.6e}  "
                f"{Fb:>16.6e}  {Fg:>16.6e}  {c.min():>10.5f}  {c.max():>10.5f}"
            )
            if has_mix:
                row += f"  {mix_energy_history[i]:>16.6e}"
            fh.write(row + "\n")
    return path


def plot_free_energy_history(
    result,
    output_dir: str | Path | None = None,
    show: bool = False,
) -> None:
    """
    Plot bulk, interfacial (gradient), and total free energy vs time.

    Parameters
    ----------
    result : CHResult
    output_dir : path, optional
        If given, saves ``free_energy_history.png`` there.
    show : bool
        Call plt.show().
    """
    t   = result.time_history
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # ── left: all three contributions (log scale on y-axis for |F|) ──────
    ax = axes[0]
    F_tot  = np.abs(result.free_energy_history)
    F_bulk = np.abs(result.bulk_energy_history)
    F_grad = np.abs(result.grad_energy_history)
    ax.plot(t, F_tot,  color="#1f77b4", lw=2,   marker="o", markersize=5,
            markerfacecolor="white", markeredgewidth=1.5, label="|$F_{total}$|")
    ax.plot(t, F_bulk, color="#d62728", lw=1.8, ls="--", marker="s", markersize=5,
            markerfacecolor="white", markeredgewidth=1.5, label="|$F_{bulk}$|")
    ax.plot(t, F_grad, color="#2ca02c", lw=1.8, ls=":",  marker="^", markersize=6,
            markerfacecolor="white", markeredgewidth=1.5, label="|$F_{grad}$|  (interfacial)")
    if result.mix_energy_history:
        dG = np.array(result.mix_energy_history)
        ax.plot(t, np.abs(dG), color="#9467bd", lw=1.8, ls="-.", marker="D", markersize=5,
                markerfacecolor="white", markeredgewidth=1.5, label="|$\\Delta G_{mix}$|")
    ax.set_yscale("log")
    ax.set_xlabel("Time  (s)", fontsize=12)
    ax.set_ylabel("|Free energy|  (J/mol)  [log scale]", fontsize=12)
    ax.set_title("Free energy contributions", fontsize=12)
    ax.legend(fontsize=11, frameon=False)
    ax.grid(False)

    # ── right: min/max composition vs time ────────────────────────────────
    ax2 = axes[1]
    c_min = [c.min() for c in result.composition_history]
    c_max = [c.max() for c in result.composition_history]
    ax2.plot(t, c_min, color="#d62728", lw=1.8, marker="s", markersize=5,
             markerfacecolor="white", markeredgewidth=1.5, label="$x_V^{min}$  (Nb-rich)")
    ax2.plot(t, c_max, color="#1f77b4", lw=1.8, marker="o", markersize=5,
             markerfacecolor="white", markeredgewidth=1.5, label="$x_V^{max}$  (V-rich)")
    ax2.set_xlabel("Time  (s)", fontsize=12)
    ax2.set_ylabel("$x_V$  (mol fraction V)", fontsize=12)
    ax2.set_title("Phase composition evolution", fontsize=12)
    ax2.legend(fontsize=11, frameon=False)
    ax2.grid(False)

    plt.tight_layout()

    if output_dir is not None:
        p = Path(output_dir) / "free_energy_history.png"
        plt.savefig(p, dpi=150, bbox_inches="tight")
        print(f"  Saved → {p}")
    if show:
        plt.show()
    plt.close()
