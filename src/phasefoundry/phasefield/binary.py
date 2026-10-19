"""Binary Cahn-Hilliard solver on a 2-D periodic grid.

The solver integrates the conserved Cahn-Hilliard equation for one composition field c:

    ∂c/∂t = ∇ · (M ∇μ),   μ = df/dc − κ ∇²c

The bulk driving force df/dc comes from one of four models: ``"double_well"``, a
synthetic double well that needs no thermodynamic data; ``"calphad"``, the exchange
chemical potential Δμ = μ_i − μ_ref from a polynomial surrogate; and
``"calphad_spline"`` and ``"calphad_gm_spline"``, which use cubic-spline surrogates of Δμ
or G_m from ``thermodynamics.fitting``. For ``"calphad"`` and ``"calphad_spline"``, c is
the mole fraction of the independent element; for ``"calphad_gm_spline"`` it is the mole
fraction of the reference element.

Time stepping is pseudo-spectral and semi-implicit (Chen 2002):

    ĉⁿ⁺¹ = (ĉⁿ + M_eff Δt k² μ̂_bulk) / (1 + M_eff κ Δt k⁴)

where k² is the Fourier-space Laplacian (real and negative) and k⁴ = (k²)². The mobility
is degenerate, M(c) = M₀ c (1 − c), and the scheme uses its spatial average
M_eff = ⟨M₀ c (1 − c)⟩, recomputed every step. The gradient term is implicit, which makes
the step unconditionally stable with respect to it; the bulk term is explicit.

References:
    1. Chen, L.-Q. (2002). Phase-field models for microstructure evolution. Annual Review
       of Materials Research, 32(1), 113-140.
    2. Provatas, N., and Elder, K. (2010). Phase-Field Methods in Materials Science and
       Engineering. Wiley-VCH.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import fft2, ifft2
from tqdm import tqdm

from .free_energy import (
    calphad_df_dc,
    calphad_df_dc_gm_spline,
    calphad_df_dc_spline,
    calphad_f,
    calphad_f_gm_spline,
    calphad_f_spline,
    double_well_df_dc,
    double_well_f,
)
from .spectral import laplacian_operator


def stable_dt(
    M: float,
    dx: float,
    calphad_fit: dict,
    temperature_K: float,
    safety: float = 0.5,
) -> float:
    """Return the largest time step that keeps the explicit bulk term stable.

    The criterion is

        (M/4) · |d²G/dx²|_max · Δt / Δx² < safety

    where M/4 is the largest value of the degenerate mobility M c (1 − c), reached at
    c = 0.5. |d²G/dx²|_max is taken over 1000 points on [0.01, 0.99]. For the G_m spline
    and the Δμ spline it is the analytic derivative of the spline. For the polynomial
    surrogate it is the second finite difference of the polynomial built from
    ``coefficients`` alone, ignoring ``feature_powers`` and ``intercept``; that is a third
    derivative of G_m, not the second.

    Args:
        M: Mobility prefactor M₀. With μ in J/mol and lengths in m, its unit is
            m²·mol/(J·s).
        dx: Grid spacing in m.
        calphad_fit: Output of ``build_gm_spline_surrogate``, ``build_spline_surrogate``
            or ``load_fits``.
        temperature_K: Temperature in K. The closest stored temperature is used, without
            a tolerance check.
        safety: Right-hand side of the criterion. 0.5 is the stability limit itself;
            smaller values leave a margin.

    Returns:
        The largest stable time step, in s for SI inputs.
    """
    stored_T = np.array(calphad_fit["temperatures_K"])
    idx = int(np.argmin(np.abs(stored_T - temperature_K)))
    T_K = stored_T[idx]

    x = np.linspace(0.01, 0.99, 1000)
    el = list(calphad_fit["fits"][T_K]["models"].keys())[0]
    mdl = calphad_fit["fits"][T_K]["models"][el]

    if "spline_dfdx" in mdl:
        d2G = mdl["spline_dfdx"].derivative()(x)
    elif "spline" in mdl:
        d2G = mdl["spline"].derivative()(x)
    else:
        h = 1e-5

        def mu(v):
            return np.polyval(mdl["coefficients"][::-1], v)  # placeholder

        d2G = (mu(x + h) - 2 * mu(x) + mu(x - h)) / h**2

    d2G_max = float(np.max(np.abs(d2G)))
    M_eff_max = M * 0.25
    dt_max = safety * dx**2 / (M_eff_max * d2G_max)
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
    """Write a human-readable parameter summary to ``sim_params.txt`` in ``output_dir``.

    Besides the inputs, the file lists derived quantities, with d²G/dx² sampled at 2000
    points on [0.01, 0.99]:

    - grid spacing Δx = Lx / N;
    - interface width δ = sqrt(κ / |d²G/dx²|_max);
    - fastest-growing spinodal wavelength λ* = 4π sqrt(κ / |d²G/dx²|_min), using the most
      negative d²G/dx²;
    - critical wavelength λ_c = 2π sqrt(2κ / |d²G/dx²|_min);
    - interface energy σ = ∫ sqrt(2κ ΔG_mix) dc, with ΔG_mix measured from the chord
      between the spinodal edges. It needs G_m, so it is only computed for the spline
      surrogates and reads "n/a" otherwise;
    - total time n_steps · dt and the stability ratio (M/4) |d²G/dx²|_max dt / Δx².

    For the polynomial surrogate, d²G/dx² is estimated as in ``stable_dt``, from
    ``coefficients`` alone.

    Args:
        output_dir: Directory for ``sim_params.txt``. It is created if missing.
        N: Grid points along each axis.
        Lx: Domain edge length in m.
        kappa: Gradient energy coefficient κ in J·m²/mol.
        M: Mobility prefactor M₀.
        dt: Time step used, in s.
        n_steps: Number of time steps.
        save_every: Snapshot interval in steps.
        calphad_fit: Surrogate fit used to evaluate G_m and d²G/dx².
        temperature_K: Temperature in K. The closest stored temperature is used.
        dt_stable: Automatic stable time step, listed next to ``dt`` if given. The file
            labels it "safety=0.4" whatever safety factor produced it.

    Returns:
        Path to the written file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dx = Lx / N

    stored_T = np.array(calphad_fit["temperatures_K"])
    T_K = stored_T[int(np.argmin(np.abs(stored_T - temperature_K)))]
    el = list(calphad_fit["fits"][T_K]["models"].keys())[0]
    mdl = calphad_fit["fits"][T_K]["models"][el]

    x = np.linspace(0.01, 0.99, 2000)
    if "spline_dfdx" in mdl:
        gm_arr = mdl["spline_gm"](x)
        d2G_arr = mdl["spline_dfdx"].derivative()(x)
    elif "spline" in mdl:
        gm_arr = mdl["spline_antideriv"](x) if "spline_antideriv" in mdl else None
        d2G_arr = mdl["spline"].derivative()(x)
    else:
        h = 1e-5

        def mu(v):
            return np.polyval(mdl["coefficients"][::-1], v)

        d2G_arr = (mu(x + h) - 2 * mu(x) + mu(x - h)) / h**2
        gm_arr = None

    d2G_max = float(np.max(np.abs(d2G_arr)))
    d2G_c0 = float(np.abs(d2G_arr[np.argmin(d2G_arr)]))

    sigma = float("nan")
    if gm_arr is not None:
        try:
            spinodal_mask = d2G_arr < 0
            if spinodal_mask.any():
                idx_lo = int(np.argmax(spinodal_mask))
                idx_hi = int(len(spinodal_mask) - 1 - np.argmax(spinodal_mask[::-1]))
                x_lo, x_hi = x[idx_lo], x[idx_hi]
                gm_lo, gm_hi = gm_arr[idx_lo], gm_arr[idx_hi]
                tangent = gm_lo + (gm_hi - gm_lo) / (x_hi - x_lo) * (x - x_lo)
                dG_mix = gm_arr - tangent
                mask = (x >= x_lo) & (x <= x_hi) & (dG_mix >= 0)
                if mask.sum() > 2:
                    from scipy.integrate import trapezoid

                    sigma = float(trapezoid(np.sqrt(2.0 * kappa * dG_mix[mask]), x[mask]))
        except Exception:
            sigma = float("nan")

    delta = float(np.sqrt(kappa / d2G_max))
    lambda_star = 4.0 * np.pi * np.sqrt(kappa / max(d2G_c0, 1e-30))
    lambda_crit = 2.0 * np.pi * np.sqrt(2.0 * kappa / max(d2G_c0, 1e-30))
    t_total = dt * n_steps
    M_eff_max = M * 0.25
    stability_ratio = M_eff_max * d2G_max * dt / dx**2

    warn_delta = "  ⚠  δ < dx — interface UNRESOLVED on grid!" if delta < dx else ""
    warn_lambda = "  ⚠  λ* < dx — spinodal wavelength UNRESOLVED!" if lambda_star < dx else ""
    warn_kappa = f"  ⚠  Suggested κ for δ = 5·dx:  κ_min ≈ {d2G_max * (5 * dx) ** 2:.3e} J·m²/mol" if delta < dx else ""

    sigma_str = f"{sigma:.3e} J/m²" if not np.isnan(sigma) else "n/a (spinodal not found)"

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
        f"  Interface width      δ        = {delta:.3e} m  ({delta / dx:.2f} × dx){('  ' + warn_delta) if warn_delta else ''}",
        f"  Interface energy     σ        = {sigma_str}",
        f"  Most unstable λ*     λ*       = {lambda_star:.3e} m  ({lambda_star / dx:.1f} × dx)"
        f"{('  ' + warn_lambda) if warn_lambda else ''}",
        f"  Critical wavelength  λ_c      = {lambda_crit:.3e} m  ({lambda_crit / dx:.1f} × dx)",
        f"  Domain / λ*                   = {Lx / lambda_star:.1f}",
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
    with out_path.open("w") as fh:
        fh.write("\n".join(lines) + "\n")

    print(f"  Parameters written → {out_path}")
    return out_path


@dataclass
class CHResult:
    """Output of ``solve_cahn_hilliard_binary``.

    The energies are averages over the domain, i.e. energies per unit volume (J/mol for
    the CALPHAD models).

    Attributes:
        composition: Final composition field, shape ``(N, N)``.
        composition_history: Snapshots of the field at step 0, every ``save_every`` steps
            and at the last step.
        time_history: Time of each snapshot, step · dt.
        free_energy_history: Total free energy ⟨f_bulk⟩ + ⟨κ/2 |∇c|²⟩ at each snapshot.
        bulk_energy_history: Bulk part ⟨f_bulk⟩ at each snapshot.
        grad_energy_history: Gradient part ⟨κ/2 |∇c|²⟩ at each snapshot, computed in
            Fourier space with Parseval's theorem.
        mix_energy_history: Mixing energy ⟨G_m(c) − G_ref(c)⟩, where G_ref is the line
            between G_m(0) and G_m(1). Filled only for ``"calphad_gm_spline"``; empty
            otherwise.
        params: Run parameters: the model, grid and time-stepping settings, ``c0``,
            ``noise``, ``seed`` and the ``dt`` actually used, plus the double-well
            parameters or the temperature and fit metadata.
        elapsed_seconds: Wall-clock time of the time-stepping loop.
    """

    composition: np.ndarray
    composition_history: list
    time_history: list
    free_energy_history: list
    bulk_energy_history: list
    grad_energy_history: list
    mix_energy_history: list
    params: dict
    elapsed_seconds: float


def solve_cahn_hilliard_binary(
    free_energy: Literal["double_well", "calphad", "calphad_spline", "calphad_gm_spline"],
    N: int = 256,
    Lx: float = 200.0,
    kappa: float = 0.5,
    M: float = 1.0,
    c0: float = 0.4,
    noise: float = 0.01,
    seed: int = 0,
    dt: float | None = None,
    dt_scale: float = 1.0,
    n_steps: int = 2000,
    rho_s: float = 5.0,
    c_a: float = 0.3,
    c_b: float = 0.7,
    calphad_fit: dict | None = None,
    temperature_K: float | None = None,
    save_every: int = 100,
    output_dir: str | Path | None = None,
    save_frames: bool = False,
) -> CHResult:
    """Solve the binary Cahn-Hilliard equation on a 2-D periodic grid.

    The initial field is c0 plus uniform noise in [−noise/2, noise/2), clipped to [0, 1].
    The mobility is the spatial average of M c (1 − c), recomputed every step. Units are
    SI for the CALPHAD models and arbitrary for the double well.

    Args:
        free_energy: Bulk model. ``"double_well"`` needs no thermodynamic data;
            ``"calphad"`` uses the polynomial Δμ surrogate from
            ``thermodynamics.fitting.load_fits``; ``"calphad_spline"`` and
            ``"calphad_gm_spline"`` use the surrogates from ``build_spline_surrogate`` and
            ``build_gm_spline_surrogate``. With ``"calphad_gm_spline"``, c is the mole
            fraction of the reference element; with the other CALPHAD models it is the
            mole fraction of the independent element.
        N: Grid points along each axis.
        Lx: Domain edge length (m for the CALPHAD models).
        kappa: Gradient energy coefficient κ (J·m²/mol for the CALPHAD models).
        M: Mobility prefactor M₀ of M(c) = M₀ c (1 − c).
        c0: Mean initial composition.
        noise: Width of the uniform initial noise around ``c0``.
        seed: Seed of the random number generator.
        dt: Time step. ``None`` picks one automatically: ``stable_dt(..., safety=0.4)``
            times ``dt_scale`` for the CALPHAD models and ``0.1 * dt_scale`` for the double
            well. A given ``dt`` larger than ``stable_dt(..., safety=0.5)`` raises a
            ``UserWarning`` for the CALPHAD models.
        dt_scale: Factor applied to the automatic time step. Ignored when ``dt`` is given.
        n_steps: Number of time steps.
        rho_s: Energy scale of the double well.
        c_a: First well minimum of the double well.
        c_b: Second well minimum of the double well.
        calphad_fit: Surrogate fit for the CALPHAD models, matching ``free_energy``.
        temperature_K: Temperature in K, required for the CALPHAD models.
        save_every: Snapshot interval in steps. Step 0 and the last step are always kept.
        output_dir: If given, the directory is created, ``.png``, ``.txt`` and ``.npy``
            files already in it are deleted, and ``free_energy_log.txt`` and
            ``free_energy_history.png`` are written there, plus ``sim_params.txt`` for the
            CALPHAD models.
        save_frames: Also save a PNG of the field at each snapshot, if ``output_dir`` is
            set.

    Returns:
        The final field, the snapshot and energy histories, and the run parameters.

    Raises:
        ValueError: If ``free_energy`` is not one of the four models, or a CALPHAD model
            is chosen without ``calphad_fit`` or ``temperature_K``.

    Examples:
        Double well, with no thermodynamic data:

        ```python
        result = solve_cahn_hilliard_binary(
            free_energy="double_well",
            N=256,
            Lx=200,
            kappa=0.5,
            M=1.0,
            c0=0.4,
            dt=0.1,
            n_steps=2000,
        )
        ```

        Nb-V at 500 °C with a polynomial fit written by
        ``thermodynamics.fitting.save_fits``:

        ```python
        from phasefoundry.thermodynamics.fitting import load_fits

        fit = load_fits("nb_v_delta_mu_fits.h5")
        result = solve_cahn_hilliard_binary(
            free_energy="calphad",
            N=256,
            Lx=100e-6,
            kappa=40e-19,
            M=1.0,
            c0=0.4,
            dt=5e-7,
            n_steps=2000,
            calphad_fit=fit,
            temperature_K=773.15,
        )
        ```
    """
    if free_energy not in ("double_well", "calphad", "calphad_spline", "calphad_gm_spline"):
        raise ValueError(f"free_energy must be 'double_well', 'calphad', or 'calphad_spline', got '{free_energy}'.")
    if free_energy in ("calphad", "calphad_spline", "calphad_gm_spline"):
        if calphad_fit is None:
            raise ValueError(
                f"calphad_fit is required when free_energy='{free_energy}'. "
                "Load it with thermodynamics.fitting.load_fits() or "
                "thermodynamics.fitting.build_spline_surrogate()."
            )
        if temperature_K is None:
            raise ValueError(f"temperature_K (in Kelvin) is required when free_energy='{free_energy}'.")

    fit = calphad_fit if calphad_fit is not None else {}
    T_K = temperature_K if temperature_K is not None else float("nan")

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        removed = 0
        for f in output_dir.iterdir():
            if f.suffix in (".png", ".txt", ".npy") or f.name == "sim_params.txt":
                f.unlink()
                removed += 1
        if removed:
            print(f"  Cleared {removed} old file(s) from {output_dir}")

    dx = Lx / N
    dt_stable_val = None
    if dt is None:
        if free_energy in ("calphad", "calphad_spline", "calphad_gm_spline"):
            dt_stable_val = stable_dt(M=M, dx=dx, calphad_fit=fit, temperature_K=T_K, safety=0.4)
            dt = dt_stable_val * dt_scale
            if dt_scale == 1.0:
                print(f"  auto dt = {dt:.4e} s  (stability criterion, safety=0.4)")
            else:
                print(f"  auto dt = {dt_stable_val:.4e} s  ×  dt_scale={dt_scale:.3g}  →  dt = {dt:.4e} s")
        else:
            dt = 0.1 * dt_scale
            print(f"  auto dt = {dt:.4e} s  (double-well default × dt_scale={dt_scale:.3g})")
    elif free_energy in ("calphad", "calphad_spline", "calphad_gm_spline"):
        dt_stable_val = stable_dt(M=M, dx=dx, calphad_fit=fit, temperature_K=T_K, safety=0.5)
        if dt > dt_stable_val:
            import warnings

            warnings.warn(
                f"dt={dt:.3e} s exceeds the stability limit dt_max={dt_stable_val:.3e} s "
                f"({dt / dt_stable_val:.2f}×). "
                "Consider setting dt=None for automatic selection.",
                UserWarning,
                stacklevel=2,
            )

    if output_dir is not None and free_energy in ("calphad", "calphad_spline", "calphad_gm_spline"):
        write_simulation_params(
            output_dir=output_dir,
            N=N,
            Lx=Lx,
            kappa=kappa,
            M=M,
            dt=dt,
            n_steps=n_steps,
            save_every=save_every,
            calphad_fit=fit,
            temperature_K=T_K,
            dt_stable=dt_stable_val,
        )

    k2 = laplacian_operator(N, Lx)
    k4 = k2**2

    rng = np.random.default_rng(seed)
    c = c0 + noise * (rng.random((N, N)) - 0.5)
    c = np.clip(c, 0.0, 1.0)

    def mobility(c: np.ndarray) -> float:
        """Spatially averaged degenerate mobility M(c) = M₀·c·(1−c)."""
        return float(np.mean(M * c * (1.0 - c)))

    if free_energy == "double_well":

        def bulk_mu(c: np.ndarray) -> np.ndarray:
            return double_well_df_dc(c, rho_s=rho_s, c_a=c_a, c_b=c_b)

        def bulk_f(c: np.ndarray) -> float:
            return float(np.mean(double_well_f(c, rho_s=rho_s, c_a=c_a, c_b=c_b)))

        bulk_gm = None

    elif free_energy == "calphad":

        def bulk_mu(c: np.ndarray) -> np.ndarray:
            return calphad_df_dc(c, fit=fit, temperature_K=T_K)

        def bulk_f(c: np.ndarray) -> float:
            return float(np.mean(calphad_f(c, fit=fit, temperature_K=T_K)))

        bulk_gm = None

    elif free_energy == "calphad_spline":

        def bulk_mu(c: np.ndarray) -> np.ndarray:
            return calphad_df_dc_spline(c, spline_fit=fit, temperature_K=T_K)

        def bulk_f(c: np.ndarray) -> float:
            return float(np.mean(calphad_f_spline(c, spline_fit=fit, temperature_K=T_K)))

        bulk_gm = None

    else:

        def bulk_mu(c: np.ndarray) -> np.ndarray:
            return calphad_df_dc_gm_spline(c, gm_spline_fit=fit, temperature_K=T_K)

        def bulk_f(c: np.ndarray) -> float:
            return float(np.mean(calphad_f_gm_spline(c, gm_spline_fit=fit, temperature_K=T_K)))

        stored_T = np.array(fit["temperatures_K"])
        T_K_idx = int(np.argmin(np.abs(stored_T - T_K)))
        T_K_key = stored_T[T_K_idx]
        el_key = list(fit["fits"][T_K_key]["models"].keys())[0]
        spl_gm = fit["fits"][T_K_key]["models"][el_key]["spline_gm"]
        gm_at_0 = float(spl_gm(0.0))
        gm_at_1 = float(spl_gm(1.0))

        def bulk_gm(c: np.ndarray) -> np.ndarray:
            """G_m(c) evaluated on the composition field."""
            return spl_gm(c)

        def linear_ref(c: np.ndarray) -> np.ndarray:
            """Mechanical mixture line: G_m(0)*(1-c) + G_m(1)*c."""
            return gm_at_0 * (1.0 - c) + gm_at_1 * c

    composition_history = []
    time_history = []
    free_energy_history = []
    bulk_energy_history = []
    grad_energy_history = []
    mix_energy_history = []

    def _snapshot(c: np.ndarray, step: int) -> None:
        """Record the energies and a copy of ``c`` at ``step``, and save a frame if asked."""
        t_phys = step * dt
        F_bulk = bulk_f(c)

        c_hat = fft2(c)
        F_grad = float(kappa / 2.0 * np.mean((-k2) * np.abs(c_hat) ** 2) / (N**2))
        F_tot = F_bulk + F_grad

        composition_history.append(c.copy())
        time_history.append(t_phys)
        free_energy_history.append(F_tot)
        bulk_energy_history.append(F_bulk)
        grad_energy_history.append(F_grad)

        if bulk_gm is not None:
            dG = float(np.mean(bulk_gm(c) - linear_ref(c)))
            mix_energy_history.append(dG)

        if output_dir is not None and save_frames:
            _save_frame(c, step, t_phys, output_dir)

    _snapshot(c, 0)
    t_start = time.perf_counter()

    for step in tqdm(range(1, n_steps + 1), desc=f"C-H ({free_energy})", unit="step"):
        M_eff = float(np.mean(M * c * (1.0 - c)))
        denom = 1.0 + M_eff * kappa * dt * k4

        mu_bulk = bulk_mu(c)
        mu_hat = fft2(mu_bulk)
        c_hat = fft2(c)

        c_hat_new = (c_hat + M_eff * dt * k2 * mu_hat) / denom
        c = np.real(ifft2(c_hat_new))
        # no clipping: it would break mass conservation

        if step % save_every == 0 or step == n_steps:
            _snapshot(c, step)

    elapsed = time.perf_counter() - t_start

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

    params = {
        "free_energy": free_energy,
        "N": N,
        "Lx": Lx,
        "kappa": kappa,
        "M": M,
        "c0": c0,
        "noise": noise,
        "seed": seed,
        "dt": dt,
        "n_steps": n_steps,
        "save_every": save_every,
    }
    if free_energy == "double_well":
        params.update(rho_s=rho_s, c_a=c_a, c_b=c_b)
    else:
        meta = fit.get("metadata", {})
        params["temperature_K"] = T_K
        if "reference_element" in meta:
            params["reference_element"] = meta["reference_element"]
        if "x_variables" in meta:
            params["x_variables"] = meta["x_variables"]
        if free_energy in ("calphad", "calphad_spline") and "poly_degree" in meta:
            pd = meta["poly_degree"]
            params["poly_degree"] = pd if isinstance(pd, str) else int(pd)

    result = CHResult(
        composition=c,
        composition_history=composition_history,
        time_history=time_history,
        free_energy_history=free_energy_history,
        bulk_energy_history=bulk_energy_history,
        grad_energy_history=grad_energy_history,
        mix_energy_history=mix_energy_history,
        params=params,
        elapsed_seconds=elapsed,
    )

    if output_dir is not None:
        plot_free_energy_history(result, output_dir=output_dir)

    return result


def plot_result(
    result: CHResult,
    output_dir: str | Path | None = None,
    show: bool = True,
) -> None:
    """Plot the final composition field and the free energy history.

    The left panel shows the final field; the right panel shows the total, bulk and
    gradient energies against time.

    Args:
        result: Output of ``solve_cahn_hilliard_binary``.
        output_dir: If given, the figure is saved there as ``result_summary.png``.
        show: Call ``plt.show()`` before closing the figure.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    im = axes[0].imshow(
        result.composition,
        cmap="RdBu_r",
        origin="lower",
    )
    plt.colorbar(im, ax=axes[0], label="Composition  (mol fraction)")
    axes[0].set_title(
        f"Composition field  —  $t$ = {result.time_history[-1]:.3g}\n[{result.params['free_energy']} model]",
        fontsize=12,
    )
    axes[0].set_xlabel("")
    axes[0].set_ylabel("")
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    axes[0].grid(False)

    mk = {"marker": "o", "markersize": 4, "markerfacecolor": "white", "markeredgewidth": 1.2}
    axes[1].plot(result.time_history, result.free_energy_history, color="#1f77b4", linewidth=1.8, label="$F_{total}$", **mk)
    axes[1].plot(
        result.time_history, result.bulk_energy_history, color="#d62728", linewidth=1.5, linestyle="--", label="$F_{bulk}$", **mk
    )
    axes[1].plot(
        result.time_history, result.grad_energy_history, color="#2ca02c", linewidth=1.5, linestyle=":", label="$F_{grad}$", **mk
    )
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
    """Save the field at one snapshot as ``frame_<step>.png`` in ``output_dir``."""
    fig, ax = plt.subplots(figsize=(5, 5))
    vmin, vmax = c.min(), c.max()
    im = ax.imshow(c, cmap="PuOr", origin="lower", vmin=vmin, vmax=vmax)
    plt.colorbar(im, ax=ax, label="$x_V$  (mol fraction V)", shrink=0.25, aspect=10, pad=0.02)
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
    """Write the energy history to ``free_energy_log.txt`` in ``output_dir``.

    Each row holds the step, time, total, bulk and gradient energies, the minimum and
    maximum of the field, and ΔG_mix when it was recorded. The step is written as
    ``i * save_every`` for the i-th snapshot.

    Returns:
        Path to the log file.
    """
    path = output_dir / "free_energy_log.txt"
    has_mix = len(mix_energy_history) > 0
    header = f"{'step':>10}  {'time_s':>14}  {'F_total':>16}  {'F_bulk':>16}  {'F_grad':>16}  {'c_min':>10}  {'c_max':>10}"
    if has_mix:
        header += f"  {'dG_mix':>16}"
    with path.open("w") as fh:
        fh.write(header + "\n")
        for i, (t, F, Fb, Fg, c) in enumerate(
            zip(
                time_history,
                free_energy_history,
                bulk_energy_history,
                grad_energy_history,
                composition_history,
                strict=True,
            )
        ):
            step = i * save_every
            row = f"{step:>10d}  {t:>14.6e}  {F:>16.6e}  {Fb:>16.6e}  {Fg:>16.6e}  {c.min():>10.5f}  {c.max():>10.5f}"
            if has_mix:
                row += f"  {mix_energy_history[i]:>16.6e}"
            fh.write(row + "\n")
    return path


def plot_free_energy_history(
    result,
    output_dir: str | Path | None = None,
    show: bool = False,
) -> None:
    """Plot the energy contributions and the composition range against time.

    The left panel shows |F_total|, |F_bulk|, |F_grad| and, when recorded, |ΔG_mix| on a
    log scale. The right panel shows the minimum and maximum of the field at each
    snapshot, labelled as x_V for the Nb-V system.

    Args:
        result: Output of ``solve_cahn_hilliard_binary``.
        output_dir: If given, the figure is saved there as ``free_energy_history.png``.
        show: Call ``plt.show()`` before closing the figure.
    """
    t = result.time_history
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    F_tot = np.abs(result.free_energy_history)
    F_bulk = np.abs(result.bulk_energy_history)
    F_grad = np.abs(result.grad_energy_history)
    ax.plot(
        t,
        F_tot,
        color="#1f77b4",
        lw=2,
        marker="o",
        markersize=5,
        markerfacecolor="white",
        markeredgewidth=1.5,
        label="|$F_{total}$|",
    )
    ax.plot(
        t,
        F_bulk,
        color="#d62728",
        lw=1.8,
        ls="--",
        marker="s",
        markersize=5,
        markerfacecolor="white",
        markeredgewidth=1.5,
        label="|$F_{bulk}$|",
    )
    ax.plot(
        t,
        F_grad,
        color="#2ca02c",
        lw=1.8,
        ls=":",
        marker="^",
        markersize=6,
        markerfacecolor="white",
        markeredgewidth=1.5,
        label="|$F_{grad}$|  (interfacial)",
    )
    if result.mix_energy_history:
        dG = np.array(result.mix_energy_history)
        ax.plot(
            t,
            np.abs(dG),
            color="#9467bd",
            lw=1.8,
            ls="-.",
            marker="D",
            markersize=5,
            markerfacecolor="white",
            markeredgewidth=1.5,
            label="|$\\Delta G_{mix}$|",
        )
    ax.set_yscale("log")
    ax.set_xlabel("Time  (s)", fontsize=12)
    ax.set_ylabel("|Free energy|  (J/mol)  [log scale]", fontsize=12)
    ax.set_title("Free energy contributions", fontsize=12)
    ax.legend(fontsize=11, frameon=False)
    ax.grid(False)

    ax2 = axes[1]
    if result.composition_history:
        c_min = [c.min() for c in result.composition_history]
        c_max = [c.max() for c in result.composition_history]
        ax2.plot(
            t,
            c_min,
            color="#d62728",
            lw=1.8,
            marker="s",
            markersize=5,
            markerfacecolor="white",
            markeredgewidth=1.5,
            label="$x_V^{min}$  (Nb-rich)",
        )
        ax2.plot(
            t,
            c_max,
            color="#1f77b4",
            lw=1.8,
            marker="o",
            markersize=5,
            markerfacecolor="white",
            markeredgewidth=1.5,
            label="$x_V^{max}$  (V-rich)",
        )
        ax2.set_xlabel("Time  (s)", fontsize=12)
        ax2.set_ylabel("$x_V$  (mol fraction V)", fontsize=12)
        ax2.set_title("Phase composition evolution", fontsize=12)
        ax2.legend(fontsize=11, frameon=False)
    else:
        ax2.text(
            0.5,
            0.5,
            "Composition history\nnot available",
            ha="center",
            va="center",
            transform=ax2.transAxes,
            fontsize=12,
            color="gray",
        )
        ax2.set_title("Phase composition evolution", fontsize=12)
    ax2.grid(False)

    plt.tight_layout()

    if output_dir is not None:
        p = Path(output_dir) / "free_energy_history.png"
        plt.savefig(p, dpi=150, bbox_inches="tight")
        print(f"  Saved → {p}")
    if show:
        plt.show()
    plt.close()


def plot_gm_with_pfm(
    result: CHResult,
    calphad_fit: dict,
    temperature_K: float,
    output_dir: str | Path | None = None,
    show: bool = False,
) -> None:
    """Plot the G_m curve next to the final phase-field composition field.

    The left panel shows G_m(x) from the CALPHAD spline with the line between G_m(0) and
    G_m(1), shades where ΔG_mix < 0 and where d²G/dx² < 0, and marks the mean initial
    composition and the minimum and maximum of the final field. The right panel shows the
    final field. Labels and title assume the Nb-V BCC system.

    Args:
        result: Output of ``solve_cahn_hilliard_binary``.
        calphad_fit: Output of ``build_gm_spline_surrogate``, with ``spline_gm`` and
            ``spline_dfdx``.
        temperature_K: Temperature in K, used to pick the fit and label the plot.
        output_dir: If given, the figure is saved there as ``gm_pfm_combined.png``.
        show: Call ``plt.show()`` before closing the figure.

    Raises:
        ValueError: If the fit at that temperature has no ``spline_gm`` or ``spline_dfdx``.
    """
    stored_T = np.array(calphad_fit["temperatures_K"])
    T_K = stored_T[int(np.argmin(np.abs(stored_T - temperature_K)))]
    el_key = list(calphad_fit["fits"][T_K]["models"].keys())[0]
    mdl = calphad_fit["fits"][T_K]["models"][el_key]

    if "spline_gm" not in mdl or "spline_dfdx" not in mdl:
        raise ValueError("plot_gm_with_pfm() requires calphad_gm_spline surrogate (spline_gm and spline_dfdx keys required).")

    spl_gm = mdl["spline_gm"]
    spl_dfdx = mdl["spline_dfdx"]

    x = np.linspace(0.001, 0.999, 600)
    gm = spl_gm(x)
    d2G = spl_dfdx.derivative()(x)

    gm_0 = float(spl_gm(0.0))
    gm_1 = float(spl_gm(1.0))
    linear_ref = gm_0 * (1.0 - x) + gm_1 * x

    c_final = result.composition
    c0_mean = result.composition_history[0].mean()
    c_min = float(c_final.min())
    c_max = float(c_final.max())
    t_final = result.time_history[-1]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    ax.plot(x, gm, color="#1f77b4", lw=2.2, label="$G_m(x)$  CALPHAD spline")
    ax.plot(x, linear_ref, color="gray", lw=1.2, ls="--", label="Mechanical mixture (ref)")

    dGmix = gm - linear_ref
    ax.fill_between(
        x, gm, linear_ref, where=(dGmix < 0), alpha=0.15, color="#d62728", label="$\\Delta G_{mix} < 0$  (driving force)"
    )

    spinodal_mask = d2G < 0
    if spinodal_mask.any():
        ax.fill_between(
            x, gm.min() - 500, gm, where=spinodal_mask, alpha=0.08, color="#9467bd", label="Spinodal region ($d^2G/dx^2 < 0$)"
        )

    ax.axvline(c0_mean, color="#2ca02c", lw=1.5, ls=":", label=f"$c_0$ = {c0_mean:.3f}")
    ax.axvline(c_min, color="#d62728", lw=1.5, ls="-.", label=f"$c_{{min}}$ = {c_min:.3f}  (Nb-rich)")
    ax.axvline(c_max, color="#1f77b4", lw=1.5, ls="-.", label=f"$c_{{max}}$ = {c_max:.3f}  (V-rich)")

    ax.set_xlabel("$x_V$  (mol fraction V)", fontsize=12)
    ax.set_ylabel("$G_m$  (J/mol)", fontsize=12)
    ax.set_title(
        f"$G_m$ landscape — {temperature_K - 273.15:.0f} °C\nNb-V BCC",
        fontsize=11,
    )
    ax.legend(fontsize=8, frameon=True, loc="lower right")
    ax.grid(False)

    ax2 = axes[1]
    im = ax2.imshow(
        c_final,
        cmap="RdBu_r",
        origin="lower",
        vmin=c_min,
        vmax=c_max,
    )
    plt.colorbar(im, ax=ax2, label="$x_V$  (mol fraction V)", shrink=0.85)
    ax2.set_title(
        f"PFM composition field\n$t$ = {t_final:.3g} s  ($c_0$ = {c0_mean:.2f})",
        fontsize=11,
    )
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.grid(False)

    plt.tight_layout()

    if output_dir is not None:
        p = Path(output_dir) / "gm_pfm_combined.png"
        plt.savefig(p, dpi=150, bbox_inches="tight")
        print(f"  Saved → {p}")
    if show:
        plt.show()
    plt.close()
