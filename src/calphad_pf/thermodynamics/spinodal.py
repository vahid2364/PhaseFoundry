"""
Spinodal decomposition boundary calculation.

A composition is inside the spinodal region when the Gibbs free energy
is concave with respect to composition, i.e. d²G/dx² < 0.

For a binary A-B system (B = reference element):
    d²G/dx_A² = ∂μ_A/∂x_A  (Jansson derivative, stored as dMU/A_A)

For a multicomponent system the spinodal condition generalises to the
eigenvalues of the Hessian matrix H_ij = ∂μ_i/∂x_j.  At least one
negative eigenvalue means the composition is inside the spinodal.

This module provides:
    spinodal_boundary_binary()   — binary: find zero-crossings of d²G/dx²
    spinodal_boundary_multi()    — multicomponent: eigenvalue analysis
    plot_spinodal()              — composition-temperature phase diagram
"""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import fsolve

from .storage import load_table


# ── binodal (common tangent) ──────────────────────────────────────────────────

def binodal_boundary_binary(
    table_paths: list[str | Path],
    independent_element: str,
    temperatures_celsius: list[float] | None = None,
    residual_tol: float = 500.0,
    min_separation: float = 0.15,
) -> dict:
    """
    Compute binodal (equilibrium / solvus) boundaries for a binary system.

    Uses the equal chemical potential condition — two phases α and β are in
    equilibrium when:

        μ_A(x_α) = μ_A(x_β)   and   μ_B(x_α) = μ_B(x_β)

    Solved numerically via ``scipy.optimize.fsolve`` with cubic-spline
    interpolants of the pycalphad chemical potential data.  Multiple initial
    guesses bracket the solution from both sides of the two-phase region.

    Parameters
    ----------
    table_paths : list of str or Path
        HDF5 tables (one per temperature) produced by generate_table().
        Must contain 'MU' datasets.
    independent_element : str
        e.g. 'NB'.  The complementary element is inferred from 'MU' keys.
    temperatures_celsius : list of float, optional
    residual_tol : float
        Maximum allowed |residual| in J/mol for a solution to be accepted.
    min_separation : float
        Minimum composition gap between the two equilibrium phases.

    Returns
    -------
    dict with keys:
        'temperatures_celsius'   list[float]
        'x_binodal_low'          list[float | None]
        'x_binodal_high'         list[float | None]
    """
    # Initial guesses: pairs (x_lo, x_hi) spanning the two-phase region
    _INIT_GUESSES = [
        (0.05, 0.95), (0.10, 0.90), (0.15, 0.85),
        (0.05, 0.85), (0.10, 0.95), (0.20, 0.80),
        (0.03, 0.97), (0.07, 0.93),
    ]

    temps, x_lo, x_hi = [], [], []

    for i, path in enumerate(table_paths):
        data = load_table(path)
        elements = data["elements"]

        if "MU" not in data:
            raise KeyError(
                f"MU not found in {path}. "
                "Re-generate with quantities=(..., 'MU')."
            )

        T_C = (
            temperatures_celsius[i]
            if temperatures_celsius is not None
            else data["metadata"].get("temperature", 273.15) - 273.15
        )
        temps.append(T_C)

        col    = elements.index(independent_element)
        x      = data["compositions"][:, col]
        mu_A   = data["MU"][independent_element]
        ref_el = [e for e in elements if e != independent_element][0]
        mu_B   = data["MU"][ref_el]

        order = np.argsort(x)
        x, mu_A, mu_B = x[order], mu_A[order], mu_B[order]

        # Cubic-spline interpolants for smooth root-finding
        f_A = interp1d(x, mu_A, kind="cubic", fill_value="extrapolate")
        f_B = interp1d(x, mu_B, kind="cubic", fill_value="extrapolate")

        def residuals(xy):
            xa, xb = xy
            return [f_A(xa) - f_A(xb), f_B(xa) - f_B(xb)]

        found_lo = found_hi = None

        for x0_lo, x0_hi in _INIT_GUESSES:
            try:
                sol, _, ier, _ = fsolve(residuals, [x0_lo, x0_hi], full_output=True)
                xa, xb = float(sol[0]), float(sol[1])
                res = residuals([xa, xb])
                resid = (res[0] ** 2 + res[1] ** 2) ** 0.5

                if (
                    ier == 1                        # solver converged
                    and resid < residual_tol        # accurate equilibrium
                    and xa < xb                     # ordered pair
                    and xb - xa >= min_separation   # two distinct phases
                    and 0.001 < xa < 0.999          # inside composition range
                    and 0.001 < xb < 0.999
                ):
                    found_lo = xa
                    found_hi = xb
                    break
            except Exception:
                continue

        x_lo.append(found_lo)
        x_hi.append(found_hi)

    return {
        "temperatures_celsius": temps,
        "x_binodal_low":        x_lo,
        "x_binodal_high":       x_hi,
    }


# ── binary ────────────────────────────────────────────────────────────────────

def spinodal_boundary_binary(
    table_paths: list[str | Path],
    independent_element: str,
    temperatures_celsius: list[float] | None = None,
) -> dict:
    """
    Compute spinodal boundaries for a binary system at multiple temperatures.

    For each temperature, finds the two compositions (low, high) where
    d²G/dx² = 0 — the spinodal limits.  Compositions between them are
    unstable to spinodal decomposition.

    Parameters
    ----------
    table_paths : list of str or Path
        HDF5 tables (one per temperature) produced by generate_table().
        Must contain the 'dMU' dataset.
    independent_element : str
        The element whose Jansson derivative ∂μ/∂x is used,
        e.g. 'NB' for the Nb-V binary.
    temperatures_celsius : list of float, optional
        Temperature labels.  If None, read from file metadata.

    Returns
    -------
    dict with keys:
        'temperatures_celsius'   list[float]          length N_T
        'x_spinodal_low'         list[float | None]   lower boundary at each T
        'x_spinodal_high'        list[float | None]   upper boundary at each T
        'x_compositions'         list[np.ndarray]     composition arrays
        'd2G_dx2'                list[np.ndarray]     d²G/dx² arrays
    """
    key = f"{independent_element}_{independent_element}"

    temps, x_lo, x_hi, all_x, all_d2g = [], [], [], [], []

    for i, path in enumerate(table_paths):
        data = load_table(path)
        elements = data["elements"]

        if "dMU" not in data or key not in data["dMU"]:
            raise KeyError(
                f"dMU/{key} not found in {path}. "
                f"Re-generate the table with quantities=(..., 'dMU')."
            )

        col = elements.index(independent_element)
        x = data["compositions"][:, col]
        d2g = data["dMU"][key]          # ∂μ_i/∂x_i = d²G/dx²

        # Sort by composition for clean interpolation
        order = np.argsort(x)
        x, d2g = x[order], d2g[order]

        T_C = (
            temperatures_celsius[i]
            if temperatures_celsius is not None
            else data["metadata"].get("temperature", 273.15) - 273.15
        )
        temps.append(T_C)
        all_x.append(x)
        all_d2g.append(d2g)

        # Find zero-crossings (sign changes) of d²G/dx²
        sign = np.sign(d2g)
        crossings = np.where(np.diff(sign) != 0)[0]

        boundaries = []
        for idx in crossings:
            # Linear interpolation between idx and idx+1
            x0, x1 = x[idx], x[idx + 1]
            f0, f1 = d2g[idx], d2g[idx + 1]
            if f1 != f0:
                x_zero = x0 - f0 * (x1 - x0) / (f1 - f0)
                boundaries.append(float(x_zero))

        if len(boundaries) >= 2:
            x_lo.append(min(boundaries))
            x_hi.append(max(boundaries))
        elif len(boundaries) == 1:
            x_lo.append(boundaries[0])
            x_hi.append(None)
            warnings.warn(f"Only one spinodal boundary found at {T_C} °C.")
        else:
            x_lo.append(None)
            x_hi.append(None)

    return {
        "temperatures_celsius": temps,
        "x_spinodal_low":       x_lo,
        "x_spinodal_high":      x_hi,
        "x_compositions":       all_x,
        "d2G_dx2":              all_d2g,
    }


# ── multicomponent ────────────────────────────────────────────────────────────

def spinodal_boundary_multi(
    table_paths: list[str | Path],
    independent_elements: list[str],
    temperatures_celsius: list[float] | None = None,
) -> dict:
    """
    Compute spinodal stability for a multicomponent system via eigenvalues
    of the Gibbs Hessian H_ij = ∂μ_i/∂x_j.

    A point is inside the spinodal when the smallest eigenvalue of H < 0.

    Parameters
    ----------
    table_paths : list of str or Path
    independent_elements : list[str]
        The N-1 independent elements (all except the reference).
    temperatures_celsius : list[float], optional

    Returns
    -------
    dict with keys:
        'temperatures_celsius'   list[float]
        'compositions'           list[np.ndarray]  shape (n_pts, N_elements)
        'min_eigenvalue'         list[np.ndarray]  shape (n_pts,)
        'is_spinodal'            list[np.ndarray]  bool, shape (n_pts,)
    """
    n = len(independent_elements)
    results = {
        "temperatures_celsius": [],
        "compositions":         [],
        "min_eigenvalue":       [],
        "is_spinodal":          [],
    }

    for i, path in enumerate(table_paths):
        data = load_table(path)

        T_C = (
            temperatures_celsius[i]
            if temperatures_celsius is not None
            else data["metadata"].get("temperature", 273.15) - 273.15
        )
        results["temperatures_celsius"].append(T_C)
        results["compositions"].append(data["compositions"])

        n_pts = len(data["compositions"])
        min_eigs = np.full(n_pts, np.nan)

        for pt in range(n_pts):
            H = np.zeros((n, n))
            valid = True
            for ri, el1 in enumerate(independent_elements):
                for ci, el2 in enumerate(independent_elements):
                    key = f"{el1}_{el2}"
                    if "dMU" not in data or key not in data["dMU"]:
                        valid = False
                        break
                    H[ri, ci] = data["dMU"][key][pt]
                if not valid:
                    break

            if valid and not np.any(np.isnan(H)):
                eigvals = np.linalg.eigvalsh(H)
                min_eigs[pt] = eigvals.min()

        results["min_eigenvalue"].append(min_eigs)
        results["is_spinodal"].append(min_eigs < 0)

    return results


# ── plotting ──────────────────────────────────────────────────────────────────

def plot_spinodal(
    spinodal_data: dict,
    x_element: str,
    ax: plt.Axes | None = None,
    figsize: tuple[float, float] = (7, 5),
    fill_color: str = "#d62728",
    fill_alpha: float = 0.15,
    save_path: str | Path | None = None,
    show: bool = True,
) -> plt.Axes:
    """
    Plot the spinodal boundary in composition-temperature space.

    Works with output from either ``spinodal_boundary_binary()`` or
    ``spinodal_boundary_multi()``.

    Parameters
    ----------
    spinodal_data : dict
        Output of spinodal_boundary_binary() or spinodal_boundary_multi().
    x_element : str
        Element label for the x-axis.
    ax : plt.Axes, optional
    figsize : tuple
    fill_color : str
        Color for the spinodal region fill.
    fill_alpha : float
    save_path : str or Path, optional
    show : bool

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    temps = np.array(spinodal_data["temperatures_celsius"])

    # ── binary path ───────────────────────────────────────────────────────
    if "x_spinodal_low" in spinodal_data:
        x_lo = spinodal_data["x_spinodal_low"]
        x_hi = spinodal_data["x_spinodal_high"]

        # Filter temperatures where both boundaries exist
        valid = [(T, lo, hi) for T, lo, hi in zip(temps, x_lo, x_hi)
                 if lo is not None and hi is not None]

        if not valid:
            warnings.warn("No spinodal region found across the given temperatures.")
            return ax

        T_v, lo_v, hi_v = zip(*valid)
        T_v  = np.array(T_v)
        lo_v = np.array(lo_v)
        hi_v = np.array(hi_v)

        # Shaded spinodal region
        ax.fill_betweenx(T_v, lo_v, hi_v, color=fill_color, alpha=fill_alpha,
                         label="Spinodal region")
        # Boundary lines
        ax.plot(lo_v, T_v, color=fill_color, linewidth=2, label="Spinodal boundary")
        ax.plot(hi_v, T_v, color=fill_color, linewidth=2)

        # Mark individual data points
        ax.scatter(lo_v, T_v, color=fill_color, s=30, zorder=4)
        ax.scatter(hi_v, T_v, color=fill_color, s=30, zorder=4)

    # ── multicomponent path (project onto one element) ────────────────────
    elif "min_eigenvalue" in spinodal_data:
        elements = None  # extracted from compositions shape implicitly
        for T, comps, is_sp in zip(
            spinodal_data["temperatures_celsius"],
            spinodal_data["compositions"],
            spinodal_data["is_spinodal"],
        ):
            sp_comps = comps[is_sp]
            if len(sp_comps) > 0:
                ax.scatter(
                    sp_comps[:, 0],   # first element — caller should pass correct x_element
                    np.full(len(sp_comps), T),
                    color=fill_color, alpha=0.4, s=15,
                )

    ax.set_xlabel(f"$x_{{\\rm {x_element.title()}}}$  (mol fraction)", fontsize=12)
    ax.set_ylabel("Temperature  (°C)", fontsize=12)
    ax.set_title(f"Spinodal region — Nb-V  BCC_A2", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(False)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved figure to {save_path}")

    if show:
        plt.show()

    return ax


def plot_phase_diagram(
    spinodal_data: dict,
    binodal_data: dict,
    x_element: str,
    system_name: str = "Nb-V  BCC_A2",
    ax: plt.Axes | None = None,
    figsize: tuple[float, float] = (7, 5),
    save_path: str | Path | None = None,
    show: bool = True,
) -> plt.Axes:
    """
    Plot binodal and spinodal boundaries together on a T–x phase diagram.

    The binodal (solid lines, blue) marks the two-phase region boundaries
    where the phases are in chemical equilibrium (common tangent).

    The spinodal (dashed lines, red) marks the limit of metastability inside
    the two-phase region; compositions inside the spinodal decompose
    spontaneously without a nucleation barrier.

    Parameters
    ----------
    spinodal_data : dict
        Output of spinodal_boundary_binary() (after remapping to x_element
        space, as done in step2_spinodal of the pipeline).
    binodal_data : dict
        Output of binodal_boundary_binary() (also in x_element space).
    x_element : str
        Element label for the x-axis (e.g. 'V').
    system_name : str
        Title suffix.
    ax : plt.Axes, optional
    figsize : tuple
    save_path : str or Path, optional
    show : bool

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    # ── binodal ───────────────────────────────────────────────────────────
    temps_b = np.array(binodal_data["temperatures_celsius"])
    b_lo    = binodal_data["x_binodal_low"]
    b_hi    = binodal_data["x_binodal_high"]

    valid_b = [(T, lo, hi) for T, lo, hi in zip(temps_b, b_lo, b_hi)
               if lo is not None and hi is not None]
    if valid_b:
        T_b, lo_b, hi_b = zip(*valid_b)
        T_b  = np.array(T_b)
        lo_b = np.array(lo_b)
        hi_b = np.array(hi_b)

        ax.fill_betweenx(T_b, lo_b, hi_b, color="#1f77b4", alpha=0.08,
                         label="Two-phase region (binodal)")
        ax.plot(lo_b, T_b, color="#1f77b4", linewidth=2.2, label="Binodal")
        ax.plot(hi_b, T_b, color="#1f77b4", linewidth=2.2)
        ax.scatter(lo_b, T_b, color="#1f77b4", s=30, zorder=5)
        ax.scatter(hi_b, T_b, color="#1f77b4", s=30, zorder=5)

    # ── spinodal ──────────────────────────────────────────────────────────
    temps_s = np.array(spinodal_data["temperatures_celsius"])
    s_lo    = spinodal_data["x_spinodal_low"]
    s_hi    = spinodal_data["x_spinodal_high"]

    valid_s = [(T, lo, hi) for T, lo, hi in zip(temps_s, s_lo, s_hi)
               if lo is not None and hi is not None]
    if valid_s:
        T_s, lo_s, hi_s = zip(*valid_s)
        T_s  = np.array(T_s)
        lo_s = np.array(lo_s)
        hi_s = np.array(hi_s)

        ax.fill_betweenx(T_s, lo_s, hi_s, color="#d62728", alpha=0.12,
                         label="Spinodal region")
        ax.plot(lo_s, T_s, color="#d62728", linewidth=2.0,
                linestyle="--", label="Spinodal")
        ax.plot(hi_s, T_s, color="#d62728", linewidth=2.0, linestyle="--")
        ax.scatter(lo_s, T_s, color="#d62728", s=30, zorder=5)
        ax.scatter(hi_s, T_s, color="#d62728", s=30, zorder=5)

    ax.set_xlabel(f"$x_{{\\rm {x_element.title()}}}$  (mol fraction)", fontsize=12)
    ax.set_ylabel("Temperature  (°C)", fontsize=12)
    ax.set_title(f"Phase diagram — {system_name}", fontsize=13)
    ax.set_xlim(0, 1)
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(False)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved figure to {save_path}")

    if show:
        plt.show()

    return ax
