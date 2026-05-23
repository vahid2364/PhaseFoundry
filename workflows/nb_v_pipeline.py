"""
Nb-V Binary Pipeline
====================
Steps
-----
1. Generate thermodynamic tables  (pycalphad → HDF5)
2. Compute spinodal boundaries     (d²G/dx² = 0)
3. Fit chemical potential surrogates (polynomial Δμ = μ_NB - μ_V)
4. Plot                             (free energy, spinodal, fit quality)

Future steps to add here
------------------------
5. Phase-field solver  (Cahn-Hilliard using surrogate Δμ)  ✓
6. Elasticity solver   (misfit strain coupling)
"""

import sys
import warnings
from pathlib import Path

# ── environment setup ─────────────────────────────────────────────────────────
import pycalphad                          # must be first — caches installed version
_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from calphad_pf.thermodynamics import (
    generate_table,
    load_table,
    fit_delta_mu,
    save_fits,
    load_fits,
)
from calphad_pf.thermodynamics.spinodal import (
    spinodal_boundary_binary, plot_spinodal,
    binodal_boundary_binary, plot_phase_diagram,
)
from calphad_pf.thermodynamics.plot import plot_free_energy_vs_composition

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG  — change only this block
# ═══════════════════════════════════════════════════════════════════════════════
CFG = dict(
    db_path        = str(_ROOT / "PhaseFoundry" / "HF_NB_TI_V.TDB"),
    elements       = ["NB", "V"],
    phase          = "BCC_A2",
    comp_ranges    = {"NB": (0.001, 0.999)},
    ref_element    = "V",
    n_points       = 200,                      # composition grid resolution
    temperatures_C = list(range(100, 1001, 100)),  # 100 … 1000 °C
    poly_degree    = 8,                        # surrogate polynomial degree
    out_dir        = _ROOT / "pipelines" / "output" / "nb_v",
)
CFG["temperatures_K"] = [T + 273.15 for T in CFG["temperatures_C"]]
CFG["out_dir"].mkdir(parents=True, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Thermodynamic tables
# ═══════════════════════════════════════════════════════════════════════════════
def step1_generate():
    print("\n── Step 1: Generate thermodynamic tables ──")
    paths = []
    for T_C, T_K in zip(CFG["temperatures_C"], CFG["temperatures_K"]):
        out = CFG["out_dir"] / f"nb_v_{T_C}C.h5"
        if out.exists():
            print(f"  {T_C} °C  →  exists, skipping")
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                generate_table(
                    db_path          = CFG["db_path"],
                    elements         = CFG["elements"],
                    phase            = CFG["phase"],
                    composition_ranges = CFG["comp_ranges"],
                    reference_element  = CFG["ref_element"],
                    n_points         = CFG["n_points"],
                    temperature      = T_K,
                    output_path      = str(out),
                    quantities       = ("GM", "MU", "dMU"),
                )
            print(f"  {T_C} °C  →  saved {out.name}")
        paths.append(out)
    return paths


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Spinodal boundaries
# ═══════════════════════════════════════════════════════════════════════════════
def step2_spinodal(paths):
    print("\n── Step 2: Spinodal + Binodal boundaries ──")

    # ── spinodal (d²G/dx² = 0) ────────────────────────────────────────────
    raw_sp = spinodal_boundary_binary(
        paths,
        independent_element  = "NB",
        temperatures_celsius = CFG["temperatures_C"],
    )
    # Convert to x_V space (x_V = 1 − x_NB)
    sp = dict(raw_sp)
    sp["x_spinodal_low"]  = [1-h if h is not None else None for h in raw_sp["x_spinodal_high"]]
    sp["x_spinodal_high"] = [1-l if l is not None else None for l in raw_sp["x_spinodal_low"]]

    print(f"  {'T (°C)':>8}  {'x_V sp_low':>12}  {'x_V sp_high':>12}")
    for T, lo, hi in zip(sp["temperatures_celsius"], sp["x_spinodal_low"], sp["x_spinodal_high"]):
        lo_s = f"{lo:.4f}" if lo is not None else "      none"
        hi_s = f"{hi:.4f}" if hi is not None else "      none"
        print(f"  {T:>8.0f}  {lo_s:>12}  {hi_s:>12}")

    # ── binodal (equal chemical potentials) ───────────────────────────────
    raw_bn = binodal_boundary_binary(
        paths,
        independent_element  = "NB",
        temperatures_celsius = CFG["temperatures_C"],
    )
    # Convert to x_V space
    bn = dict(raw_bn)
    bn["x_binodal_low"]  = [1-h if h is not None else None for h in raw_bn["x_binodal_high"]]
    bn["x_binodal_high"] = [1-l if l is not None else None for l in raw_bn["x_binodal_low"]]

    print(f"\n  {'T (°C)':>8}  {'x_V bn_low':>12}  {'x_V bn_high':>12}")
    for T, lo, hi in zip(bn["temperatures_celsius"], bn["x_binodal_low"], bn["x_binodal_high"]):
        lo_s = f"{lo:.4f}" if lo is not None else "      none"
        hi_s = f"{hi:.4f}" if hi is not None else "      none"
        print(f"  {T:>8.0f}  {lo_s:>12}  {hi_s:>12}")

    # ── plots ─────────────────────────────────────────────────────────────
    # Spinodal only (legacy)
    plot_spinodal(
        sp, x_element="V",
        save_path=CFG["out_dir"] / "spinodal.png",
        show=False,
    )
    # Combined binodal + spinodal phase diagram
    plot_phase_diagram(
        spinodal_data = sp,
        binodal_data  = bn,
        x_element     = "V",
        system_name   = "Nb-V  BCC_A2",
        save_path     = CFG["out_dir"] / "phase_diagram.png",
        show          = False,
    )
    print(f"  Phase diagram saved → {CFG['out_dir'] / 'phase_diagram.png'}")

    return sp, bn


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3 — Polynomial surrogate for Δμ
# ═══════════════════════════════════════════════════════════════════════════════
def step3_fit(paths):
    print(f"\n── Step 3: Fit Δμ surrogate (degree {CFG['poly_degree']}) ──")
    fits_path = CFG["out_dir"] / "delta_mu_fits.h5"

    results = fit_delta_mu(
        paths,
        reference_element = CFG["ref_element"],
        poly_degree       = CFG["poly_degree"],
        temperatures_K    = CFG["temperatures_K"],
    )
    save_fits(results, fits_path, source_db=CFG["db_path"])
    return load_fits(fits_path)


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 4 — Plots
# ═══════════════════════════════════════════════════════════════════════════════
def step4_plot(paths, fits):
    print("\n── Step 4: Plots ──")

    # Free energy — all temperatures
    plot_free_energy_vs_composition(
        paths, x_element="V", quantity="GM",
        temperatures_celsius=CFG["temperatures_C"],
        save_path=CFG["out_dir"] / "GM_all.png",
        show=False,
    )

    # Free energy — 500 °C standalone
    idx500 = CFG["temperatures_C"].index(500)
    plot_free_energy_vs_composition(
        [paths[idx500]], x_element="V", quantity="GM",
        temperatures_celsius=[500],
        save_path=CFG["out_dir"] / "GM_500C.png",
        show=False,
    )

    # Fit quality — R² and RMSE vs temperature
    import matplotlib.pyplot as plt
    temps  = sorted(fits["fits"].keys())
    rmse   = [fits["fits"][T]["models"]["NB"]["rmse"] for T in temps]
    r2     = [fits["fits"][T]["models"]["NB"]["r2"]   for T in temps]
    temps_C = [T - 273.15 for T in temps]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(temps_C, rmse, "o-", color="#1f77b4")
    ax1.set_xlabel("Temperature (°C)")
    ax1.set_ylabel("RMSE  (J/mol)")
    ax1.set_title(f"Surrogate fit quality — degree {CFG['poly_degree']}")
    ax1.grid(False)

    ax2.plot(temps_C, r2, "o-", color="#2ca02c")
    ax2.set_xlabel("Temperature (°C)")
    ax2.set_ylabel("R²")
    ax2.set_title(f"Surrogate R²")
    ax2.set_ylim(0.8, 1.0)
    ax2.grid(False)

    plt.tight_layout()
    plt.savefig(CFG["out_dir"] / "fit_quality.png", dpi=150, bbox_inches="tight")
    plt.close()

    # d²G/dx² vs temperature at fixed x_V = 0.3
    import h5py
    import numpy as np
    x_V_target = 0.3
    d2G_vals = []
    for p in paths:
        with h5py.File(p, "r") as f:
            comps  = f["compositions"][:]          # (n_pts, 2): [NB, V]
            x_V    = comps[:, 1]                   # V column
            dmu_nb = f["dMU/NB_NB"][:]             # ∂²G/∂x²_NB = ∂μ_NB/∂x_NB
        idx = int(np.argmin(np.abs(x_V - x_V_target)))
        d2G_vals.append(dmu_nb[idx])

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(CFG["temperatures_C"], d2G_vals, "o-", color="#d62728", linewidth=1.8)
    ax.axhline(0, color="black", linewidth=1.2, linestyle="--")
    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel(r"$\partial^2 G / \partial x_{\rm V}^2$  (J/mol)")
    ax.set_title(r"Second derivative of $G$ at $x_{\rm V} = 0.3$")
    ax.grid(False)
    plt.tight_layout()
    plt.savefig(CFG["out_dir"] / "d2G_vs_T_xV03.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"  Figures saved to {CFG['out_dir']}")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 5 — Phase-field simulation
# ═══════════════════════════════════════════════════════════════════════════════
def step5_phase_field(fits):
    print("\n── Step 5: Phase-field simulation ──")
    from calphad_pf.phasefield.binary import solve_cahn_hilliard_binary, plot_result

    # ── (a) double-well benchmark (dimensionless) ─────────────────────────
    # Synthetic double-well — all quantities in arbitrary units.
    # Parameters chosen to produce well-resolved spinodal decomposition
    # in ~2000 steps for benchmarking purposes.
    print("  (a) Double-well benchmark ...")
    r_dw = solve_cahn_hilliard_binary(
        free_energy = "double_well",
        N      = 256,        # grid points
        Lx     = 200.0,      # domain length  [a.u.]
        kappa  = 0.1,        # gradient energy coefficient  [a.u.]  ← notebook value
        M      = 1.0,        # mobility  [a.u.]
        c0     = 0.4,        # mean composition
        noise  = 0.01,       # initial perturbation amplitude
        seed   = 0,
        dt     = 0.1,        # time step  [a.u.]
        n_steps    = 2000,
        save_every = 200,
        save_frames = True,
        rho_s  = 5.0,        # double-well barrier height
        c_a    = 0.3,        # well minimum (phase α)
        c_b    = 0.7,        # well minimum (phase β)
        output_dir = CFG["out_dir"] / "pf_double_well",
    )
    plot_result(r_dw, output_dir=CFG["out_dir"] / "pf_double_well", show=False)
    print(f"     elapsed: {r_dw.elapsed_seconds:.1f} s  |  "
          f"final F = {r_dw.free_energy_history[-1]:.4f}")

    # ── (b) CALPHAD-coupled at 800 °C (SI units) ──────────────────────────
    # Domain size: 1 μm × 1 μm — captures early-stage spinodal wavelengths
    #   in Nb-V BCC (characteristic length ~ 5–50 nm).
    # κ = 1×10⁻¹⁸ J·m²/mol — estimated from interface energy ~0.05 J/m²,
    #   interface width ~2 nm, molar volume ~10⁻⁵ m³/mol.
    # M = 5×10⁻²¹ m⁵/(J·s) — consistent with interdiffusion coefficient
    #   D ~ M × |∂²G/∂x²| ≈ 5×10⁻²¹ × 2000 ≈ 10⁻¹⁷ m²/s at 800 °C.
    # dt = 5 s — set by explicit stability: M|∂²G/∂x²|Δt/Δx² ≈ 0.16 < 0.5.
    print("  (b) CALPHAD surrogate (800 °C) ...")
    r_ca = solve_cahn_hilliard_binary(
        free_energy   = "calphad",
        N      = 256,        # grid points
        Lx     = 1e-6,       # domain length  [m]   1 μm × 1 μm
        kappa  = 1e-18,      # gradient energy coefficient  [J·m²/mol]
        M      = 5e-21,      # mobility  [m⁵/(J·s)]
        c0     = 0.4,        # mean composition  (mol fraction V)
        noise  = 0.02,       # initial perturbation amplitude
        seed   = 0,
        dt     = 5.0,        # time step  [s]
        n_steps    = 3000,
        save_every = 150,                  # doubled frames: 21 snapshots
        save_frames = True,
        calphad_fit   = fits,
        temperature_K = 773.15,            # 500 °C
        output_dir    = CFG["out_dir"] / "pf_calphad_500C",
    )
    plot_result(r_ca, output_dir=CFG["out_dir"] / "pf_calphad_500C", show=False)
    print(f"     elapsed: {r_ca.elapsed_seconds:.1f} s  |  "
          f"final F = {r_ca.free_energy_history[-1]:.4e}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 60)
    print("  Nb-V Pipeline")
    print(f"  Phase : {CFG['phase']}   Ref: μ_{CFG['ref_element']}")
    print(f"  T     : {CFG['temperatures_C'][0]}–{CFG['temperatures_C'][-1]} °C")
    print(f"  Grid  : {CFG['n_points']} pts   Poly degree: {CFG['poly_degree']}")
    print("=" * 60)

    paths    = step1_generate()
    sp, bn   = step2_spinodal(paths)
    fits     = step3_fit(paths)
    step4_plot(paths, fits)
    step5_phase_field(fits)

    print("\nDone. Outputs →", CFG["out_dir"])
