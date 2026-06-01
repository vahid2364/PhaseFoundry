"""
3D Cahn-Hilliard verification — binary Nb50V50 at 500 °C
=========================================================
Runs a short 3D phase-field simulation and checks:

  1. Mass conservation  : mean(c_Nb) stays within 1e-10 of c0 = 0.50
  2. Energy monotonicity: total free energy is non-increasing
  3. Phase separation   : c_min drops and c_max rises over time
  4. Mid-plane figures  : XY, XZ, YZ slices saved for visual inspection

Uses N=64 (64³ grid, ~262k points) and a short physical time (200 s)
so the test completes in a few minutes on a laptop.

Usage
-----
    python tests/run_3d_nbv_500C.py
    python tests/run_3d_nbv_500C.py --N 32 --t 50   # fast smoke test
    python tests/run_3d_nbv_500C.py --N 128 --t 800  # production quality
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from calphad_pf.thermodynamics.storage     import load_table
from calphad_pf.thermodynamics.interpolant import build_interpolant
from calphad_pf.phasefield.solver          import solve_cahn_hilliard, stable_dt
from calphad_pf.phasefield.visualize       import save_slices_2d, save_pyvista_3d


# ── parameters ────────────────────────────────────────────────────────────────

TDB_STEM  = "HfNbTiV_BCC_PhaseForge_v3"
SYSTEM    = "NbV"
ELEMENTS  = ["NB", "V"]
REF_EL    = "V"
T_TARGET  = 773.15   # 500 °C

M_diag = [5.0e-21]   # Nb mobility  [mol·m²/(J·s)]
KAPPA  = [1.0e-12]   # Nb gradient energy coefficient  [J·m²/mol]
C0     = [0.50]      # x_NB = 0.50
Lx     = 1.0e-6      # domain side length [m]


# ── helpers ───────────────────────────────────────────────────────────────────


def check_mass_conservation(c_history: list, c0: float,
                             tol: float = 1e-10) -> bool:
    """
    Verify mean composition is conserved relative to the initial snapshot.

    The initial noise field has mean ≈ c0 but not exactly c0 (random noise
    does not cancel perfectly at finite N).  The solver conserves whatever
    initial mean it starts from, so we compare every snapshot against the
    initial mean rather than the target c0.
    """
    mean0  = float(c_history[0][0].mean())
    errors = [abs(snap[0].mean() - mean0) for snap in c_history]
    max_err = max(errors)
    ok = max_err < tol
    status = "PASS" if ok else "FAIL"
    print(f"  Mass conservation [{status}]  "
          f"initial mean = {mean0:.6f}  (target c0 = {c0})  "
          f"max |mean(c) - mean0| = {max_err:.2e}  (tol = {tol:.0e})")
    return ok


def check_energy_monotonicity(F_hist: list, tol: float = 1e-6) -> bool:
    """Verify free energy is non-increasing (allow tiny numerical noise)."""
    F = np.array(F_hist)
    violations = np.sum(np.diff(F) > tol)
    ok = violations == 0
    status = "PASS" if ok else "FAIL"
    print(f"  Energy monotonicity [{status}]  violations = {violations}  "
          f"ΔF range = [{np.diff(F).min():.3e}, {np.diff(F).max():.3e}] J/mol")
    return ok


def check_phase_separation(c_history: list, threshold: float = 0.05) -> bool:
    """Verify that composition range widens (spinodal decomposition occurring)."""
    c_init = c_history[0][0]
    c_final = c_history[-1][0]
    range_init  = float(c_init.max()  - c_init.min())
    range_final = float(c_final.max() - c_final.min())
    ok = range_final > range_init + threshold
    status = "PASS" if ok else "WARN"
    print(f"  Phase separation  [{status}]  "
          f"composition range: {range_init:.4f} → {range_final:.4f}  "
          f"(need Δ > {threshold})")
    return ok


# ── main ──────────────────────────────────────────────────────────────────────

def run(N: int = 64, t_phys: float = 200.0, n_snapshots: int = 5,
        dt_safety: float = 1.0, use_gpu: bool = False) -> None:

    out_dir = _ROOT / "results" / TDB_STEM / SYSTEM / "500C" / "kinetics" / "pf_3d_c50_kappa1e12_tdbv3"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── load thermodynamic table ──────────────────────────────────────────────
    table_dir = _ROOT / "data" / TDB_STEM / SYSTEM
    tables = sorted(table_dir.glob("*.h5"))
    if not tables:
        raise FileNotFoundError(f"No HDF5 tables in {table_dir}. Run step 1 first.")
    best = min(tables, key=lambda p: abs(
        float(load_table(p)["metadata"]["temperature"]) - T_TARGET))
    T_K = float(load_table(best)["metadata"]["temperature"])
    print(f"Table     : {best.name}  ({T_K - 273.15:.0f} °C)")

    # ── build interpolant ─────────────────────────────────────────────────────
    interp = build_interpolant(
        [best], elements=ELEMENTS, reference_element=REF_EL,
        temperature_K=T_K, derivative_method="jansson",
    )

    # ── time step ─────────────────────────────────────────────────────────────
    dx = Lx / N
    dt = stable_dt(np.array(M_diag), dx, interp, c0=np.array(C0),
                   safety=dt_safety)
    n_steps    = int(np.ceil(t_phys / dt))
    save_every = max(1, n_steps // n_snapshots)

    print(f"\nGrid      : {N}³  =  {N**3:,} points")
    print(f"dt        : {dt:.4e} s  |  n_steps = {n_steps:,}  |  t_phys = {n_steps*dt:.1f} s")
    print(f"Snapshots : every {save_every} steps  (~{n_snapshots} frames)")
    print(f"Output    : {out_dir}\n")

    # ── run 3D simulation ─────────────────────────────────────────────────────
    t0  = time.time()
    res = solve_cahn_hilliard(
        interpolant = interp,
        M           = [[M_diag[0]]],
        kappa       = KAPPA,
        c0          = C0,
        N           = N,
        Lx          = Lx,
        noise       = 0.01,
        seed        = 42,
        dt          = dt,
        n_steps     = n_steps,
        save_every  = save_every,
        output_dir  = None,
        save_frames = False,
        ndim        = 3,
        adaptive_dt = True,
        use_gpu     = use_gpu,
    )
    elapsed = time.time() - t0
    print(f"\nWall time : {elapsed:.1f} s  ({elapsed/60:.1f} min)")

    # ── verification checks ───────────────────────────────────────────────────
    print("\n── Verification ─────────────────────────────────────────────────────")
    all_pass = True
    all_pass &= check_mass_conservation(res.compositions_history, C0[0])
    all_pass &= check_energy_monotonicity(res.free_energy_history)
    all_pass &= check_phase_separation(res.compositions_history)

    c_final = res.compositions[0]
    print(f"\n  Final x_Nb : min={c_final.min():.4f}  "
          f"mean={c_final.mean():.4f}  max={c_final.max():.4f}")
    print(f"  Eq. targets: BCC_1 x_Nb ≈ 0.99  |  BCC_2 x_Nb ≈ 0.01")

    # ── save figures ──────────────────────────────────────────────────────────
    print("\n── Figures ──────────────────────────────────────────────────────────")

    SYS_TITLE = f"Nb$_{{50}}$V$_{{50}}$ 500 °C — 3D simulation"

    # initial and final 2D mid-plane slices
    for c_snap, t_snap in [
        (res.compositions_history[0][0], res.time_history[0]),
        (c_final,                        res.time_history[-1]),
    ]:
        p = save_slices_2d(c_snap, t_snap, out_dir,
                           label="x_Nb", title=SYS_TITLE)
        print(f"  Saved: {p.name}")

    # 3D PyVista renders (slices, surface, isosurface)
    for p in save_pyvista_3d(c_final, res.time_history[-1], out_dir,
                              label="x_Nb", bar_title="Nb Composition",
                              title="Nb50V50 500 C"):
        print(f"  Saved: {p.name}")

    # free energy history
    fig, ax = plt.subplots(figsize=(6, 4))
    t_arr = np.array(res.time_history)
    F_arr = np.array(res.free_energy_history)
    ax.plot(t_arr, F_arr, "o-", lw=2, color="#1f77b4")
    ax.set_xlabel("Physical time  (s)", fontsize=12)
    ax.set_ylabel("$F_{\\rm total}$  (J/mol)", fontsize=12)
    ax.set_title("Free energy — 3D Nb$_{50}$V$_{50}$ 500 °C", fontsize=12)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_dir / "free_energy_3d.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: free_energy_3d.png")

    # save final field as npz
    np.savez_compressed(
        out_dir / "final_field_3d.npz",
        x_nb    = c_final,
        x_v     = 1.0 - c_final,
        t_phys  = res.time_history[-1],
        t_wall  = elapsed,
        N       = N,
        T_K     = T_K,
    )
    print(f"  Saved: final_field_3d.npz  (shape {c_final.shape})")

    # ── summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"  {'ALL CHECKS PASSED' if all_pass else 'SOME CHECKS FAILED'}")
    print(f"  Results → {out_dir}")
    print("=" * 60)


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="3D Cahn-Hilliard verification — binary Nb50V50 at 500 °C"
    )
    parser.add_argument("--N",  type=int,   default=64,
                        help="Grid points per side (default: 64 → 64³)")
    parser.add_argument("--t",  type=float, default=200.0,
                        help="Physical end time in seconds (default: 200)")
    parser.add_argument("--snapshots", type=int, default=5,
                        help="Number of snapshots (default: 5)")
    parser.add_argument("--safety", type=float, default=1.0,
                        help="dt safety factor (default: 1.0)")
    parser.add_argument("--gpu", action="store_true", default=False,
                        help="Use CuPy GPU-accelerated FFT (requires: pip install cupy-cuda12x)")
    args = parser.parse_args()

    run(N=args.N, t_phys=args.t, n_snapshots=args.snapshots,
        dt_safety=args.safety, use_gpu=args.gpu)
