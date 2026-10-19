"""3D Cahn-Hilliard simulation — ternary Nb50Hf5V45  |  500 °C  |  TDB v3
=======================================================================
Extends the 2D ternary run (NbHfV_500C.py) to a 3D domain.

Checks
------
  1. Mass conservation  : mean(x_Nb) and mean(x_Hf) conserved to 1e-10
  2. Energy monotonicity: total free energy is non-increasing
  3. Phase separation   : composition range widens for both Nb and Hf

Figures saved (per element: Nb, Hf, V)
---------------------------------------
  slices_*              2D mid-plane XY/XZ/YZ matplotlib figure
  pyvista_slices_*      PyVista orthogonal slice mesh in 3D
  pyvista_surface_*     PyVista smooth-shaded outer surface
  pyvista_iso_*         PyVista isosurface at mean composition

Usage
-----
    python runs/v3/run_3d_nbhfv_500C.py
    python runs/v3/run_3d_nbhfv_500C.py --N 32 --t 50   # fast smoke test
    python runs/v3/run_3d_nbhfv_500C.py --N 64 --t 200  # default
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
sys.path.insert(0, str(_ROOT / "src"))

from phasefoundry.phasefield.solver import solve_cahn_hilliard, stable_dt
from phasefoundry.phasefield.visualize import save_pyvista_3d, save_slices_2d
from phasefoundry.thermodynamics.interpolant import build_interpolant
from phasefoundry.thermodynamics.storage import load_table

TDB_STEM = "HfNbTiV_BCC_PhaseForge_v3"
SYSTEM = "NbHfV"
ELEMENTS = ["NB", "HF", "V"]
REF_EL = "V"
T_TARGET = 773.15  # 500 °C
RUN_NAME = "pf_3d_cNb50Hf05_kappa1e12_tdbv3"

M_diag = [5.0e-21, 5.0e-23]  # Nb, Hf  [mol·m²/(J·s)]
KAPPA = [1.0e-12, 7.0e-12]  # Nb, Hf  [J·m²/mol]
C0 = [0.50, 0.05]  # x_NB=0.50, x_HF=0.05, x_V=0.45
Lx = 1.0e-6  # domain side length [m]


def check_mass_conservation(c_history: list, c0: list, tol: float = 1e-10) -> bool:
    """Verify mean of each component is conserved relative to initial snapshot."""
    n_ind = len(c0)
    ok = True
    for i in range(n_ind):
        mean0 = float(c_history[0][i].mean())
        errors = [abs(snap[i].mean() - mean0) for snap in c_history]
        max_err = max(errors)
        passed = max_err < tol
        ok &= passed
        status = "PASS" if passed else "FAIL"
        print(
            f"  Mass conservation [{status}]  component {i}  "
            f"initial mean = {mean0:.6f}  (target = {c0[i]})  "
            f"max |Δmean| = {max_err:.2e}  (tol = {tol:.0e})"
        )
    return ok


def check_energy_monotonicity(F_hist: list, tol: float = 1e-6) -> bool:
    """Verify free energy is non-increasing (allow tiny numerical noise)."""
    F = np.array(F_hist)
    vio = int(np.sum(np.diff(F) > tol))
    ok = vio == 0
    status = "PASS" if ok else "FAIL"
    print(
        f"  Energy monotonicity [{status}]  violations = {vio}  ΔF range = [{np.diff(F).min():.3e}, {np.diff(F).max():.3e}] J/mol"
    )
    return ok


def check_phase_separation(c_history: list, threshold: float = 0.02) -> bool:
    """Verify composition range widens for both independent components."""
    ok = True
    labels = ["Nb", "Hf"]
    for i, lbl in enumerate(labels):
        r0 = float(c_history[0][i].max() - c_history[0][i].min())
        rf = float(c_history[-1][i].max() - c_history[-1][i].min())
        passed = rf > r0 + threshold
        ok &= passed
        status = "PASS" if passed else "WARN"
        print(f"  Phase separation  [{status}]  x_{lbl}  range: {r0:.4f} → {rf:.4f}  (need Δ > {threshold})")
    return ok


def run(
    N: int = 64,
    t_phys: float = 200.0,
    n_snapshots: int = 5,
    dt_safety: float = 1.0,
    use_gpu: bool = False,
    adaptive_dt: bool = True,
) -> None:

    out_dir = _ROOT / "results" / TDB_STEM / SYSTEM / "500C" / "kinetics" / RUN_NAME
    out_dir.mkdir(parents=True, exist_ok=True)

    table_dir = _ROOT / "data" / TDB_STEM / SYSTEM
    tables = sorted(table_dir.glob("*.h5"))
    if not tables:
        raise FileNotFoundError(f"No HDF5 tables in {table_dir}. Run step 1 first.")
    best = min(tables, key=lambda p: abs(float(load_table(p)["metadata"]["temperature"]) - T_TARGET))
    T_K = float(load_table(best)["metadata"]["temperature"])
    print(f"Table     : {best.name}  ({T_K - 273.15:.0f} °C)")

    interp = build_interpolant(
        [best],
        elements=ELEMENTS,
        reference_element=REF_EL,
        temperature_K=T_K,
        derivative_method="jansson",
    )

    dx = Lx / N
    dt = stable_dt(np.array(M_diag), dx, interp, c0=np.array(C0), safety=dt_safety)
    n_steps = int(np.ceil(t_phys / dt))
    save_every = max(1, n_steps // n_snapshots)

    print(f"\nGrid      : {N}³  =  {N**3:,} points")
    print(f"dt        : {dt:.4e} s  |  n_steps = {n_steps:,}  |  t_phys = {n_steps * dt:.1f} s")
    print(f"Snapshots : every {save_every} steps  (~{n_snapshots} frames)")
    print(f"Output    : {out_dir}\n")

    t0 = time.time()
    res = solve_cahn_hilliard(
        interpolant=interp,
        M=[[M_diag[0], 0.0], [0.0, M_diag[1]]],
        kappa=KAPPA,
        c0=C0,
        N=N,
        Lx=Lx,
        noise=0.01,
        seed=42,
        dt=dt,
        n_steps=n_steps,
        save_every=save_every,
        output_dir=None,
        save_frames=False,
        ndim=3,
        adaptive_dt=adaptive_dt,
        use_gpu=use_gpu,
    )
    elapsed = time.time() - t0
    print(f"\nWall time : {elapsed:.1f} s  ({elapsed / 60:.1f} min)")

    x_nb = res.compositions[0]
    x_hf = res.compositions[1]
    x_v = 1.0 - x_nb - x_hf

    print("\n── Verification ─────────────────────────────────────────────────────")
    all_pass = True
    all_pass &= check_mass_conservation(res.compositions_history, C0)
    all_pass &= check_energy_monotonicity(res.free_energy_history)
    all_pass &= check_phase_separation(res.compositions_history)

    print(f"\n  Final x_Nb : min={x_nb.min():.4f}  mean={x_nb.mean():.4f}  max={x_nb.max():.4f}")
    print(f"  Final x_Hf : min={x_hf.min():.4f}  mean={x_hf.mean():.4f}  max={x_hf.max():.4f}")
    print(f"  Final x_V  : min={x_v.min():.4f}   mean={x_v.mean():.4f}   max={x_v.max():.4f}")
    print("  Eq. targets: BCC_1 Nb~91% Hf~9% V~0.3%  |  BCC_2 V~99%")

    print("\n── Figures ──────────────────────────────────────────────────────────")

    t_final = res.time_history[-1]
    SYS_TITLE = "Nb$_{50}$Hf$_{5}$V$_{45}$ 500 °C — 3D simulation"

    fields = [
        (x_nb, "x_Nb", "Nb Composition"),
        (x_hf, "x_Hf", "Hf Composition"),
        (x_v, "x_V", "V Composition"),
    ]

    for field, label, bar_title in fields:
        p = save_slices_2d(field, t_final, out_dir, label=label, title=SYS_TITLE)
        print(f"  Saved: {p.name}")

        for p in save_pyvista_3d(field, t_final, out_dir, label=label, bar_title=bar_title, title="Nb50Hf5V45 500 C"):
            print(f"  Saved: {p.name}")

    # free energy history
    fig, ax = plt.subplots(figsize=(6, 4))
    t_arr = np.array(res.time_history)
    F_arr = np.array(res.free_energy_history)
    ax.plot(t_arr, F_arr, "o-", lw=2, color="#1f77b4")
    ax.set_xlabel("Physical time  (s)", fontsize=12)
    ax.set_ylabel("$F_{\\rm total}$  (J/mol)", fontsize=12)
    ax.set_title("Free energy — 3D Nb$_{50}$Hf$_{5}$V$_{45}$ 500 °C", fontsize=12)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_dir / "free_energy_3d.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: free_energy_3d.png")

    np.savez_compressed(
        out_dir / "final_field_3d.npz",
        x_nb=x_nb,
        x_hf=x_hf,
        x_v=x_v,
        t_phys=t_final,
        t_wall=elapsed,
        N=N,
        T_K=T_K,
    )
    print(f"  Saved: final_field_3d.npz  (shape {x_nb.shape})")

    print("\n" + "=" * 60)
    print(f"  {'ALL CHECKS PASSED' if all_pass else 'SOME CHECKS FAILED'}")
    print(f"  Results → {out_dir}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="3D Cahn-Hilliard simulation — ternary Nb50Hf5V45 at 500 °C")
    parser.add_argument("--N", type=int, default=64, help="Grid points per side (default: 64 → 64³)")
    parser.add_argument("--t", type=float, default=200.0, help="Physical end time in seconds (default: 200)")
    parser.add_argument("--snapshots", type=int, default=5, help="Number of snapshots (default: 5)")
    parser.add_argument("--safety", type=float, default=1.0, help="dt safety factor (default: 1.0)")
    parser.add_argument(
        "--gpu", action="store_true", default=False, help="Use CuPy GPU-accelerated FFT (requires: pip install cupy-cuda12x)"
    )
    parser.add_argument(
        "--adaptive",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable/disable adaptive time-stepping (default: enabled)",
    )
    args = parser.parse_args()

    run(N=args.N, t_phys=args.t, n_snapshots=args.snapshots, dt_safety=args.safety, use_gpu=args.gpu, adaptive_dt=args.adaptive)
