"""
Ternary Nb-Hf-V Jansson simulation to t=800 s physical time
============================================================
Runs Jansson-mode Cahn-Hilliard for Nb50Hf5V45 BCC_A2 at 600 °C.
dt is computed automatically from stable_dt(safety=0.7).

At 600°C/xHf=0.05:
  H_NB   =  -5547 J/mol  (Nb-only spinodal driver, ~2× stronger than 700°C)
  H_HF   = +87720 J/mol  (Hf stable — no negative-composition risk)
  lam_min = -18823 J/mol (~1.4× stronger driving force than 700°C)
  tau_Nb  ~  1.1 s        (t=800s ~ 700 tau — excellent evolution)

Equilibrium two-phase tie-line (from pycalphad):
  BCC_1 (Nb+Hf-rich): Nb~90%  Hf~9%  V~1%
  BCC_2 (V-rich):     Nb~1%   Hf~0.1% V~99%

Outputs saved to:  tests/results_ternary_jansson_600C/
  - final_field.npz       : final compositions (x_NB, x_HF, x_V) + metadata
  - snapshots.npz         : periodic composition snapshots (~10 frames)
  - snapshots/            : per-frame 3-panel PDFs (xNB, xHF, xV)
  - micrograph_final.pdf  : final 3-panel micrograph
  - free_energy.npz       : free energy history
  - free_energy.csv       : free energy history (t, F) in CSV
"""

from __future__ import annotations
import sys, time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from phasefoundry.thermodynamics.storage import load_table
from phasefoundry.thermodynamics.interpolant import build_interpolant
from phasefoundry.phasefield.solver import solve_cahn_hilliard, stable_dt

# ── parameters ────────────────────────────────────────────────────────────────
ELEMENTS    = ["NB", "HF", "V"]
REF_EL      = "V"
T_TARGET    = 873.15        # 600 °C
T_PHYS      = 5000.0        # target physical time (s)
DT_SAFETY   = 10.0          # well above classical stability limit — testing Eyre scheme robustness
N_SNAPSHOTS = 10

M_diag = [5.0e-21, 5.0e-23]   # Nb, Hf — Hf ~100x slower diffuser in BCC
KAPPA  = [1.0e-12, 7.0e-12]   # Nb, Hf — scaled to |H_HF|/|H_NB| ratio
N      = 256
Lx     = 1.0e-6
C0     = [0.50, 0.05]    # x_NB=0.50, x_HF=0.05; x_V=0.45

OUT_DIR = _ROOT / "tests" / "results_ternary_jansson_600C"
import shutil
if OUT_DIR.exists():
    shutil.rmtree(OUT_DIR)
    print(f"Cleared: {OUT_DIR}")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── load table ────────────────────────────────────────────────────────────────
table_dir = _ROOT / "data" / "nb_hf_v"
tables    = sorted(table_dir.glob("*.h5"))
best = min(tables, key=lambda p: abs(
    float(load_table(p)["metadata"]["temperature"]) - T_TARGET))
T_K = float(load_table(best)["metadata"]["temperature"])
print(f"Table: {best.name}  ({T_K - 273.15:.0f} °C)")

# ── build interpolant ─────────────────────────────────────────────────────────
interp = build_interpolant(
    [best], elements=ELEMENTS, reference_element=REF_EL,
    temperature_K=T_K, derivative_method="jansson")

# ── compute dt from stability criterion ──────────────────────────────────────
dx = Lx / N
dt = stable_dt(np.array(M_diag), dx, interp, c0=np.array(C0), safety=DT_SAFETY)
N_STEPS    = int(np.ceil(T_PHYS / dt))
SAVE_EVERY = max(1, N_STEPS // N_SNAPSHOTS)

print(f"dt (safety={DT_SAFETY}) = {dt:.4e} s")
print(f"n_steps = {N_STEPS}  →  t_phys = {N_STEPS * dt:.2f} s")
print(f"Snapshots every {SAVE_EVERY} steps  (~{N_SNAPSHOTS} frames)")

SIM = dict(
    N          = N,
    Lx         = Lx,
    c0         = C0,
    noise      = 0.01,
    seed       = 42,
    kappa      = KAPPA,
    M          = [[M_diag[0], 0.0], [0.0, M_diag[1]]],
    dt         = dt,
    n_steps    = N_STEPS,
    save_every = SAVE_EVERY,
)

# ── helper: save one 3-panel micrograph (xNB, xHF, xV) ──────────────────────
matplotlib.rcParams.update({
    "font.family": "serif",
    "font.serif":  ["Times New Roman", "DejaVu Serif"],
    "font.size": 8,
})

def save_micrograph(xnb, xhf, xv, t_s, out_path):
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5))
    panels = [
        (xnb, r"$x_{\rm Nb}$", "RdBu_r"),
        (xhf, r"$x_{\rm Hf}$", "RdBu_r"),
        (xv,  r"$x_{\rm V}$",  "RdBu_r"),
    ]
    for ax, (field, label, cmap) in zip(axes, panels):
        vmid  = field.mean()
        vspan = max(abs(field.min() - vmid), abs(field.max() - vmid), 1e-6)
        im = ax.imshow(field, origin="lower", extent=[0, 1, 0, 1],
                       cmap=cmap, vmin=vmid - vspan, vmax=vmid + vspan,
                       interpolation="nearest", aspect="equal")
        plt.colorbar(im, ax=ax, label=label, fraction=0.046, pad=0.04)
        ax.set_xlabel(r"$L_x$ ($\mu$m)")
        ax.set_ylabel(r"$L_y$ ($\mu$m)")
        ax.set_title(
            f"Nb$_{{50}}$Hf$_{{5}}$V$_{{45}}$ 600$^{{\\circ}}$C — {label}\n"
            f"$t = {t_s:.1f}$ s")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"Saved: {out_path}")

# ── run ───────────────────────────────────────────────────────────────────────
snap_dir = OUT_DIR / "snapshots"
snap_dir.mkdir(parents=True, exist_ok=True)

def _save_snapshot_pdf(c_arr, t_s):
    """Called during the run to save a 3-panel PDF snapshot to disk."""
    snb = c_arr[0]
    shf = c_arr[1]
    sv  = 1.0 - snb - shf
    frame_idx = _save_snapshot_pdf.counter
    out = snap_dir / f"snapshot_{frame_idx:03d}_t{t_s:.1f}s.pdf"
    save_micrograph(snb, shf, sv, t_s, out)
    _save_snapshot_pdf.counter += 1
_save_snapshot_pdf.counter = 0

t0  = time.time()
res = solve_cahn_hilliard(
    interpolant=interp, output_dir=None, save_frames=False,
    snapshot_callback=_save_snapshot_pdf, **SIM)
elapsed = time.time() - t0

x_nb = res.compositions[0]
x_hf = res.compositions[1]
x_v  = 1.0 - x_nb - x_hf
t_phys_actual = N_STEPS * dt

print(f"\nDone in {elapsed:.0f} s wall time  ({elapsed/60:.1f} min)")
print(f"x_NB: [{x_nb.min():.4f}, {x_nb.mean():.4f}, {x_nb.max():.4f}]")
print(f"x_HF: [{x_hf.min():.4f}, {x_hf.mean():.4f}, {x_hf.max():.4f}]")
print(f"x_V:  [{x_v.min():.4f},  {x_v.mean():.4f},  {x_v.max():.4f}]")

# ── save final field ──────────────────────────────────────────────────────────
npz_path = OUT_DIR / "final_field.npz"
np.savez_compressed(
    npz_path,
    x_nb      = x_nb,
    x_hf      = x_hf,
    x_v       = x_v,
    t_phys    = t_phys_actual,
    t_wall    = elapsed,
    dt        = dt,
    dt_safety = DT_SAFETY,
    n_steps   = N_STEPS,
    c0_nb     = C0[0],
    c0_hf     = C0[1],
    T_K       = T_K,
)
print(f"Saved: {npz_path}")

# ── save periodic snapshots (npz + micrograph PDFs) ──────────────────────────
if hasattr(res, "compositions_history") and res.compositions_history:
    snap       = np.array(res.compositions_history)   # (n_frames, 2, N, N)
    snap_times = np.array(res.time_history)
    snap_path  = OUT_DIR / "snapshots.npz"
    np.savez_compressed(snap_path,
                        compositions=snap,
                        times=snap_times,
                        elements=["NB", "HF"])
    print(f"Saved: {snap_path}  ({snap.shape[0]} frames)")
    print(f"  ({snap.shape[0]} snapshot PDFs already saved to {snap_dir})")
else:
    print("Note: compositions_history not available — only final field saved.")

# ── save free energy ──────────────────────────────────────────────────────────
fe_path = OUT_DIR / "free_energy.npz"
np.savez_compressed(fe_path, free_energy=np.array(res.free_energy_history))
print(f"Saved: {fe_path}")

fe_csv_path = OUT_DIR / "free_energy.csv"
fe_arr   = np.array(res.free_energy_history)
fe_times = np.linspace(0, t_phys_actual, len(fe_arr))
np.savetxt(fe_csv_path, np.column_stack([fe_times, fe_arr]),
           delimiter=",", header="t_phys_s,free_energy_J_mol", comments="")
print(f"Saved: {fe_csv_path}")

# ── save final micrograph ─────────────────────────────────────────────────────
save_micrograph(x_nb, x_hf, x_v, t_phys_actual,
                OUT_DIR / "micrograph_final.pdf")

print("\nAll done.")
