"""
Ternary Nb-Hf-V Jansson simulation to t=200 s physical time
============================================================
Runs Jansson-mode Cahn-Hilliard for Nb50Hf10V40 BCC_A2 at 500 °C.
dt is computed automatically from stable_dt(safety=0.1).

Outputs saved to:  tests/results_ternary_jansson_200s/
  - final_field.npz       : final compositions (x_NB, x_HF, x_V) + metadata
  - snapshots.npz         : periodic composition snapshots (~20 frames)
  - micrograph_xNB.pdf    : x_NB composition field
  - micrograph_xHF.pdf    : x_HF composition field
  - micrograph_xV.pdf     : x_V  composition field
  - free_energy.npz       : free energy history
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

from calphad_pf.thermodynamics.storage import load_table
from calphad_pf.thermodynamics.interpolant import build_interpolant
from calphad_pf.phasefield.solver import solve_cahn_hilliard, stable_dt

# ── parameters ────────────────────────────────────────────────────────────────
ELEMENTS  = ["NB", "HF", "V"]
REF_EL    = "V"
T_TARGET  = 773.15        # 500 °C
T_PHYS    = 400.0         # target physical time (s)
DT_SAFETY = 0.1           # CFL safety factor (was 0.3)
N_SNAPSHOTS = 10          # number of periodic composition snapshots to save

M_diag = [5.0e-21, 5.0e-21]
N      = 256
Lx     = 1.0e-6
C0     = [0.50, 0.10]    # x_NB=0.50, x_HF=0.10; x_V=0.40

OUT_DIR = _ROOT / "tests" / "results_ternary_jansson_200s"
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
    kappa      = [1.0e-12, 1.0e-12],
    M          = [[M_diag[0], 0.0], [0.0, M_diag[1]]],
    dt         = dt,
    n_steps    = N_STEPS,
    save_every = SAVE_EVERY,
)

# ── run ───────────────────────────────────────────────────────────────────────
t0  = time.time()
res = solve_cahn_hilliard(
    interpolant=interp, output_dir=None, save_frames=False, **SIM)
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

# ── save periodic snapshots ───────────────────────────────────────────────────
# res.comp_history shape: (n_frames, N_ind, N, N)
# Saving ~20 frames lets you reconstruct the full microstructural evolution
# without storing all N_STEPS frames (which would be ~tens of GB).
if hasattr(res, "comp_history") and res.comp_history is not None:
    snap = np.array(res.comp_history)   # (n_frames, 2, N, N)
    snap_times = np.arange(1, snap.shape[0] + 1) * SAVE_EVERY * dt
    snap_path = OUT_DIR / "snapshots.npz"
    np.savez_compressed(snap_path,
                        compositions=snap,
                        times=snap_times,
                        elements=["NB", "HF"])
    print(f"Saved: {snap_path}  ({snap.shape[0]} frames, shape {snap.shape})")
else:
    print("Note: comp_history not available in solver result — only final field saved.")

fe_path = OUT_DIR / "free_energy.npz"
np.savez_compressed(fe_path, free_energy=np.array(res.free_energy_history))
print(f"Saved: {fe_path}")

# ── save micrographs ──────────────────────────────────────────────────────────
matplotlib.rcParams.update({
    "font.family": "serif",
    "font.serif":  ["Times New Roman", "DejaVu Serif"],
    "font.size": 8,
})

fields = {"xNB": (x_nb, r"$x_{\rm Nb}$", "RdBu_r"),
          "xHF": (x_hf, r"$x_{\rm Hf}$", "RdBu_r"),
          "xV":  (x_v,  r"$x_{\rm V}$",  "RdBu_r")}

for name, (field, label, cmap) in fields.items():
    fig, ax = plt.subplots(figsize=(4, 4))
    vmid  = field.mean()
    vspan = max(abs(field.min()-vmid), abs(field.max()-vmid))
    im = ax.imshow(field, origin="lower", extent=[0,1,0,1],
                   cmap=cmap, vmin=vmid-vspan, vmax=vmid+vspan,
                   interpolation="nearest", aspect="equal")
    plt.colorbar(im, ax=ax, label=label, fraction=0.046, pad=0.04)
    ax.set_xlabel(r"$L_x$ (µm)")
    ax.set_ylabel(r"$L_y$ (µm)")
    ax.set_title(f"Nb$_{{50}}$Hf$_{{10}}$V$_{{40}}$ — Jansson\n"
                 f"$t = {t_phys_actual:.0f}$ s,  wall = {elapsed:.0f} s,  "
                 f"dt={dt:.2e} s")
    out = OUT_DIR / f"micrograph_{name}.pdf"
    fig.savefig(out, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved: {out}")

print("\nAll done.")
