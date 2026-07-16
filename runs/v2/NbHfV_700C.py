"""
Ternary Nb50Hf5V45  |  BCC_A2  |  700 °C  |  TDB: HfNbTiV_BCC_PhaseForge_v2_Tdep
=============================================================================
Cahn-Hilliard phase-field simulation via Eyre convex-splitting (spectral).

Thermodynamics (700 °C, Nb50Hf5V45, BCC_A2):
  H_NB = -2744 J/mol  (mild Nb-only spinodal)
  H_HF = +113484 J/mol (Hf stable — spectator)
  tau_Nb ~ 26 s  |  lambda* ~ 119 nm
  Equilibrium tie-line:
    BCC_1 (Nb+Hf-rich)  Nb~90%  Hf~9%  V~1%
    BCC_2 (V-rich)      Nb~2%   Hf~0.1% V~98%

Output: results/HfNbTiV_BCC_PhaseForge_v2_Tdep/NbHfV/700C/
"""

from __future__ import annotations
import sys, time, shutil
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))

from phasefoundry.thermodynamics.storage import load_table
from phasefoundry.thermodynamics.interpolant import build_interpolant
from phasefoundry.phasefield.solver import solve_cahn_hilliard, stable_dt

# ── parameters ────────────────────────────────────────────────────────────────
TDB_STEM    = "HfNbTiV_BCC_PhaseForge_v2_Tdep"
SYSTEM      = "NbHfV"
ELEMENTS    = ["NB", "HF", "V"]
REF_EL      = "V"
T_TARGET    = 973.15        # 700 °C
T_LABEL     = "700C"
T_PHYS      = 800.0         # target physical time (s)
DT_SAFETY   = 1.0
N_SNAPSHOTS = 10

M_diag = [5.0e-21, 5.0e-23]
KAPPA  = [1.0e-12, 7.0e-12]
N      = 256
Lx     = 1.0e-6
C0     = [0.50, 0.05]

RUN_NAME = "pf_cNb50Hf05_kappa1e12_tdbv2"
OUT_DIR  = _ROOT / "results" / TDB_STEM / SYSTEM / T_LABEL / "kinetics" / RUN_NAME
if OUT_DIR.exists():
    shutil.rmtree(OUT_DIR)
    print(f"Cleared: {OUT_DIR}")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── load thermodynamic table ──────────────────────────────────────────────────
table_dir = _ROOT / "data" / "HfNbTiV_BCC_PhaseForge_v2_Tdep" / "NbHfV"
tables    = sorted(table_dir.glob("*.h5"))
best = min(tables, key=lambda p: abs(
    float(load_table(p)["metadata"]["temperature"]) - T_TARGET))
T_K = float(load_table(best)["metadata"]["temperature"])
print(f"Table: {best.name}  ({T_K - 273.15:.0f} °C)")

# ── build interpolant ─────────────────────────────────────────────────────────
interp = build_interpolant(
    [best], elements=ELEMENTS, reference_element=REF_EL,
    temperature_K=T_K, derivative_method="jansson")

# ── print key thermodynamic quantities ───────────────────────────────────────
c0_field = np.array(C0)[:, np.newaxis, np.newaxis] * np.ones((len(C0), 1, 1))
H = interp.d2g(c0_field)[:, :, 0, 0]
for i, el in enumerate(ELEMENTS[:-1]):
    print(f"H_{el} = {float(H[i,i]):.0f} J/mol")

# ── time-step ────────────────────────────────────────────────────────────────
dx = Lx / N
dt = stable_dt(np.array(M_diag), dx, interp, c0=np.array(C0), safety=DT_SAFETY)
N_STEPS    = int(np.ceil(T_PHYS / dt))
SAVE_EVERY = max(1, N_STEPS // N_SNAPSHOTS)
print(f"dt (safety={DT_SAFETY}) = {dt:.4e} s")
print(f"n_steps = {N_STEPS}  ->  t_phys = {N_STEPS * dt:.2f} s")
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

# ── micrograph helper ─────────────────────────────────────────────────────────
matplotlib.rcParams.update({
    "font.family": "serif",
    "font.serif":  ["Times New Roman", "DejaVu Serif"],
    "font.size": 8,
})

def save_micrograph(xnb, xhf, xv, t_s, out_path):
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5))
    panels = [(xnb, r"$x_{\rm Nb}$", "RdBu_r"),
              (xhf, r"$x_{\rm Hf}$", "RdBu_r"),
              (xv,  r"$x_{\rm V}$",  "RdBu_r")]
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
            f"Nb$_{{50}}$Hf$_{{5}}$V$_{{45}}$ 700$^{{\\circ}}$C — {label}\n"
            f"$t = {t_s:.1f}$ s")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"Saved: {out_path}")

# ── snapshot callback ─────────────────────────────────────────────────────────
snap_dir = OUT_DIR / "snapshots"
snap_dir.mkdir(parents=True, exist_ok=True)

def _save_snapshot(c_arr, t_s):
    snb = c_arr[0]
    shf = c_arr[1]
    sv  = 1.0 - snb - shf
    out = snap_dir / f"snapshot_{_save_snapshot.counter:03d}_t{t_s:.1f}s.pdf"
    save_micrograph(snb, shf, sv, t_s, out)
    _save_snapshot.counter += 1
_save_snapshot.counter = 0

# ── run ───────────────────────────────────────────────────────────────────────
t0  = time.time()
res = solve_cahn_hilliard(
    interpolant=interp, output_dir=None, save_frames=False,
    snapshot_callback=_save_snapshot, **SIM)
elapsed = time.time() - t0

x_nb = res.compositions[0]
x_hf = res.compositions[1]
x_v  = 1.0 - x_nb - x_hf
t_phys_actual = N_STEPS * dt

print(f"\nDone in {elapsed:.0f} s wall time  ({elapsed/60:.1f} min)")
print(f"x_NB: [{x_nb.min():.4f}, {x_nb.mean():.4f}, {x_nb.max():.4f}]")
print(f"x_HF: [{x_hf.min():.4f}, {x_hf.mean():.4f}, {x_hf.max():.4f}]")
print(f"x_V:  [{x_v.min():.4f},  {x_v.mean():.4f},  {x_v.max():.4f}]")

# ── save outputs ──────────────────────────────────────────────────────────────
np.savez_compressed(
    OUT_DIR / "final_field.npz",
    x_nb=x_nb, x_hf=x_hf, x_v=x_v,
    t_phys=t_phys_actual, t_wall=elapsed,
    dt=dt, dt_safety=DT_SAFETY, n_steps=N_STEPS,
    c0_nb=C0[0], c0_hf=C0[1], T_K=T_K,
)
print(f"Saved: {OUT_DIR / 'final_field.npz'}")

if hasattr(res, "compositions_history") and res.compositions_history:
    snap = np.array(res.compositions_history)
    np.savez_compressed(OUT_DIR / "snapshots.npz",
                        compositions=snap,
                        times=np.array(res.time_history),
                        elements=["NB", "HF"])
    print(f"Saved: {OUT_DIR / 'snapshots.npz'}  ({snap.shape[0]} frames)")

fe_arr   = np.array(res.free_energy_history)
fe_times = np.linspace(0, t_phys_actual, len(fe_arr))
np.savez_compressed(OUT_DIR / "free_energy.npz", free_energy=fe_arr)
np.savetxt(OUT_DIR / "free_energy.csv",
           np.column_stack([fe_times, fe_arr]),
           delimiter=",", header="t_phys_s,free_energy_J_mol", comments="")
print(f"Saved: {OUT_DIR / 'free_energy.csv'}")

save_micrograph(x_nb, x_hf, x_v, t_phys_actual, OUT_DIR / "micrograph_final.pdf")

print("\nAll done.")
