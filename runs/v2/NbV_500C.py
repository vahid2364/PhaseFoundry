"""Binary Nb50V50  |  BCC_A2  |  500 °C  |  TDB: HfNbTiV_BCC_PhaseForge_v2_Tdep
=========================================================================
Cahn-Hilliard phase-field simulation via Eyre convex-splitting (spectral).

Thermodynamics (500 °C, x_NB = 0.50, BCC_A2 miscibility gap):
  Equilibrium tie-line:
    BCC_1 (Nb-rich)  x_NB ~ 0.99 | x_V ~ 0.01
    BCC_2 (V-rich)   x_NB ~ 0.01 | x_V ~ 0.99

Output: results/HfNbTiV_BCC_PhaseForge_v2_Tdep/NbV/500C/
"""

from __future__ import annotations

import logging
import shutil
import sys
import time
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))

from phasefoundry.phasefield.solver import solve_cahn_hilliard, stable_dt
from phasefoundry.thermodynamics.interpolant import build_interpolant
from phasefoundry.thermodynamics.storage import load_table

logging.basicConfig(format="%(message)s", stream=sys.stdout)
logging.getLogger("phasefoundry").setLevel(logging.INFO)

TDB_STEM = "HfNbTiV_BCC_PhaseForge_v2_Tdep"
SYSTEM = "NbV"
ELEMENTS = ["NB", "V"]
REF_EL = "V"
T_TARGET = 773.15  # 500 °C
T_LABEL = "500C"
T_PHYS = 800.0  # target physical time (s)
DT_SAFETY = 1.0
N_SNAPSHOTS = 10

M_diag = [5.0e-21]  # Nb mobility  [mol·m²/(J·s)]
KAPPA = [1.0e-12]  # Nb gradient energy coefficient  [J·m²/mol]
N = 256
Lx = 1.0e-6  # domain side length  [m]
C0 = [0.50]  # x_NB = 0.50

RUN_NAME = "pf_c50_kappa1e12_tdbv2"
OUT_DIR = _ROOT / "results" / TDB_STEM / SYSTEM / T_LABEL / "kinetics" / RUN_NAME
if OUT_DIR.exists():
    shutil.rmtree(OUT_DIR)
    print(f"Cleared: {OUT_DIR}")
OUT_DIR.mkdir(parents=True, exist_ok=True)

table_dir = _ROOT / "data" / "HfNbTiV_BCC_PhaseForge_v2_Tdep" / "NbV"
tables = sorted(table_dir.glob("*.h5"))
best = min(tables, key=lambda p: abs(float(load_table(p)["metadata"]["temperature"]) - T_TARGET))
T_K = float(load_table(best)["metadata"]["temperature"])
print(f"Table: {best.name}  ({T_K - 273.15:.0f} °C)")

interp = build_interpolant([best], elements=ELEMENTS, reference_element=REF_EL, temperature_K=T_K, derivative_method="jansson")

c0_field = np.array([[[C0[0]]]])
H = interp.d2g(c0_field)[:, :, 0, 0]
H_NB = float(H[0, 0])
A_NB = max(0.0, -H_NB)
sigma_max = M_diag[0] * H_NB**2 / (4 * KAPPA[0])
tau = 1.0 / abs(sigma_max)
lam_star = 2 * np.pi * np.sqrt(2 * KAPPA[0] / abs(H_NB))
print(f"H_NB     = {H_NB:.0f} J/mol")
print(f"A_NB     = {A_NB:.0f} J/mol  (Eyre stabilizer, H_eff = {H_NB + A_NB:.0f} J/mol)")
print(f"tau      = {tau:.1f} s  |  t_phys/tau = {T_PHYS / tau:.1f}")
print(f"lambda*  = {lam_star * 1e9:.1f} nm  ({lam_star / (Lx / N):.1f} grid pts)")

dx = Lx / N
dt = stable_dt(np.array(M_diag), dx, interp, c0=np.array(C0), safety=DT_SAFETY)
N_STEPS = int(np.ceil(T_PHYS / dt))
SAVE_EVERY = max(1, N_STEPS // N_SNAPSHOTS)
print(f"dt (safety={DT_SAFETY}) = {dt:.4e} s")
print(f"n_steps = {N_STEPS}  ->  t_phys = {N_STEPS * dt:.2f} s")
print(f"Snapshots every {SAVE_EVERY} steps  (~{N_SNAPSHOTS} frames)")

SIM = dict(
    N=N,
    Lx=Lx,
    c0=C0,
    noise=0.01,
    seed=42,
    kappa=KAPPA,
    M=[[M_diag[0]]],
    dt=dt,
    n_steps=N_STEPS,
    save_every=SAVE_EVERY,
)

matplotlib.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 8,
    }
)


def save_micrograph(xnb, xv, t_s, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))
    panels = [(xnb, r"$x_{\rm Nb}$", "RdBu_r"), (xv, r"$x_{\rm V}$", "RdBu_r")]
    for ax, (field, label, cmap) in zip(axes, panels):
        vmid = field.mean()
        vspan = max(abs(field.min() - vmid), abs(field.max() - vmid), 1e-6)
        im = ax.imshow(
            field,
            origin="lower",
            extent=[0, 1, 0, 1],
            cmap=cmap,
            vmin=vmid - vspan,
            vmax=vmid + vspan,
            interpolation="nearest",
            aspect="equal",
        )
        plt.colorbar(im, ax=ax, label=label, fraction=0.046, pad=0.04)
        ax.set_xlabel(r"$L_x$ ($\mu$m)")
        ax.set_ylabel(r"$L_y$ ($\mu$m)")
        ax.set_title(f"Nb$_{{50}}$V$_{{50}}$ 500$^{{\\circ}}$C — {label}\n$t = {t_s:.1f}$ s  ($t/\\tau = {t_s / tau:.1f}$)")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"Saved: {out_path}")


snap_dir = OUT_DIR / "snapshots"
snap_dir.mkdir(parents=True, exist_ok=True)


def _save_snapshot(c_arr, t_s):
    snb = c_arr[0]
    sv = 1.0 - snb
    out = snap_dir / f"snapshot_{_save_snapshot.counter:03d}_t{t_s:.1f}s.pdf"
    save_micrograph(snb, sv, t_s, out)
    _save_snapshot.counter += 1


_save_snapshot.counter = 0

t0 = time.time()
res = solve_cahn_hilliard(interpolant=interp, output_dir=None, save_frames=False, snapshot_callback=_save_snapshot, **SIM)
elapsed = time.time() - t0

x_nb = res.compositions[0]
x_v = 1.0 - x_nb
t_phys_actual = N_STEPS * dt

print(f"\nDone in {elapsed:.0f} s wall time  ({elapsed / 60:.1f} min)")
print(f"x_NB: [{x_nb.min():.4f}, {x_nb.mean():.4f}, {x_nb.max():.4f}]")
print(f"x_V:  [{x_v.min():.4f},  {x_v.mean():.4f},  {x_v.max():.4f}]")
print("Eq. target: BCC_1 x_NB~0.99 | BCC_2 x_NB~0.01")
print(f"Decomposition progress: {(x_nb.max() - x_nb.min()) / (0.99 - 0.01) * 100:.1f}%")

np.savez_compressed(
    OUT_DIR / "final_field.npz",
    x_nb=x_nb,
    x_v=x_v,
    t_phys=t_phys_actual,
    t_wall=elapsed,
    dt=dt,
    dt_safety=DT_SAFETY,
    n_steps=N_STEPS,
    c0_nb=C0[0],
    T_K=T_K,
    H_NB=H_NB,
    A_NB=A_NB,
    tau=tau,
    lam_star=lam_star,
)
print(f"Saved: {OUT_DIR / 'final_field.npz'}")

if hasattr(res, "compositions_history") and res.compositions_history:
    snap = np.array(res.compositions_history)
    np.savez_compressed(OUT_DIR / "snapshots.npz", compositions=snap, times=np.array(res.time_history), elements=["NB"])
    print(f"Saved: {OUT_DIR / 'snapshots.npz'}  ({snap.shape[0]} frames)")

fe_arr = np.array(res.free_energy_history)
fe_times = np.linspace(0, t_phys_actual, len(fe_arr))
np.savez_compressed(OUT_DIR / "free_energy.npz", free_energy=fe_arr)
np.savetxt(
    OUT_DIR / "free_energy.csv",
    np.column_stack([fe_times, fe_arr]),
    delimiter=",",
    header="t_phys_s,free_energy_J_mol",
    comments="",
)
print(f"Saved: {OUT_DIR / 'free_energy.csv'}")

save_micrograph(x_nb, x_v, t_phys_actual, OUT_DIR / "micrograph_final.pdf")

print("\nAll done.")
