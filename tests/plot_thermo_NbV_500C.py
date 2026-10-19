"""
Thermodynamic properties of binary Nb50V50 BCC_A2 at 500 °C
=============================================================
Plots from the pre-computed HDF5 table (HfNbTiV_BCC_PhaseForge_v3):

  Panel 1 — Molar Gibbs energy G_m vs x_V
  Panel 2 — Chemical potentials mu_NB and mu_V vs x_V
  Panel 3 — Hessian d2G/dx_NB^2 (= dmu_NB/dx_NB) vs x_V
             with spinodal region (H < 0) shaded

Output: tests/results_thermo_NbV_500C/thermo_NbV_500C.pdf
        tests/results_thermo_NbV_500C/thermo_NbV_500C.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import ticker

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from phasefoundry.thermodynamics.storage import load_table

TDB_STEM = "HfNbTiV_BCC_PhaseForge_v3"
table_path = _ROOT / "data" / TDB_STEM / "NbV" / "nb_v_bcc_500C.h5"
data = load_table(table_path)

x_nb  = data["compositions"][:, 0]        # x_NB (independent element is first)
gm    = data["GM"]                         # J/mol
mu_nb = data["MU"]["NB"]                   # J/mol
mu_v  = data["MU"]["V"]                    # J/mol
h_nb  = data["dMU"]["NB_NB"]              # d mu_NB / d x_NB  [J/mol]

# sort by x_V for clean line plots (x_V = 1 - x_NB)
x_v   = 1.0 - x_nb
idx   = np.argsort(x_v)
x_v   = x_v[idx]
gm    = gm[idx]
mu_nb = mu_nb[idx]
mu_v  = mu_v[idx]
h_nb  = h_nb[idx]

T_C = float(data["metadata"]["temperature"]) - 273.15

matplotlib.rcParams.update({
    "font.family":       "serif",
    "font.serif":        ["Times New Roman", "DejaVu Serif"],
    "font.size":         9,
    "axes.labelsize":    10,
    "axes.titlesize":    10,
    "legend.fontsize":   8,
    "xtick.labelsize":   8,
    "ytick.labelsize":   8,
    "lines.linewidth":   1.5,
    "axes.linewidth":    0.8,
    "xtick.direction":   "in",
    "ytick.direction":   "in",
    "xtick.top":         True,
    "ytick.right":       True,
    "figure.dpi":        300,
})

fig, axes = plt.subplots(1, 3, figsize=(10, 3.4))
fig.subplots_adjust(wspace=0.38)

spinodal_mask = h_nb < 0
# find the x_V range where H < 0 (interior region only)
interior = (x_v > 0.02) & (x_v < 0.98)
sp_x = x_v[spinodal_mask & interior]
sp_lo = sp_x.min() if sp_x.size else 0.0
sp_hi = sp_x.max() if sp_x.size else 1.0

SPINODAL_COLOR = "#ffd966"
SPINODAL_ALPHA = 0.35

for ax in axes:
    ax.set_xlabel(r"$x_{\rm V}$")
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.2))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.1))

ax = axes[0]
gm_plot = gm - gm.min()
ax.axvspan(sp_lo, sp_hi, color=SPINODAL_COLOR, alpha=SPINODAL_ALPHA, label="Spinodal")
ax.plot(x_v, gm_plot / 1e3, color="#1f4e79", lw=1.6)
ax.set_ylabel(r"$G_m - G_m^{\min}$  (kJ mol$^{-1}$)")
ax.set_title(r"Molar Gibbs Energy")
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))
ax.axvline(0.50, color="gray", lw=0.8, ls="--", alpha=0.7)
ax.legend(loc="upper right", frameon=True, framealpha=0.85, edgecolor="0.8")
ax.text(0.52, 0.97, r"$x_{\rm V}=0.50$",
        fontsize=7, color="gray", va="top", transform=ax.transAxes)

ax = axes[1]
ax.axvspan(sp_lo, sp_hi, color=SPINODAL_COLOR, alpha=SPINODAL_ALPHA, label="Spinodal")
ax.plot(x_v, mu_nb / 1e3, color="#1f4e79", lw=1.6, label=r"$\mu_{\rm Nb}$")
ax.plot(x_v, mu_v  / 1e3, color="#c00000", lw=1.6, label=r"$\mu_{\rm V}$", ls="--")
ax.set_ylabel(r"$\mu_i$  (kJ mol$^{-1}$)")
ax.set_title(r"Chemical Potentials")
ax.axvline(0.50, color="gray", lw=0.8, ls="--", alpha=0.7)
ax.legend(loc="lower right", frameon=True, framealpha=0.85, edgecolor="0.8")

ax = axes[2]
ax.axvspan(sp_lo, sp_hi, color=SPINODAL_COLOR, alpha=SPINODAL_ALPHA)
ax.fill_between(x_v, h_nb / 1e3, 0,
                where=spinodal_mask,
                color=SPINODAL_COLOR, alpha=0.4)
ax.plot(x_v, h_nb / 1e3, color="#375623", lw=1.6, label=r"$H_{\rm NbNb}$")
ax.axhline(0, color="black", lw=0.8, ls="-")
ax.axvline(0.50, color="gray", lw=0.8, ls="--", alpha=0.7)
ax.set_ylabel(r"$\partial\mu_{\rm Nb}/\partial x_{\rm Nb}$  (kJ mol$^{-1}$)")
ax.set_title(r"Hessian $\partial^2 G_m / \partial x_{\rm Nb}^2$")
ax.legend(loc="upper left", frameon=True, framealpha=0.85, edgecolor="0.8")
# restrict y to interior range (exclude near-boundary divergence)
x_mask = (x_v > 0.05) & (x_v < 0.95)
h_interior = h_nb[x_mask]
y_lo = h_interior.min() / 1e3 * 1.15
y_hi = max(h_interior.max() / 1e3 * 1.15, abs(y_lo) * 0.3)
ax.set_xlim(0.05, 0.95)
ax.set_ylim(y_lo, y_hi)

fig.suptitle(
    f"Nb–V  BCC$_{{\\rm A2}}$  |  $T = {T_C:.0f}\\,^{{\\circ}}$C  "
    f"|  {TDB_STEM}",
    fontsize=10, y=1.01)

OUT_DIR = _ROOT / "tests" / "results_thermo_NbV_500C"
OUT_DIR.mkdir(parents=True, exist_ok=True)

for ext in ("pdf", "png"):
    out = OUT_DIR / f"thermo_NbV_500C.{ext}"
    fig.savefig(out, bbox_inches="tight", dpi=300)
    print(f"Saved: {out}")

plt.close(fig)
print("Done.")
