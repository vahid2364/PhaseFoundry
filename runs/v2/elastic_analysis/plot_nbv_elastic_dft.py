"""runs/v2/elastic_analysis/plot_nbv_elastic_dft.py
================================
Interpret SQS/DFT elastic constants for the Nb-V binary extracted from
the HfNbVTi quaternary dataset (PhaseForge v2).

Plots
-----
1. Elastic stiffness constants C11, C12, C44 vs x_Nb
2. Derived moduli: G_VRH, bulk modulus, Poisson's ratio vs x_Nb
3. Zener anisotropy ratio A_Z = 2C44/(C11-C12) vs x_Nb
4. Lattice parameter a(c) and Vegard eigenstrain eps*(c) vs x_Nb
5. Mechanical stability indicators: C44 > 0, C11 > C12, C11+2C12 > 0
6. Table of key compositions for PFM

Usage:
    python runs/v2/elastic_analysis/plot_nbv_elastic_dft.py

Output:
    runs/v2/elastic_analysis/nbv_elastic_dft.pdf
"""

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

CSV = Path(__file__).resolve().parents[2] / "data/HfNbTiV_BCC_PhaseForge_v2/sqs_elastic_constants_HfNbVTi-orb-v2.csv"
OUT = Path(__file__).parent / "nbv_elastic_dft.pdf"

df = pd.read_csv(CSV)
nbv = df[(df.x_Hf == 0) & (df.x_Ti == 0)].sort_values("x_Nb").reset_index(drop=True)

x = nbv["x_Nb"].values  # independent axis: Nb mole fraction
C11 = nbv["C11_GPa"].values
C12 = nbv["C12_GPa"].values
C44 = nbv["C44_GPa"].values
G = nbv["shear_modulus_vrh_GPa"].values
B = nbv["bulk_modulus_vrh_GPa"].values
nu = nbv["poisson_ratio"].values
a = nbv["a_relaxed_A"].values

A_Z = 2 * C44 / (C11 - C12)  # Zener anisotropy
a_Nb = a[-1]  # reference: pure Nb (x_Nb=1)
eps_star = (a - a_Nb) / a_Nb  # Vegard eigenstrain relative to Nb

# Mechanical stability criteria (Born criteria for cubic)
stable_C44 = C44 > 0
stable_C11_C12 = C11 > C12
stable_bulk = (C11 + 2 * C12) > 0

fig = plt.figure(figsize=(14, 12))
gs = GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

mk = dict(marker="o", ms=5, lw=1.5)

# Panel 1 — stiffness constants
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(x, C11, "b-", label="$C_{11}$", **mk)
ax1.plot(x, C12, "r-", label="$C_{12}$", **mk)
ax1.plot(x, C44, "g-", label="$C_{44}$", **mk)
ax1.axhline(0, color="k", lw=0.8, ls="--", alpha=0.4)
ax1.fill_between(x, C44, 0, where=(C44 < 0), color="red", alpha=0.12, label="$C_{44}<0$ (unstable)")
ax1.set_xlabel("$x_{\\rm Nb}$")
ax1.set_ylabel("Elastic constant (GPa)")
ax1.set_title("Stiffness constants $C_{ij}$ vs composition")
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.3)

# Panel 2 — derived moduli
ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(x, G, "m-", label="$G_{\\rm VRH}$", **mk)
ax2.plot(x, B, "c-", label="$B_{\\rm VRH}$", **mk)
ax2.axhline(0, color="k", lw=0.8, ls="--", alpha=0.4)
ax2.fill_between(x, G, 0, where=(G < 0), color="red", alpha=0.12, label="$G<0$ (unstable)")
ax2b = ax2.twinx()
ax2b.plot(x, nu, "k--", label="$\\nu$", **mk)
ax2b.set_ylabel("Poisson's ratio $\\nu$", color="k")
ax2b.set_ylim(0, 0.8)
ax2.set_xlabel("$x_{\\rm Nb}$")
ax2.set_ylabel("Modulus (GPa)")
ax2.set_title("Derived moduli and Poisson's ratio")
lines1, labs1 = ax2.get_legend_handles_labels()
lines2, labs2 = ax2b.get_legend_handles_labels()
ax2.legend(lines1 + lines2, labs1 + labs2, fontsize=8)
ax2.grid(True, alpha=0.3)

# Panel 3 — Zener anisotropy
ax3 = fig.add_subplot(gs[1, 0])
ax3.plot(x, A_Z, "darkorange", **mk)
ax3.axhline(1.0, color="k", lw=1.0, ls="--", label="Isotropic ($A_Z=1$)")
ax3.axhline(0.0, color="r", lw=0.8, ls=":", alpha=0.6, label="$A_Z=0$ (C44=0)")
ax3.fill_between(x, A_Z, 1.0, where=(A_Z < 1.0), color="blue", alpha=0.10, label="$A_Z<1$: diamond morphology")
ax3.fill_between(x, A_Z, 1.0, where=(A_Z > 1.0), color="green", alpha=0.10, label="$A_Z>1$: square morphology")
ax3.set_xlabel("$x_{\\rm Nb}$")
ax3.set_ylabel("Zener anisotropy $A_Z = 2C_{44}/(C_{11}-C_{12})$")
ax3.set_title("Elastic anisotropy — shape selection")
ax3.legend(fontsize=7)
ax3.grid(True, alpha=0.3)
ax3.set_ylim(-0.7, 0.8)

# Panel 4 — lattice parameter and Vegard eigenstrain
ax4 = fig.add_subplot(gs[1, 1])
ax4.plot(x, a, "b-", **mk, label="$a$ (DFT)")
p = np.polyfit(x, a, 1)
ax4.plot(x, np.polyval(p, x), "b--", lw=1, alpha=0.5, label=f"Vegard fit: $a={p[0]:.4f}\\,x_{{Nb}}+{p[1]:.4f}$")
ax4.set_xlabel("$x_{\\rm Nb}$")
ax4.set_ylabel("Lattice parameter $a$ (Å)", color="b")
ax4.tick_params(axis="y", colors="b")
ax4b = ax4.twinx()
ax4b.plot(x, eps_star * 100, "r-", **mk, label=r"$\varepsilon^*(c)$ rel. Nb")
ax4b.axhline(0, color="r", lw=0.8, ls="--", alpha=0.4)
ax4b.set_ylabel(r"Vegard eigenstrain $\varepsilon^*$ (%)", color="r")
ax4b.tick_params(axis="y", colors="r")
ax4.set_title("Lattice parameter and Vegard eigenstrain")
lines1, labs1 = ax4.get_legend_handles_labels()
lines2, labs2 = ax4b.get_legend_handles_labels()
ax4.legend(lines1 + lines2, labs1 + labs2, fontsize=7)
ax4.grid(True, alpha=0.3)

# Panel 5 — mechanical stability map
ax5 = fig.add_subplot(gs[2, 0])
ax5.fill_between(x, 0, 1, where=stable_C44, step="mid", color="green", alpha=0.25, label="$C_{44}>0$")
ax5.fill_between(x, 0, 1, where=~stable_C44, step="mid", color="red", alpha=0.25, label="$C_{44}\\leq 0$ (unstable)")
ax5.fill_between(x, 1, 2, where=stable_C11_C12, step="mid", color="green", alpha=0.25, label="$C_{11}>C_{12}$")
ax5.fill_between(x, 1, 2, where=~stable_C11_C12, step="mid", color="red", alpha=0.25)
ax5.fill_between(x, 2, 3, where=stable_bulk, step="mid", color="green", alpha=0.25, label="$C_{11}+2C_{12}>0$")
ax5.fill_between(x, 2, 3, where=~stable_bulk, step="mid", color="red", alpha=0.25)
ax5.set_yticks([0.5, 1.5, 2.5], ["$C_{44}>0$", "$C_{11}>C_{12}$", "$C_{11}+2C_{12}>0$"])
ax5.set_xlabel("$x_{\\rm Nb}$")
ax5.set_title("Born mechanical stability criteria")
ax5.set_xlim(0, 1)
ax5.set_ylim(0, 3)
handles = [
    plt.Rectangle((0, 0), 1, 1, color="green", alpha=0.4, label="Stable"),
    plt.Rectangle((0, 0), 1, 1, color="red", alpha=0.4, label="Unstable"),
]
ax5.legend(handles=handles, fontsize=8)
ax5.grid(True, alpha=0.3, axis="x")

# Panel 6 — summary table of PFM-relevant values
ax6 = fig.add_subplot(gs[2, 1])
ax6.axis("off")
# Key compositions for PFM: endpoints and mean
rows = []
for xn in [1.0, 0.8, 0.6, 0.4, 0.2, 0.1, 0.0]:
    row = nbv[np.isclose(nbv["x_Nb"], xn, atol=0.01)]
    if len(row):
        r = row.iloc[0]
        az = 2 * r.C44_GPa / (r.C11_GPa - r.C12_GPa)
        rows.append(
            [
                f"{r.x_Nb:.2f}",
                f"{r.x_V:.2f}",
                f"{r.a_relaxed_A:.4f}",
                f"{r.C11_GPa:.1f}",
                f"{r.C12_GPa:.1f}",
                f"{r.C44_GPa:.1f}",
                f"{r.shear_modulus_vrh_GPa:.1f}",
                f"{az:.3f}",
            ]
        )

cols = ["$x_{Nb}$", "$x_V$", "$a$ (Å)", "$C_{11}$", "$C_{12}$", "$C_{44}$", "$G_{VRH}$", "$A_Z$"]
tbl = ax6.table(cellText=rows, colLabels=cols, loc="center", cellLoc="center")
tbl.auto_set_font_size(False)
tbl.set_fontsize(7.5)
tbl.scale(1, 1.35)
ax6.set_title("Key compositions for PFM (moduli in GPa)", fontsize=9, pad=12)

fig.suptitle(
    "Nb-V BCC: SQS/DFT Elastic Constants (PhaseForge v2)\n"
    r"$A_Z < 1$ throughout stable range → spinodal morphology favors "
    r"diamond (⟨100⟩ corner) | $C_{44} < 0$ for $x_V \gtrsim 0.9$: BCC V mechanically unstable",
    fontsize=10,
)

fig.savefig(OUT, dpi=150, bbox_inches="tight")
print(f"Saved: {OUT}")

print("\n── PFM Option A endpoint values ────────────────────────────────────")
for label, xn in [("C_mat  (x_Nb=1.0, Nb-rich)", 1.0), ("C_ppt  (x_Nb=0.1, V-rich, C44>0)", 0.1)]:
    row = nbv[np.isclose(nbv["x_Nb"], xn, atol=0.01)].iloc[0]
    az = 2 * row.C44_GPa / (row.C11_GPa - row.C12_GPa)
    print(f"  {label}")
    print(f"    a={row.a_relaxed_A:.4f} Å  C11={row.C11_GPa:.1f}  C12={row.C12_GPa:.1f}  C44={row.C44_GPa:.1f} GPa  A_Z={az:.3f}")
eps = (
    nbv[np.isclose(nbv["x_Nb"], 0.1, atol=0.01)].iloc[0].a_relaxed_A
    - nbv[np.isclose(nbv["x_Nb"], 1.0, atol=0.01)].iloc[0].a_relaxed_A
) / nbv[np.isclose(nbv["x_Nb"], 1.0, atol=0.01)].iloc[0].a_relaxed_A
print(f"\n  Vegard eps_T (V-rich vs Nb): {eps:.5f} ({eps * 100:.2f}%)")
print("  (use a_matrix=3.3246, a_ppt=3.0390 in config for DFT-accurate eigenstrain)")
