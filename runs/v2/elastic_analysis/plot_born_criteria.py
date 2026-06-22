"""
runs/v2/elastic_analysis/plot_born_criteria.py
==============================
Visualize Born mechanical stability criteria for the Nb-V binary
extracted from the HfNbVTi SQS/DFT dataset (PhaseForge v2).

Born criteria for cubic crystals:
    (1)  C44 > 0              shear stability on {100} planes
    (2)  C11 - C12 > 0        tetragonal shear stability
    (3)  C11 + 2*C12 > 0      bulk (hydrostatic) stability

Each panel shows where the criterion is satisfied (green) or violated (red).
The combined stability region is marked on every panel.

Usage:
    python runs/v2/elastic_analysis/plot_born_criteria.py

Output:
    runs/v2/elastic_analysis/born_criteria_nbv.pdf
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

CSV = Path(__file__).resolve().parents[2] / \
      "data/HfNbTiV_BCC_PhaseForge_v2/sqs_elastic_constants_HfNbVTi-orb-v2.csv"
OUT = Path(__file__).parent / "born_criteria_nbv.pdf"

# ── load NbV binary ───────────────────────────────────────────────────────────
df  = pd.read_csv(CSV)
nbv = df[(df.x_Hf == 0) & (df.x_Ti == 0)].sort_values("x_Nb").reset_index(drop=True)

xNb = nbv["x_Nb"].values
xV  = nbv["x_V"].values
C11 = nbv["C11_GPa"].values
C12 = nbv["C12_GPa"].values
C44 = nbv["C44_GPa"].values
G   = nbv["shear_modulus_vrh_GPa"].values
B   = nbv["bulk_modulus_vrh_GPa"].values
nu  = nbv["poisson_ratio"].values

# Born criteria (pointwise)
crit1 = C44                    # > 0
crit2 = C11 - C12              # > 0
crit3 = C11 + 2 * C12         # > 0
all_stable = (crit1 > 0) & (crit2 > 0) & (crit3 > 0)

# find transition compositions
def first_violation(arr, stable_sign=1):
    """Return x_Nb of first point where stable_sign*arr <= 0."""
    for i, v in enumerate(arr):
        if stable_sign * v <= 0:
            return xNb[i]
    return None

x_c44_violation  = first_violation(crit1)
x_c11c12_violation = first_violation(crit2)

# ── figure ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 10))
gs  = GridSpec(2, 3, figure=fig, hspace=0.50, wspace=0.38)

COLORS = dict(stable="#2ca02c", unstable="#d62728", neutral="#888888")
mk = dict(marker="o", ms=6, lw=2.0, zorder=5)

def shade_regions(ax, x, criterion, ymin, ymax):
    """Shade green where criterion > 0, red where <= 0."""
    for i in range(len(x) - 1):
        col = COLORS["stable"] if criterion[i] > 0 else COLORS["unstable"]
        ax.axvspan(x[i], x[i+1], alpha=0.15, color=col, lw=0)
    # cap edges
    if criterion[0] > 0:
        ax.axvspan(0, x[0], alpha=0.15, color=COLORS["stable"], lw=0)
    else:
        ax.axvspan(0, x[0], alpha=0.15, color=COLORS["unstable"], lw=0)
    if criterion[-1] > 0:
        ax.axvspan(x[-1], 1, alpha=0.15, color=COLORS["stable"], lw=0)
    else:
        ax.axvspan(x[-1], 1, alpha=0.15, color=COLORS["unstable"], lw=0)

def add_legend(ax):
    p1 = mpatches.Patch(color=COLORS["stable"],   alpha=0.5, label="Stable")
    p2 = mpatches.Patch(color=COLORS["unstable"], alpha=0.5, label="Unstable / violated")
    ax.legend(handles=[p1, p2], fontsize=8, loc="best")

def mark_zero(ax):
    ax.axhline(0, color="k", lw=1.0, ls="--", alpha=0.6, zorder=3)

def set_axes(ax, ylabel, title):
    ax.set_xlabel("$x_{\\rm Nb}$ (Nb mole fraction)", fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_xlim(0, 1)
    ax.grid(True, alpha=0.25, zorder=0)
    ax.tick_params(labelsize=8)
    # secondary x-axis showing x_V
    ax2 = ax.twiny()
    ax2.set_xlim(1, 0)
    ax2.set_xlabel("$x_{\\rm V}$ (V mole fraction)", fontsize=8, color="#555555")
    ax2.tick_params(labelsize=7, colors="#555555")


# ─── Panel 1: Criterion 1 — C44 > 0 ─────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
shade_regions(ax1, xNb, crit1, C44.min(), C44.max())
ax1.plot(xNb, C44, color="#1f77b4", label="$C_{44}$", **mk)
mark_zero(ax1)
if x_c44_violation is not None:
    ax1.axvline(x_c44_violation, color=COLORS["unstable"], lw=1.5, ls=":",
                label=f"Violation at $x_{{Nb}}={x_c44_violation:.2f}$")
ax1.fill_between(xNb, C44, 0, where=(C44 < 0),
                 color=COLORS["unstable"], alpha=0.35, label="$C_{44} \\leq 0$")
ax1.text(0.04, 5, "UNSTABLE\n(shear collapse)", color=COLORS["unstable"],
         fontsize=7.5, ha="left", va="bottom", style="italic")
ax1.text(0.25, 20, "STABLE", color=COLORS["stable"],
         fontsize=7.5, ha="left", va="bottom", style="italic")
set_axes(ax1, "$C_{44}$ (GPa)", "Criterion 1:  $C_{44} > 0$\n(shear stability on {100})")
ax1.legend(fontsize=7, loc="upper left")


# ─── Panel 2: Criterion 2 — C11 - C12 > 0 ───────────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
c2_vals = crit2
shade_regions(ax2, xNb, c2_vals, c2_vals.min() - 5, c2_vals.max() + 5)
ax2.plot(xNb, c2_vals, color="#ff7f0e", label="$C_{11}-C_{12}$", **mk)
mark_zero(ax2)
set_axes(ax2, "$C_{11} - C_{12}$ (GPa)", "Criterion 2:  $C_{11} - C_{12} > 0$\n(tetragonal shear stability)")
ax2.text(0.5, (c2_vals.max()+c2_vals.min())/2, "STABLE\neverywhere",
         color=COLORS["stable"], fontsize=8, ha="center", style="italic")
add_legend(ax2)


# ─── Panel 3: Criterion 3 — C11 + 2·C12 > 0 ─────────────────────────────────
ax3 = fig.add_subplot(gs[0, 2])
c3_vals = crit3
shade_regions(ax3, xNb, c3_vals, c3_vals.min() - 5, c3_vals.max() + 5)
ax3.plot(xNb, c3_vals, color="#9467bd", label="$C_{11}+2C_{12}$", **mk)
mark_zero(ax3)
set_axes(ax3, "$C_{11} + 2C_{12}$ (GPa)", "Criterion 3:  $C_{11}+2C_{12} > 0$\n(bulk / hydrostatic stability)")
ax3.text(0.5, c3_vals.mean(), "STABLE\neverywhere",
         color=COLORS["stable"], fontsize=8, ha="center", style="italic")
add_legend(ax3)


# ─── Panel 4: All three criteria together ────────────────────────────────────
ax4 = fig.add_subplot(gs[1, 0])
# normalise each criterion to [0,1] for overlay
def norm(arr):
    r = arr.max() - arr.min()
    return (arr - arr.min()) / r if r > 0 else arr * 0 + 0.5

ax4.plot(xNb, norm(crit1), color="#1f77b4", label="$C_{44}$ (norm.)",       **mk)
ax4.plot(xNb, norm(crit2), color="#ff7f0e", label="$C_{11}-C_{12}$ (norm.)", **mk)
ax4.plot(xNb, norm(crit3), color="#9467bd", label="$C_{11}+2C_{12}$ (norm.)",**mk)
# shade overall stability
for i in range(len(xNb) - 1):
    col = COLORS["stable"] if all_stable[i] else COLORS["unstable"]
    ax4.axvspan(xNb[i], xNb[i+1], alpha=0.18, color=col, lw=0)
ax4.axhline(0, color="k", lw=1.0, ls="--", alpha=0.5)
set_axes(ax4, "Criterion value (normalized)", "All Born criteria\n(combined stability)")
ax4.legend(fontsize=7, loc="lower right")
add_legend(ax4)
ax4.set_ylim(-0.05, 1.05)


# ─── Panel 5: Mismatch — where PFM model is valid ────────────────────────────
ax5 = fig.add_subplot(gs[1, 1])
# Show C11, C12, C44 together with stability shading
shade_regions(ax5, xNb, crit1, -30, 290)  # shade by C44 criterion (most restrictive)
ax5.plot(xNb, C11, "b-",  label="$C_{11}$", **mk)
ax5.plot(xNb, C12, "r-",  label="$C_{12}$", **mk)
ax5.plot(xNb, C44, "g-",  label="$C_{44}$", **mk)
mark_zero(ax5)

# mark c0=0.6 (mean composition) and PFM endpoints
for xmark, label, col in [(0.6, "$x_{Nb}=0.6$\n(mean $c_0$)", "navy"),
                           (0.1, "$x_{Nb}=0.1$\n(C_ppt)", "darkgreen"),
                           (1.0, "$x_{Nb}=1.0$\n(C_mat)", "darkblue")]:
    ax5.axvline(xmark, color=col, lw=1.2, ls="-.", alpha=0.8)
    ax5.text(xmark + 0.01, 230, label, color=col, fontsize=6.5, va="top")

ax5.fill_between(xNb, C44, 0, where=(C44 < 0),
                 color=COLORS["unstable"], alpha=0.4, label="$C_{44}\\leq 0$: avoid")
set_axes(ax5, "Elastic constant (GPa)", "Stiffness constants\nwith stability and PFM points")
ax5.legend(fontsize=7, loc="upper right")
ax5.set_ylim(-35, 295)


# ─── Panel 6: Summary — stability diagram as a function of x_V ──────────────
ax6 = fig.add_subplot(gs[1, 2])

# three horizontal bands, one per criterion
bands = [
    (crit1, "C44 > 0\n(shear)", 0),
    (crit2, "C11−C12 > 0\n(tetragonal)", 1),
    (crit3, "C11+2C12 > 0\n(bulk)", 2),
]
for arr, label, row in bands:
    for i in range(len(xNb)):
        col = COLORS["stable"] if arr[i] > 0 else COLORS["unstable"]
        ax6.barh(row, 1/len(xNb), left=xNb[i], height=0.7,
                 color=col, alpha=0.75, edgecolor="none")

ax6.set_yticks([0, 1, 2])
ax6.set_yticklabels(["$C_{44}>0$\n(shear)", "$C_{11}-C_{12}>0$\n(tetragonal)",
                     "$C_{11}+2C_{12}>0$\n(bulk)"], fontsize=8)
ax6.set_xlabel("$x_{\\rm Nb}$ →  |  ←  $x_{\\rm V}$", fontsize=9)
ax6.set_xlim(0, 1)
ax6.set_title("Born Stability Map\n(red = violated, green = satisfied)", fontsize=10, fontweight="bold")
ax6.axvline(0.1, color="k", lw=1.5, ls="--", label="$x_{Nb}=0.1$ (C_ppt cutoff)")
ax6.text(0.10, 2.55, "← avoid", fontsize=7, color="red", ha="right")
ax6.text(0.11, 2.55, "safe →", fontsize=7, color=COLORS["stable"])
ax6.legend(fontsize=7)
ax6.grid(True, axis="x", alpha=0.25)
add_legend(ax6)


# ── global title ─────────────────────────────────────────────────────────────
fig.suptitle(
    "Nb-V BCC — Born Mechanical Stability Criteria (SQS/DFT, PhaseForge v2)\n"
    "Only $C_{44}>0$ is violated, for $x_{\\rm V} \\gtrsim 0.90$ ($x_{\\rm Nb} \\lesssim 0.10$). "
    "Use $x_{\\rm Nb}=0.10$ as safe PFM precipitate endpoint.",
    fontsize=11,
)

fig.savefig(OUT, dpi=150, bbox_inches="tight")
print(f"Saved: {OUT}")

# ── console summary ───────────────────────────────────────────────────────────
print("\nBorn criteria check for NbV binary:")
print(f"{'x_Nb':>6} {'x_V':>6} {'C44':>8} {'C11-C12':>10} {'C11+2C12':>11}  "
      f"{'C44>0':>6} {'C11>C12':>8} {'Bulk>0':>7} {'ALL':>5}")
print("-" * 75)
for i in range(len(xNb)):
    ok1 = "YES" if crit1[i] > 0 else "NO "
    ok2 = "YES" if crit2[i] > 0 else "NO "
    ok3 = "YES" if crit3[i] > 0 else "NO "
    ok  = "OK" if all_stable[i] else "FAIL"
    print(f"{xNb[i]:6.2f} {xV[i]:6.2f} {crit1[i]:8.1f} {crit2[i]:10.1f} "
          f"{crit3[i]:11.1f}  {ok1:>6} {ok2:>8} {ok3:>7} {ok:>5}")
