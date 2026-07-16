"""Generate the PhaseFoundry pipeline schematic (Figure 1 of the software paper)."""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

INPUT = "#dbe9f6"   # light blue: external inputs
STAGE = "#fdf2d0"   # light amber: pipeline stages
DATA = "#e8e8e8"    # gray: data products
OUT = "#ddebdd"     # light green: outputs
EDGE = "#555555"


def box(ax, x, y, w, h, text, fc, fs=8.0):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012",
                                fc=fc, ec=EDGE, lw=0.8))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)


def arrow(ax, xy0, xy1, style="-|>", color=EDGE, lw=1.1, rad=0.0):
    ax.add_patch(FancyArrowPatch(xy0, xy1, arrowstyle=style, color=color,
                                 lw=lw, mutation_scale=11,
                                 connectionstyle=f"arc3,rad={rad}"))


fig, ax = plt.subplots(figsize=(7.4, 4.1))
ax.set_xlim(0, 10)
ax.set_ylim(0, 5.6)
ax.axis("off")

# Row 1: inputs
box(ax, 0.2, 4.7, 2.6, 0.7, "CALPHAD TDB\n(PhaseForge / assessed)", INPUT)
box(ax, 3.7, 4.7, 2.6, 0.7, "SQS/MLIP elastic tables\n$C_{ij}(\\mathbf{x})$, $a(\\mathbf{x})$", INPUT)
box(ax, 7.2, 4.7, 2.6, 0.7, "YAML configuration\n(system, grid, solver, elasticity)", INPUT)

# Row 2: sampling stages
box(ax, 0.2, 3.4, 2.6, 0.8,
    "Step 1 — pycalphad sampling\n$G_m$, $\\mu_i-\\mu_n$, Jansson $H_{ij}$", STAGE)
box(ax, 3.7, 3.4, 2.6, 0.8,
    "Steps 1b/1c — elastic lookup\n+ CALPHAD tie-line endpoints", STAGE)

# Row 3: surrogate
box(ax, 0.2, 2.1, 2.6, 0.8,
    "HDF5 thermodynamic\ntables (per $T$)", DATA)
box(ax, 3.7, 2.1, 2.6, 0.8,
    "Step 2 — surrogate interpolant\n+ gauge-invariant spinodal map", STAGE)

# Row 4: solver
box(ax, 3.7, 0.65, 2.6, 0.95,
    "Step 3 — Cahn–Hilliard solver\npseudo-spectral, Eyre-stabilized\n(CPU / GPU)", STAGE)
box(ax, 7.2, 0.65, 2.6, 0.95,
    "Moulinec–Suquet\nelasticity solve\n$\\mu^{\\rm el}_k$ every $n_{\\rm el}$ steps", STAGE)

# Row 5: outputs
box(ax, 0.2, 0.65, 2.6, 0.95,
    "Steps 4–6 — post-processing,\nderivative validation,\nchemical vs. elasto-chemical", OUT)

# Arrows
arrow(ax, (1.5, 4.7), (1.5, 4.2))          # TDB -> step1
arrow(ax, (5.0, 4.7), (5.0, 4.2))          # elastic csv -> step1b
arrow(ax, (8.5, 4.7), (8.5, 1.6))          # config -> solver
arrow(ax, (1.5, 3.4), (1.5, 2.9))          # step1 -> hdf5
arrow(ax, (2.8, 2.5), (3.7, 2.5))          # hdf5 -> step2
arrow(ax, (5.0, 2.1), (5.0, 1.6))          # step2 -> step3
arrow(ax, (5.6, 3.4), (5.6, 2.9))          # step1b -> step2 (endpoints)
arrow(ax, (6.3, 1.28), (7.2, 1.28))        # step3 <-> elasticity
arrow(ax, (7.2, 1.02), (6.3, 1.02))
arrow(ax, (3.7, 1.12), (2.8, 1.12))        # step3 -> post
fig.tight_layout()
fig.savefig("figs/pipeline.pdf", bbox_inches="tight")
print("wrote figs/pipeline.pdf")
