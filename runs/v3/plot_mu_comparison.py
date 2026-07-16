"""
runs/v3/plot_mu_comparison.py
==============================
Run 100 CH + elastic steps from a circular precipitate initial condition
and plot the chemical potential (mu_ch) and elastic potential (mu_el)
side-by-side at steps 0, 50, and 100.

Usage:
    python runs/v3/plot_mu_comparison.py

Output:
    runs/v3/mu_comparison.pdf
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from phasefoundry.elasticity import solve_elasticity, build_isotropic_stiffness
from phasefoundry.elasticity.elastic_potential import compute_mu_el_from_result

# ── parameters ────────────────────────────────────────────────────────────────
N        = 128
Lx       = 128.0
R        = 22.0
A_b      = 1.0
kappa    = 0.5
M        = 0.05
dt       = 0.05
G_matrix = 100.0
nu       = 0.3
eps_T    = 0.1
delta    = 0.5          # soft precipitate
sigma_11 = 0.0          # elastic force disabled (zero applied stress)

SNAP_STEPS = [0, 1000, 2500, 5000]

# ── initial composition ───────────────────────────────────────────────────────
x  = np.linspace(0, Lx, N, endpoint=False)
xx, yy = np.meshgrid(x, x, indexing="ij")
r  = np.sqrt((xx - Lx/2)**2 + (yy - Lx/2)**2)
c  = np.clip(0.5 * (1.0 - np.tanh((r - R) / 2.0)), 0.0, 1.0)

# ── elasticity setup ─────────────────────────────────────────────────────────
C_mat = build_isotropic_stiffness(G_matrix, nu)
C_ppt = build_isotropic_stiffness(delta * G_matrix, nu)
eig   = eps_T * np.eye(3)

sigma_applied          = np.zeros((2, 2))
sigma_applied[0, 0]    = sigma_11

# ── spectral CH setup ─────────────────────────────────────────────────────────
freqs    = np.fft.fftfreq(N, d=Lx / N)
kv       = 2.0 * np.pi * freqs
kx, ky   = np.meshgrid(kv, kv, indexing="ij")
k2       = kx**2 + ky**2

f_pp  = 2.0 * A_b
denom = 1.0 + dt * M * (f_pp * k2 + kappa * k2**2)

# ── storage ───────────────────────────────────────────────────────────────────
snaps = {}   # step -> {"c": ..., "mu_ch": ..., "mu_el": ...}

# ── time integration ──────────────────────────────────────────────────────────
u_prev = None
for step in range(5001):
    mu_el = np.zeros_like(c)   # elastic disabled

    # chemical potential
    mu_ch = A_b * (2*c*(1-c)**2 - 2*c**2*(1-c))

    # gradient energy term: -κ ∇²c  (spectral)
    mu_grad = -kappa * np.real(np.fft.ifftn(-k2 * np.fft.fftn(c)))

    if step in SNAP_STEPS:
        snaps[step] = dict(c=c.copy(), mu_ch=mu_ch.copy(),
                           mu_el=mu_el.copy(), mu_grad=mu_grad.copy())
        print(f"step {step:3d}  |mu_ch|={np.abs(mu_ch).max():.3e}"
              f"  |mu_grad|={np.abs(mu_grad).max():.3e}"
              f"  |mu_el|={np.abs(mu_el).max():.3e}"
              f"  el/ch={np.abs(mu_el).max()/np.abs(mu_ch).max():.3f}")

    if step == 5000:
        break

    # Eyre convex-splitting CH update
    mu_exp = (mu_ch - f_pp * c) + mu_el
    c_hat  = np.fft.fftn(c)
    c = np.clip(np.real(np.fft.ifftn(
        (c_hat - dt * M * k2 * np.fft.fftn(mu_exp)) / denom
    )), 0.0, 1.0)

# ── plot ──────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 10))
# rows: step 0 / step 50 / step 100
# cols: c | mu_ch | mu_el | mu_el/mu_ch
col_labels = [r"Composition $c$",
              r"Chemical $\mu_{ch}$",
              r"Gradient $-\kappa\nabla^2 c$",
              r"Elastic $\mu_{el}$",
              r"Ratio $\mu_{el}/|\mu_{ch}|$"]
n_rows = len(SNAP_STEPS)
n_cols = 5
gs = GridSpec(n_rows, n_cols, figure=fig, hspace=0.4, wspace=0.35)

for row, step in enumerate(SNAP_STEPS):
    s    = snaps[step]
    c_s  = s["c"]
    mch  = s["mu_ch"]
    mgr  = s["mu_grad"]
    mel  = s["mu_el"]

    lim_ch  = max(np.abs(mch).max(), 1e-12)
    lim_gr  = max(np.abs(mgr).max(), 1e-12)
    lim_el  = max(np.abs(mel).max(), 1e-12)
    ratio   = mel / (np.abs(mch) + 1e-30)
    lim_rat = min(np.abs(ratio).max(), 5.0)

    datas  = [c_s, mch, mgr, mel, ratio]
    cmaps  = ["RdBu_r", "RdBu_r", "RdBu_r", "RdBu_r", "PiYG"]
    vlims  = [(0, 1), (-lim_ch, lim_ch), (-lim_gr, lim_gr),
              (-lim_el, lim_el), (-lim_rat, lim_rat)]

    for col in range(n_cols):
        ax = fig.add_subplot(gs[row, col])
        im = ax.imshow(datas[col].T, origin="lower", cmap=cmaps[col],
                       vmin=vlims[col][0], vmax=vlims[col][1], aspect="equal")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        if row == 0:
            ax.set_title(col_labels[col], fontsize=9)
        ax.set_ylabel(f"step {step}", fontsize=8) if col == 0 else None
        ax.set_xticks([]); ax.set_yticks([])

fig.suptitle(
    r"$\mu_{ch}$ vs $\mu_{el}$ — soft precipitate ($\delta=0.5$), "
    r"$\sigma_{11}=-1.50$ (+200%), $\eta=G\varepsilon_T^2/A_b=1$",
    fontsize=10
)

out = Path(__file__).parent / "mu_comparison.pdf"
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"\nSaved: {out}")
