"""runs/v3/fig2_cubic_anisotropy/run_fig2.py
==========================================
Replicates Fig. 2 of Gururajan & Abinandanan (2007):

    A circular precipitate evolves toward an equilibrium shape that depends
    on the Zener elastic anisotropy ratio A_Z = 2 C44 / (C11 - C12):

        A_Z = 1.0  →  isotropic  →  stays circular
        A_Z > 1.0  →  ⟨100⟩ soft →  square aligned with x/y axes
        A_Z < 1.0  →  ⟨110⟩ soft →  square rotated 45° (diamond)

    No applied stress. Shape change is driven purely by the elastic free
    energy anisotropy via mu_el (Eq. 36).

Parameter scaling
-----------------
    C11 = 100, C12 = 50  (fixed)
    C44 = A_Z * (C11 - C12) / 2 = 25 * A_Z
    eps_T = 0.1, A_b = 1.0  →  eta = C11 * eps_T^2 / A_b = 1.0
    No applied stress (sigma_app = 0)

Usage:
    python runs/v3/fig2_cubic_anisotropy/run_fig2.py

Output:
    runs/v3/fig2_cubic_anisotropy/fig2_morphology.pdf
    runs/v3/fig2_cubic_anisotropy/fig2_mu_el.pdf
    runs/v3/fig2_cubic_anisotropy/fig2_radial_profile.pdf
"""

import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from phasefoundry.elasticity import (
    build_cubic_stiffness,
    solve_elasticity,
)
from phasefoundry.elasticity.elastic_potential import compute_mu_el_from_result

OUT_DIR = Path(__file__).parent

N = 128
Lx = 128.0
R = 22.0
A_b = 1.0
kappa = 0.5
M = 0.05
dt = 0.05
C11 = 100.0
C12 = 50.0  # A_Z = 2*C44/(C11-C12) = 2*C44/50
eps_T = 0.1
N_STEPS = 8000
SAVE_EVERY = 1000

A_Z_VALUES = [0.5, 1.0, 2.0, 3.0]


def make_circle(N, R, Lx, width=2.0, c_matrix=0.05):
    """Circular precipitate (c=1 inside) in a slightly supersaturated matrix (c=c_matrix)."""
    x = np.linspace(0, Lx, N, endpoint=False)
    xx, yy = np.meshgrid(x, x, indexing="ij")
    r = np.sqrt((xx - Lx / 2) ** 2 + (yy - Lx / 2) ** 2)
    c_ppt = 0.5 * (1.0 - np.tanh((r - R) / width))  # 1 inside, 0 outside
    return np.clip(c_matrix + (1.0 - c_matrix) * c_ppt, 0.0, 1.0)


def radial_profile(c, level=0.5):
    """Extract r(theta) of the c=level contour using matplotlib.contour
    for smooth, sub-pixel accurate interface positions.
    Returns (thetas, r_vals) sorted by angle, covering [0, 2*pi).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    N = c.shape[0]
    cx = cy = N / 2.0
    fig_tmp, ax_tmp = plt.subplots()
    cs = ax_tmp.contour(c.T, levels=[level])
    plt.close(fig_tmp)

    pts = []
    for path in cs.get_paths():
        verts = path.vertices  # (n, 2) in pixel coords
        pts.append(verts)
    if not pts:
        return np.array([]), np.array([])

    pts = np.vstack(pts)
    dx = pts[:, 0] - cx
    dy = pts[:, 1] - cy
    th = np.arctan2(dy, dx)  # [-pi, pi]
    th = np.where(th < 0, th + 2 * np.pi, th)  # [0, 2*pi]
    r = np.sqrt(dx**2 + dy**2)

    order = np.argsort(th)
    return th[order], r[order]


def run_case(A_Z):
    C44 = A_Z * (C11 - C12) / 2.0
    print(f"\n  A_Z = {A_Z:.1f}  →  C44 = {C44:.1f}")

    C_mat = build_cubic_stiffness(C11, C12, C44)
    C_ppt = build_cubic_stiffness(C11, C12, C44)  # same elastic constants, only eigenstrain differs
    eig = eps_T * np.eye(3)

    # spectral setup
    freqs = np.fft.fftfreq(N, d=Lx / N)
    kv = 2.0 * np.pi * freqs
    kx, ky = np.meshgrid(kv, kv, indexing="ij")
    k2 = kx**2 + ky**2
    f_pp = 2.0 * A_b
    denom = 1.0 + dt * M * (f_pp * k2 + kappa * k2**2)

    # Scale supersaturation with A_Z to compensate higher elastic energy density
    # at large C44: W_el ~ C44*eps_T^2 ~ 25*A_Z*0.01, so c_matrix ~ 0.05 + 0.05*A_Z
    c_mat = min(0.05 + 0.05 * A_Z, 0.3)
    c = make_circle(N, R, Lx, c_matrix=c_mat)
    print(f"    c_matrix = {c_mat:.3f}")
    u_prev = None
    snaps = {}

    for step in range(N_STEPS + 1):
        el = solve_elasticity(c, C_mat, C_ppt, eig, Lx=Lx, tol=1e-6, max_iter=500, u_init=u_prev)
        u_prev = el["displacement"]
        mu_el = compute_mu_el_from_result(c, el, C_mat, C_ppt, eig)

        if step % SAVE_EVERY == 0 or step == N_STEPS:
            snaps[step] = dict(c=c.copy(), mu_el=mu_el.copy())
            print(f"    step {step:5d}  |mu_el|_max={np.abs(mu_el).max():.3e}")

        if step == N_STEPS:
            break

        # CH update (Eyre convex-splitting)
        mu_ch = A_b * (2 * c * (1 - c) ** 2 - 2 * c**2 * (1 - c))
        mu_exp = (mu_ch - f_pp * c) + mu_el
        c_hat = np.fft.fftn(c)
        c = np.clip(np.real(np.fft.ifftn((c_hat - dt * M * k2 * np.fft.fftn(mu_exp)) / denom)), 0.0, 1.0)

    return snaps


print("=" * 60)
print("G&A (2007) Fig. 2 — Cubic elastic anisotropy")
print(f"C11={C11}  C12={C12}  eps_T={eps_T}  eta={C11 * eps_T**2 / A_b:.2f}")
print("=" * 60)

results = {}
for az in A_Z_VALUES:
    results[az] = run_case(az)

fig, axes = plt.subplots(1, len(A_Z_VALUES), figsize=(4 * len(A_Z_VALUES), 4.2))
for ax, az in zip(axes, A_Z_VALUES):
    snap = results[az][N_STEPS]
    ax.imshow(snap["c"].T, origin="lower", cmap="RdBu_r", vmin=0, vmax=1, aspect="equal")
    ax.set_title(f"$A_Z = {az}$", fontsize=10)
    ax.axis("off")

fig.suptitle(
    f"Equilibrium precipitate shape vs Zener anisotropy $A_Z$\n"
    f"(step {N_STEPS}, $\\eta = C_{{11}}\\varepsilon_T^2/A_b = {C11 * eps_T**2 / A_b:.1f}$, "
    r"no applied stress)",
    fontsize=10,
)
fig.tight_layout()
out = OUT_DIR / "fig2_morphology.pdf"
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"\nSaved: {out}")

fig, axes = plt.subplots(1, len(A_Z_VALUES), figsize=(4 * len(A_Z_VALUES), 4.2))
for ax, az in zip(axes, A_Z_VALUES):
    snap = results[az][N_STEPS]
    mel = snap["mu_el"]
    lim = max(np.abs(mel).max(), 1e-12)
    im = ax.imshow(mel.T, origin="lower", cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="equal")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(f"$A_Z = {az}$", fontsize=10)
    ax.axis("off")

fig.suptitle(
    r"Elastic chemical potential $\mu_{el}$ at final step — "
    "anisotropy drives shape change",
    fontsize=10,
)
fig.tight_layout()
out = OUT_DIR / "fig2_mu_el.pdf"
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved: {out}")

# Shows shape change: circle has flat r(θ), square has 4-fold variation.
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

ax_pol = axes[0]
for az in A_Z_VALUES:
    th, rv = radial_profile(results[az][N_STEPS]["c"])
    mask = ~np.isnan(rv)
    ax_pol.plot(np.degrees(th[mask]), rv[mask], label=f"$A_Z = {az}$")
ax_pol.set_xlabel("Angle θ (degrees)")
ax_pol.set_ylabel("Interface radius r(θ) (grid units)")
ax_pol.set_title("Interface shape: r(θ) at final step")
ax_pol.set_xticks([0, 45, 90, 135, 180, 225, 270, 315, 360])
ax_pol.legend(fontsize=8)
ax_pol.grid(True, alpha=0.3)

# same data in first quadrant only (0–90°) for clarity
ax_q1 = axes[1]
for az in A_Z_VALUES:
    th, rv = radial_profile(results[az][N_STEPS]["c"])
    mask = (~np.isnan(rv)) & (th <= np.pi / 2)
    ax_q1.plot(np.degrees(th[mask]), rv[mask], "-o", ms=3, label=f"$A_Z = {az}$")
ax_q1.axvline(45, color="k", lw=0.8, ls="--", alpha=0.5, label="45°")
ax_q1.set_xlabel("Angle θ (degrees)")
ax_q1.set_ylabel("Interface radius r(θ) (grid units)")
ax_q1.set_title("First quadrant (0–90°)")
ax_q1.set_xticks([0, 15, 30, 45, 60, 75, 90])
ax_q1.legend(fontsize=8)
ax_q1.grid(True, alpha=0.3)

fig.suptitle(
    r"Interface shape $r(\theta)$ — "
    r"$A_Z>1$: peak at 0°/90° (square), $A_Z<1$: peak at 45° (diamond)",
    fontsize=9,
)
fig.tight_layout()
out = OUT_DIR / "fig2_radial_profile.pdf"
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved: {out}")
