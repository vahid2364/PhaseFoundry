"""runs/v3/test_elasticity_circular_precipitate.py
================================================
Benchmark: replicate Fig. 1 of Gururajan & Abinandanan (2007).

Two sub-tests
-------------
1. δ = 1  (homogeneous inclusion, G^p = G^m):
   Exact Eshelby analytic solution applies — used as the primary validation.

2. δ = 0.5  (soft inhomogeneous precipitate, G^p = 0.5 G^m):
   Reproduces the physical setup of Fig. 1 in the paper.
   Analytic reference from the Mura equivalent-inclusion method.

Analytical solution
-------------------
All formulae are plane-strain (ε₃₃ = 0).

For a circular inhomogeneity with eigenstrain ε* = ε₀ I  (dilatational),
zero far-field stress, the equivalent-inclusion method (Mura 1987) gives:

  Eshelby tensor for circle (plane strain, isotropic):
      S₁₁₁₁ = S₂₂₂₂ = (5 - 4ν) / [8(1-ν)]
      S₁₁₂₂ = S₂₂₁₁ = (4ν - 1) / [8(1-ν)]

  For δ = 1 (homogeneous):
      σ₁₁^in = σ₂₂^in = -σ₀  where  σ₀ = G^m ε₀ / (1-ν)
      Outside along x-axis:
          σ₁₁ = -σ₀ (R/r)²
          σ₂₂ = +σ₀ (R/r)²

  For δ ≠ 1 (inhomogeneous), the equivalent inclusion replaces ε* by ε**:
      ε** = C_hom^{-1} C_ppt ε*  (scalar for dilatational, isotropic case)
      where C_hom corresponds to the equivalent homogeneous matrix problem.

Usage
-----
    python runs/v3/test_elasticity_circular_precipitate.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from phasefoundry.elasticity import build_isotropic_stiffness, solve_elasticity

N = 512  # grid points per axis — matches the paper exactly (Lx=Ly=512)
Lx = 512.0  # domain size — matches the paper exactly (Δx=Δy=1)
R = 25.8  # precipitate radius — matches the paper exactly

nu = 0.3  # Poisson's ratio
Gm = 1.0  # matrix shear modulus (dimensionless)
eps0 = 0.01  # dilatational misfit strain
eigenstrain = eps0 * np.eye(3)

# Normalisation matching the paper exactly (G&A 2007, Fig. 1 caption):
#   "the stress normalized by the characteristic stress G^m ε^T"
sigma0_paper = Gm * eps0
# Our own derivation normalization (used for the δ=1 sanity check below):
#   σ₀_eshelby = G^m ε₀ / (1-ν);  σ₀_paper = (1-ν) * σ₀_eshelby
sigma0 = Gm * eps0 / (1.0 - nu)
print(f"σ₀ (G&A convention, = G^m ε^T)  = {sigma0_paper:.6f}")
print(f"σ₀ (Eshelby convention, /(1-ν)) = {sigma0:.6f}")


def make_circle(N, R, Lx):
    x = np.linspace(0, Lx, N, endpoint=False)
    xx, yy = np.meshgrid(x, x, indexing="ij")
    cx, cy = Lx / 2.0, Lx / 2.0
    return ((xx - cx) ** 2 + (yy - cy) ** 2 <= R**2).astype(np.float64)


c = make_circle(N, R, Lx)
print(f"Grid: {N}×{N},  R = {R:.5f},  f_ppt = {c.mean():.5f}  (target π R²/L² = {np.pi * R**2 / Lx**2:.5f})")


# Eshelby tensor for a circular cylinder in plane strain (Qu & Cherkaoui 2006):
#
#   S₁₁₁₁ = S₂₂₂₂ = (5 - 4ν) / [8(1-ν)]
#   S₁₁₂₂ = S₂₂₁₁ = (4ν - 1) / [8(1-ν)]
#
# For a dilatational eigenstrain ε* = ε₀ I, the constrained strain inside is:
#   ε^c₁₁ = S₁₁₁₁ ε₀ + S₁₁₂₂ ε₀ = S_dil * ε₀
#
# where S_dil = (S₁₁₁₁ + S₁₁₂₂) = (5-4ν+4ν-1) / [8(1-ν)] = 4/[8(1-ν)] = 1/[2(1-ν)]
#
# Verification:  σ₁₁ = (2λ+2G)(ε^c - ε₀) = (2λ+2G)(S_dil - 1)ε₀
#              = 2G/(1-2ν) * (2ν-1)/[2(1-ν)] * ε₀
#              = -G ε₀/(1-ν) = -σ₀  ✓
S_dil = 1.0 / (2.0 * (1.0 - nu))
lam = 2.0 * Gm * nu / (1.0 - 2.0 * nu)
eps_c = S_dil * eps0
de = eps_c - eps0
sig_in_hom = (lam * 2 + 2 * Gm) * de  # (2λ+2G)(εc-ε₀)
print(f"S_dil = {S_dil:.6f}  (= ν/(1-ν) = {nu / (1 - nu):.6f})")
print(f"Analytic interior stress (δ=1): σ₁₁/σ₀ = {sig_in_hom / sigma0:.4f}  (should be -1.0)")


def analytic_line(r_arr, R, sigma0, eps0, delta=1.0, nu=0.3, Gm=1.0):
    """Analytic stress profile along x-axis (θ=0) for a circular inhomogeneity
    with dilatational eigenstrain in an isotropic infinite matrix (plane strain).

    Algorithm: Mura / Eshelby equivalent inclusion method.

    For δ=1 (homogeneous Eshelby):
      Inside:  σ₁₁ = σ₂₂ = -σ₀   (σ₀ = G^m ε₀ / (1-ν))
      Outside (x-axis, θ=0):
               σ₁₁ = -σ₀ (R/r)²
               σ₂₂ = +σ₀ (R/r)²

    For δ≠1, the equivalent eigenstrain ε** satisfies:
      C^ppt (S_dil ε** - ε*) = C^mat (S_dil ε** - ε**)
    Solution (all moduli scale by δ for same-ν case):
      ε** = δ ε₀ / [1 - S_dil(1-δ)]
    Interior stress uses C^mat:
      σ_int = (2λ_m + 2G_m)(S_dil - 1) ε** = -σ₀ δ / [1 - S_dil(1-δ)]
    Outside amplitude scales with ε**/ε₀.
    """
    lam_m = 2.0 * Gm * nu / (1.0 - 2.0 * nu)
    Cm = lam_m + 2.0 * Gm  # (λ+2G) for the diagonal modulus

    # Equivalent eigenstrain (see derivation above)
    eps_ss = delta * eps0 / (1.0 - S_dil * (1.0 - delta))

    eps_int = S_dil * eps_ss
    de_int = eps_int - eps_ss  # elastic strain in the equivalent inclusion
    # σ₁₁ = C₁₁₁₁(εc-ε**) + C₁₁₂₂(εc-ε**) = [(λ+2G) + λ] de = (2λ+2G) de
    sig_int = (2 * lam_m + 2 * Gm) * de_int

    # Outside amplitude scales with ε**/ε₀:
    #   σ₀_eff = Gm * ε** / (1-ν)  (same formula as σ₀ but with ε**)
    # Outside along x-axis (θ=0):
    #   σ₁₁ = σ_rr = -σ₀_eff * (R/r)²   [continuous with interior at r=R]
    #   σ₂₂ = σ_θθ = +σ₀_eff * (R/r)²   [jumps by 2σ₀_eff at r=R]
    amp = Gm * eps_ss / (1.0 - nu)  # σ₀_eff = σ₀ * (ε**/ε₀)

    s11 = np.where(r_arr <= R, sig_int, -amp * (R / r_arr) ** 2)
    s22 = np.where(r_arr <= R, sig_int, +amp * (R / r_arr) ** 2)
    return s11, s22


cases = [
    {"delta": 1.0, "label": r"$\delta = 1.0$ (homogeneous, exact Eshelby validation)"},
    {"delta": 0.5, "label": r"$\delta = 0.5$ (soft precipitate, Fig. 1 in G\&A 2007)"},
]

fig, axes = plt.subplots(2, 2, figsize=(12, 9))
residual_histories = {}  # delta -> list of RMS Δε per iteration

for row, case in enumerate(cases):
    delta = case["delta"]
    Gp = delta * Gm
    C_mat = build_isotropic_stiffness(Gm, nu)
    C_ppt = build_isotropic_stiffness(Gp, nu)

    print(f"\n--- δ = {delta} ---")
    result = solve_elasticity(
        c=c,
        C_mat=C_mat,
        C_ppt=C_ppt,
        eigenstrain=eigenstrain,
        sigma_applied=None,
        Lx=Lx,
        tol=1e-8,
        max_iter=3000,
        verbose=True,
    )
    print(f"  n_iter = {result['n_iter']},  residual = {result['residual']:.3e}")
    residual_histories[delta] = result["residual_history"]

    sigma = result["stress"]  # (N, N, 2, 2)
    iy = N // 2

    x_arr = np.linspace(0, Lx, N, endpoint=False)
    cx_dom = Lx / 2.0
    mask = x_arr >= cx_dom
    r_fft = x_arr[mask] - cx_dom

    s11_fft = sigma[mask, iy, 0, 0]
    s22_fft = sigma[mask, iy, 1, 1]

    # Print interior stress and strain for verification
    mask_in = c.astype(bool)
    s11_in_fft = sigma[mask_in, 0, 0].mean()
    e11_in_fft = result["strain"][mask_in, 0, 0].mean()
    s11_an_in, _ = analytic_line(np.array([0.5 * R]), R, sigma0, eps0, delta=delta, nu=nu, Gm=Gm)
    Gp = delta * Gm
    lm = 2 * Gm * nu / (1 - 2 * nu)
    S_dil_ex = 1.0 / (2 * (1 - nu))
    eps_ss_ex = delta * eps0 / (1 - S_dil_ex * (1 - delta))
    eps_int_ex = S_dil_ex * eps_ss_ex
    print(f"  E (macro strain):     [{result['E'][0, 0]:.4e}, {result['E'][1, 1]:.4e}]")
    print(f"  Interior ε₁₁:  FFT = {e11_in_fft:.6f},  Eshelby = {eps_int_ex:.6f}")
    print(
        f"  Interior σ₁₁/σ₀ (paper norm):  FFT = {s11_in_fft / sigma0_paper:.4f},  "
        f"analytic = {s11_an_in[0] / sigma0_paper:.4f}  (G&A Fig.1 shows ≈ -1.1 for δ=0.5)"
    )

    r_an = np.linspace(0.005 * R, 6.0 * R, 600)
    s11_an, s22_an = analytic_line(r_an, R, sigma0, eps0, delta=delta, nu=nu, Gm=Gm)

    # Pixel-digitized points extracted directly from Fig. 1a/1b of the paper
    # (G&A 2007, p. 5020). Curve pixels were traced from a 4x-rendered PNG of
    # the page (literature/digitized/fig1{a,b}_box.png) using axis-tick pixel
    # calibration; see literature/digitized/ga2007_fig1.npz. Only for δ=0.5
    # (their Fig. 1 setup), already in their σ₀ = G^m ε^T normalization.
    if delta == 0.5:
        _ga_path = _ROOT / "literature" / "digitized" / "ga2007_fig1.npz"
        _ga = np.load(_ga_path)
        r_ga_norm = _ga["r"]  # r/R values (already normalized)
        s11_ga_norm = _ga["s11"]  # already in σ₀=G^m ε^T units
        s22_ga_norm = _ga["s22"]

    for col, (s_fft, s_an, lbl) in enumerate(
        zip([s11_fft, s22_fft], [s11_an, s22_an], [r"$\sigma_{11}/\sigma_0$", r"$\sigma_{22}/\sigma_0$"])
    ):
        ax = axes[row, col]
        ax.plot(r_fft / R, s_fft / sigma0_paper, "b.", ms=3, label="FFT (this work)")
        ax.plot(r_an / R, s_an / sigma0_paper, "r--", lw=1.8, label="Eshelby/Mura (analytic)")
        if delta == 0.5:
            s_ga_norm = s11_ga_norm if col == 0 else s22_ga_norm
            ax.plot(r_ga_norm, s_ga_norm, "k^", ms=6, mfc="none", mew=1.3, label="G\\&A (2007) Fig. 1 [pixel-digitized]")
        ax.axvline(1.0, color="gray", ls=":", lw=1.0)
        ax.axhline(0.0, color="k", ls="-", lw=0.5)
        ax.set_xlabel(r"$r/R$", fontsize=12)
        ax.set_ylabel(lbl, fontsize=12)
        ax.set_xlim(0, 6.0)
        ax.set_ylim(-1.5, 1.5)
        ax.legend(fontsize=8, loc="upper right" if col == 1 else "lower right")
        if col == 0:
            ax.set_title(f"{lbl} — {case['label']}", fontsize=10)

fig.suptitle(
    rf"FFT elasticity benchmark — circular precipitate ($\nu={nu}$, $N={N}$, $\varepsilon_0={eps0}$)",
    fontsize=12,
)
fig.tight_layout()

out = _ROOT / "results" / "elasticity_benchmark"
out.mkdir(parents=True, exist_ok=True)
fpath = out / "circular_precipitate_stress.pdf"
fig.savefig(fpath, bbox_inches="tight")
print(f"\nFigure saved to {fpath}")


fig_c, ax_c = plt.subplots(figsize=(6, 4.5))
colors = {1.0: "tab:blue", 0.5: "tab:orange"}
for delta, hist in residual_histories.items():
    iters = np.arange(1, len(hist) + 1)
    ax_c.semilogy(iters, hist, "o-", ms=4, lw=1.3, color=colors.get(delta), label=rf"$\delta = {delta}$  ({len(hist)} iters)")
ax_c.axhline(1e-8, color="k", ls="--", lw=1.0, label="tol = 1e-8")
ax_c.set_xlabel("Iteration", fontsize=12)
ax_c.set_ylabel(r"RMS $\Delta\varepsilon$  (strain change per iteration)", fontsize=12)
ax_c.set_title("FFT elasticity solver — convergence history", fontsize=12)
ax_c.legend(fontsize=9)
ax_c.grid(True, which="both", alpha=0.3)
fig_c.tight_layout()

fpath_c = out / "circular_precipitate_convergence.pdf"
fig_c.savefig(fpath_c, bbox_inches="tight")
print(f"Convergence plot saved to {fpath_c}")


delta = 0.5
C_mat2 = build_isotropic_stiffness(Gm, nu)
C_ppt2 = build_isotropic_stiffness(delta * Gm, nu)
result2 = solve_elasticity(c, C_mat2, C_ppt2, eigenstrain, Lx=Lx, tol=1e-8, verbose=False)
sigma2 = result2["stress"]

fig2, axes2 = plt.subplots(1, 2, figsize=(10, 4.5))
theta_c = np.linspace(0, 2 * np.pi, 300)
for ax, s_comp, title in zip(
    axes2,
    [sigma2[:, :, 0, 0] / sigma0, sigma2[:, :, 1, 1] / sigma0],
    [r"$\sigma_{11}/\sigma_0$  ($\delta=0.5$)", r"$\sigma_{22}/\sigma_0$  ($\delta=0.5$)"],
):
    im = ax.imshow(s_comp.T, origin="lower", extent=[0, Lx, 0, Lx], cmap="RdBu_r", vmin=-2.0, vmax=2.0)
    ax.plot(Lx / 2 + R * np.cos(theta_c), Lx / 2 + R * np.sin(theta_c), "k-", lw=1.0)
    plt.colorbar(im, ax=ax)
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("x")
    ax.set_ylabel("y")

fig2.suptitle(r"Stress field maps ($\delta=0.5$, $\nu=0.3$)", fontsize=12)
fig2.tight_layout()
fpath2 = out / "circular_precipitate_stress_map.pdf"
fig2.savefig(fpath2, bbox_inches="tight")
print(f"Map   saved to {fpath2}")
