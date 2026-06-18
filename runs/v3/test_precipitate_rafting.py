"""
runs/v3/test_precipitate_rafting.py
=====================================
Precipitate rafting validation — reproduces the qualitative result of
Gururajan & Abinandanan (2007) Section 5:

    A circular precipitate (δ < 1, soft) under uniaxial COMPRESSIVE stress σ₁₁ < 0
    elongates parallel to the stress axis (x-direction), i.e. the aspect ratio
    a_x / a_y > 1 and grows with time.  (Under tension the direction reverses.)

    For a stiff precipitate (δ > 1) under compression, the precipitate elongates
    perpendicular to the applied stress (a_x / a_y < 1).  Both cases are tested here.

    Sign convention: σ_app < 0 → compression along x (matches G&A Section 5 Fig. 5).

Physics
-------
    μ_total = μ_ch + μ_el
    μ_el from Eq. 36 (elastic_potential.py)
    CH dynamics: ∂c/∂t = ∇·(M ∇μ_total) − κ ∇⁴c

Parameters chosen to be dimensionless and fast:
    G^m = 1.0,  ε_T = 0.1,  ν = 0.3,  A_b = 1e-2
    σ_app = 0.5 G^m ε_T (along x)
    Scale ratio G ε_T² / A_b ≈ 1  →  elastic and chemical energies comparable.
    δ = 0.5  (soft precipitate)

Run with:
    python runs/v3/test_precipitate_rafting.py

Outputs (in runs/v3/):
    rafting_morphology.pdf   — composition snapshots at t = 0, t_mid, t_final
    rafting_aspect_ratio.pdf — a_x / a_y vs step for soft and stiff cases
"""

from __future__ import annotations
import sys
from pathlib import Path
from dataclasses import dataclass, field

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from calphad_pf.elasticity import solve_elasticity, build_isotropic_stiffness
from calphad_pf.elasticity.elastic_potential import compute_mu_el_from_result


# ── synthetic double-well "interpolant" (no TDB needed) ──────────────────────

@dataclass
class _FakeInterpolant:
    A_b: float = 1.0
    N_ind: int = 1
    ind_elements: list = field(default_factory=lambda: ["A"])
    reference_element: str = "B"
    grid_bounds: object = None
    derivative_method: str = "analytic"
    temperature_K: float = 500.0

    def mu(self, c):
        c0 = c[0]
        dmu = self.A_b * (2 * c0 * (1 - c0) ** 2 - 2 * c0 ** 2 * (1 - c0))
        return dmu[np.newaxis, ...]

    def d2g(self, c):
        c0 = c[0]
        d2 = self.A_b * (2 * (1 - c0) ** 2 - 8 * c0 * (1 - c0) + 2 * c0 ** 2)
        return d2.reshape(1, 1, -1)


# ── composition initializer ──────────────────────────────────────────────────

def _make_circle(N: int, R: float, Lx: float, width: float = 2.5) -> np.ndarray:
    """Smooth (tanh) circular composition profile, c=1 inside, c=0 outside."""
    x = np.linspace(0, Lx, N, endpoint=False)
    xx, yy = np.meshgrid(x, x, indexing="ij")
    cx, cy = Lx / 2.0, Lx / 2.0
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    return 0.5 * (1.0 - np.tanh((r - R) / width))


# ── aspect ratio from second moments ─────────────────────────────────────────

def _aspect_ratio(c: np.ndarray, N: int, Lx: float, threshold: float = 0.5) -> float:
    """
    Compute a_x / a_y from the second moments of the precipitate mask (c > threshold).
    Returns 1.0 if precipitate is too small or degenerate.
    """
    mask = c > threshold
    if mask.sum() < 4:
        return 1.0
    x = np.linspace(0, Lx, N, endpoint=False)
    xx, yy = np.meshgrid(x, x, indexing="ij")
    cx = (xx[mask]).mean()
    cy = (yy[mask]).mean()
    m20 = ((xx[mask] - cx) ** 2).mean()
    m02 = ((yy[mask] - cy) ** 2).mean()
    if m02 < 1e-30:
        return 1.0
    return float(np.sqrt(m20 / m02))


# ── Cahn-Hilliard + elastic time integration ─────────────────────────────────

def run_rafting(
    delta:       float,
    sigma_app:   float,
    N:           int   = 128,
    Lx:          float = 128.0,
    R:           float = 20.0,
    G_matrix:    float = 1.0,
    nu:          float = 0.3,
    eps_T:       float = 0.01,
    kappa:       float = 2.0,
    M:           float = 0.1,
    A_b:         float = 1.0,
    dt:          float = 0.2,
    n_steps:     int   = 2000,
    save_every:  int   = 200,
    elastic_tol: float = 1e-6,
    elastic_max_iter: int = 500,
) -> dict:
    """
    Run CH + elastic coupling.  Returns dict with:
        steps, aspect_ratios, compositions (list of snapshots)
    """
    interp = _FakeInterpolant(A_b=A_b)

    # Build stiffness tensors
    C_mat = build_isotropic_stiffness(G_matrix, nu)
    C_ppt = build_isotropic_stiffness(delta * G_matrix, nu)
    eigenstrain = eps_T * np.eye(3)

    # Applied stress: uniaxial σ₁₁ = sigma_app (tension along x)
    sigma_applied = np.zeros((2, 2))
    sigma_applied[0, 0] = sigma_app

    # Initial composition
    c = _make_circle(N, R, Lx)
    c = c[np.newaxis]           # (1, N, N)

    # Pre-compute FFT wave-vectors for CH update
    freqs = np.fft.fftfreq(N, d=Lx / N)
    kv    = 2.0 * np.pi * freqs
    kx, ky = np.meshgrid(kv, kv, indexing="ij")
    k2 = kx**2 + ky**2         # (N, N)

    # Eyre convex-splitting stabilization: f''_+ = max eigenvalue of d²g/dc²
    # For double-well g = A_b c²(1-c)², d²g/dc² ∈ [-A_b, +2*A_b]
    f_pp = 2.0 * A_b
    denom = 1.0 + dt * M * kappa * k2**2 + dt * M * f_pp * k2  # (N, N)

    steps_out:      list[int]   = []
    ratios_out:     list[float] = []
    comps_out:      list[np.ndarray] = []

    u_prev = None   # warm-start displacement

    for step in range(n_steps + 1):
        # snapshot
        if step % save_every == 0:
            steps_out.append(step)
            ratios_out.append(_aspect_ratio(c[0], N, Lx))
            comps_out.append(c[0].copy())
            print(f"  step {step:5d}  aspect = {ratios_out[-1]:.4f}")

        if step == n_steps:
            break

        # elastic solve (warm-started)
        el_result = solve_elasticity(
            c[0], C_mat, C_ppt, eigenstrain,
            sigma_applied=sigma_applied, Lx=Lx,
            tol=elastic_tol, max_iter=elastic_max_iter,
            verbose=False, u_init=u_prev,
        )
        u_prev = el_result["displacement"]

        mu_el = compute_mu_el_from_result(
            c[0], el_result, C_mat, C_ppt, eigenstrain,
        )

        # chemical potential
        mu_ch = interp.mu(c)     # (1, N, N)
        mu_total = mu_ch + mu_el[np.newaxis]  # (1, N, N)

        # spectral CH update (semi-implicit Eyre)
        c_hat  = np.fft.fftn(c[0])
        mu_hat = np.fft.fftn(mu_total[0])
        c_new_hat = (c_hat - dt * M * k2 * mu_hat) / denom
        c[0] = np.real(np.fft.ifftn(c_new_hat))

    return {
        "steps":        np.array(steps_out),
        "aspect_ratios": np.array(ratios_out),
        "compositions": comps_out,
    }


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    out_dir = Path(__file__).parent
    print("=" * 60)
    print("Precipitate Rafting Validation")
    print("Gururajan & Abinandanan (2007) — Section 5")
    print("=" * 60)

    # Shared params
    # Scale choices: interface width l ~ sqrt(kappa/A_b) ~ 1.4 grid units.
    # Elastic energy scale G*eps_T^2 must be comparable to A_b for rafting.
    # With A_b=1e-2, G=1.0, eps_T=0.1: G*eps_T^2/A_b ~ 1.0 (comparable).
    common = dict(
        N=128, Lx=128.0, R=20.0,
        G_matrix=1.0, nu=0.3, eps_T=0.1,
        kappa=2.0, M=0.1, A_b=1e-2,
        dt=0.5, n_steps=6000, save_every=500,
        elastic_tol=1e-6, elastic_max_iter=500,
    )
    # G&A Section 5 uses COMPRESSIVE stress along x.
    # Under compression: soft (δ<1) rafts ∥ σ (AR > 1), stiff (δ>1) rafts ⊥ σ (AR < 1).
    sigma_app = -0.5 * 1.0 * 0.1   # compressive along x (negative = compression)

    print("\n── Case 1: soft precipitate  (δ = 0.5) ──")
    res_soft = run_rafting(delta=0.5, sigma_app=sigma_app, **common)

    print("\n── Case 2: stiff precipitate (δ = 2.0) ──")
    res_stiff = run_rafting(delta=2.0, sigma_app=sigma_app, **common)

    # ── Figure 1: aspect ratio vs step ─────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(res_soft["steps"],  res_soft["aspect_ratios"],
            "b-o", ms=4, label=r"Soft ($\delta=0.5$): elongates $\parallel$ $\sigma_{11}$")
    ax.plot(res_stiff["steps"], res_stiff["aspect_ratios"],
            "r-s", ms=4, label=r"Stiff ($\delta=2.0$): elongates $\perp$ $\sigma_{11}$")
    ax.axhline(1.0, color="k", lw=0.8, ls="--", label="Circular (AR = 1)")
    ax.set_xlabel("CH step")
    ax.set_ylabel(r"Aspect ratio $a_x / a_y$")
    ax.set_title("Precipitate Rafting under Uniaxial Compression\n"
                 r"$\sigma_{11}^{\rm app} = -0.5\,G^m \varepsilon^T$ (G\&A Sec. 5)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "rafting_aspect_ratio.pdf", dpi=150)
    print(f"\n  Saved: {out_dir / 'rafting_aspect_ratio.pdf'}")

    # ── Figure 2: morphology snapshots ─────────────────────────────────────
    snap_indices = [0, len(res_soft["compositions"]) // 2, -1]
    n_snaps = len(snap_indices)

    fig = plt.figure(figsize=(4 * n_snaps, 4.5 * 2))
    gs  = GridSpec(2, n_snaps, figure=fig, hspace=0.35, wspace=0.1)

    for row, (res, label) in enumerate(
        [(res_soft,  r"Soft $\delta=0.5$"),
         (res_stiff, r"Stiff $\delta=2.0$")]
    ):
        for col, idx in enumerate(snap_indices):
            ax = fig.add_subplot(gs[row, col])
            c_snap = res["compositions"][idx]
            step   = res["steps"][idx]
            ar     = res["aspect_ratios"][idx]
            ax.imshow(c_snap.T, origin="lower", cmap="RdBu_r",
                      vmin=0, vmax=1, aspect="equal")
            ax.set_title(f"{label}\nstep {step}  AR={ar:.3f}", fontsize=8)
            ax.axis("off")

    fig.suptitle("Precipitate Rafting Morphology\n"
                 r"Compressive stress $\sigma_{11} < 0$ along x (horizontal)", fontsize=10)
    fig.savefig(out_dir / "rafting_morphology.pdf", dpi=150)
    print(f"  Saved: {out_dir / 'rafting_morphology.pdf'}")

    # ── Summary ────────────────────────────────────────────────────────────
    soft_final  = res_soft["aspect_ratios"][-1]
    stiff_final = res_stiff["aspect_ratios"][-1]
    print(f"\n  Final AR — soft  (δ=0.5): {soft_final:.4f}  "
          f"({'PASS ✓' if soft_final > 1.0 else 'FAIL ✗'} > 1, elongated ∥ σ₁₁)")
    print(f"  Final AR — stiff (δ=2.0): {stiff_final:.4f}  "
          f"({'PASS ✓' if stiff_final < 1.0 else 'FAIL ✗'} < 1, elongated ⊥ σ₁₁)")


if __name__ == "__main__":
    main()
