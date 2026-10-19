"""runs/v3/test_precipitate_rafting.py
=====================================
Precipitate rafting validation — reproduces the qualitative result of
Gururajan & Abinandanan (2007) Section 5:

    A circular precipitate (δ < 1, soft) under uniaxial COMPRESSIVE stress σ₁₁ < 0
    elongates parallel to the stress axis (x-direction), i.e. a_x/a_y > 1.

    A stiff precipitate (δ > 1) under compression elongates perpendicular to
    the applied stress (a_x/a_y < 1).

    Sign convention: σ_app < 0 → compression along x (matches G&A Section 5 Fig. 5).

Physics
-------
    μ_total = μ_ch + μ_el
    μ_el from Eq. 36 (elastic_potential.py)
    CH dynamics: ∂c/∂t = ∇·(M ∇μ_total) − κ ∇⁴c  (Eyre convex-splitting)

Parameter scaling
-----------------
    To see measurable rafting the elastic energy must be comparable to the
    chemical barrier.  The key dimensionless ratio is:

        η = G^m ε_T² / A_b

    With G^m = 100, ε_T = 0.1, A_b = 1.0 → η = 1.0  (elastic ≈ chemical)

    Interface width  W = sqrt(2 κ / A_b) ≈ 1–2 grid units  →  R/W >> 1
    so the precipitate is geometrically well-resolved and Gibbs-Thomson
    curvature dissolution is negligible.

    Eyre convex-splitting CH update (unconditionally stable):
        ĉ_{n+1} = [ĉ_n - dt M k² (μ_ch⁻ + μ_el)] / [1 + dt M k² (f''₊ + κk²)]
    where μ_ch⁻ = μ_ch − f''₊ c  (explicit part only; f''₊ c handled implicitly).

Run with:
    python runs/v3/test_precipitate_rafting.py

Outputs (in runs/v3/):
    rafting_aspect_ratio.pdf — a_x/a_y vs time for soft and stiff cases
    rafting_morphology.pdf   — composition snapshots at t = 0, mid, final
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from phasefoundry.elasticity import build_isotropic_stiffness, solve_elasticity
from phasefoundry.elasticity.elastic_potential import compute_mu_el_from_result


def _make_circle(N: int, R: float, Lx: float, width: float = 2.0) -> np.ndarray:
    """Smooth (tanh) circular profile, c=1 inside, c=0 outside."""
    x = np.linspace(0, Lx, N, endpoint=False)
    xx, yy = np.meshgrid(x, x, indexing="ij")
    r = np.sqrt((xx - Lx / 2) ** 2 + (yy - Lx / 2) ** 2)
    return 0.5 * (1.0 - np.tanh((r - R) / width))


def _aspect_ratio(c: np.ndarray, threshold: float = 0.5) -> float:
    """a_x / a_y from second moments of the c > threshold mask."""
    N = c.shape[0]
    x = np.arange(N, dtype=float)
    xx, yy = np.meshgrid(x, x, indexing="ij")
    mask = c > threshold
    if mask.sum() < 4:
        return 1.0
    cx = xx[mask].mean()
    cy = yy[mask].mean()
    m20 = ((xx[mask] - cx) ** 2).mean()
    m02 = ((yy[mask] - cy) ** 2).mean()
    return float(np.sqrt(m20 / (m02 + 1e-30)))


def run_rafting(
    delta: float,
    sigma_app: float,
    N: int = 128,
    Lx: float = 128.0,
    R: float = 22.0,
    G_matrix: float = 100.0,
    nu: float = 0.3,
    eps_T: float = 0.1,
    kappa: float = 0.5,
    M: float = 0.05,
    A_b: float = 1.0,
    dt: float = 0.05,
    n_steps: int = 12000,
    save_every: int = 1000,
    elastic_solve_every: int = 1,
    elastic_tol: float = 1e-6,
    elastic_max_iter: int = 500,
) -> dict:
    """Run CH + elastic coupling from a single circular precipitate initial condition.

    Returns dict: steps, aspect_ratios, compositions (snapshots), times.
    """
    C_mat = build_isotropic_stiffness(G_matrix, nu)
    C_ppt = build_isotropic_stiffness(delta * G_matrix, nu)
    eigenstrain = eps_T * np.eye(3)

    sigma_applied = np.zeros((2, 2))
    sigma_applied[0, 0] = sigma_app  # compression < 0 along x

    c = _make_circle(N, R, Lx, width=2.0)

    # Spectral wave-vectors
    freqs = np.fft.fftfreq(N, d=Lx / N)
    kv = 2.0 * np.pi * freqs
    kx, ky = np.meshgrid(kv, kv, indexing="ij")
    k2 = kx**2 + ky**2

    # Eyre convex-splitting: f = f_+ - f_-
    # For g = A_b c²(1-c)², d²g/dc² = A_b(2 - 12c + 12c²) ∈ [-1, +2]*A_b
    # f''_+ = 2*A_b  (upper bound of second derivative)
    f_pp = 2.0 * A_b
    # Denominator: implicit terms (gradient energy + f''_+)
    denom = 1.0 + dt * M * (f_pp * k2 + kappa * k2**2)

    steps_out: list[int] = []
    times_out: list[float] = []
    ratios_out: list[float] = []
    comps_out: list[np.ndarray] = []

    u_prev = None  # warm-start displacement

    t = 0.0
    for step in range(n_steps + 1):
        if step % save_every == 0:
            steps_out.append(step)
            times_out.append(t)
            ar = _aspect_ratio(c)
            ratios_out.append(ar)
            comps_out.append(c.copy())
            print(f"  step {step:6d}  t = {t:8.1f}  AR = {ar:.4f}")

        if step == n_steps:
            break

        # elastic solve (warm-started, quasi-static)
        if step % elastic_solve_every == 0 or step == 0:
            el_result = solve_elasticity(
                c,
                C_mat,
                C_ppt,
                eigenstrain,
                sigma_applied=sigma_applied,
                Lx=Lx,
                tol=elastic_tol,
                max_iter=elastic_max_iter,
                verbose=False,
                u_init=u_prev,
            )
            u_prev = el_result["displacement"]
            mu_el = compute_mu_el_from_result(c, el_result, C_mat, C_ppt, eigenstrain)

        # chemical potential  μ_ch = dg/dc = A_b(2c(1-c)² - 2c²(1-c))
        mu_ch = A_b * (2 * c * (1 - c) ** 2 - 2 * c**2 * (1 - c))

        # Eyre convex-splitting CH update
        # Explicit part of μ_ch: μ_ch⁻ = μ_ch − f''_+ c  (concave contribution)
        mu_ch_exp = mu_ch - f_pp * c
        mu_total_exp = mu_ch_exp + mu_el  # both explicit

        c_hat = np.fft.fftn(c)
        mu_hat = np.fft.fftn(mu_total_exp)
        c_new = np.real(np.fft.ifftn((c_hat - dt * M * k2 * mu_hat) / denom))

        # clamp to [0, 1] to prevent numerical overshoot at sharp interfaces
        c = np.clip(c_new, 0.0, 1.0)
        t += dt

    return {
        "steps": np.array(steps_out),
        "times": np.array(times_out),
        "aspect_ratios": np.array(ratios_out),
        "compositions": comps_out,
    }


def main():
    out_dir = Path(__file__).parent
    print("=" * 60)
    print("Precipitate Rafting Validation")
    print("Gururajan & Abinandanan (2007) — Section 5")
    print("=" * 60)

    # η = G^m ε_T² / A_b = 100 * 0.01 / 1.0 = 1.0  (elastic ≈ chemical)
    # W = sqrt(2κ/A_b) ≈ 1.0 grid unit  →  R/W ≈ 22  (well-resolved)
    common = dict(
        N=128,
        Lx=128.0,
        R=22.0,
        G_matrix=100.0,
        nu=0.3,
        eps_T=0.1,
        kappa=0.5,
        M=0.05,
        A_b=1.0,
        dt=0.05,
        n_steps=3000,
        save_every=300,
        elastic_solve_every=1,
        elastic_tol=1e-6,
        elastic_max_iter=500,
    )
    # Compressive stress: |σ_app| = 0.05 G^m ε_T = 0.5
    # (η = G ε_T²/A_b = 1.0 → elastic energy ≈ chemical barrier)
    sigma_app = -0.05 * common["G_matrix"] * common["eps_T"]
    print(f"\n  σ_app = {sigma_app:.3f}  (compressive along x)")
    print(f"  η = G·ε_T²/A_b = {common['G_matrix'] * common['eps_T'] ** 2 / common['A_b']:.2f}")
    print(f"  |σ_app|/(G·ε_T) = {abs(sigma_app) / (common['G_matrix'] * common['eps_T']):.3f}")
    print(f"  Interface width W ≈ {np.sqrt(2 * common['kappa'] / common['A_b']):.2f} (grid units)")

    print("\n── Case 1: soft precipitate  (δ = 0.5) ──")
    res_soft = run_rafting(delta=0.5, sigma_app=sigma_app, **common)

    print("\n── Case 2: stiff precipitate (δ = 2.0) ──")
    res_stiff = run_rafting(delta=2.0, sigma_app=sigma_app, **common)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(
        res_soft["times"],
        res_soft["aspect_ratios"],
        "b-o",
        ms=4,
        label=r"Soft ($\delta=0.5$): elongates $\parallel$ $\sigma_{11}$",
    )
    ax.plot(
        res_stiff["times"],
        res_stiff["aspect_ratios"],
        "r-s",
        ms=4,
        label=r"Stiff ($\delta=2.0$): elongates $\perp$ $\sigma_{11}$",
    )
    ax.axhline(1.0, color="k", lw=0.8, ls="--", label="Circular (AR = 1)")
    ax.set_xlabel("Time (a.u.)")
    ax.set_ylabel(r"Aspect ratio $a_x / a_y$")
    ax.set_title(
        r"Precipitate Rafting under Uniaxial Compression"
        "\n"
        r"$\sigma_{11}^{\rm app} = -0.5\,G^m \varepsilon^T$"
        r"   ($\eta = G\varepsilon_T^2/A_b = 1$)"
    )
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "rafting_aspect_ratio.pdf", dpi=150)
    print(f"\n  Saved: {out_dir / 'rafting_aspect_ratio.pdf'}")

    snap_idx = [0, len(res_soft["compositions"]) // 2, -1]
    fig = plt.figure(figsize=(4 * len(snap_idx), 4.5 * 2))
    gs = GridSpec(2, len(snap_idx), figure=fig, hspace=0.35, wspace=0.1)

    for row, (res, label) in enumerate(
        [
            (res_soft, r"Soft $\delta=0.5$"),
            (res_stiff, r"Stiff $\delta=2.0$"),
        ]
    ):
        for col, idx in enumerate(snap_idx):
            ax = fig.add_subplot(gs[row, col])
            ax.imshow(res["compositions"][idx].T, origin="lower", cmap="RdBu_r", vmin=0, vmax=1, aspect="equal")
            ax.set_title(f"{label}\nstep {res['steps'][idx]}  AR={res['aspect_ratios'][idx]:.3f}", fontsize=8)
            ax.axis("off")

    fig.suptitle(r"Precipitate Rafting Morphology — compression $\sigma_{11} < 0$ (horizontal)", fontsize=10)
    fig.savefig(out_dir / "rafting_morphology.pdf", dpi=150)
    print(f"  Saved: {out_dir / 'rafting_morphology.pdf'}")

    ar_soft = res_soft["aspect_ratios"][-1]
    ar_stiff = res_stiff["aspect_ratios"][-1]
    print(
        f"\n  Final AR — soft  (δ=0.5): {ar_soft:.4f}  "
        f"({'PASS' if ar_soft > 1.02 else 'MARGINAL' if ar_soft > 1.0 else 'FAIL'})"
        f"  (expected > 1, elongated ∥ σ₁₁)"
    )
    print(
        f"  Final AR — stiff (δ=2.0): {ar_stiff:.4f}  "
        f"({'PASS' if ar_stiff < 0.98 else 'MARGINAL' if ar_stiff < 1.0 else 'FAIL'})"
        f"  (expected < 1, elongated ⊥ σ₁₁)"
    )


if __name__ == "__main__":
    logging.basicConfig(format="%(message)s", stream=sys.stdout)
    logging.getLogger("phasefoundry").setLevel(logging.INFO)
    main()
