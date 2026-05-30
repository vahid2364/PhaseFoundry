"""
Ternary analytical (three-well polynomial) Cahn-Hilliard simulation
====================================================================
Uses the `fchem_three_well` free energy from the PhaseFoundry notebook:

    f(c1,c2) = rho_s * (c1² c2² + c2² c3² + c3² c1²),   c3 = 1 - c1 - c2

This is a symmetric three-well potential with minima at the pure-component
corners of the Gibbs triangle — a standard analytical benchmark for ternary
spinodal decomposition solvers.

Outputs saved to:  tests/results_ternary_analytical/
  - final_field.npz
  - micrograph_c1.pdf / c2 / c3
"""
from __future__ import annotations
import sys, time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from numpy.fft import fft2, ifft2
from tqdm import tqdm

OUT_DIR = Path(__file__).resolve().parent / "results_ternary_analytical"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── parameters ────────────────────────────────────────────────────────────────
N      = 256
Lx     = 200.0
Kappa  = 20.0        # gradient energy coefficient
M      = 1.0         # mobility
rho_s  = 20.0        # free energy prefactor
dt     = 0.1
tmax   = 2000.0      # physical time
SEED   = 42
C0     = [0.25, 0.25]   # mean compositions of c1, c2; c3 = 0.50
NOISE  = 0.01

n_steps   = int(tmax / dt)
save_every = max(1, n_steps // 200)

print(f"N={N}, dt={dt}, n_steps={n_steps}, tmax={tmax}")

# ── spectral operators ────────────────────────────────────────────────────────
k  = np.concatenate([np.arange(0, N//2), [0], np.arange(-N//2+1, 0)])
g  = 2 * np.pi / Lx
kx, ky = np.meshgrid(g*1j*k, g*1j*k)
k2 = kx**2 + ky**2          # (N,N) complex, but imaginary part is 0 for k2

# ── free energy derivatives (analytical) ─────────────────────────────────────
def dfdc1(c1, c2):
    c3 = 1.0 - c1 - c2
    return rho_s * (2*c1*c2**2 + 2*c1*c3**2 - 2*c1**2*c3)

def dfdc2(c1, c2):
    c3 = 1.0 - c1 - c2
    return rho_s * (2*c2*c1**2 + 2*c2*c3**2 - 2*c2**2*c3)

# ── initial conditions ────────────────────────────────────────────────────────
rng = np.random.default_rng(SEED)
c1  = C0[0] + NOISE * (rng.random((N, N)) - 0.5)
c2  = C0[1] + NOISE * (rng.random((N, N)) - 0.5)
c3  = 1.0 - c1 - c2

free_energy_hist = []

# ── time integration ──────────────────────────────────────────────────────────
t0 = time.time()
for step in tqdm(range(1, n_steps + 1)):
    mu1 = dfdc1(c1, c2)
    mu2 = dfdc2(c1, c2)

    mu1_hat = fft2(mu1)
    mu2_hat = fft2(mu2)
    c1_hat  = fft2(c1)
    c2_hat  = fft2(c2)

    denom = 1.0 + 2.0 * M * Kappa * dt * k2**2
    c1 = np.real(ifft2((c1_hat + M * dt * k2 * mu1_hat) / denom))
    c2 = np.real(ifft2((c2_hat + M * dt * k2 * mu2_hat) / denom))

    # no clipping needed — analytical free energy is well-behaved
    c3 = 1.0 - c1 - c2

    if step % save_every == 0:
        fe = rho_s * (c1**2 * c2**2 + c2**2 * c3**2 + c3**2 * c1**2)
        free_energy_hist.append(float(fe.mean()))

elapsed = time.time() - t0
print(f"\nDone in {elapsed:.1f} s")
print(f"c1: [{c1.min():.4f}, {c1.mean():.4f}, {c1.max():.4f}]")
print(f"c2: [{c2.min():.4f}, {c2.mean():.4f}, {c2.max():.4f}]")
print(f"c3: [{c3.min():.4f}, {c3.mean():.4f}, {c3.max():.4f}]")

# ── save npz ─────────────────────────────────────────────────────────────────
np.savez_compressed(OUT_DIR / "final_field.npz",
                    c1=c1, c2=c2, c3=c3,
                    free_energy=np.array(free_energy_hist),
                    t_phys=tmax, dt=dt, n_steps=n_steps)
print(f"Saved: {OUT_DIR / 'final_field.npz'}")

# ── micrographs ───────────────────────────────────────────────────────────────
matplotlib.rcParams.update({
    "font.family": "serif",
    "font.serif":  ["Times New Roman", "DejaVu Serif"],
    "font.size": 9,
})

fields = {
    "c1": (c1, r"$c_1$", "Reds"),
    "c2": (c2, r"$c_2$", "Greens"),
    "c3": (c3, r"$c_3$", "Blues"),
}

for key, (field, label, cmap) in fields.items():
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(field, origin="lower", cmap=cmap,
                   vmin=0, vmax=1, interpolation="nearest", aspect="equal")
    plt.colorbar(im, ax=ax, label=label, fraction=0.046, pad=0.04)
    ax.set_title(f"Three-well ternary — {label}\n$t = {tmax:.0f}$,  wall = {elapsed:.0f} s")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    out = OUT_DIR / f"micrograph_{key}.pdf"
    fig.savefig(out, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"Saved: {out}")

print("\nAll done.")
