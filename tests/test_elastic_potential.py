"""Tests for ``elastic_potential``, the elastic chemical potential of Eq. 36.

Gururajan and Abinandanan (2007), Eq. 36:

    N_V μ_el = ½ α'(c) ΔC_ijkl ε^el_kl ε^el_ij − β'(c) (ε* : σ)

The tests use a smooth circular precipitate with isotropic stiffness. They check that
μ_el is zero without misfit, zero in the bulk phases, equal to a finite-difference
derivative of the elastic energy, and the same at symmetric points on the interface.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from phasefoundry.elasticity import (
    build_isotropic_stiffness,
    compute_mu_el_from_result,
    elastic_free_energy_density,
    elastic_potential,
    elastic_strain,
    solve_elasticity,
)


def _make_circle(N: int, R: float, Lx: float, width: float = 0.025) -> np.ndarray:
    """Return a tanh circular profile, with c = 1 inside radius R and c = 0 outside."""
    x = np.linspace(0, Lx, N, endpoint=False)
    xx, yy = np.meshgrid(x, x, indexing="ij")
    cx, cy = Lx / 2.0, Lx / 2.0
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    return 0.5 * (1.0 - np.tanh((r - R) / width))


NU = 0.3
GM = 1.0
EPS0 = 0.01
EIGENSTRAIN = EPS0 * np.eye(3)


def test_trivial_case_zero_mismatch():
    """With equal stiffnesses and zero eigenstrain, μ_el is zero everywhere."""
    N, Lx, R = 48, 1.0, 0.15
    c = _make_circle(N, R, Lx)
    C_mat = build_isotropic_stiffness(GM, NU)
    C_ppt = build_isotropic_stiffness(GM, NU)  # delta = 1

    result = solve_elasticity(c, C_mat, C_ppt, np.zeros((3, 3)), Lx=Lx, tol=1e-9, verbose=False)
    mu_el = compute_mu_el_from_result(c, result, C_mat, C_ppt, np.zeros((3, 3)))

    assert np.allclose(mu_el, 0.0, atol=1e-14), f"Expected mu_el=0 for trivial case, got max|mu_el|={np.abs(mu_el).max():.3e}"


def test_bulk_vanishing():
    """μ_el is zero in the pure phases, where the cubic h' vanishes, and nonzero at the interface."""
    N, Lx, R = 48, 1.0, 0.15
    c = _make_circle(N, R, Lx)
    C_mat = build_isotropic_stiffness(GM, NU)
    C_ppt = build_isotropic_stiffness(0.5 * GM, NU)  # delta = 0.5

    result = solve_elasticity(c, C_mat, C_ppt, EIGENSTRAIN, Lx=Lx, tol=1e-9, verbose=False)
    mu_el = compute_mu_el_from_result(c, result, C_mat, C_ppt, EIGENSTRAIN)

    bulk_mask = (c < 1e-6) | (c > 1.0 - 1e-6)
    assert bulk_mask.sum() > 0, "Test setup error: no pure-bulk pixels found"
    assert np.abs(mu_el[bulk_mask]).max() < 1e-10, (
        f"Expected mu_el=0 in bulk, got max|mu_el|={np.abs(mu_el[bulk_mask]).max():.3e}"
    )
    # sanity: it should be genuinely nonzero at the interface
    assert np.abs(mu_el).max() > 1e-6


@pytest.mark.parametrize("delta", [0.5, 2.0])
def test_finite_difference_consistency(delta):
    """μ_el matches the finite-difference derivative of the elastic energy F_el.

    The pixel with the largest |μ_el| is perturbed by ±1e-4 and elasticity is solved again.
    The central difference of F_el must agree with the analytic μ_el to 1 %.
    """
    N, Lx, R = 48, 1.0, 0.15
    c = _make_circle(N, R, Lx)
    C_mat = build_isotropic_stiffness(GM, NU)
    C_ppt = build_isotropic_stiffness(delta * GM, NU)

    def total_F_el(c_field: np.ndarray) -> float:
        """Solve elasticity for ``c_field`` and return the total elastic energy Σ f_el dV."""
        res = solve_elasticity(c_field, C_mat, C_ppt, EIGENSTRAIN, Lx=Lx, tol=1e-10, max_iter=5000, verbose=False)
        eps_el = elastic_strain(res["strain"], c_field, EIGENSTRAIN)
        f_el = elastic_free_energy_density(eps_el, res["stress"])
        dV = (Lx / N) ** 2
        return float(np.sum(f_el) * dV)

    result0 = solve_elasticity(c, C_mat, C_ppt, EIGENSTRAIN, Lx=Lx, tol=1e-10, max_iter=5000, verbose=False)
    F0 = total_F_el(c)
    mu_el = elastic_potential(c, result0["strain"], result0["stress"], C_mat, C_ppt, EIGENSTRAIN)

    # pick the interfacial pixel with the largest |mu_el| (most sensitive test)
    iy, ix = np.unravel_index(np.argmax(np.abs(mu_el)), mu_el.shape)

    dV = (Lx / N) ** 2
    dc = 1e-4
    c_plus, c_minus = c.copy(), c.copy()
    c_plus[iy, ix] += dc
    c_minus[iy, ix] -= dc

    F_plus = total_F_el(c_plus)
    F_minus = total_F_el(c_minus)
    mu_el_numeric = (F_plus - F_minus) / (2.0 * dc * dV)

    rel_err = abs(mu_el_numeric - mu_el[iy, ix]) / abs(mu_el[iy, ix])
    assert rel_err < 1e-2, (
        f"delta={delta}: finite-difference mu_el={mu_el_numeric:.6e} vs "
        f"analytic mu_el={mu_el[iy, ix]:.6e}  (rel_err={rel_err:.2e})"
    )


def test_radial_symmetry():
    """For an isotropic circular precipitate, μ_el agrees to 5 % at four symmetric points on the interface."""
    N, Lx, R = 64, 1.0, 0.15
    c = _make_circle(N, R, Lx)
    C_mat = build_isotropic_stiffness(GM, NU)
    C_ppt = build_isotropic_stiffness(0.5 * GM, NU)

    result = solve_elasticity(c, C_mat, C_ppt, EIGENSTRAIN, Lx=Lx, tol=1e-9, verbose=False)
    mu_el = compute_mu_el_from_result(c, result, C_mat, C_ppt, EIGENSTRAIN)

    # Compare mu_el at four symmetric points around the interface ring
    cx = cy = Lx / 2.0
    dx = Lx / N
    r_interface_idx = int(round(R / dx))
    i0 = N // 2
    pts = [
        (i0 + r_interface_idx, i0),
        (i0 - r_interface_idx, i0),
        (i0, i0 + r_interface_idx),
        (i0, i0 - r_interface_idx),
    ]
    vals = [mu_el[p] for p in pts]
    spread = (max(vals) - min(vals)) / (abs(np.mean(vals)) + 1e-30)
    assert spread < 0.05, f"mu_el not radially symmetric: values={vals}"


if __name__ == "__main__":
    print("Testing elastic_potential() (Eq. 36, Gururajan & Abinandanan 2007)\n")

    test_trivial_case_zero_mismatch()
    print("  [PASS] trivial case (delta=1, eigenstrain=0) -> mu_el = 0")

    test_bulk_vanishing()
    print("  [PASS] bulk vanishing (mu_el = 0 away from interface)")

    for delta in [0.5, 2.0]:
        test_finite_difference_consistency(delta)
        print(f"  [PASS] finite-difference consistency (delta={delta})")

    test_radial_symmetry()
    print("  [PASS] radial symmetry (isotropic circular precipitate)")

    print("\nALL TESTS PASSED")
