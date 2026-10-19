"""
tests/test_elastic_coupling_integration.py
============================================
Integration validation for the elastic-chemical-potential coupling wired
into ``phasefoundry.phasefield.solver.solve_cahn_hilliard`` (Eq. 2 with
mu_total = mu_ch + mu_el).

Uses a self-contained synthetic double-well "interpolant" (no pycalphad/TDB
dependency needed) so these tests run fast and in isolation.

Validations
-----------
1. Mass conservation: <c> is exactly conserved over time with elastic
   coupling switched on (mu_el flows through the same flux divergence as
   mu_ch, so this is structurally guaranteed -- this test catches any
   accidental breakage of that property).
2. Off-switch equivalence: elastic_enabled=False is byte-identical to the
   pre-existing no-elasticity code path (regression guard for the wiring).
3. Degenerate-coupling equivalence: elastic_enabled=True with delta=1 and
   eps_T=0 (no actual elastic mismatch) should evolve identically to
   elastic_enabled=False, since mu_el=0 pointwise in that case.

Run with:
    python tests/test_elastic_coupling_integration.py
    pytest tests/test_elastic_coupling_integration.py -v
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from phasefoundry.elasticity import build_vegard_tensor
from phasefoundry.phasefield.solver import solve_cahn_hilliard


@dataclass
class _FakeInterpolant:
    """
    Minimal stand-in for GmInterpolant: a simple double-well free energy
    g(c) = A_b * c^2 * (1-c)^2 with mu(c) = dg/dc, d2g(c) = d2g/dc2.
    Just enough API surface for solve_cahn_hilliard to run.
    """
    A_b: float = 1.0
    N_ind: int = 1
    ind_elements: list = field(default_factory=lambda: ["A"])
    reference_element: str = "B"
    grid_bounds: object = None
    derivative_method: str = "analytic"
    temperature_K: float = 500.0

    def gm(self, c):
        c0 = c[0] if c.ndim == 3 else c
        return self.A_b * c0**2 * (1.0 - c0)**2

    def mu(self, c):
        c0 = c[0] if c.ndim == 3 else c
        dmu = self.A_b * (2*c0*(1-c0)**2 - 2*c0**2*(1-c0))
        return dmu[np.newaxis, ...] if c.ndim == 3 else dmu[np.newaxis, ...]

    def d2g(self, c):
        # c: (N_ind, n_pts) -> Hessian (N_ind, N_ind, n_pts)
        c0 = c[0]
        d2 = self.A_b * (2*(1-c0)**2 - 8*c0*(1-c0) + 2*c0**2)
        return d2.reshape(1, 1, -1)


def _common_kwargs(N=32, n_steps=20, save_every=10, **overrides):
    kwargs = dict(
        interpolant=_FakeInterpolant(A_b=5.0),
        M=1.0e-2,
        kappa=1.0e-3,
        c0=0.5,
        N=N,
        Lx=1.0,
        noise=0.02,
        seed=7,
        dt=1.0e-3,
        n_steps=n_steps,
        save_every=save_every,
        output_dir=None,
        save_frames=False,
        ndim=2,
    )
    kwargs.update(overrides)
    return kwargs


def test_mass_conservation_with_elastic_coupling():
    Lambda = build_vegard_tensor({"A": 0.087}, ["A"], ndim=2)
    result = solve_cahn_hilliard(
        **_common_kwargs(
            elastic_enabled=True,
            elastic_G_matrix=1.0,
            elastic_delta=0.5,
            elastic_nu=0.3,
            elastic_Lambda=Lambda,
            elastic_N_V=1.0,
            elastic_solve_every=1,
            elastic_max_iter=200,
        )
    )
    means = [float(c[0].mean()) for c in result.compositions_history]
    spread = max(means) - min(means)
    assert spread < 1e-10, (
        f"Mass not conserved with elastic coupling on: "
        f"<c> ranged over {spread:.3e} (means={means})"
    )
    print(f"  <c> history: {means}")


def test_off_switch_matches_pre_existing_path():
    """elastic_enabled=False must be identical to never having added the kwarg."""
    kwargs = _common_kwargs()
    result_explicit_off = solve_cahn_hilliard(elastic_enabled=False, **kwargs)
    result_default       = solve_cahn_hilliard(**kwargs)   # elastic_enabled defaults to False

    assert np.array_equal(result_explicit_off.compositions, result_default.compositions), \
        "elastic_enabled=False should be bit-identical to the default (no elastic kwarg) path"


def test_degenerate_elastic_coupling_equals_off():
    """
    With delta=1 (no modulus mismatch) and Lambda=0 (no eigenstrain), mu_el is
    identically zero everywhere, so the CH evolution should match the
    elastic_enabled=False run closely.

    Note: not bit-identical, because the elastic solve still runs (and its
    own internal arithmetic introduces O(1e-16) floating point noise into
    mu_el before it's added to mu_bulk), but should agree to a tight
    numerical tolerance.
    """
    kwargs = _common_kwargs(n_steps=30, save_every=15)

    result_off = solve_cahn_hilliard(elastic_enabled=False, **kwargs)
    Lambda_zero = np.zeros((1, 2, 2))
    result_degenerate = solve_cahn_hilliard(
        elastic_enabled=True,
        elastic_G_matrix=1.0,
        elastic_delta=1.0,     # no modulus mismatch
        elastic_Lambda=Lambda_zero,  # zero eigenstrain → mu_el = 0
        elastic_N_V=1.0,
        elastic_max_iter=200,
        **kwargs,
    )

    max_diff = float(np.abs(result_off.compositions - result_degenerate.compositions).max())
    assert max_diff < 1e-8, (
        f"Degenerate elastic coupling (delta=1, eps_T=0) should match the "
        f"off path closely, but max|Δc| = {max_diff:.3e}"
    )
    print(f"  max|Δc| (degenerate vs off) = {max_diff:.3e}")


if __name__ == "__main__":
    print("Validating elastic-chemical-potential coupling in solve_cahn_hilliard\n")

    test_mass_conservation_with_elastic_coupling()
    print("  [PASS] mass conservation with elastic coupling enabled\n")

    test_off_switch_matches_pre_existing_path()
    print("  [PASS] off-switch (elastic_enabled=False) matches pre-existing path\n")

    test_degenerate_elastic_coupling_equals_off()
    print("  [PASS] degenerate coupling (delta=1, eps_T=0) matches off path\n")

    print("ALL TESTS PASSED")
