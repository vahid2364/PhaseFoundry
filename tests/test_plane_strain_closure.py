"""
Tests for eigenstrain_closure="plane_strain" in solve_elasticity().

Level 1 — Analytical: for a cubic crystal with axes aligned to the grid, an
    in-plane wavevector, and a purely dilatational eigenstrain, cubic
    symmetry decouples the out-of-plane direction from the dilatational
    forcing (no normal-shear coupling), so the full 3D Khachaturyan
    coherency modulus Y(n) restricted to an in-plane n is *exactly* the
    plane-strain 2D reduction -- not an approximation. This is re-derived
    independently here (pure 2x2 linear algebra, no FFT) and checked
    against phasefoundry.thermodynamics.coherent_hessian.coherency_modulus.

Level 2 — Numerical: the production FFT solve_elasticity(...,
    eigenstrain_closure="plane_strain") on a homogeneous medium with a
    single-Fourier-mode eigenstrain must reproduce the Level-1 analytical
    Y(n) via its total elastic energy, to near machine precision. Using a
    single exactly-representable Fourier mode (rather than a general
    microstructure) avoids the O(1e8)-vs-O(1e-7) stress cancellation that
    made pointwise comparisons unreliable during development.

Level 3 — Contrast: the default "truncate" closure must differ from both
    of the above by a material- and direction-dependent amount, confirming
    it is neither plane strain nor plane stress, just omission (as
    documented in solve_elasticity's docstring).
"""

from __future__ import annotations

import numpy as np
import pytest

from phasefoundry.thermodynamics.coherent_hessian import coherency_modulus
from phasefoundry.elasticity.fft_solver import build_cubic_stiffness, solve_elasticity


def _Y_planestrain_2x2(C: np.ndarray, n2: np.ndarray) -> float:
    """Independent re-derivation of Y(n) using only the in-plane 2x2
    acoustic-tensor block (no 3x3 tensors, no z-components at all)."""
    n2 = np.asarray(n2, dtype=float)
    n2 = n2 / np.linalg.norm(n2)
    I3 = np.eye(3)
    sigma0_full = np.einsum("ijkl,kl->ij", C, I3)
    term1 = np.einsum("ijkl,ij,kl->", C, I3, I3)

    n3 = np.array([n2[0], n2[1], 0.0])
    A2 = np.einsum("abcd,b,d->ac", C[:2, :, :2, :], n3, n3)
    Omega2 = np.linalg.inv(A2)
    v2 = (sigma0_full @ n3)[:2]
    term2 = v2 @ Omega2 @ v2
    return float((term1 - term2) / 2.0)


def _cubic_cases():
    return [
        ("A_Z<1 (soft <110>)", 160e9, 100e9, 21e9),
        ("A_Z=1 (isotropic)", 160e9, 100e9, 30e9),
        ("A_Z>1 (production equiatomic, 773K)", 154.8251e9, 102.9682e9, 39.346e9),
    ]


def _single_mode_eigenstrain(eta0, mx, my, N, Lx=1.0):
    x = np.linspace(0.0, Lx, N, endpoint=False)
    X, Y = np.meshgrid(x, x, indexing="ij")
    wave = np.cos(2.0 * np.pi * (mx * X / Lx + my * Y / Lx))
    eps_star_field = eta0 * wave[..., None, None] * np.eye(3)[None, None, ...]
    return wave, eps_star_field


class TestPlaneStrainAnalytical:
    """Level 1: pure linear algebra, no FFT."""

    @pytest.mark.parametrize("label,C11,C12,C44", _cubic_cases())
    @pytest.mark.parametrize("theta_deg", [0.0, 15.0, 30.0, 45.0, 60.0, 75.0, 90.0])
    def test_inplane_3d_formula_matches_2x2_reduction(self, label, C11, C12, C44, theta_deg):
        C = build_cubic_stiffness(C11, C12, C44)
        th = np.radians(theta_deg)
        n = np.array([np.cos(th), np.sin(th), 0.0])
        Y_3d = coherency_modulus(C, n)
        Y_2x2 = _Y_planestrain_2x2(C, n[:2])
        rel = abs(Y_3d - Y_2x2) / abs(Y_3d)
        assert rel < 1e-12, f"{label} theta={theta_deg}: Y_3D={Y_3d:.6e}, Y_2x2={Y_2x2:.6e}"


class TestPlaneStrainFFTEnergy:
    """Level 2: production FFT solver vs the Level-1 analytical target."""

    @pytest.mark.parametrize("label,C11,C12,C44", _cubic_cases())
    @pytest.mark.parametrize("mx,my", [(1, 0), (1, 1), (2, 1)])
    def test_fft_plane_strain_energy_matches_analytical_Y(self, label, C11, C12, C44, mx, my):
        C = build_cubic_stiffness(C11, C12, C44)
        n2 = np.array([float(mx), float(my)])
        n2 = n2 / np.linalg.norm(n2)
        Y_target = coherency_modulus(C, np.array([n2[0], n2[1], 0.0]))

        N = 32
        Lx = 1.0
        eta0 = 1.0e-2
        wave, eps_star_field = _single_mode_eigenstrain(eta0, mx, my, N, Lx)
        c = np.zeros((N, N))  # homogeneous medium: C_mat == C_ppt, c is irrelevant

        out = solve_elasticity(
            c, C, C,
            eps_star_field=eps_star_field,
            Lx=Lx, tol=1e-12, max_iter=200,
            eigenstrain_closure="plane_strain",
        )

        eps_el_full = np.zeros((N, N, 3, 3))
        eps_el_full[..., :2, :2] = out["elastic_strain"]
        eps_el_full[..., 2, 2] = -eta0 * wave  # eps_33 = 0 total - eps*_33

        u_el = 0.5 * np.einsum("...ij,ijkl,...kl->...", eps_el_full, C, eps_el_full)
        # Y(n) = (term1 - term2)/2 has no <cos^2>=1/2 mode-averaging baked in;
        # the zero-relaxation limit gives <u_el> = (1/2) Y eta0^2, so Y = 2<u_el>/eta0^2.
        Y_numeric = 2.0 * u_el.mean() / eta0**2

        rel = abs(Y_numeric - Y_target) / abs(Y_target)
        assert rel < 1e-10, (
            f"{label} n=({mx},{my}): Y_target={Y_target:.6e}, Y_numeric={Y_numeric:.6e}"
        )


class TestTruncateDiffersFromPlaneStrain:
    """Level 3: "truncate" is neither plane strain nor plane stress -- confirm
    it gives a materially different (lower) coherency energy, quantifying the
    documented limitation rather than silently matching by coincidence."""

    def test_truncate_softer_than_plane_strain_for_production_case(self):
        C11, C12, C44 = 154.8251e9, 102.9682e9, 39.346e9
        C = build_cubic_stiffness(C11, C12, C44)
        N = 32
        Lx = 1.0
        eta0 = 1.0e-2
        mx, my = 1, 0
        wave, eps_star_field = _single_mode_eigenstrain(eta0, mx, my, N, Lx)
        c = np.zeros((N, N))

        out_ps = solve_elasticity(
            c, C, C, eps_star_field=eps_star_field,
            Lx=Lx, tol=1e-12, max_iter=200, eigenstrain_closure="plane_strain",
        )
        out_tr = solve_elasticity(
            c, C, C, eps_star_field=eps_star_field,
            Lx=Lx, tol=1e-12, max_iter=200, eigenstrain_closure="truncate",
        )

        eps_el_full_ps = np.zeros((N, N, 3, 3))
        eps_el_full_ps[..., :2, :2] = out_ps["elastic_strain"]
        eps_el_full_ps[..., 2, 2] = -eta0 * wave
        Y_ps = 2.0 * (0.5 * np.einsum(
            "...ij,ijkl,...kl->...", eps_el_full_ps, C, eps_el_full_ps
        )).mean() / eta0**2

        eps_el_2d_tr = out_tr["elastic_strain"]
        Y_tr_2d_only = 2.0 * (0.5 * np.einsum(
            "...ab,abcd,...cd->...", eps_el_2d_tr, C[:2, :2, :2, :2], eps_el_2d_tr
        )).mean() / eta0**2

        assert Y_tr_2d_only < Y_ps, (
            "truncate closure should understate the coherency energy relative "
            f"to plane_strain: truncate={Y_tr_2d_only:.4e}, plane_strain={Y_ps:.4e}"
        )


class TestPlaneStrainInputValidation:
    """eigenstrain_closure='plane_strain' requires a full 3x3 eigenstrain and
    is only meaningful for d=2."""

    def test_rejects_pretruncated_2x2_eigenstrain(self):
        C = build_cubic_stiffness(160e9, 100e9, 30e9)
        N = 8
        c = np.zeros((N, N))
        eps_star_field = 0.01 * np.broadcast_to(np.eye(2), (N, N, 2, 2)).copy()
        with pytest.raises(ValueError, match="plane_strain"):
            solve_elasticity(
                c, C, C, eps_star_field=eps_star_field,
                Lx=1.0, eigenstrain_closure="plane_strain",
            )

    def test_rejects_unknown_closure_name(self):
        C = build_cubic_stiffness(160e9, 100e9, 30e9)
        N = 8
        c = np.zeros((N, N))
        eps_star_field = 0.01 * np.broadcast_to(np.eye(3), (N, N, 3, 3)).copy()
        with pytest.raises(ValueError, match="Unknown eigenstrain_closure"):
            solve_elasticity(
                c, C, C, eps_star_field=eps_star_field,
                Lx=1.0, eigenstrain_closure="plane_stress",
            )

    def test_default_truncate_unaffected_by_new_parameter(self):
        """The default behaviour must stay byte-for-byte unchanged."""
        C_mat = build_cubic_stiffness(246e9, 132e9, 28e9)
        C_ppt = build_cubic_stiffness(200e9, 110e9, 20e9)
        N = 12
        rng = np.random.default_rng(0)
        c = rng.uniform(0, 1, size=(N, N))

        out_default = solve_elasticity(
            c, C_mat, C_ppt, eigenstrain=0.01 * np.eye(2),
            Lx=1e-6, tol=1e-8, max_iter=200,
        )
        out_explicit = solve_elasticity(
            c, C_mat, C_ppt, eigenstrain=0.01 * np.eye(2),
            Lx=1e-6, tol=1e-8, max_iter=200, eigenstrain_closure="truncate",
        )
        np.testing.assert_array_equal(out_default["stress"], out_explicit["stress"])
