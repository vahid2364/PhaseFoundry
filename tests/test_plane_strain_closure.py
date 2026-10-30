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

Level 3 — Contrast: the default "in_plane_only" closure must differ from
    both of the above by a material- and direction-dependent amount,
    confirming it is neither plane strain nor plane stress, just omission
    (as documented in solve_elasticity's docstring).

Level 4 — Classical circular-inclusion (Eshelby disk): a uniform
    dilatational eigenstrain filling a circular region, in an elastically
    HOMOGENEOUS isotropic medium (same modulus inside and outside, so the
    only thing under test is the eigenstrain closure, not a stiffness
    contrast), has a known closed-form uniform interior stress -- one
    formula for a genuinely-2D material ("in_plane_only"), a different one
    (ratio 1+nu) for true plane strain. This is the same kind of check
    Gururajan & Abinandanan (2007) themselves ran against Mura's analytical
    solutions. See docs/tutorial_plane_strain_closure.md Step 5a for the
    by-hand derivation of both formulas.
"""

from __future__ import annotations

import numpy as np
import pytest

from phasefoundry.thermodynamics.coherent_hessian import coherency_modulus
from phasefoundry.elasticity.fft_solver import (
    build_cubic_stiffness, build_isotropic_stiffness, solve_elasticity,
)


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


class TestInPlaneOnlyDiffersFromPlaneStrain:
    """Level 3: "in_plane_only" is neither plane strain nor plane stress --
    confirm it gives a materially different (lower) coherency energy,
    quantifying the documented limitation rather than silently matching by
    coincidence."""

    def test_in_plane_only_softer_than_plane_strain_for_production_case(self):
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
        out_ip = solve_elasticity(
            c, C, C, eps_star_field=eps_star_field,
            Lx=Lx, tol=1e-12, max_iter=200, eigenstrain_closure="in_plane_only",
        )

        eps_el_full_ps = np.zeros((N, N, 3, 3))
        eps_el_full_ps[..., :2, :2] = out_ps["elastic_strain"]
        eps_el_full_ps[..., 2, 2] = -eta0 * wave
        Y_ps = 2.0 * (0.5 * np.einsum(
            "...ij,ijkl,...kl->...", eps_el_full_ps, C, eps_el_full_ps
        )).mean() / eta0**2

        eps_el_2d_ip = out_ip["elastic_strain"]
        Y_ip_2d_only = 2.0 * (0.5 * np.einsum(
            "...ab,abcd,...cd->...", eps_el_2d_ip, C[:2, :2, :2, :2], eps_el_2d_ip
        )).mean() / eta0**2

        assert Y_ip_2d_only < Y_ps, (
            "in_plane_only closure should understate the coherency energy "
            f"relative to plane_strain: in_plane_only={Y_ip_2d_only:.4e}, "
            f"plane_strain={Y_ps:.4e}"
        )


class TestCircularInclusion:
    """Level 4: classical Eshelby-disk check (see docs/tutorial_plane_strain_closure.md
    Step 5a for the by-hand derivation of both closed-form target stresses)."""

    @staticmethod
    def _sigma_rr_analytical(G, nu, eps0, closure):
        if closure == "in_plane_only":
            return -G * eps0 / (1.0 - nu)
        if closure == "plane_strain":
            return -G * (1.0 + nu) * eps0 / (1.0 - nu)
        raise ValueError(closure)

    @staticmethod
    def _run_disk(G, nu, eps0, a_over_Lx, N, Lx=1.0):
        C = build_isotropic_stiffness(G, nu)
        x = np.linspace(0.0, Lx, N, endpoint=False)
        X, Y = np.meshgrid(x, x, indexing="ij")
        x0 = y0 = Lx / 2.0
        dx = np.minimum(np.abs(X - x0), Lx - np.abs(X - x0))
        dy = np.minimum(np.abs(Y - y0), Lx - np.abs(Y - y0))
        r = np.sqrt(dx**2 + dy**2)
        a = a_over_Lx * Lx
        disk = (r < a).astype(float)

        eps_star_field = eps0 * disk[..., None, None] * np.eye(3)[None, None, ...]
        results = {}
        for closure in ("in_plane_only", "plane_strain"):
            out = solve_elasticity(
                disk, C, C, eps_star_field=eps_star_field,
                Lx=Lx, tol=1e-10, max_iter=1000, eigenstrain_closure=closure,
            )
            inner = r < 0.5 * a
            results[closure] = float(out["stress"][..., 0, 0][inner].mean())
        return results

    def test_fft_matches_each_closures_own_analytical_target(self):
        """Both closures must land near their own (different) analytical
        answer -- the residual is a periodic-image finite-domain effect
        (shrinks with a/Lx, not with resolution), not solver error."""
        G, nu, eps0 = 50e9, 0.3, 0.01
        sigma = self._run_disk(G, nu, eps0, a_over_Lx=0.05, N=256)
        for closure in ("in_plane_only", "plane_strain"):
            target = self._sigma_rr_analytical(G, nu, eps0, closure)
            rel = abs(sigma[closure] - target) / abs(target)
            assert rel < 0.02, (
                f"{closure}: FFT={sigma[closure]:.4e}, analytical={target:.4e}, "
                f"rel diff={rel:.4f}"
            )

    def test_ratio_matches_one_plus_nu(self):
        """The ratio plane_strain/in_plane_only must match the closed-form
        1+nu prediction, independent of the periodic-image error common to
        both."""
        G, nu, eps0 = 50e9, 0.3, 0.01
        sigma = self._run_disk(G, nu, eps0, a_over_Lx=0.08, N=256)
        ratio = sigma["plane_strain"] / sigma["in_plane_only"]
        assert abs(ratio - (1.0 + nu)) < 1e-3, f"ratio={ratio:.6f}, expected {1+nu}"

    def test_error_shrinks_with_smaller_inclusion_not_with_resolution(self):
        """Confirms the residual gap to the analytical (infinite-matrix)
        target is a finite-domain-size artifact: it tracks a/Lx, and is
        essentially unchanged by doubling the grid resolution."""
        G, nu, eps0 = 50e9, 0.3, 0.01

        def err(a_over_Lx, N):
            sigma = self._run_disk(G, nu, eps0, a_over_Lx, N)
            target = self._sigma_rr_analytical(G, nu, eps0, "in_plane_only")
            return abs(sigma["in_plane_only"] - target) / abs(target)

        err_large = err(0.12, 256)
        err_small = err(0.03, 256)
        assert err_small < err_large / 10, (
            f"shrinking a/Lx should shrink the error roughly linearly: "
            f"a/Lx=0.12 -> {err_large:.4f}, a/Lx=0.03 -> {err_small:.4f}"
        )

        err_N256 = err(0.08, 256)
        err_N512 = err(0.08, 512)
        assert abs(err_N512 - err_N256) < 0.01, (
            f"doubling resolution at fixed a/Lx should barely move the error: "
            f"N=256 -> {err_N256:.4f}, N=512 -> {err_N512:.4f}"
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

    def test_default_in_plane_only_unaffected_by_new_parameter(self):
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
            Lx=1e-6, tol=1e-8, max_iter=200, eigenstrain_closure="in_plane_only",
        )
        np.testing.assert_array_equal(out_default["stress"], out_explicit["stress"])
