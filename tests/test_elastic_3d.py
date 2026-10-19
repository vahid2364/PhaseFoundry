"""Tests for the FFT elasticity solver in 2-D and 3-D.

Checks the stress for a uniform eigenstrain, the output shapes and the stress symmetry,
and runs a short 3-D Cahn-Hilliard simulation with elastic coupling.
"""

from __future__ import annotations

import numpy as np
import pytest

from phasefoundry.elasticity.fft_solver import (
    build_cubic_stiffness,
    build_isotropic_stiffness,
    solve_elasticity,
)


def _cubic():
    """Return a cubic stiffness with Nb-like constants (C11 = 246, C12 = 132, C44 = 28 GPa), in Pa."""
    return build_cubic_stiffness(
        C11=246e9,
        C12=132e9,
        C44=28e9,
    )


def _isotropic():
    """Return an isotropic stiffness with G = 50 GPa and ν = 0.3, in Pa."""
    return build_isotropic_stiffness(G=50e9, nu=0.30)


class TestHomogeneousEigenstrain:
    """Uniform eigenstrain field with no stiffness contrast (``C_mat`` equal to ``C_ppt``)."""

    @pytest.mark.parametrize("N,ndim", [(16, 2), (16, 3)])
    def test_uniform_eigenstrain_stress(self, N, ndim):
        """Mean σ_11 equals C : (−ε*), the stress at zero total strain, with ε* passed as ``eps_star_field``."""
        C = _cubic()
        eps_T = 0.01
        # Uniform eigenstrain field: eps*(x) = eps_T * I for every grid point
        es = eps_T * np.eye(ndim)
        eps_star_field = np.broadcast_to(es, (*(N,) * ndim, ndim, ndim)).copy()

        c = 0.5 * np.ones((N,) * ndim)  # composition irrelevant when using eps_star_field
        result = solve_elasticity(
            c,
            C,
            C,
            eps_star_field=eps_star_field,
            Lx=1e-6,
            tol=1e-10,
            max_iter=200,
        )
        stress = result["stress"]

        # Expected: sigma_ij = C_ijkl * (-eps_T * delta_kl)  (summed over k,l)
        expected_sigma_11 = sum(-C[0, 0, k_, l_] * eps_T * (k_ == l_) for k_ in range(ndim) for l_ in range(ndim))
        sigma_11_mean = float(stress[..., 0, 0].mean())
        assert abs(sigma_11_mean - expected_sigma_11) < 1e6, (
            f"ndim={ndim}: σ_11 = {sigma_11_mean:.3e}, expected {expected_sigma_11:.3e}"
        )

    @pytest.mark.parametrize("ndim", [2, 3])
    def test_stress_spatially_uniform(self, ndim):
        """The stress is uniform in space, with a standard deviation below 1 kPa in every component."""
        N = 16
        C = _isotropic()
        eps_T = 0.005
        es = eps_T * np.eye(ndim)
        eps_star_field = np.broadcast_to(es, (*(N,) * ndim, ndim, ndim)).copy()
        c = np.zeros((N,) * ndim)

        result = solve_elasticity(
            c,
            C,
            C,
            eps_star_field=eps_star_field,
            Lx=1e-6,
            tol=1e-10,
            max_iter=500,
        )
        stress = result["stress"]
        # Spatial variance of stress should be negligible
        for i in range(ndim):
            for j in range(ndim):
                std = float(stress[..., i, j].std())
                assert std < 1e3, f"ndim={ndim}: σ_{i}{j} not uniform, std={std:.2e} Pa"


class TestConsistency2D3D:
    """Output shapes and stress symmetry in 2-D and 3-D."""

    def test_output_shapes_correct_2d_and_3d(self):
        """Strain, stress, displacement and E have the expected shapes, and the stress is finite."""
        for N, ndim in [(16, 2), (8, 3)]:
            C_mat = _cubic()
            C_ppt = build_cubic_stiffness(C11=200e9, C12=110e9, C44=20e9)
            c = np.random.default_rng(42).uniform(0, 1, size=(N,) * ndim)
            result = solve_elasticity(
                c,
                C_mat,
                C_ppt,
                eigenstrain=0.01 * np.eye(ndim),
                Lx=1e-6,
                tol=1e-5,
                max_iter=100,
            )
            spatial = (N,) * ndim
            assert result["strain"].shape == (*spatial, ndim, ndim), f"ndim={ndim} strain shape"
            assert result["stress"].shape == (*spatial, ndim, ndim), f"ndim={ndim} stress shape"
            assert result["displacement"].shape == (*spatial, ndim), f"ndim={ndim} displ shape"
            assert result["E"].shape == (ndim, ndim), f"ndim={ndim} E shape"
            assert np.all(np.isfinite(result["stress"])), f"ndim={ndim} NaN in stress"

    def test_stress_tensor_symmetric_3d(self):
        """The 3-D Cauchy stress is symmetric, σ_ij = σ_ji, to a relative RMS of 1e-6."""
        N = 12
        C_mat = _cubic()
        C_ppt = build_cubic_stiffness(C11=200e9, C12=110e9, C44=20e9)
        rng = np.random.default_rng(2)
        c3 = rng.uniform(0, 1, size=(N, N, N))

        result = solve_elasticity(
            c3,
            C_mat,
            C_ppt,
            eigenstrain=0.01 * np.eye(3),
            Lx=1e-6,
            tol=1e-8,
            max_iter=500,
        )
        stress = result["stress"]
        for i, j in [(0, 1), (0, 2), (1, 2)]:
            asym = float(np.sqrt(np.mean((stress[..., i, j] - stress[..., j, i]) ** 2)))
            mag = float(np.sqrt(np.mean(stress[..., i, i] ** 2))) + 1.0
            assert asym < 1e-6 * mag, f"stress not symmetric: σ_{i}{j} - σ_{j}{i} rms = {asym:.2e}"


class TestElastic3DIntegration:
    """Short 3-D Cahn-Hilliard run with elastic coupling."""

    def test_ch_elastic_3d_no_nan(self):
        """A 100-step 3-D run with a double-well free energy stays finite and has shape (1, N, N, N)."""
        from dataclasses import dataclass
        from dataclasses import field as dc_field

        from phasefoundry.elasticity import build_vegard_tensor
        from phasefoundry.phasefield.solver import solve_cahn_hilliard

        @dataclass
        class _FakeInterp3D:
            """Double-well stand-in for GmInterpolant, g(c) = 10 c² (1 − c)²."""

            N_ind: int = 1
            ind_elements: list = dc_field(default_factory=lambda: ["A"])
            reference_element: str = "B"
            grid_bounds: object = None
            derivative_method: str = "analytic"
            temperature_K: float = 773.0  # noqa: N815  (mirrors GmInterpolant.temperature_K)

            def gm(self, c):
                """Return g(c)."""
                c0 = c[0] if c.ndim >= 2 else c
                return 10.0 * c0**2 * (1 - c0) ** 2

            def mu(self, c):
                """Return dg/dc with a leading component axis."""
                c0 = c[0]
                dmu = 10.0 * (2 * c0 * (1 - c0) ** 2 - 2 * c0**2 * (1 - c0))
                return dmu[np.newaxis, ...]

            def d2g(self, c):
                """Return d²g/dc², shape ``(1, 1, n_pts)``."""
                c0 = c[0]
                d2 = 10.0 * (2 * (1 - c0) ** 2 - 8 * c0 * (1 - c0) + 2 * c0**2)
                return d2.reshape(1, 1, -1)

        N = 12
        C_mat = _cubic()
        C_ppt = build_cubic_stiffness(C11=200e9, C12=110e9, C44=20e9)
        Lambda = build_vegard_tensor({"A": 0.05}, ["A"], ndim=3)

        result = solve_cahn_hilliard(
            interpolant=_FakeInterp3D(),
            M=1e-2,
            kappa=1e-3,
            c0=0.5,
            N=N,
            Lx=1.0,
            noise=0.02,
            seed=7,
            dt=1e-3,
            n_steps=100,
            save_every=100,
            output_dir=None,
            save_frames=False,
            ndim=3,
            elastic_enabled=True,
            elastic_C_mat=C_mat,
            elastic_C_ppt=C_ppt,
            elastic_Lambda=Lambda,
            elastic_N_V=1.0,
            elastic_solve_every=10,
            elastic_tol=1e-4,
            elastic_max_iter=200,
        )

        c_final = result.compositions
        assert np.all(np.isfinite(c_final)), "3D elastic CH: NaN in final field"
        assert c_final.shape == (1, N, N, N)
