"""
calphad_pf.elasticity.models
==============================
Eigenstrain and stiffness models for the FFT elastic solver.

Two models are provided; the choice is set in the config via:

    elastic:
      eigenstrain_model: fixed_vegard    # (default) constant α from step 1b
      #                  local_vegard    # α(c) and ε*(c) from SQS interpolant

fixed_vegard (FixedVegardModel)
---------------------------------
  ε*(x) = Σ_k α_k c_k(x) I
  μ_el_k = −σ(x):Λ_k / N_V          (Λ_k = α_k I)
  C_eff  = ½(C_mat + C_ppt)          (Voigt average of endpoint stiffnesses)
  C(x)   = (1−h(c_ref)) C_mat + h(c_ref) C_ppt   (h = interpolation scheme)

  α_k are fixed scalars evaluated at the midpoint between tie-line endpoints
  by step 1b.  This is the current (pre-generalisation) behaviour and is the
  default so that existing configs work unchanged.

local_vegard (LocalVegardModel)
---------------------------------
  ε*(x) = [a(c(x)) − a(c₀)] / a(c₀) · I
  μ_el_k = −α_k(c₀) tr(σ(x)) / N_V   (α_k evaluated at c₀, not midpoint)
  C_eff  = C(c₀)                       (SQS stiffness at parent composition)
  C(x)   = SQS interpolant evaluated at c(x) pointwise

  ε* vanishes everywhere when c(x)=c₀ (correct for a uniform parent phase).
  Captures nonlinear Vegard behaviour across the full composition range.

  Requires the SQS CSV path (from elastic.elastic_csv in config) and the
  filtered DataFrame from sqs_interpolant.build_batch_interpolators().

Academic note
-------------
  Both models use the Lippmann–Schwinger / Moulinec–Suquet spectral scheme
  with a homogeneous reference C_eff.  The stiffness-contrast correction
  (G&A Eq. 36 term 1, ½ ∂C/∂c_k : ε_el : ε_el) is neglected in both modes
  as is standard in spectral phase-field codes when the stiffness contrast
  is moderate.  local_vegard additionally neglects spatial variation of α_k
  in μ_el (α is evaluated at c₀, not at each grid point) — this is a
  second-order correction that can be added when the batch gradient of the
  SQS interpolant is available.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

from .fft_solver import build_cubic_stiffness, _interp_stiffness
from .interpolation import get_scheme
from .sqs_interpolant import (
    load_and_filter,
    build_batch_interpolators,
    build_interpolators,
    query_composition,
    BatchInterp,
)


# ── protocol (structural typing) ──────────────────────────────────────────────

@runtime_checkable
class EigenstrainModel(Protocol):
    """Minimal interface every eigenstrain model must satisfy."""

    def eps_star_field(self, c_all: np.ndarray, ndim: int) -> np.ndarray:
        """
        Eigenstrain field ε*(x).

        Parameters
        ----------
        c_all : (N_ind, *spatial)   all independent composition fields
        ndim  : spatial dimension (2 or 3)

        Returns
        -------
        eps_star : (*spatial, ndim, ndim)
        """
        ...

    def mu_el(self, sigma: np.ndarray, c_all: np.ndarray,
              N_V: float) -> np.ndarray:
        """
        Elastic chemical potential μ_el_k(x) = δF_el/δc_k.

        Parameters
        ----------
        sigma : (*spatial, d, d)    Cauchy stress from solve_elasticity
        c_all : (N_ind, *spatial)
        N_V   : float               molar density [mol/m³]

        Returns
        -------
        mu_el : (N_ind, *spatial)   [J/mol]
        """
        ...

    def C_eff(self, C_mat: np.ndarray, C_ppt: np.ndarray) -> np.ndarray:
        """Reference stiffness (3,3,3,3) for the Green's operator."""
        ...

    def C_field(self, c_all: np.ndarray,
                C_mat: np.ndarray, C_ppt: np.ndarray,
                scheme_name: str = "cubic") -> np.ndarray:
        """
        Spatially varying stiffness field C(x).

        Returns
        -------
        C_field : (*spatial, 3, 3, 3, 3)
        """
        ...


# ── FixedVegardModel ──────────────────────────────────────────────────────────

class FixedVegardModel:
    """
    Current (pre-generalisation) behaviour.  α_k are fixed scalars from step 1b.
    ε*(x) = Σ_k α_k c_k(x) I — linear in composition fields.
    """

    def __init__(self, Lambda: np.ndarray,
                 C_mat: np.ndarray, C_ppt: np.ndarray) -> None:
        """
        Parameters
        ----------
        Lambda : (N_ind, ndim, ndim)   Vegard tensor from build_vegard_tensor()
        C_mat  : (3,3,3,3)             matrix stiffness [Pa]
        C_ppt  : (3,3,3,3)             precipitate stiffness [Pa]
        """
        self.Lambda = Lambda
        self._C_mat = C_mat
        self._C_ppt = C_ppt

    # ── EigenstrainModel interface ────────────────────────────────────────────

    def eps_star_field(self, c_all: np.ndarray, ndim: int) -> np.ndarray:
        Lam = self.Lambda[:, :ndim, :ndim]      # (N_ind, d, d)
        return np.einsum('k...,kij->...ij', c_all, Lam)

    def mu_el(self, sigma: np.ndarray, c_all: np.ndarray,
              N_V: float) -> np.ndarray:
        d   = sigma.shape[-1]
        Lam = self.Lambda[:, :d, :d]
        return -np.einsum('...ij,kij->k...', sigma, Lam) / N_V

    def C_eff(self, C_mat: np.ndarray, C_ppt: np.ndarray) -> np.ndarray:
        return 0.5 * (C_mat + C_ppt)

    def C_field(self, c_all: np.ndarray,
                C_mat: np.ndarray, C_ppt: np.ndarray,
                scheme_name: str = "cubic") -> np.ndarray:
        scheme = get_scheme(scheme_name)
        # order parameter = reference element fraction
        c_ref = np.clip(1.0 - c_all.sum(axis=0), 0.0, 1.0)
        return _interp_stiffness(c_ref, C_mat, C_ppt, scheme)

    def __repr__(self) -> str:  # pragma: no cover
        n_ind = self.Lambda.shape[0]
        alphas = {f"el{k}": round(self.Lambda[k, 0, 0], 6) for k in range(n_ind)}
        return f"FixedVegardModel(alpha={alphas})"


# ── LocalVegardModel ──────────────────────────────────────────────────────────

class LocalVegardModel:
    """
    Composition-dependent eigenstrain via SQS lattice-parameter interpolant.

    ε*(x) = [a(c(x)) − a(c₀)] / a(c₀) · I

    α_k for μ_el is evaluated at c₀ (first-order approximation; avoids
    per-point gradient calls on the scattered interpolant).

    C(x) is evaluated from the SQS interpolant pointwise across the grid.
    C_eff = C(c₀).
    """

    def __init__(self,
                 batch_interps: dict[str, BatchInterp],
                 ind_elements: list[str],
                 c0: np.ndarray,
                 delta_fd: float = 0.02) -> None:
        """
        Parameters
        ----------
        batch_interps : dict from build_batch_interpolators()
        ind_elements  : independent element names (same order as c_all axis 0)
        c0            : (N_ind,)  parent composition (mean composition)
        delta_fd      : finite-difference step for α_k at c₀
        """
        self._interps     = batch_interps
        self._ind_elements = ind_elements
        self._N_ind       = len(ind_elements)
        self._c0          = np.asarray(c0, dtype=float)

        # a_ref = a(c₀)
        c0_q          = self._c0.reshape(1, -1)
        self._a_ref   = float(batch_interps["a_relaxed_A"](c0_q)[0]) * 1e-10  # → m

        # α_k at c₀: (∂a/∂c_k)(c₀) / a(c₀)  via central FD
        self._alpha_c0 = self._compute_alpha_at(self._c0, delta_fd)

        # C_eff = C(c₀)
        self._C_eff_cache = self._build_C_at(self._c0)

    # ── internal helpers ──────────────────────────────────────────────────────

    def _eval_a_field(self, c_all: np.ndarray) -> np.ndarray:
        """a(c(x)) in metres, shape (*spatial,)."""
        spatial_shape = c_all.shape[1:]
        x_ind = c_all.reshape(self._N_ind, -1).T        # (n_pts, N_ind)
        a_flat = self._interps["a_relaxed_A"](x_ind)    # (n_pts,) in Å
        return a_flat.reshape(spatial_shape) * 1e-10     # → m

    def _eval_C_field(self, c_all: np.ndarray) -> np.ndarray:
        """C(x) shape (*spatial, 3,3,3,3) in Pa."""
        spatial_shape = c_all.shape[1:]
        n_pts = int(np.prod(spatial_shape))
        x_ind = c_all.reshape(self._N_ind, -1).T        # (n_pts, N_ind)

        C11 = self._interps["C11_GPa"](x_ind) * 1e9    # (n_pts,) Pa
        C12 = self._interps["C12_GPa"](x_ind) * 1e9
        C44 = self._interps["C44_GPa"](x_ind) * 1e9

        C = np.zeros((n_pts, 3, 3, 3, 3), dtype=np.float64)
        for i in range(3):
            C[:, i, i, i, i] = C11
            for j in range(3):
                if i != j:
                    C[:, i, i, j, j] = C12
                    C[:, i, j, i, j] = C44
                    C[:, i, j, j, i] = C44
        return C.reshape(*spatial_shape, 3, 3, 3, 3)

    def _build_C_at(self, c_vec: np.ndarray) -> np.ndarray:
        """Stiffness tensor (3,3,3,3) at a single composition vector."""
        q = c_vec.reshape(1, -1)
        C11 = float(self._interps["C11_GPa"](q)[0]) * 1e9
        C12 = float(self._interps["C12_GPa"](q)[0]) * 1e9
        C44 = float(self._interps["C44_GPa"](q)[0]) * 1e9
        return build_cubic_stiffness(C11, C12, C44)

    def _compute_alpha_at(self, c_vec: np.ndarray,
                           delta_fd: float) -> np.ndarray:
        """α_k = (∂a/∂c_k)(c_vec) / a(c_vec) for all k.  Returns (N_ind,)."""
        c_q   = c_vec.reshape(1, -1)
        a_ref = float(self._interps["a_relaxed_A"](c_q)[0]) * 1e-10
        alpha = np.zeros(self._N_ind)
        for k in range(self._N_ind):
            xp = c_vec.copy(); xp[k] = min(1.0, xp[k] + delta_fd)
            xm = c_vec.copy(); xm[k] = max(0.0, xm[k] - delta_fd)
            step = xp[k] - xm[k]
            if step < 1e-12:
                continue
            ap = float(self._interps["a_relaxed_A"](xp.reshape(1, -1))[0]) * 1e-10
            am = float(self._interps["a_relaxed_A"](xm.reshape(1, -1))[0]) * 1e-10
            alpha[k] = (ap - am) / step / a_ref
        return alpha

    # ── EigenstrainModel interface ────────────────────────────────────────────

    def eps_star_field(self, c_all: np.ndarray, ndim: int) -> np.ndarray:
        a_field = self._eval_a_field(c_all)               # (*spatial,) in m
        eps_T   = (a_field - self._a_ref) / self._a_ref   # (*spatial,)
        I       = np.eye(ndim)
        return eps_T[..., None, None] * I                  # (*spatial, d, d)

    def mu_el(self, sigma: np.ndarray, c_all: np.ndarray,
              N_V: float) -> np.ndarray:
        d        = sigma.shape[-1]
        tr_sigma = np.einsum('...ii->...', sigma)          # (*spatial,)
        # α_k at c₀ (fixed) — shape (N_ind, 1, ..., 1) for broadcasting
        alpha    = self._alpha_c0                          # (N_ind,)
        n_extra  = tr_sigma.ndim
        alpha_bc = alpha.reshape(self._N_ind, *([1] * n_extra))
        return -alpha_bc * tr_sigma[None, ...] / N_V

    def C_eff(self, C_mat: np.ndarray, C_ppt: np.ndarray) -> np.ndarray:
        return self._C_eff_cache

    def C_field(self, c_all: np.ndarray,
                C_mat: np.ndarray, C_ppt: np.ndarray,
                scheme_name: str = "cubic") -> np.ndarray:
        return self._eval_C_field(c_all)

    def __repr__(self) -> str:  # pragma: no cover
        alphas = {el: round(a, 6)
                  for el, a in zip(self._ind_elements, self._alpha_c0)}
        return (f"LocalVegardModel(a_ref={self._a_ref*1e10:.4f} Å, "
                f"alpha_c0={alphas})")


# ── factory ───────────────────────────────────────────────────────────────────

def build_eigenstrain_model(
    model_name   : str,
    el_cfg       : dict,
    elastic_params: dict,
    ind_elements : list[str],
    c0           : list | np.ndarray,
    ndim         : int,
    Lambda       : np.ndarray,
    C_mat        : np.ndarray,
    C_ppt        : np.ndarray,
    all_elements : list[str] | None = None,
) -> EigenstrainModel:
    """
    Instantiate the correct EigenstrainModel from config.

    Parameters
    ----------
    model_name    : value of elastic.eigenstrain_model in config
    el_cfg        : the elastic sub-dict from simulation config
    elastic_params: loaded elastic_params.yaml (may be empty for isotropic)
    ind_elements  : independent element names (no reference element)
    c0            : mean composition (N_ind,)
    ndim          : spatial dimension
    Lambda        : Vegard tensor (N_ind, ndim, ndim) — always built by caller
    C_mat, C_ppt  : endpoint stiffness tensors
    all_elements  : full element list including reference (used by local_vegard
                    to filter the SQS CSV to the correct subspace)
    """
    c0 = np.asarray(c0, dtype=float)

    if model_name in ("fixed_vegard", "uniform", "fixed"):
        return FixedVegardModel(Lambda, C_mat, C_ppt)

    if model_name == "local_vegard":
        csv_path = Path(el_cfg.get("elastic_csv", ""))
        if not csv_path.exists():
            raise FileNotFoundError(
                f"local_vegard model requires elastic.elastic_csv pointing to "
                f"the SQS CSV file.  Not found: {csv_path}"
            )
        # Determine which elements define the active subspace for CSV filtering.
        # Priority: all_elements arg > elastic_params meta > ind_elements + ref hint
        meta_sys = elastic_params.get("meta", {}).get("system", None)
        if all_elements is not None:
            csv_elements = [e.upper() for e in all_elements]
        elif meta_sys:
            csv_elements = [e.upper() for e in meta_sys]
        else:
            ref_hint = el_cfg.get("reference_element", "")
            csv_elements = [e.upper() for e in ind_elements]
            if ref_hint:
                csv_elements.append(ref_hint.upper())

        df = load_and_filter(csv_path, csv_elements)
        batch_interps = build_batch_interpolators(df, ind_elements)
        model = LocalVegardModel(batch_interps, ind_elements, c0)
        print(f"  LocalVegardModel: a_ref = {model._a_ref*1e10:.4f} Å")
        for k, el in enumerate(ind_elements):
            print(f"    α_{el}(c₀) = {model._alpha_c0[k]:.6f}")
        return model

    raise ValueError(
        f"Unknown eigenstrain_model: {model_name!r}. "
        "Choose 'fixed_vegard' (default) or 'local_vegard'."
    )
