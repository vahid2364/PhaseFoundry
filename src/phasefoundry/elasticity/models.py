"""Eigenstrain and stiffness models for the FFT elastic solver.

The config chooses the model with ``elastic.eigenstrain_model``:

```yaml
elastic:
  eigenstrain_model: fixed_vegard    # default: constant α_k from step 1b
  # eigenstrain_model: local_vegard  # α(c) and ε*(c) from the SQS interpolant
```

``fixed_vegard`` (``FixedVegardModel``):

    ε*(x) = Σ_k α_k c_k(x) I
    μ_el,k = −σ(x) : Λ_k / N_V,   with Λ_k = α_k I
    C_eff = (C_mat + C_ppt) / 2
    C(x) = (1 − h(c_ref)) C_mat + h(c_ref) C_ppt

The α_k are fixed numbers that step 1b evaluates at the midpoint between the tie-line
endpoints. This model is the default.

``local_vegard`` (``LocalVegardModel``):

    ε*(x) = [a(c(x)) − a(c₀)] / a(c₀) I
    μ_el,k = −α_k(c₀) tr(σ(x)) / N_V
    C_eff = C(c₀), the SQS stiffness at the parent composition
    C(x) = the SQS interpolant evaluated at each grid point

ε* is zero wherever c(x) = c₀, as it should be for a uniform parent phase, and the model
follows a nonlinear a(c) across the whole composition range. It needs the SQS elastic
table (``elastic.elastic_csv``, or the MLIP table from step 1b). With
``elastic.lattice_model: bond``, a(x) comes from the bond-length model instead of the SQS
interpolant.

Both models use a spectral (Lippmann-Schwinger, Moulinec-Suquet) scheme with a
homogeneous reference stiffness C_eff. Both leave out the stiffness-contrast term of
Eq. 36, ½ ∂C/∂c_k : ε^el : ε^el, as spectral phase-field codes usually do when the
contrast is moderate. ``local_vegard`` also evaluates α_k at c₀ instead of at each grid
point; adding the spatial variation is a second-order correction that needs the gradient
of the SQS interpolant.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np

from .fft_solver import _interp_stiffness, build_cubic_stiffness
from .interpolation import get_scheme
from .sqs_interpolant import (
    BatchInterp,
    build_batch_interpolators,
    load_and_filter,
)

if TYPE_CHECKING:
    from .bond_model import BondLatticeModel


@runtime_checkable
class EigenstrainModel(Protocol):
    """Interface that every eigenstrain model implements."""

    def eps_star_field(self, c_all: np.ndarray, ndim: int) -> np.ndarray:
        """Return the eigenstrain field ε*(x).

        Args:
            c_all: All independent composition fields, shape ``(n_ind, *spatial)``.
            ndim: Spatial dimension, 2 or 3.

        Returns:
            Eigenstrain, shape ``(*spatial, ndim, ndim)``.
        """
        ...

    def mu_el(self, sigma: np.ndarray, c_all: np.ndarray, N_V: float) -> np.ndarray:
        """Return the elastic chemical potential μ_el,k(x) = δF_el/δc_k.

        Args:
            sigma: Cauchy stress from ``solve_elasticity``, shape ``(*spatial, d, d)``.
            c_all: All independent composition fields, shape ``(n_ind, *spatial)``.
            N_V: Molar density in mol/m³.

        Returns:
            μ_el in J/mol, shape ``(n_ind, *spatial)``.
        """
        ...

    def C_eff(self, C_mat: np.ndarray, C_ppt: np.ndarray) -> np.ndarray:
        """Return the reference stiffness for the Green's operator.

        Args:
            C_mat: Matrix stiffness, shape ``(3, 3, 3, 3)``.
            C_ppt: Precipitate stiffness, shape ``(3, 3, 3, 3)``.

        Returns:
            Reference stiffness, shape ``(3, 3, 3, 3)``.
        """
        ...

    def C_field(self, c_all: np.ndarray, C_mat: np.ndarray, C_ppt: np.ndarray, scheme_name: str = "cubic") -> np.ndarray:
        """Return the stiffness field C(x).

        Args:
            c_all: All independent composition fields, shape ``(n_ind, *spatial)``.
            C_mat: Matrix stiffness, shape ``(3, 3, 3, 3)``.
            C_ppt: Precipitate stiffness, shape ``(3, 3, 3, 3)``.
            scheme_name: Interpolation scheme, for models that blend ``C_mat`` and ``C_ppt``.

        Returns:
            Stiffness field, shape ``(*spatial, 3, 3, 3, 3)``.
        """
        ...


class FixedVegardModel:
    """Eigenstrain linear in composition, with fixed Vegard coefficients from step 1b.

    ε*(x) = Σ_k α_k c_k(x) I. The stiffness field blends ``C_mat`` and ``C_ppt`` by the
    reference element's fraction.

    Attributes:
        Lambda: Vegard tensor, shape ``(n_ind, ndim, ndim)``.
    """

    def __init__(self, Lambda: np.ndarray, C_mat: np.ndarray, C_ppt: np.ndarray) -> None:
        """Initialize the model.

        Args:
            Lambda: Vegard tensor from ``build_vegard_tensor``, shape ``(n_ind, ndim, ndim)``.
            C_mat: Matrix stiffness in Pa, shape ``(3, 3, 3, 3)``.
            C_ppt: Precipitate stiffness in Pa, shape ``(3, 3, 3, 3)``.
        """
        self.Lambda = Lambda
        self._C_mat = C_mat
        self._C_ppt = C_ppt

    def eps_star_field(self, c_all: np.ndarray, ndim: int) -> np.ndarray:
        """Return ε*(x) = Σ_k Λ_k c_k(x), shape ``(*spatial, ndim, ndim)``."""
        Lam = self.Lambda[:, :ndim, :ndim]
        return np.einsum("k...,kij->...ij", c_all, Lam)

    def mu_el(self, sigma: np.ndarray, c_all: np.ndarray, N_V: float) -> np.ndarray:
        """Return μ_el,k = −σ : Λ_k / N_V, shape ``(n_ind, *spatial)``."""
        d = sigma.shape[-1]
        Lam = self.Lambda[:, :d, :d]
        return -np.einsum("...ij,kij->k...", sigma, Lam) / N_V

    def C_eff(self, C_mat: np.ndarray, C_ppt: np.ndarray) -> np.ndarray:
        """Return the mean of ``C_mat`` and ``C_ppt``."""
        return 0.5 * (C_mat + C_ppt)

    def C_field(self, c_all: np.ndarray, C_mat: np.ndarray, C_ppt: np.ndarray, scheme_name: str = "cubic") -> np.ndarray:
        """Blend ``C_mat`` and ``C_ppt`` by h of the reference element's fraction, 1 − Σ_k c_k."""
        scheme = get_scheme(scheme_name)
        c_ref = np.clip(1.0 - c_all.sum(axis=0), 0.0, 1.0)
        return _interp_stiffness(c_ref, C_mat, C_ppt, scheme)

    def __repr__(self) -> str:  # pragma: no cover
        """Return the Vegard coefficients α_k, labelled by component index."""
        n_ind = self.Lambda.shape[0]
        alphas = {f"el{k}": round(self.Lambda[k, 0, 0], 6) for k in range(n_ind)}
        return f"FixedVegardModel(alpha={alphas})"


class LocalVegardModel:
    """Composition-dependent eigenstrain from the SQS lattice-parameter interpolant.

    ε*(x) = [a(c(x)) − a(c₀)] / a(c₀) I. For μ_el, α_k is evaluated once at c₀, a
    first-order approximation that avoids gradient calls on the scattered interpolant at
    every grid point. The stiffness field comes from the SQS interpolant at each grid
    point, and the reference stiffness is C(c₀).
    """

    def __init__(
        self, batch_interps: dict[str, BatchInterp], ind_elements: list[str], c0: np.ndarray, delta_fd: float = 0.02
    ) -> None:
        """Evaluate a(c₀), α_k(c₀) and C(c₀) from the interpolants.

        Args:
            batch_interps: Output of ``build_batch_interpolators``, with ``a_relaxed_A``
                in Å and ``C11_GPa``, ``C12_GPa`` and ``C44_GPa`` in GPa.
            ind_elements: Independent elements, in the order of the composition fields.
            c0: Parent (mean) composition, shape ``(n_ind,)``.
            delta_fd: Step of the central finite difference for α_k at c₀.
        """
        self._interps = batch_interps
        self._ind_elements = ind_elements
        self._N_ind = len(ind_elements)
        self._c0 = np.asarray(c0, dtype=float)

        c0_q = self._c0.reshape(1, -1)
        self._a_ref = float(batch_interps["a_relaxed_A"](c0_q)[0]) * 1e-10

        self._alpha_c0 = self._compute_alpha_at(self._c0, delta_fd)

        self._C_eff_cache = self._build_C_at(self._c0)

    def _eval_a_field(self, c_all: np.ndarray) -> np.ndarray:
        """Return a(c(x)) in metres, shape ``(*spatial,)``."""
        spatial_shape = c_all.shape[1:]
        x_ind = c_all.reshape(self._N_ind, -1).T
        a_flat = self._interps["a_relaxed_A"](x_ind)
        return a_flat.reshape(spatial_shape) * 1e-10

    def _eval_C_field(self, c_all: np.ndarray) -> np.ndarray:
        """Return the cubic stiffness field C(x) in Pa, shape ``(*spatial, 3, 3, 3, 3)``."""
        spatial_shape = c_all.shape[1:]
        n_pts = int(np.prod(spatial_shape))
        x_ind = c_all.reshape(self._N_ind, -1).T

        C11 = self._interps["C11_GPa"](x_ind) * 1e9
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
        """Return the cubic stiffness tensor in Pa at one composition."""
        q = c_vec.reshape(1, -1)
        C11 = float(self._interps["C11_GPa"](q)[0]) * 1e9
        C12 = float(self._interps["C12_GPa"](q)[0]) * 1e9
        C44 = float(self._interps["C44_GPa"](q)[0]) * 1e9
        return build_cubic_stiffness(C11, C12, C44)

    def _compute_alpha_at(self, c_vec: np.ndarray, delta_fd: float) -> np.ndarray:
        """Return α_k = (∂a/∂c_k) / a at one composition, by central finite differences.

        The perturbed fractions are clipped to [0, 1], and a component whose clipped step
        is zero gets α_k = 0.

        Returns:
            α_k of every independent element, shape ``(n_ind,)``.
        """
        c_q = c_vec.reshape(1, -1)
        a_ref = float(self._interps["a_relaxed_A"](c_q)[0]) * 1e-10
        alpha = np.zeros(self._N_ind)
        for k in range(self._N_ind):
            xp = c_vec.copy()
            xp[k] = min(1.0, xp[k] + delta_fd)
            xm = c_vec.copy()
            xm[k] = max(0.0, xm[k] - delta_fd)
            step = xp[k] - xm[k]
            if step < 1e-12:
                continue
            ap = float(self._interps["a_relaxed_A"](xp.reshape(1, -1))[0]) * 1e-10
            am = float(self._interps["a_relaxed_A"](xm.reshape(1, -1))[0]) * 1e-10
            alpha[k] = (ap - am) / step / a_ref
        return alpha

    def eps_star_field(self, c_all: np.ndarray, ndim: int) -> np.ndarray:
        """Return ε*(x) = [a(c(x)) − a(c₀)] / a(c₀) I, shape ``(*spatial, ndim, ndim)``."""
        a_field = self._eval_a_field(c_all)
        eps_T = (a_field - self._a_ref) / self._a_ref
        identity = np.eye(ndim)
        return eps_T[..., None, None] * identity

    def mu_el(self, sigma: np.ndarray, c_all: np.ndarray, N_V: float) -> np.ndarray:
        """Return μ_el,k = −α_k(c₀) tr(σ) / N_V, shape ``(n_ind, *spatial)``."""
        tr_sigma = np.einsum("...ii->...", sigma)
        alpha = self._alpha_c0
        n_extra = tr_sigma.ndim
        alpha_bc = alpha.reshape(self._N_ind, *([1] * n_extra))
        return -alpha_bc * tr_sigma[None, ...] / N_V

    def C_eff(self, C_mat: np.ndarray, C_ppt: np.ndarray) -> np.ndarray:
        """Return the SQS stiffness at c₀; ``C_mat`` and ``C_ppt`` are ignored."""
        return self._C_eff_cache

    def C_field(self, c_all: np.ndarray, C_mat: np.ndarray, C_ppt: np.ndarray, scheme_name: str = "cubic") -> np.ndarray:
        """Return the SQS stiffness at each grid point; the other arguments are ignored."""
        return self._eval_C_field(c_all)

    def __repr__(self) -> str:  # pragma: no cover
        """Return a(c₀) in Å and the α_k(c₀) values."""
        alphas = {el: round(a, 6) for el, a in zip(self._ind_elements, self._alpha_c0, strict=True)}
        return f"LocalVegardModel(a_ref={self._a_ref * 1e10:.4f} Å, alpha_c0={alphas})"


def build_eigenstrain_model(
    model_name: str,
    el_cfg: dict,
    elastic_params: dict,
    ind_elements: list[str],
    c0: list | np.ndarray,
    ndim: int,
    Lambda: np.ndarray | None,
    C_mat: np.ndarray | None,
    C_ppt: np.ndarray | None,
    all_elements: list[str] | None = None,
    table_path: str | Path | None = None,
    lattice_model: BondLatticeModel | None = None,
    reference_element: str | None = None,
) -> EigenstrainModel:
    """Create the eigenstrain model named in the config.

    Args:
        model_name: Value of ``elastic.eigenstrain_model``: ``"fixed_vegard"`` (also
            accepted as ``"fixed"`` or ``"uniform"``) or ``"local_vegard"``.
        el_cfg: The ``simulation.elastic`` block.
        elastic_params: Contents of ``elastic_params.yaml``; may be empty.
        ind_elements: Independent elements, without the reference element.
        c0: Mean composition, shape ``(n_ind,)``. A scalar is accepted for binaries.
        ndim: Spatial dimension. Neither model uses it at present.
        Lambda: Vegard tensor, shape ``(n_ind, ndim, ndim)``, used by ``fixed_vegard``.
        C_mat: Matrix stiffness, used by ``fixed_vegard``.
        C_ppt: Precipitate stiffness, used by ``fixed_vegard``.
        all_elements: All elements, including the reference. ``local_vegard`` uses them to
            select the rows of the SQS table. Without them it falls back to
            ``elastic_params["meta"]["system"]``, then to ``ind_elements`` plus
            ``el_cfg["reference_element"]``.
        table_path: SQS elastic table. Defaults to ``elastic.elastic_csv``; see
            ``sources.elastic_table_path``.
        lattice_model: Bond-length model that replaces the SQS a(x) interpolant
            (``local_vegard`` only). Needs ``reference_element``.
        reference_element: Dependent element. Required with ``lattice_model``.

    Returns:
        The eigenstrain model.

    Raises:
        FileNotFoundError: If ``local_vegard`` is chosen and the SQS table does not exist.
        ValueError: If ``model_name`` is unknown, ``fixed_vegard`` is chosen without ``Lambda``,
            ``C_mat`` and ``C_ppt``, or ``lattice_model`` is given without
            ``reference_element``.
    """
    c0 = np.atleast_1d(np.asarray(c0, dtype=float))

    if model_name in ("fixed_vegard", "uniform", "fixed"):
        if Lambda is None or C_mat is None or C_ppt is None:
            raise ValueError(f"{model_name} requires Lambda, C_mat and C_ppt")
        return FixedVegardModel(Lambda, C_mat, C_ppt)

    if model_name == "local_vegard":
        csv_path = Path(table_path or el_cfg.get("elastic_csv", ""))
        if not csv_path.exists():
            raise FileNotFoundError(
                f"local_vegard model requires the SQS elastic table "
                f"(elastic.elastic_csv, or step 1b with elastic.source: mlip).  "
                f"Not found: {csv_path}"
            )
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
        if lattice_model is not None:
            if reference_element is None:
                raise ValueError("reference_element is required with a lattice_model")
            batch_interps["a_relaxed_A"] = lattice_model.batch_interp(ind_elements, reference_element)
            print(f"  Lattice parameter a(x): bond-length model ({lattice_model.structure.upper()})")
        model = LocalVegardModel(batch_interps, ind_elements, c0)
        print(f"  LocalVegardModel: a_ref = {model._a_ref * 1e10:.4f} Å")
        for k, el in enumerate(ind_elements):
            print(f"    α_{el}(c₀) = {model._alpha_c0[k]:.6f}")
        return model

    raise ValueError(f"Unknown eigenstrain_model: {model_name!r}. Choose 'fixed_vegard' (default) or 'local_vegard'.")
