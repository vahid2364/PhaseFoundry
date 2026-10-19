"""Bond-length model of the lattice parameter (Tandoc et al., Materialia 2025).

The lattice parameter is proportional to the composition-weighted first-nearest-neighbour
(FNN) bond length:

    d̄(x) = Σ_ij x_i x_j d_ij
    a(x) = K · d̄(x),   with K = 2/√3 (BCC), √2 (FCC) or 1 (HCP)

The pair bond lengths d_ij come from MLIP relaxations of pure-element cells (i = j) and
binary ordered cells (B2, L1_0 or D0_19, i ≠ j). ``phasefoundry.atomistic.compute_bond_table``
computes them and stores a symmetric matrix CSV in the format of MaterialsFramework's
``BondLatticeParameter.from_csv``:

    ,Hf,Nb,Ti,V
    Hf,3.0912,2.9978,...
    ...

Because a(x) is a quadratic form, its gradient with respect to the independent mole
fractions (with the reference element as the dependent one) has a closed form:

    ∂a/∂x_k = 2K [(D x)_k − (D x)_ref]

The module needs only NumPy and pandas, so the simulation steps can use it without
MaterialsFramework installed.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

STRUCTURE_FACTOR = {
    "bcc": 2.0 / np.sqrt(3.0),
    "fcc": np.sqrt(2.0),
    "hcp": 1.0,
}
"""Factor K in a = K · d̄ for each crystal structure. The HCP value assumes the ideal c/a."""


class BondLatticeModel:
    """Lattice parameter a(x) and its gradient from a pairwise bond-length matrix.

    Element names are matched case-insensitively, so config names (``NB``) and pymatgen
    symbols (``Nb``) both work. The methods take arrays of compositions.

    Attributes:
        elements: Element symbols, capitalised (``Nb``), in matrix order.
        bonds: Symmetric FNN bond-length matrix in Å, shape ``(n_el, n_el)``.
        structure: Crystal structure, ``"bcc"``, ``"fcc"`` or ``"hcp"``.
        K: Structure factor from ``STRUCTURE_FACTOR``.
    """

    def __init__(self, elements: list[str], bonds: ArrayLike, structure: str = "bcc") -> None:
        """Initialize the model and check the bond matrix.

        Args:
            elements: Element symbols in the row and column order of ``bonds``.
            bonds: Symmetric FNN bond lengths d_ij in Å, shape ``(n_el, n_el)``.
            structure: Crystal structure, ``"bcc"``, ``"fcc"`` or ``"hcp"``.

        Raises:
            ValueError: If ``structure`` is unknown, or ``bonds`` has the wrong shape or is
                not symmetric.
        """
        structure = structure.lower()
        if structure not in STRUCTURE_FACTOR:
            raise ValueError(f"Unknown structure {structure!r}. Choose from {sorted(STRUCTURE_FACTOR)}.")
        bonds = np.asarray(bonds, dtype=float)
        n_el = len(elements)
        if bonds.shape != (n_el, n_el):
            raise ValueError(f"bonds must have shape ({n_el}, {n_el}), got {bonds.shape}")
        if not np.allclose(bonds, bonds.T):
            raise ValueError("bonds matrix must be symmetric")

        self.elements = [e.capitalize() for e in elements]
        self.bonds = bonds
        self.structure = structure
        self.K = STRUCTURE_FACTOR[structure]
        self._index = {e.upper(): i for i, e in enumerate(self.elements)}

    @classmethod
    def from_csv(cls, path: str | Path, structure: str = "bcc") -> BondLatticeModel:
        """Load a bond-length matrix CSV in MaterialsFramework format.

        Args:
            path: CSV with element symbols as both the index column and the header row.
            structure: Crystal structure, ``"bcc"``, ``"fcc"`` or ``"hcp"``.

        Returns:
            The bond-length model.

        Raises:
            ValueError: If the row and column labels differ.
        """
        df = pd.read_csv(path, index_col=0)
        if list(df.index) != list(df.columns):
            raise ValueError(f"{path}: row and column element labels differ")
        return cls(list(df.columns), df.values, structure)

    def to_csv(self, path: str | Path) -> Path:
        """Write the bond matrix in MaterialsFramework ``from_csv`` format.

        Args:
            path: Output CSV path. Missing parent directories are created.

        Returns:
            The output path.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(self.bonds, index=self.elements, columns=self.elements).to_csv(path)
        return path

    def _full_composition(self, x_ind: np.ndarray, ind_elements: list[str], reference_element: str) -> np.ndarray:
        """Expand independent mole fractions to full compositions in ``self.elements`` order.

        Takes shape ``(n_pts, n_ind)`` and returns ``(n_pts, n_el)``. The reference element
        gets one minus the sum of the others, and elements in neither list get zero.
        """
        x_ind = np.atleast_2d(np.asarray(x_ind, dtype=float))
        x = np.zeros((x_ind.shape[0], len(self.elements)))
        for k, el in enumerate(ind_elements):
            x[:, self._idx(el)] = x_ind[:, k]
        x[:, self._idx(reference_element)] = 1.0 - x_ind.sum(axis=1)
        return x

    def _idx(self, el: str) -> int:
        """Return the matrix index of an element, ignoring case."""
        try:
            return self._index[el.upper()]
        except KeyError:
            raise KeyError(f"Element {el!r} not in bond table {self.elements}") from None

    def a(self, x: ArrayLike) -> np.ndarray:
        """Return the lattice parameter in Å for full compositions.

        Args:
            x: Mole fractions in ``self.elements`` order, shape ``(..., n_el)``.

        Returns:
            Lattice parameters, shape ``x.shape[:-1]``.
        """
        x = np.asarray(x, dtype=float)
        return self.K * np.einsum("...i,ij,...j->...", x, self.bonds, x)

    def a_ind(self, x_ind: np.ndarray, ind_elements: list[str], reference_element: str) -> np.ndarray:
        """Return the lattice parameter in Å for independent mole fractions.

        Args:
            x_ind: Independent mole fractions, shape ``(n_pts, n_ind)``.
            ind_elements: Independent elements, in the column order of ``x_ind``.
            reference_element: Dependent element, whose fraction is one minus the others.

        Returns:
            Lattice parameters, shape ``(n_pts,)``.
        """
        return self.a(self._full_composition(x_ind, ind_elements, reference_element))

    def grad_a_ind(self, x_ind: np.ndarray, ind_elements: list[str], reference_element: str) -> np.ndarray:
        """Return the analytic gradient ∂a/∂x_k in Å for each independent element k.

        Args:
            x_ind: Independent mole fractions, shape ``(n_pts, n_ind)``.
            ind_elements: Independent elements, in the column order of ``x_ind``.
            reference_element: Dependent element, whose fraction is one minus the others.

        Returns:
            Gradient, shape ``(n_pts, n_ind)``.
        """
        x = self._full_composition(x_ind, ind_elements, reference_element)
        Dx = x @ self.bonds  # (n_pts, n_el)
        r = self._idx(reference_element)
        cols = [self._idx(el) for el in ind_elements]
        return 2.0 * self.K * (Dx[:, cols] - Dx[:, [r]])

    def batch_interp(self, ind_elements: list[str], reference_element: str):
        """Wrap ``a_ind`` in a function with the ``BatchInterp`` call signature.

        The function maps shape ``(n_pts, n_ind)`` to ``(n_pts,)``, so the bond model can
        replace the SQS ``a_relaxed_A`` interpolant.

        Args:
            ind_elements: Independent elements, in the column order of the query array.
            reference_element: Dependent element, whose fraction is one minus the others.

        Returns:
            Function from independent mole fractions to lattice parameters in Å.
        """

        def _call(x_ind: np.ndarray) -> np.ndarray:
            return self.a_ind(x_ind, ind_elements, reference_element)

        return _call

    def __repr__(self) -> str:  # pragma: no cover
        """Return ``BondLatticeModel(<elements>, <structure>)``."""
        return f"BondLatticeModel({'-'.join(self.elements)}, {self.structure})"
