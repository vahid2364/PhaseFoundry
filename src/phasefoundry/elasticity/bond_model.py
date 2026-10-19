"""phasefoundry.elasticity.bond_model
==================================
Bond-length lattice-parameter model (Tandoc et al., Materialia 2025).

    d̄(x) = Σ_ij x_i x_j d_ij              (composition-weighted FNN bond length)
    a(x)  = K · d̄(x)                       K = 2/√3 (BCC), √2 (FCC), 1 (HCP)

The pair bond lengths d_ij come from MLIP relaxations of pure-element cells
(i = j) and binary ordered cells (B2 / L1_0 / D0_19, i ≠ j); they are
computed by ``phasefoundry.atomistic.compute_bond_table`` and stored as a
symmetric matrix CSV in the same format as MaterialsFramework's
``BondLatticeParameter.from_csv``:

    ,Hf,Nb,Ti,V
    Hf,3.0912,2.9978,...
    ...

Because a(x) is a quadratic form, its gradient with respect to the
independent mole fractions (reference element dependent) is analytic:

    ∂a/∂x_k = 2K [(D x)_k − (D x)_ref]

This module is pure NumPy so the simulation steps can use it without
MaterialsFramework installed.

Public API
----------
BondLatticeModel(elements, bonds, structure)
BondLatticeModel.from_csv(path, structure)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# K in a = K · d̄ for each crystal structure (HCP: exact for ideal c/a)
STRUCTURE_FACTOR = {
    "bcc": 2.0 / np.sqrt(3.0),
    "fcc": np.sqrt(2.0),
    "hcp": 1.0,
}


class BondLatticeModel:
    """Vectorized a(x) and ∂a/∂x from a pairwise bond-length matrix.

    Element names are matched case-insensitively, so config names (``NB``)
    and pymatgen symbols (``Nb``) are interchangeable.
    """

    def __init__(self, elements: list[str], bonds: np.ndarray, structure: str = "bcc") -> None:
        """Parameters
        ----------
        elements  : element symbols, in the row/column order of *bonds*
        bonds     : (n_el, n_el) symmetric FNN bond lengths d_ij [Å]
        structure : 'bcc' | 'fcc' | 'hcp'
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
        """Load a symmetric bond-length matrix CSV (MaterialsFramework format)."""
        df = pd.read_csv(path, index_col=0)
        if list(df.index) != list(df.columns):
            raise ValueError(f"{path}: row and column element labels differ")
        return cls(list(df.columns), df.values, structure)

    def to_csv(self, path: str | Path) -> Path:
        """Write the bond matrix in MaterialsFramework ``from_csv`` format."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(self.bonds, index=self.elements, columns=self.elements).to_csv(path)
        return path

    def _full_composition(self, x_ind: np.ndarray, ind_elements: list[str], reference_element: str) -> np.ndarray:
        """(n_pts, n_ind) independent fractions → (n_pts, n_el) full fractions."""
        x_ind = np.atleast_2d(np.asarray(x_ind, dtype=float))
        x = np.zeros((x_ind.shape[0], len(self.elements)))
        for k, el in enumerate(ind_elements):
            x[:, self._idx(el)] = x_ind[:, k]
        x[:, self._idx(reference_element)] = 1.0 - x_ind.sum(axis=1)
        return x

    def _idx(self, el: str) -> int:
        try:
            return self._index[el.upper()]
        except KeyError:
            raise KeyError(f"Element {el!r} not in bond table {self.elements}") from None

    def a(self, x: np.ndarray) -> np.ndarray:
        """Lattice parameter [Å] from full compositions.

        x : (..., n_el) mole fractions in ``self.elements`` order
        """
        x = np.asarray(x, dtype=float)
        return self.K * np.einsum("...i,ij,...j->...", x, self.bonds, x)

    def a_ind(self, x_ind: np.ndarray, ind_elements: list[str], reference_element: str) -> np.ndarray:
        """Lattice parameter [Å] from (n_pts, n_ind) independent fractions."""
        return self.a(self._full_composition(x_ind, ind_elements, reference_element))

    def grad_a_ind(self, x_ind: np.ndarray, ind_elements: list[str], reference_element: str) -> np.ndarray:
        """Analytic ∂a/∂x_k [Å] for each independent element k.

        Returns (n_pts, n_ind).
        """
        x = self._full_composition(x_ind, ind_elements, reference_element)
        Dx = x @ self.bonds  # (n_pts, n_el)
        r = self._idx(reference_element)
        cols = [self._idx(el) for el in ind_elements]
        return 2.0 * self.K * (Dx[:, cols] - Dx[:, [r]])

    def batch_interp(self, ind_elements: list[str], reference_element: str):
        """Callable with the ``BatchInterp`` signature, (n_pts, n_ind) → (n_pts,),
        so the bond model can stand in for the SQS ``a_relaxed_A`` interpolant.
        """

        def _call(x_ind: np.ndarray) -> np.ndarray:
            return self.a_ind(x_ind, ind_elements, reference_element)

        return _call

    def __repr__(self) -> str:  # pragma: no cover
        return f"BondLatticeModel({'-'.join(self.elements)}, {self.structure})"
