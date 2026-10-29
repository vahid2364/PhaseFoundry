"""
Isotropic coherent-Hessian stability analysis.

Chemical spinodal stability is governed by the eigenvalues of the chemical
Gibbs energy Hessian, H^chem_ij = d(mu_i - mu_ref)/dx_j (see
``phasefoundry.thermodynamics.spinodal``). Coherency between differently
strained regions raises this curvature; the classical isotropic
approximation (Cahn, Acta Metall. 9 (1961) 795) adds a rank-1 correction
built from a single scalar modulus and a per-element Vegard eigenstrain
vector:

    H^coh_ij = H^chem_ij + 2 * V_m * M * eta_i * eta_j

This module provides that calculation end to end (TDB -> chemical Hessian,
SQS/MLIP elastic-constants CSV -> eta/M/a_ref, correction, gauge-invariant
eigenvalues), replacing a cell-by-cell notebook calculation that previously
queried an untracked TDB and an untracked elastic-constants CSV, with values
hand-transcribed into the notebook rather than read from any file in this
repository.

Important limitation: this is an isotropic, single-composition, linear
(infinitesimal-fluctuation) screening estimate. It is NOT the same
calculation as the anisotropic, composition-dependent C_ijkl(x) elasticity
actually solved by the elasto-chemical phase-field coupling in
``phasefoundry.elasticity`` (Moulinec-Suquet FFT micromechanics with a
two-endpoint Vegard-tensor eigenstrain). A positive lambda_min here rules
out spontaneous linear decomposition from the exact evaluated composition;
it does not by itself establish stability of a finite-amplitude phase-field
simulation started from that composition.

This module provides:
    KADIRVEL_HF_NB_TI_O          — gauge-invariant orthogonal projection
                                    (Hf, Nb, Ti | V-reference systems only)
    chemical_hessian()           — H^chem via pycalphad Jansson derivatives
    isotropic_coherent_correction() — the 2*V_m*M*(eta (x) eta) term
    project_gauge_invariant()    — apply O^T H O
    hessian_eigenvalues()        — sorted eigenvalues, gauge-invariant if O given
    evaluate_coherent_stability() — end-to-end, one composition/temperature
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

_N_A = 6.02214076e23  # Avogadro's number [1/mol]


# ── gauge-invariant projection ──────────────────────────────────────────────────

# Orthogonal projection from the (Hf, Nb, Ti | V-reference) composition-
# simplex basis to an orthonormal Cartesian basis, so that Hessian
# eigenvalues/eigenvectors do not depend on the arbitrary choice of which
# element is treated as dependent. Table 1, Kadirvel et al., Scripta
# Materialia 212 (2022) 114521. Valid only for a 3-independent-element
# system in exactly this (HF, NB, TI) order, referenced against V.
KADIRVEL_HF_NB_TI_O = np.array([
    [-1.0, -1.0 / np.sqrt(3), -1.0 / np.sqrt(6)],
    [ 1.0, -1.0 / np.sqrt(3), -1.0 / np.sqrt(6)],
    [ 0.0,  2.0 / np.sqrt(3), -1.0 / np.sqrt(6)],
])


def project_gauge_invariant(H: np.ndarray, O: np.ndarray) -> np.ndarray:
    """Apply the gauge-invariant orthogonal projection O^T H O."""
    return O.T @ H @ O


def hessian_eigenvalues(H: np.ndarray, O: np.ndarray | None = None) -> np.ndarray:
    """
    Sorted (ascending) eigenvalues of a Hessian matrix.

    If ``O`` is given, project via ``project_gauge_invariant()`` first so the
    eigenvalues do not depend on which element was chosen as the dependent
    (reference) component. If ``O`` is omitted, eigenvalues are computed
    directly on ``H`` and are NOT guaranteed gauge-invariant.
    """
    if O is not None:
        H = project_gauge_invariant(H, O)
    return np.sort(np.linalg.eigvalsh(H))


# ── chemical Hessian ─────────────────────────────────────────────────────────────

def chemical_hessian(
    db_path: str | Path,
    elements: list[str],
    phase: str,
    composition: dict[str, float],
    temperature_K: float,
    reference_element: str,
    pressure: float = 101325.0,
) -> np.ndarray:
    """
    Gibbs-Duhem-reduced chemical Hessian H_ij = d(mu_i - mu_ref)/dx_j at a
    single composition and temperature, via pycalphad Jansson derivatives.
    Same convention as ``phasefoundry.thermodynamics.spinodal.
    spinodal_boundary_multi()``.

    Parameters
    ----------
    db_path : str or Path
        TDB file.
    elements : list[str]
        All elements including the reference, e.g. ["HF", "NB", "TI", "V"].
    phase : str
        Phase name, e.g. "BCC_A2".
    composition : dict[str, float]
        Independent-element mole fractions (all elements except the
        reference), e.g. {"HF": 0.25, "NB": 0.25, "TI": 0.25}.
    temperature_K : float
    reference_element : str
        The dependent element; conditions are specified for the others.
    pressure : float
        Pa, default 101325.

    Returns
    -------
    np.ndarray, shape (n_ind, n_ind)
        H_ij, with rows/columns in the order of ``elements`` minus
        ``reference_element``.
    """
    from pycalphad import Database, Workspace, variables as v
    from pycalphad.property_framework import IsolatedPhase

    independent_elements = [e for e in elements if e != reference_element]
    missing = set(independent_elements) - set(composition)
    if missing:
        raise ValueError(
            f"composition is missing independent element(s): {sorted(missing)}"
        )

    cond = {v.T: temperature_K, v.P: pressure}
    for el in independent_elements:
        cond[v.X(el)] = float(composition[el])

    db = Database(str(db_path))
    wks = Workspace(database=db, components=elements, phases=[phase], conditions=cond)
    ip = IsolatedPhase(phase, wks=wks)

    def dmu(el1: str, el2: str) -> float:
        return float(wks.get(ip(f"MU({el1}).X({el2})")).squeeze())

    n = len(independent_elements)
    H = np.zeros((n, n))
    for ri, eli in enumerate(independent_elements):
        for ci, elj in enumerate(independent_elements):
            H[ri, ci] = dmu(eli, elj) - dmu(reference_element, elj)
    return H


# ── isotropic coherency correction ───────────────────────────────────────────────

def isotropic_coherent_correction(
    eta: np.ndarray | dict[str, float],
    M_Pa: float,
    a_ref_angstrom: float,
    independent_elements: list[str] | None = None,
    atoms_per_cell: float = 2.0,
) -> np.ndarray:
    """
    Classical isotropic (Cahn 1961) coherency correction to the chemical
    Hessian: 2 * V_m * M * (eta (x) eta), where M is a scalar "biaxial
    modulus" and eta_k = (1/a) * da/dx_k is the per-element Vegard
    eigenstrain coefficient.

    NOTE: this is an isotropic, single-direction approximation, distinct
    from (and generally more stabilizing than) the full anisotropic,
    composition-dependent C_ijkl(x) elasticity actually solved by the
    elasto-chemical phase-field coupling in ``phasefoundry.elasticity``.
    Use this for fast linear-stability screening only.

    Parameters
    ----------
    eta : array-like or dict[str, float]
        Per-independent-element Vegard coefficients. If a dict, pass
        ``independent_elements`` to fix the output ordering.
    M_Pa : float
        Scalar biaxial modulus, in Pa (not GPa).
    a_ref_angstrom : float
        Reference lattice parameter, in angstrom.
    independent_elements : list[str], optional
        Required if ``eta`` is a dict; gives eta's row/column order.
    atoms_per_cell : float
        Atoms per unit cell used for the molar-volume conversion
        (V_m = a^3 * N_A / atoms_per_cell). Default 2.0 (BCC).

    Returns
    -------
    np.ndarray, shape (n, n)
        The correction matrix, in the same units as the chemical Hessian
        (J/mol when M is in Pa and a_ref in angstrom).
    """
    if isinstance(eta, dict):
        if independent_elements is None:
            raise ValueError(
                "independent_elements is required when eta is a dict."
            )
        eta_arr = np.array([eta[el] for el in independent_elements], dtype=float)
    else:
        eta_arr = np.asarray(eta, dtype=float)

    Vm = (a_ref_angstrom * 1e-10) ** 3 * _N_A / atoms_per_cell
    eta_col = eta_arr.reshape(-1, 1)
    return 2.0 * M_Pa * Vm * (eta_col @ eta_col.T)


# ── end-to-end evaluation ─────────────────────────────────────────────────────────

def evaluate_coherent_stability(
    db_path: str | Path,
    elastic_csv_path: str | Path,
    elements: list[str],
    phase: str,
    composition: dict[str, float],
    temperature_K: float,
    reference_element: str,
    O: np.ndarray | None = None,
    delta_fd: float = 0.02,
    atoms_per_cell: float = 2.0,
) -> dict:
    """
    End-to-end isotropic coherent-Hessian stability evaluation at one
    composition and temperature: chemical Hessian from the TDB, eta/M/a_ref
    from the SQS/MLIP elastic-constants CSV, combined via the classical
    isotropic (Cahn 1961) coherency correction.

    Replaces the manual, multi-cell notebook calculation (TDB query, eta/M/a
    lookup, correction, projection, eigendecomposition) with a single call
    against files actually tracked in this repository.

    Parameters
    ----------
    db_path : str or Path
        TDB file.
    elastic_csv_path : str or Path
        SQS/MLIP elastic-constants CSV (same format consumed by
        ``workflows/01b_lookup_elastic``).
    elements : list[str]
        All elements including the reference, e.g. ["HF", "NB", "TI", "V"].
    phase : str
        Phase name, e.g. "BCC_A2".
    composition : dict[str, float]
        Independent-element mole fractions.
    temperature_K : float
    reference_element : str
    O : np.ndarray, optional
        Gauge-invariant orthogonal projection (e.g. ``KADIRVEL_HF_NB_TI_O``
        for a 3-independent-element Hf/Nb/Ti system). If omitted, raw
        (non-gauge-invariant) eigenvalues are returned.
    delta_fd : float
        Finite-difference step for the Vegard eta coefficients (default 0.02).
    atoms_per_cell : float
        Atoms per unit cell for the molar-volume conversion (default 2.0, BCC).

    Returns
    -------
    dict with keys:
        'chemical_hessian', 'coherent_hessian'   (n,n) ndarrays
        'eta'                                     dict[str, float]
        'M_GPa', 'a_ref_A'                         float
        'chemical_eigenvalues', 'coherent_eigenvalues'  (n,) ndarrays, ascending
        'lambda_min_chemical', 'lambda_min_coherent'    float
    """
    from phasefoundry.elasticity.sqs_interpolant import (
        load_and_filter, build_interpolators, query_composition,
        vegard_eta_coefficients,
    )

    independent_elements = [e for e in elements if e != reference_element]

    H_chem = chemical_hessian(
        db_path, elements, phase, composition, temperature_K, reference_element,
    )

    df = load_and_filter(Path(elastic_csv_path), elements)
    interps = build_interpolators(df, independent_elements)
    props = query_composition(interps, composition, independent_elements)
    eta = vegard_eta_coefficients(
        interps, composition, independent_elements, delta_fd=delta_fd,
    )

    correction = isotropic_coherent_correction(
        eta,
        props["biaxial_modulus_M_GPa"] * 1e9,
        props["a_relaxed_A"],
        independent_elements=independent_elements,
        atoms_per_cell=atoms_per_cell,
    )
    H_coh = H_chem + correction

    chem_eig = hessian_eigenvalues(H_chem, O)
    coh_eig = hessian_eigenvalues(H_coh, O)

    return {
        "chemical_hessian":     H_chem,
        "coherent_hessian":     H_coh,
        "eta":                  eta,
        "M_GPa":                props["biaxial_modulus_M_GPa"],
        "a_ref_A":              props["a_relaxed_A"],
        "chemical_eigenvalues": chem_eig,
        "coherent_eigenvalues": coh_eig,
        "lambda_min_chemical":  float(chem_eig[0]),
        "lambda_min_coherent":  float(coh_eig[0]),
    }
