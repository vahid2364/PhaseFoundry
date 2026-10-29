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


# ── anisotropic (direction-dependent) coherency modulus ──────────────────────────

def coherency_modulus(C: np.ndarray, n: np.ndarray) -> float:
    """
    Direction-dependent coherency modulus Y(n) for a dilatational eigenstrain
    in an anisotropic elastic medium (Khachaturyan microelasticity theory),
    generalizing ``isotropic_coherent_correction()``'s single scalar M to the
    actual wavevector-direction dependence a cubic crystal has.

    Y(n) = [C_ijkl delta_ij delta_kl - n_i sigma0_ij Omega_jk(n) sigma0_kl n_k] / 2

    where sigma0 = C : I is the "eigenstress" for a unit dilatational
    eigenstrain, and Omega(n) = [C_ijkl n_j n_l]^-1 is the Green's-function
    (acoustic-tensor inverse) -- the same acoustic tensor construction as
    ``phasefoundry.elasticity.fft_solver._build_green_operator``, evaluated
    here at a single direction instead of a Fourier grid.

    Validated: reduces to a direction-independent constant for isotropic C
    (Zener ratio A_Z=1), and Y([1,0,0]) reproduces the closed-form cubic
    <100> result C11 + C12 - 2*C12**2/C11 exactly (see
    ``tests/test_coherent_hessian.py``).

    Parameters
    ----------
    C : ndarray (3, 3, 3, 3)
        Stiffness tensor [Pa].
    n : array-like, shape (3,)
        Wavevector direction (need not be pre-normalized).

    Returns
    -------
    float
        Y(n) [Pa], same units as C.
    """
    n = np.asarray(n, dtype=float)
    n = n / np.linalg.norm(n)
    I = np.eye(3)
    sigma0 = np.einsum("ijkl,kl->ij", C, I)
    A = np.einsum("ijkl,j,l->ik", C, n, n)
    Omega = np.linalg.inv(A)
    term1 = np.einsum("ijkl,ij,kl->", C, I, I)
    v = np.einsum("ij,j->i", sigma0, n)
    term2 = np.einsum("i,ij,j->", v, Omega, v)
    return float((term1 - term2) / 2.0)


def softest_direction(C: np.ndarray, n_theta: int = 100, n_phi: int = 200) -> dict:
    """
    Scan a dense grid of directions on the unit sphere and return the one
    minimizing the coherency modulus Y(n) -- the direction that most
    destabilizes a coherent composition fluctuation, and therefore the
    relevant worst case for a linear-stability test.

    For a cubic crystal the global minimum is known analytically to sit at
    <100> when A_Z > 1 and at <110> when A_Z < 1; this function finds it
    numerically rather than assuming which high-symmetry direction applies,
    so it also works for lower-symmetry or strongly non-cubic C.

    Parameters
    ----------
    C : ndarray (3, 3, 3, 3)
    n_theta, n_phi : int
        Grid resolution in polar/azimuthal angle.

    Returns
    -------
    dict with keys 'Y_min', 'n_min', 'Y_max', 'n_max'
    """
    thetas = np.linspace(0.0, np.pi, n_theta)
    phis = np.linspace(0.0, 2.0 * np.pi, n_phi)
    Y_min, n_min = np.inf, None
    Y_max, n_max = -np.inf, None
    for th in thetas:
        st, ct = np.sin(th), np.cos(th)
        for ph in phis:
            n = np.array([st * np.cos(ph), st * np.sin(ph), ct])
            Y = coherency_modulus(C, n)
            if Y < Y_min:
                Y_min, n_min = Y, n
            if Y > Y_max:
                Y_max, n_max = Y, n
    return {"Y_min": Y_min, "n_min": n_min, "Y_max": Y_max, "n_max": n_max}


def anisotropic_coherent_correction(
    C: np.ndarray,
    n: np.ndarray,
    eta: np.ndarray | dict[str, float],
    a_ref_angstrom: float,
    independent_elements: list[str] | None = None,
    atoms_per_cell: float = 2.0,
) -> np.ndarray:
    """
    Coherency correction 2 * V_m * Y(n) * (eta (x) eta), using the
    direction-dependent modulus Y(n) (``coherency_modulus()``) in place of
    ``isotropic_coherent_correction()``'s single scalar M. Same structure
    otherwise: rank-1, positive semi-definite, built from the Vegard eta
    vector.

    Parameters
    ----------
    C : ndarray (3, 3, 3, 3)
        Stiffness tensor [Pa] at the composition of interest.
    n : array-like, shape (3,)
        Wavevector direction to evaluate the coherency modulus at. Pass
        ``softest_direction(C)['n_min']`` for the worst-case (most
        destabilizing) direction.
    eta, a_ref_angstrom, independent_elements, atoms_per_cell :
        Same as ``isotropic_coherent_correction()``.

    Returns
    -------
    np.ndarray, shape (n_ind, n_ind)
    """
    Y_n = coherency_modulus(C, n)
    return isotropic_coherent_correction(
        eta, Y_n, a_ref_angstrom,
        independent_elements=independent_elements,
        atoms_per_cell=atoms_per_cell,
    )


def evaluate_anisotropic_coherent_stability(
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
    n_theta: int = 100,
    n_phi: int = 200,
) -> dict:
    """
    End-to-end coherent-Hessian stability evaluation using the direction-
    dependent, worst-case coherency modulus Y(n_min) in place of
    ``evaluate_coherent_stability()``'s single scalar M -- the reviewer-
    requested generalization H_coh(n) = H + 2*V_m*Y(n)*(eta (x) eta),
    evaluated at the direction that most destabilizes the homogeneous state.

    Builds the full cubic stiffness tensor from the interpolated C11/C12/C44
    at this composition (rather than reading the CSV's own
    ``biaxial_modulus_M_GPa`` column, which does not reproduce the
    closed-form cubic <100> result -- see module tests), scans directions
    via ``softest_direction()``, and applies the correction at the softest
    one via ``anisotropic_coherent_correction()``.

    Parameters
    ----------
    Same as ``evaluate_coherent_stability()``, plus:
    n_theta, n_phi : int
        Direction-scan grid resolution (see ``softest_direction()``).

    Returns
    -------
    dict with keys:
        'chemical_hessian', 'coherent_hessian'   (n,n) ndarrays
        'eta'                                     dict[str, float]
        'C11_GPa', 'C12_GPa', 'C44_GPa', 'a_ref_A'  float
        'A_Z'                                      float, Zener anisotropy ratio
        'Y_min_GPa', 'n_min'                       float, ndarray -- softest direction
        'chemical_eigenvalues', 'coherent_eigenvalues'  (n,) ndarrays, ascending
        'lambda_min_chemical', 'lambda_min_coherent'    float
    """
    from phasefoundry.elasticity.fft_solver import build_cubic_stiffness
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

    C11, C12, C44 = props["C11_GPa"], props["C12_GPa"], props["C44_GPa"]
    C = build_cubic_stiffness(C11 * 1e9, C12 * 1e9, C44 * 1e9)
    A_Z = 2.0 * C44 / (C11 - C12)

    scan = softest_direction(C, n_theta=n_theta, n_phi=n_phi)
    Y_min_Pa = scan["Y_min"]

    correction = anisotropic_coherent_correction(
        C, scan["n_min"], eta, props["a_relaxed_A"],
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
        "C11_GPa":              C11,
        "C12_GPa":              C12,
        "C44_GPa":              C44,
        "a_ref_A":              props["a_relaxed_A"],
        "A_Z":                  A_Z,
        "Y_min_GPa":            Y_min_Pa / 1e9,
        "n_min":                scan["n_min"],
        "chemical_eigenvalues": chem_eig,
        "coherent_eigenvalues": coh_eig,
        "lambda_min_chemical":  float(chem_eig[0]),
        "lambda_min_coherent":  float(coh_eig[0]),
    }


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
