"""
tests/test_coherent_hessian.py
================================
Regression tests for phasefoundry.thermodynamics.coherent_hessian.

Pins today's (2026-10-28) verified, production-consistent numbers for the
quaternary Hf-Nb-Ti-V coherent-Hessian stability analysis, so that a future
change to the TDB, the elastic-constants CSV, the Vegard-coefficient step
size, or the gauge-invariant projection convention is caught here instead
of resurfacing silently somewhere else. See
docs/tutorial_coherent_hessian_quaternary.md for the full derivation and
provenance story.

Run with:
    python tests/test_coherent_hessian.py
    pytest tests/test_coherent_hessian.py -v
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from phasefoundry.thermodynamics.coherent_hessian import (
    chemical_hessian,
    isotropic_coherent_correction,
    hessian_eigenvalues,
    project_gauge_invariant,
    evaluate_coherent_stability,
    coherency_modulus,
    softest_direction,
    anisotropic_coherent_correction,
    evaluate_anisotropic_coherent_stability,
    KADIRVEL_HF_NB_TI_O,
)

DB_PATH = _ROOT / "data/tdb/HfNbTiV_BCC_PhaseForge_v2_Tdep.tdb"
ELASTIC_CSV_PATH = (
    _ROOT / "data/HfNbTiV_BCC_PhaseForge_v2/sqs_elastic_constants_HfNbVTi-orb-v2.csv"
)
ELEMENTS = ["HF", "NB", "TI", "V"]
PHASE = "BCC_A2"
REFERENCE_ELEMENT = "V"
EQUIATOMIC = {"HF": 0.25, "NB": 0.25, "TI": 0.25}
T_500C_K = 773.15


# ── provenance: catch exactly the problem this module was written to fix ──────

def test_production_data_files_exist():
    """
    The notebook this module replaces (stability_analysis_nonisotropic.ipynb)
    queried a TDB and an elastic-constants CSV that do not exist anywhere in
    this repository. Guard against that recurring: the production files this
    module and workflows/01e_coherent_hessian depend on must actually be
    tracked here.
    """
    assert DB_PATH.exists(), f"Production TDB missing: {DB_PATH}"
    assert ELASTIC_CSV_PATH.exists(), f"Production elastic CSV missing: {ELASTIC_CSV_PATH}"


# ── end-to-end regression: the verified reconciliation numbers ────────────────

def test_equiatomic_500C_production_inputs():
    """
    Equiatomic Hf25Nb25Ti25V25 at 500 C / 773.15 K, using the production TDB
    and production elastic CSV throughout.

    Swapping only the TDB (keeping the notebook's original, untraceable
    eta/M) gives a *negative* coherent eigenvalue instead of the positive
    one below -- a false lead investigated and ruled out this session (see
    test_tdb_only_swap_gives_opposite_sign() below). Both eta/M AND the TDB
    must be production-consistent to get this sign.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = evaluate_coherent_stability(
            db_path=DB_PATH,
            elastic_csv_path=ELASTIC_CSV_PATH,
            elements=ELEMENTS,
            phase=PHASE,
            composition=EQUIATOMIC,
            temperature_K=T_500C_K,
            reference_element=REFERENCE_ELEMENT,
            O=KADIRVEL_HF_NB_TI_O,
        )

    # production eta/M/a_ref at equiatomic
    assert result["a_ref_A"] == pytest.approx(3.2918, abs=5e-4)
    assert result["M_GPa"] == pytest.approx(144.595, abs=5e-3)
    assert result["eta"]["HF"] == pytest.approx(0.145488, abs=5e-6)
    assert result["eta"]["NB"] == pytest.approx(0.073429, abs=5e-6)
    assert result["eta"]["TI"] == pytest.approx(0.060820, abs=5e-6)

    # the reconciliation result itself
    assert result["lambda_min_chemical"] == pytest.approx(-39095.6, rel=1e-3)
    assert result["lambda_min_coherent"] == pytest.approx(11214.0, rel=1e-2)
    assert result["lambda_min_chemical"] < 0, "chemical Hessian should be unstable here"
    assert result["lambda_min_coherent"] > 0, (
        "coherent Hessian should be stable here with fully production-"
        "consistent inputs -- this is the reconciliation finding itself"
    )


def test_tdb_only_swap_gives_opposite_sign():
    """
    The investigated-and-ruled-out false lead: production TDB with the
    *original notebook's* (untraceable) eta/M/a_ref gives a NEGATIVE
    coherent eigenvalue at the same composition/temperature -- opposite
    sign from test_equiatomic_500C_production_inputs() above. Pinned here
    so the distinction between "right TDB" and "right TDB + right eta/M"
    doesn't silently get lost.
    """
    eta_notebook = {"HF": 0.08620911, "NB": 0.07105138, "TI": 0.06485024}
    M_notebook_Pa = 149.323894513184e9
    a_ref_notebook = 3.2726

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        H_chem = chemical_hessian(
            DB_PATH, ELEMENTS, PHASE, EQUIATOMIC, T_500C_K, REFERENCE_ELEMENT,
        )
    correction = isotropic_coherent_correction(
        eta_notebook, M_notebook_Pa, a_ref_notebook,
        independent_elements=["HF", "NB", "TI"],
    )
    H_coh = H_chem + correction

    chem_eig = hessian_eigenvalues(H_chem, KADIRVEL_HF_NB_TI_O)
    coh_eig = hessian_eigenvalues(H_coh, KADIRVEL_HF_NB_TI_O)

    assert chem_eig[0] == pytest.approx(-39095.6, rel=1e-3)
    assert coh_eig[0] == pytest.approx(-18786.2, rel=1e-2)
    assert coh_eig[0] < 0


# ── unit tests on the building blocks ──────────────────────────────────────────

def test_isotropic_correction_is_rank_one_and_psd():
    """2*Vm*M*(eta (x) eta) is symmetric, positive semi-definite, rank 1:
    exactly one nonzero eigenvalue, equal to 2*Vm*M*|eta|^2."""
    eta = np.array([0.1, 0.05, 0.02])
    M_Pa = 150e9
    a_ref_A = 3.3

    corr = isotropic_coherent_correction(eta, M_Pa, a_ref_A)

    assert np.allclose(corr, corr.T), "correction matrix must be symmetric"
    eigvals = np.sort(np.linalg.eigvalsh(corr))
    assert eigvals[0] == pytest.approx(0.0, abs=1e-6)
    assert eigvals[1] == pytest.approx(0.0, abs=1e-6)
    assert eigvals[2] > 0, "the single nonzero eigenvalue must be positive (PSD)"

    N_A = 6.02214076e23
    Vm = (a_ref_A * 1e-10) ** 3 * N_A / 2.0
    expected_nonzero = 2.0 * M_Pa * Vm * float(eta @ eta)
    assert eigvals[2] == pytest.approx(expected_nonzero, rel=1e-6)


def test_isotropic_correction_accepts_dict_or_array_equivalently():
    eta_dict = {"HF": 0.1, "NB": 0.05, "TI": 0.02}
    ind_els = ["HF", "NB", "TI"]
    eta_arr = np.array([eta_dict[el] for el in ind_els])

    corr_dict = isotropic_coherent_correction(
        eta_dict, 150e9, 3.3, independent_elements=ind_els,
    )
    corr_arr = isotropic_coherent_correction(eta_arr, 150e9, 3.3)

    assert np.allclose(corr_dict, corr_arr)


def test_isotropic_correction_dict_requires_independent_elements():
    with pytest.raises(ValueError):
        isotropic_coherent_correction({"HF": 0.1, "NB": 0.05}, 150e9, 3.3)


def test_hessian_eigenvalues_sorted_ascending():
    H = np.diag([5.0, -3.0, 1.0])
    eig = hessian_eigenvalues(H)
    assert list(eig) == sorted(eig)
    assert eig[0] == pytest.approx(-3.0)
    assert eig[-1] == pytest.approx(5.0)


def test_project_gauge_invariant_matches_manual_OT_H_O():
    H = np.array([[4.0, 1.0, 0.5], [1.0, 3.0, 0.2], [0.5, 0.2, 2.0]])
    O = KADIRVEL_HF_NB_TI_O
    projected = project_gauge_invariant(H, O)
    assert np.allclose(projected, O.T @ H @ O)


def test_chemical_hessian_requires_full_composition():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(ValueError):
            chemical_hessian(
                DB_PATH, ELEMENTS, PHASE, {"HF": 0.25, "NB": 0.25},  # missing TI
                T_500C_K, REFERENCE_ELEMENT,
            )


# ── anisotropic (direction-dependent) coherency modulus Y(n) ──────────────────

def _build_cubic_C(C11, C12, C44):
    C = np.zeros((3, 3, 3, 3))
    for i in range(3):
        C[i, i, i, i] = C11
        for j in range(3):
            if i != j:
                C[i, i, j, j] = C12
                C[i, j, i, j] = C44
                C[i, j, j, i] = C44
    return C


def test_coherency_modulus_isotropic_is_direction_independent():
    """At A_Z=1 (2*C44 = C11-C12), Y(n) must be the same for every n."""
    C11, C12 = 150.0e9, 50.0e9
    C44 = (C11 - C12) / 2.0
    C = _build_cubic_C(C11, C12, C44)
    directions = [[1, 0, 0], [1, 1, 0], [1, 1, 1], [1, 2, 3]]
    values = [coherency_modulus(C, n) for n in directions]
    assert all(v == pytest.approx(values[0], rel=1e-9) for v in values)


def test_coherency_modulus_cubic_100_matches_closed_form():
    """Y([100]) must equal the textbook Cahn <100> result C11+C12-2*C12^2/C11."""
    C11, C12, C44 = 167.0903e9, 112.2250e9, 35.4109e9  # equiatomic matrix, production CSV
    C = _build_cubic_C(C11, C12, C44)
    Y_100 = coherency_modulus(C, [1, 0, 0])
    expected = C11 + C12 - 2 * C12 ** 2 / C11
    assert Y_100 == pytest.approx(expected, rel=1e-9)


def test_softest_direction_matches_known_cubic_rule():
    """For A_Z > 1, the cubic coherency modulus is minimized at <100>."""
    C11, C12, C44 = 167.0903e9, 112.2250e9, 35.4109e9  # A_Z = 1.29 > 1
    C = _build_cubic_C(C11, C12, C44)
    scan = softest_direction(C, n_theta=60, n_phi=120)
    assert np.allclose(np.abs(scan["n_min"]), [0, 0, 1], atol=1e-2)
    assert scan["Y_min"] == pytest.approx(coherency_modulus(C, [1, 0, 0]), rel=1e-3)


def test_equiatomic_500C_anisotropic_worst_case():
    """
    The reviewer-requested generalization: H_coh(n) = H + 2*Vm*Y(n)*eta*eta^T,
    evaluated at the worst-case (softest) direction instead of a single
    scalar M. Weaker correction than the (non-physical) CSV M value, but
    still not enough to flip the sign at this composition/temperature.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = evaluate_anisotropic_coherent_stability(
            db_path=DB_PATH,
            elastic_csv_path=ELASTIC_CSV_PATH,
            elements=ELEMENTS,
            phase=PHASE,
            composition=EQUIATOMIC,
            temperature_K=T_500C_K,
            reference_element=REFERENCE_ELEMENT,
            O=KADIRVEL_HF_NB_TI_O,
        )

    assert result["A_Z"] == pytest.approx(1.2908, abs=5e-4)
    assert np.allclose(np.abs(result["n_min"]), [0, 0, 1], atol=1e-2)
    assert result["Y_min_GPa"] == pytest.approx(128.565, abs=5e-3)
    assert result["lambda_min_chemical"] == pytest.approx(-39095.6, rel=1e-3)
    assert result["lambda_min_coherent"] == pytest.approx(9839.0, rel=1e-2)
    assert result["lambda_min_coherent"] > 0, (
        "still stable at the worst-case direction -- rules out 'isotropic M "
        "hides a softer direction' as the Fig. 4(a)/Fig. 7(b) explanation"
    )


def test_anisotropic_correction_matches_isotropic_at_same_Y():
    """anisotropic_coherent_correction(C, n, ...) must reduce exactly to
    isotropic_coherent_correction(..., M=Y(n), ...) -- same formula, just
    with Y(n) computed instead of supplied."""
    C11, C12, C44 = 167.0903e9, 112.2250e9, 35.4109e9
    C = _build_cubic_C(C11, C12, C44)
    n = [1, 0, 0]
    eta = {"HF": 0.1, "NB": 0.05, "TI": 0.02}
    ind = ["HF", "NB", "TI"]
    a_ref = 3.3

    aniso = anisotropic_coherent_correction(C, n, eta, a_ref, independent_elements=ind)
    iso = isotropic_coherent_correction(
        eta, coherency_modulus(C, n), a_ref, independent_elements=ind,
    )
    assert np.allclose(aniso, iso)


if __name__ == "__main__":
    import subprocess
    subprocess.run([sys.executable, "-m", "pytest", __file__, "-v"])
