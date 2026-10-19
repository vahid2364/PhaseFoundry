"""
tests/test_atomistic.py
========================
Validates the MLIP-backed atomistic inputs (phasefoundry.atomistic) and their
consumers (phasefoundry.elasticity.bond_model / sources).

Tests
-----
1. BondLatticeModel reproduces a = K Σ x_i x_j d_ij and its analytic
   gradient matches central finite differences.
2. Bond matrix CSV round-trips and matches MaterialsFramework's
   BondLatticeParameter.from_csv / predict.
3. simplex_compositions gives the expected point counts with exact sums.
4. Config resolution: elastic table / bond table paths and calculator
   override merging.
5. LocalVegardModel with lattice_model=bond uses a(c0) and α_k from the
   bond model.
6. With ASE's EMT potential standing in for an MLIP (Cu-Ni FCC):
   a. the SQS elastic table is computed, resumes without recomputing, and
      refuses to extend a table built with different settings;
   b. the bond matrix agrees with the SQS lattice parameters;
   c. sqs2tdb produces a TDB that pycalphad loads (needs ATAT on PATH).

Tests in 2 and 6 are skipped when MaterialsFramework (Python 3.12) is not
installed.

Run with:
    pytest tests/test_atomistic.py -v
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from phasefoundry.atomistic.elastic import simplex_compositions
from phasefoundry.atomistic.settings import mlip_label, mlip_settings
from phasefoundry.elasticity.bond_model import BondLatticeModel
from phasefoundry.elasticity.models import build_eigenstrain_model
from phasefoundry.elasticity.sources import (
    bond_table_path,
    elastic_table_path,
    load_lattice_model,
)

ORB_CSV = _ROOT / "data/HfNbTiV_BCC_PhaseForge_v2/sqs_elastic_constants_HfNbVTi-orb-v2.csv"

# Representative BCC bond lengths [Å] for Hf, Nb, V (order matters below)
BONDS = np.array([
    [3.10, 2.98, 2.85],
    [2.98, 2.86, 2.74],
    [2.85, 2.74, 2.60],
])


@pytest.fixture
def bond_model() -> BondLatticeModel:
    return BondLatticeModel(["Hf", "Nb", "V"], BONDS, "bcc")


def _cfg(tmp_path: Path, **elastic) -> dict:
    return {
        "system": {"elements": ["NB", "HF", "V"], "reference_element": "V"},
        "output": {"data_dir": str(tmp_path / "data")},
        "mlip":   {"calculator": "orb", "params": {"model": "orb-v2", "device": "cpu"}},
        "simulation": {"elastic": elastic},
    }


def test_bond_model_formula(bond_model):
    x = np.array([0.2, 0.5, 0.3])
    expected = 2.0 / np.sqrt(3.0) * sum(x[i] * x[j] * BONDS[i, j]
                                        for i in range(3) for j in range(3))
    assert bond_model.a(x) == pytest.approx(expected, rel=1e-12)
    # pure element: a = d_ii · 2/√3
    assert bond_model.a([0, 1, 0]) == pytest.approx(2.86 * 2 / np.sqrt(3))


def test_bond_model_independent_fractions_and_gradient(bond_model):
    ind, ref = ["NB", "HF"], "V"                 # config order and case
    x_ind = np.array([[0.40, 0.25], [0.10, 0.70]])
    a = bond_model.a_ind(x_ind, ind, ref)
    assert a[0] == pytest.approx(bond_model.a([0.25, 0.40, 0.35]))

    grad = bond_model.grad_a_ind(x_ind, ind, ref)
    h = 1e-6
    for k in range(len(ind)):
        dx = np.zeros(2)
        dx[k] = h
        fd = (bond_model.a_ind(x_ind + dx, ind, ref)
              - bond_model.a_ind(x_ind - dx, ind, ref)) / (2 * h)
        np.testing.assert_allclose(grad[:, k], fd, rtol=1e-7)


def test_bond_model_rejects_asymmetric():
    with pytest.raises(ValueError, match="symmetric"):
        BondLatticeModel(["A", "B"], [[1.0, 2.0], [2.1, 1.0]])


def test_bond_csv_roundtrip(bond_model, tmp_path):
    path = bond_model.to_csv(tmp_path / "bonds.csv")
    loaded = BondLatticeModel.from_csv(path, "bcc")
    assert loaded.elements == bond_model.elements
    np.testing.assert_array_equal(loaded.bonds, bond_model.bonds)


def test_bond_csv_matches_materialsframework(bond_model, tmp_path):
    mf = pytest.importorskip("materialsframework.tools.bond_lattice_parameter")
    path = bond_model.to_csv(tmp_path / "bonds.csv")
    ref = mf.BondLatticeParameter.from_csv("bcc", path)
    comp = {"Hf": 0.2, "Nb": 0.5, "V": 0.3}
    assert bond_model.a(list(comp.values())) == pytest.approx(ref.predict(comp), rel=1e-12)


@pytest.mark.parametrize("n_el, n_pts", [(2, 21), (3, 231), (4, 1771)])
def test_simplex_compositions(n_el, n_pts):
    grid = simplex_compositions(n_el, 0.05)
    assert grid.shape == (n_pts, n_el)
    np.testing.assert_allclose(grid.sum(axis=1), 1.0, atol=1e-12)
    assert (grid >= 0).all()


def test_simplex_compositions_rejects_uneven_step():
    with pytest.raises(ValueError):
        simplex_compositions(3, 0.3)


def test_mlip_settings_override():
    cfg = {"mlip": {"calculator": "orb", "params": {"model": "orb-v2", "device": "cuda"}}}
    # same calculator: params merge
    s = mlip_settings(cfg, {"params": {"device": "cpu"}})
    assert s == {"calculator": "orb", "params": {"model": "orb-v2", "device": "cpu"}}
    # different calculator: top-level params are not inherited
    s = mlip_settings(cfg, {"calculator": "grace", "params": {"model": "GRACE-2L-OMAT"}})
    assert s == {"calculator": "grace", "params": {"model": "GRACE-2L-OMAT"}}
    assert mlip_label(s) == "GRACE-2L-OMAT"
    with pytest.raises(ValueError, match="No MLIP calculator"):
        mlip_settings({}, {})


def test_table_paths(tmp_path):
    cfg = _cfg(tmp_path, elastic_csv="data/x.csv")
    assert elastic_table_path(cfg, tmp_path) == tmp_path / "data/x.csv"

    cfg = _cfg(tmp_path, source="mlip")
    assert elastic_table_path(cfg, tmp_path) == \
        tmp_path / "data" / "sqs_elastic_constants_orb-v2.csv"
    assert bond_table_path(cfg, tmp_path) == tmp_path / "data" / "bond_lengths_orb-v2.csv"

    cfg = _cfg(tmp_path, source="mlip", sqs={"table": "cache/el.csv"},
               bond={"calculator": "grace", "params": {"model": "GRACE-2L-OMAT"}})
    assert elastic_table_path(cfg, tmp_path) == tmp_path / "cache/el.csv"
    assert bond_table_path(cfg, tmp_path).name == "bond_lengths_GRACE-2L-OMAT.csv"

    with pytest.raises(ValueError, match="source"):
        elastic_table_path(_cfg(tmp_path, source="dft"), tmp_path)


def test_load_lattice_model(bond_model, tmp_path):
    assert load_lattice_model(_cfg(tmp_path), tmp_path) is None
    cfg = _cfg(tmp_path, lattice_model="bond", bond={"table": "bonds.csv"})
    with pytest.raises(FileNotFoundError, match="Bond-length table"):
        load_lattice_model(cfg, tmp_path)
    bond_model.to_csv(tmp_path / "bonds.csv")
    assert load_lattice_model(cfg, tmp_path).elements == ["Hf", "Nb", "V"]


@pytest.mark.skipif(not ORB_CSV.exists(), reason="ORB-v2 SQS table not present")
def test_local_vegard_uses_bond_model(bond_model):
    ind, ref, c0 = ["NB", "HF"], "V", np.array([0.5, 0.25])
    eye = np.zeros((2, 2, 2))
    model = build_eigenstrain_model(
        "local_vegard", {}, {}, ind, c0, ndim=2, Lambda=eye,
        C_mat=None, C_ppt=None, all_elements=["NB", "HF", "V"],
        table_path=ORB_CSV, lattice_model=bond_model, reference_element=ref,
    )
    a0 = bond_model.a_ind(c0[None, :], ind, ref)[0]
    assert model._a_ref == pytest.approx(a0 * 1e-10, rel=1e-12)
    alpha = bond_model.grad_a_ind(c0[None, :], ind, ref)[0] / a0
    np.testing.assert_allclose(model._alpha_c0, alpha, rtol=1e-9)
    # eigenstrain vanishes for the uniform parent composition
    c_uniform = np.broadcast_to(c0[:, None, None], (2, 4, 4)).copy()
    np.testing.assert_allclose(model.eps_star_field(c_uniform, 2), 0.0, atol=1e-14)


@pytest.fixture
def emt_calculator():
    pytest.importorskip("materialsframework")
    pytest.importorskip("sqsgenerator")
    from ase.calculators.emt import EMT
    from materialsframework.tools.calculator import BaseCalculator

    class EMTCalculator(BaseCalculator):
        AVAILABLE_PROPERTIES = ["energy", "forces", "stress"]

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self._calc = None

        @property
        def calculator(self):
            if self._calc is None:
                self._calc = EMT()
            return self._calc

    return EMTCalculator(fmax=0.01)


EMT_SETTINGS = {"calculator": "emt", "params": {}}


def test_elastic_table_emt(emt_calculator, tmp_path, monkeypatch):
    from phasefoundry.atomistic import elastic as el

    path = tmp_path / "el.csv"
    kw = dict(step=0.5, crystal_structure="fcc", supercell=(2, 2, 2),
              calculator=emt_calculator)
    df = el.build_elastic_table(["CU", "NI"], path, EMT_SETTINGS, **kw)

    assert list(df.columns[:4]) == ["x_Cu", "x_Ni", "m_Cu", "m_Ni"]
    assert set(el.PROPERTY_COLUMNS) <= set(df.columns)
    assert len(df) == 3
    cu = df[df.x_Cu == 1.0].iloc[0]
    assert cu.m_Cu == 32 and cu.m_Ni == 0
    assert cu.a_relaxed_A == pytest.approx(3.59, abs=0.02)        # EMT Cu
    assert cu.C11_GPa > cu.C12_GPa > 0 and cu.C44_GPa > 0
    assert cu.biaxial_modulus_M_GPa == pytest.approx(
        cu.youngs_modulus_GPa / (1 - cu.poisson_ratio))

    # resume: a finer grid only computes the new points
    computed = []
    real = el._compute_point
    monkeypatch.setattr(el, "_compute_point",
                        lambda *a, **k: computed.append(1) or real(*a, **k))
    df = el.build_elastic_table(["CU", "NI"], path, EMT_SETTINGS,
                                **{**kw, "step": 0.25})
    assert len(computed) == 2 and len(df) == 5

    # different settings → refuse to extend
    with pytest.raises(ValueError, match="different settings"):
        el.build_elastic_table(["CU", "NI"], path, EMT_SETTINGS,
                               **{**kw, "supercell": (3, 3, 3)})


def test_bond_table_emt(emt_calculator, tmp_path):
    from phasefoundry.atomistic import compute_bond_table

    model = compute_bond_table(["CU", "NI"], tmp_path / "bonds.csv", EMT_SETTINGS,
                               structure="fcc", calculator=emt_calculator)
    assert (tmp_path / "bonds.meta.yaml").exists()
    # pure FCC: d = a / √2 with EMT a_Cu ≈ 3.59 Å, a_Ni ≈ 3.49 Å
    assert model.a([1, 0]) == pytest.approx(3.59, abs=0.02)
    assert model.a([0, 1]) == pytest.approx(3.49, abs=0.02)
    assert model.a([1, 0]) > model.a([0.5, 0.5]) > model.a([0, 1])


@pytest.mark.skipif(shutil.which("sqs2tdb") is None
                    or not (Path.home() / ".atat.rc").exists(),
                    reason="ATAT sqs2tdb not installed")
def test_generate_tdb_emt(emt_calculator, tmp_path):
    from pycalphad import Database

    from phasefoundry.atomistic import generate_tdb

    out = generate_tdb(["CU", "NI"], tmp_path / "CuNi.tdb", EMT_SETTINGS,
                       work_dir=tmp_path / "work", calculator=emt_calculator,
                       lattices=["FCC_A1"], level=1, fmax=0.01)
    db = Database(str(out))
    assert "FCC_A1" in db.phases
    assert {"CU", "NI"} <= set(db.elements)
