"""phasefoundry.atomistic.bond
===========================
Compute the bond-length matrix for ``BondLatticeModel`` with an MLIP, using
MaterialsFramework's ``BondLatticeParameter`` (Tandoc et al., Materialia 2025):

  d_ii from relaxed pure cells        (BCC / FCC / HCP)
  d_ij from relaxed binary ordered cells  (B2 / L1_0 / D0_19)

The matrix is written in the MaterialsFramework ``from_csv`` format, with a
``<table>.meta.yaml`` sidecar recording the calculator used.

Public API
----------
compute_bond_table(elements, path, settings, …)  → BondLatticeModel
ensure_bond_table(cfg, root)                     → Path
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import yaml

from phasefoundry.elasticity.bond_model import BondLatticeModel

from .settings import build_calculator, mlip_settings, require_materialsframework


def compute_bond_table(
    elements: list[str],
    path: str | Path,
    settings: dict,
    *,
    structure: str = "bcc",
    initial_a: dict[str, float] | None = None,
    calculator: Any = None,
) -> BondLatticeModel:
    """Relax pure and binary ordered cells and write the bond-length matrix.

    Parameters
    ----------
    elements   : system elements (any case)
    path       : output CSV
    settings   : ``mlip_settings`` output, recorded in the metadata sidecar
    structure  : 'bcc' | 'fcc' | 'hcp'
    initial_a  : starting lattice parameters [Å] for elements missing from
                 MaterialsFramework's built-in defaults
    calculator : pre-built calculator (default: built from *settings*)
    """
    require_materialsframework()
    from materialsframework.tools.bond_lattice_parameter import BondLatticeParameter

    symbols = [e.capitalize() for e in elements]
    if initial_a:
        initial_a = {el.capitalize(): float(a) for el, a in initial_a.items()}
    if calculator is None:
        calculator = build_calculator(settings)

    print(
        f"  Bond-length model ({structure.upper()}, {'-'.join(symbols)}): "
        f"{len(symbols)} pure + {len(symbols) * (len(symbols) - 1) // 2} binary relaxations"
    )
    blp = BondLatticeParameter(structure, symbols, initial_a=initial_a, calculator=calculator)
    bonds = blp.calculate()["bonds"]

    D = np.array([[bonds[tuple(sorted((ei, ej)))] for ej in symbols] for ei in symbols])
    model = BondLatticeModel(symbols, D, structure)

    path = Path(path)
    model.to_csv(path)
    with open(path.with_suffix(".meta.yaml"), "w") as fh:
        yaml.safe_dump({"elements": symbols, "structure": structure, "calculator": settings}, fh, sort_keys=False)
    print(f"    Written: {path}")
    return model


def ensure_bond_table(cfg: dict, root: Path, calculator: Any = None) -> Path:
    """Compute the bond table for ``simulation.elastic.bond`` unless it already exists."""
    from phasefoundry.elasticity.sources import bond_structure, bond_table_path

    path = bond_table_path(cfg, root)
    if path.exists():
        print(f"  Bond-length table (cached): {path}")
        return path

    bond_cfg = cfg["simulation"]["elastic"].get("bond", {}) or {}
    compute_bond_table(
        elements=cfg["system"]["elements"],
        path=path,
        settings=mlip_settings(cfg, bond_cfg),
        structure=bond_structure(cfg),
        initial_a=bond_cfg.get("initial_a"),
        calculator=calculator,
    )
    return path
