"""Bond-length matrix for ``BondLatticeModel``, computed with an MLIP.

The matrix comes from MaterialsFramework's ``BondLatticeParameter`` (Tandoc et al.,
Materialia 2025). The diagonal entries d_ii come from relaxed pure-element cells (BCC,
FCC or HCP) and the off-diagonal entries d_ij from relaxed binary ordered cells (B2,
L1_0 or D0_19). The matrix is written in MaterialsFramework's ``from_csv`` format, next
to a ``<table>.meta.yaml`` file that records the elements, structure and calculator.
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

    Args:
        elements: System elements, in any case.
        path: Output CSV path. The metadata file is written next to it.
        settings: Output of ``mlip_settings``, stored in the metadata file.
        structure: Crystal structure, ``"bcc"``, ``"fcc"`` or ``"hcp"``.
        initial_a: Starting lattice parameters in Å for elements that have no built-in
            default in MaterialsFramework.
        calculator: Calculator to use. Defaults to one built from ``settings``.

    Returns:
        The bond-length model built from the computed matrix.

    Raises:
        ImportError: If MaterialsFramework is not installed.
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
    blp = BondLatticeParameter(structure, symbols, initial_a=initial_a, calculator=calculator)  # ty: ignore[invalid-argument-type]
    bonds = blp.calculate()["bonds"]

    D = np.array([[bonds[tuple(sorted((ei, ej)))] for ej in symbols] for ei in symbols])
    model = BondLatticeModel(symbols, D, structure)

    path = Path(path)
    model.to_csv(path)
    with path.with_suffix(".meta.yaml").open("w") as fh:
        yaml.safe_dump({"elements": symbols, "structure": structure, "calculator": settings}, fh, sort_keys=False)
    print(f"    Written: {path}")
    return model


def ensure_bond_table(cfg: dict, root: Path, calculator: Any = None) -> Path:
    """Compute the bond table configured in ``simulation.elastic.bond`` if it does not exist yet.

    Args:
        cfg: Nested config dict.
        root: Repository root used to resolve relative paths.
        calculator: Calculator to use. Defaults to one built from the config.

    Returns:
        Path to the bond table CSV.
    """
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
