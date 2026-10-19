"""phasefoundry.elasticity.sources
===============================
Resolve where the elastic-constant table and the lattice-parameter model
come from, so step 1b (which may compute them) and step 3 (which only reads
them) agree on the same files.

Config (``simulation.elastic``)
-------------------------------
    source:        csv | mlip      # csv: read elastic_csv (default)
                                   # mlip: compute the SQS table with an MLIP (step 1b)
    lattice_model: sqs | bond      # a(x) from the SQS table (default) or the
                                   # bond-length model (Tandoc et al. 2025)
    elastic_csv:   path            # used when source: csv
    sqs:
      table: path                  # cache for source: mlip
                                   # default <data_dir>/sqs_elastic_constants_<model>.csv
    bond:
      table: path                  # bond-length matrix CSV
                                   # default <data_dir>/bond_lengths_<model>.csv
      crystal_structure: bcc

Public API
----------
elastic_source(cfg)              → 'csv' | 'mlip'
lattice_model_name(cfg)          → 'sqs' | 'bond'
elastic_table_path(cfg, root)    → Path
bond_table_path(cfg, root)       → Path
bond_structure(cfg)              → 'bcc' | 'fcc' | 'hcp'
load_lattice_model(cfg, root)    → BondLatticeModel | None
"""

from __future__ import annotations

from pathlib import Path

from phasefoundry.atomistic.settings import mlip_label, mlip_settings

from .bond_model import BondLatticeModel


def _el_cfg(cfg: dict) -> dict:
    return cfg["simulation"].get("elastic", {}) or {}


def _resolve(path: str | Path, root: Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else Path(root) / path


def _data_dir(cfg: dict, root: Path) -> Path:
    return _resolve(cfg["output"]["data_dir"], root)


def elastic_source(cfg: dict) -> str:
    source = str(_el_cfg(cfg).get("source", "csv")).lower()
    if source not in ("csv", "mlip"):
        raise ValueError(f"elastic.source must be 'csv' or 'mlip', got {source!r}")
    return source


def lattice_model_name(cfg: dict) -> str:
    name = str(_el_cfg(cfg).get("lattice_model", "sqs")).lower()
    if name not in ("sqs", "bond"):
        raise ValueError(f"elastic.lattice_model must be 'sqs' or 'bond', got {name!r}")
    return name


def elastic_table_path(cfg: dict, root: Path) -> Path:
    """SQS elastic-constant table: the user CSV, or the MLIP cache file."""
    el_cfg = _el_cfg(cfg)
    if elastic_source(cfg) == "csv":
        csv_rel = el_cfg.get("elastic_csv", "")
        if not csv_rel:
            raise ValueError("elastic.elastic_csv must be set when elastic.source is 'csv'")
        return _resolve(csv_rel, root)

    sqs_cfg = el_cfg.get("sqs", {}) or {}
    if sqs_cfg.get("table"):
        return _resolve(sqs_cfg["table"], root)
    label = mlip_label(mlip_settings(cfg, sqs_cfg))
    return _data_dir(cfg, root) / f"sqs_elastic_constants_{label}.csv"


def bond_table_path(cfg: dict, root: Path) -> Path:
    """Bond-length matrix CSV: explicit ``bond.table`` or the MLIP cache file."""
    bond_cfg = _el_cfg(cfg).get("bond", {}) or {}
    if bond_cfg.get("table"):
        return _resolve(bond_cfg["table"], root)
    label = mlip_label(mlip_settings(cfg, bond_cfg))
    return _data_dir(cfg, root) / f"bond_lengths_{label}.csv"


def bond_structure(cfg: dict) -> str:
    el_cfg = _el_cfg(cfg)
    bond_cfg = el_cfg.get("bond", {}) or {}
    sqs_cfg = el_cfg.get("sqs", {}) or {}
    return str(bond_cfg.get("crystal_structure", sqs_cfg.get("crystal_structure", "bcc"))).lower()


def load_lattice_model(cfg: dict, root: Path) -> BondLatticeModel | None:
    """Return the bond model when ``lattice_model: bond``, else None (use SQS a(x))."""
    if lattice_model_name(cfg) != "bond":
        return None
    path = bond_table_path(cfg, root)
    if not path.exists():
        raise FileNotFoundError(
            f"Bond-length table not found: {path}\n"
            "Run step 1b with an 'mlip' section in the config to compute it, "
            "or point elastic.bond.table at an existing bond matrix CSV."
        )
    return BondLatticeModel.from_csv(path, bond_structure(cfg))
