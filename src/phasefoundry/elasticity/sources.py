"""Locations of the elastic-constant table and the lattice-parameter model.

Step 1b may compute these files and step 3 only reads them. Both steps resolve the paths
here, so they always use the same files. The settings live in ``simulation.elastic``:

```yaml
source: csv            # csv (default): read elastic_csv
                       # mlip: compute the SQS table with an MLIP in step 1b
lattice_model: sqs     # sqs (default): a(x) from the SQS table
                       # bond: bond-length model (Tandoc et al. 2025)
elastic_csv: path      # used when source is csv
sqs:
  table: path          # cache file when source is mlip,
                       # default <data_dir>/sqs_elastic_constants_<model>.csv
bond:
  table: path          # bond-length matrix CSV,
                       # default <data_dir>/bond_lengths_<model>.csv
  crystal_structure: bcc
```
"""

from __future__ import annotations

from pathlib import Path

from phasefoundry.atomistic.settings import mlip_label, mlip_settings
from phasefoundry.elasticity.bond_model import BondLatticeModel
from phasefoundry.errors import ConfigError


def _el_cfg(cfg: dict) -> dict:
    """Return ``simulation.elastic``, or an empty dict if it is missing or empty."""
    return cfg["simulation"].get("elastic", {}) or {}


def _resolve(path: str | Path, root: Path) -> Path:
    """Join a relative path to ``root`` and leave an absolute path unchanged."""
    path = Path(path)
    return path if path.is_absolute() else Path(root) / path


def _data_dir(cfg: dict, root: Path) -> Path:
    """Return ``output.data_dir`` resolved against ``root``."""
    return _resolve(cfg["output"]["data_dir"], root)


def elastic_source(cfg: dict) -> str:
    """Return where the elastic-constant table comes from.

    Args:
        cfg: Nested config dict.

    Returns:
        ``"csv"`` (the default) or ``"mlip"``.

    Raises:
        ConfigError: If ``elastic.source`` has any other value.
    """
    source = str(_el_cfg(cfg).get("source", "csv")).lower()
    if source not in ("csv", "mlip"):
        raise ConfigError(f"elastic.source must be 'csv' or 'mlip', got {source!r}")
    return source


def lattice_model_name(cfg: dict) -> str:
    """Return which model gives the lattice parameter a(x).

    Args:
        cfg: Nested config dict.

    Returns:
        ``"sqs"`` (the default) or ``"bond"``.

    Raises:
        ConfigError: If ``elastic.lattice_model`` has any other value.
    """
    name = str(_el_cfg(cfg).get("lattice_model", "sqs")).lower()
    if name not in ("sqs", "bond"):
        raise ConfigError(f"elastic.lattice_model must be 'sqs' or 'bond', got {name!r}")
    return name


def elastic_table_path(cfg: dict, root: Path) -> Path:
    """Return the path of the SQS elastic-constant table.

    With ``source: csv`` this is ``elastic.elastic_csv``. With ``source: mlip`` it is
    ``elastic.sqs.table`` if set, and ``<data_dir>/sqs_elastic_constants_<model>.csv``
    otherwise.

    Args:
        cfg: Nested config dict.
        root: Repository root used to resolve relative paths.

    Returns:
        Path to the table.

    Raises:
        ConfigError: If ``source`` is ``csv`` and ``elastic.elastic_csv`` is not set, or
            ``source`` is ``mlip``, no table is set and no MLIP calculator is configured.
    """
    el_cfg = _el_cfg(cfg)
    if elastic_source(cfg) == "csv":
        csv_rel = el_cfg.get("elastic_csv", "")
        if not csv_rel:
            raise ConfigError("elastic.elastic_csv must be set when elastic.source is 'csv'")
        return _resolve(csv_rel, root)

    sqs_cfg = el_cfg.get("sqs", {}) or {}
    if sqs_cfg.get("table"):
        return _resolve(sqs_cfg["table"], root)
    label = mlip_label(mlip_settings(cfg, sqs_cfg))
    return _data_dir(cfg, root) / f"sqs_elastic_constants_{label}.csv"


def bond_table_path(cfg: dict, root: Path) -> Path:
    """Return the path of the bond-length matrix CSV.

    This is ``elastic.bond.table`` if set, and ``<data_dir>/bond_lengths_<model>.csv``
    otherwise.

    Args:
        cfg: Nested config dict.
        root: Repository root used to resolve relative paths.

    Returns:
        Path to the bond-length matrix.

    Raises:
        ConfigError: If no table is set and no MLIP calculator is configured.
    """
    bond_cfg = _el_cfg(cfg).get("bond", {}) or {}
    if bond_cfg.get("table"):
        return _resolve(bond_cfg["table"], root)
    label = mlip_label(mlip_settings(cfg, bond_cfg))
    return _data_dir(cfg, root) / f"bond_lengths_{label}.csv"


def bond_structure(cfg: dict) -> str:
    """Return the crystal structure of the bond-length model.

    The value is ``elastic.bond.crystal_structure``, then ``elastic.sqs.crystal_structure``,
    then ``"bcc"``, whichever is set first.

    Args:
        cfg: Nested config dict.

    Returns:
        Lowercase structure name, e.g. ``"bcc"``.
    """
    el_cfg = _el_cfg(cfg)
    bond_cfg = el_cfg.get("bond", {}) or {}
    sqs_cfg = el_cfg.get("sqs", {}) or {}
    return str(bond_cfg.get("crystal_structure", sqs_cfg.get("crystal_structure", "bcc"))).lower()


def load_lattice_model(cfg: dict, root: Path) -> BondLatticeModel | None:
    """Load the bond-length model if ``lattice_model`` is ``bond``.

    Args:
        cfg: Nested config dict.
        root: Repository root used to resolve relative paths.

    Returns:
        The bond-length model, or ``None`` if ``lattice_model`` is ``sqs`` and a(x) comes
        from the SQS table.

    Raises:
        FileNotFoundError: If the bond-length table does not exist.
    """
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
