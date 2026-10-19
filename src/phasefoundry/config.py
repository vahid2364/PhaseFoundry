"""YAML configuration loading and command-line overrides.

A config file is a nested YAML mapping. ``--set`` overrides reach nested keys with dot
notation, so ``simulation.kappa=1e-10`` sets ``cfg["simulation"]["kappa"]``. Override
values are converted to the closest Python type: ``true`` and ``false`` become bool,
``none`` and ``null`` become None, numbers become int or float, comma-separated values
become a list (``thermo.temperatures_C=100,200,300``), and anything else stays a string.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict:
    """Load a YAML config file.

    Args:
        path: Path to the YAML file.

    Returns:
        The config as a nested dict.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If the file is empty.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open() as fh:
        cfg = yaml.safe_load(fh)
    if cfg is None:
        raise ValueError(f"Config file is empty: {path}")
    return cfg


_LITERALS = {"true": True, "false": False, "none": None, "null": None}
"""Override values, compared case-insensitively, that map to a Python constant."""


def _coerce(value: str) -> Any:
    """Convert an override value to bool, None, int, float, list or str.

    Comma-separated input gives a list whose items are converted the same way.
    """
    if value.lower() in _LITERALS:
        return _LITERALS[value.lower()]
    if "," in value:
        return [_coerce(v.strip()) for v in value.split(",")]
    for convert in (int, float):
        try:
            return convert(value)
        except ValueError:
            pass
    return value


def apply_overrides(cfg: dict, overrides: list[str]) -> dict:
    """Apply ``--set`` overrides to a copy of a config.

    Args:
        cfg: Nested config dict, as returned by ``load_config``.
        overrides: Strings of the form ``"section.key=value"`` or
            ``"section.subsection.key=value"``. Each key path must already exist in ``cfg``.

    Returns:
        A deep copy of ``cfg`` with the overrides applied. ``cfg`` itself is not modified.

    Raises:
        ValueError: If an override has no ``=``, or its key path is not in ``cfg``.

    Examples:
        ```python
        cfg = load_config("configs/nb_v_pipeline_test.yaml")
        cfg = apply_overrides(cfg, ["simulation.kappa=1e-10", "simulation.n_steps=5000"])
        ```
    """
    cfg = copy.deepcopy(cfg)
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Invalid --set override {override!r}: expected 'key.subkey=value'")
        key_path, raw_value = override.split("=", 1)
        keys = key_path.strip().split(".")
        value = _coerce(raw_value.strip())

        # Walk to the parent node
        node = cfg
        for k in keys[:-1]:
            if k not in node:
                raise ValueError(f"Override key {key_path!r}: section {k!r} not found in config.")
            node = node[k]
        leaf = keys[-1]
        if leaf not in node:
            raise ValueError(f"Override key {key_path!r}: key {leaf!r} not found in section.")
        old = node[leaf]
        node[leaf] = value
        print(f"  [config] {key_path}: {old!r} → {value!r}")

    return cfg


def build_run_name(cfg: dict) -> str:
    """Build a run name from the simulation parameters.

    The name has the form ``pf_{ndim}d_c{composition}_kappa{kappa}_M{M}_{tdb}[_elastic]``.
    The composition tag gives the mole percent and symbol of each independent element.
    ``kappa`` and ``M`` are formatted with ``_sci_tag``, using only the first entry when
    they are lists. The TDB tag is ``tdbv<N>`` when the database file name contains
    ``_v<N>``, and ``tdb`` otherwise.

    Args:
        cfg: Nested config dict with ``system`` and ``simulation`` sections.

    Returns:
        The run name, e.g. ``pf_2d_c50Nb_kappa1e12_M5e21_tdbv3`` or
        ``pf_3d_c50Nb05Hf_kappa1e12_M5e21_tdbv3``.
    """
    sim = cfg["simulation"]
    db_path = str(cfg["system"].get("db_path", ""))
    elements = cfg["system"]["elements"]
    ref_el = cfg["system"]["reference_element"]
    ind_els = [e for e in elements if e != ref_el]

    ndim = int(sim.get("dimensions", 2))

    # composition tag: always includes element symbol to avoid ambiguity
    # binary: c60Nb  |  ternary: c60Nb05Hf
    c0 = sim["c0"]
    if not isinstance(c0, list):
        c0 = [c0]
    parts = []
    for xi, el in zip(c0, ind_els, strict=True):
        parts.append(f"{round(xi * 100):02d}{el.capitalize()}")
    comp_tag = "c" + "".join(parts)

    # kappa tag: 1e-12 → kappa1e12
    kappa = sim["kappa"]
    if isinstance(kappa, list):
        kappa = kappa[0]
    kappa_tag = "kappa" + _sci_tag(kappa)

    # mobility tag: 5e-21 → M5e21
    M = sim["M"]
    if isinstance(M, list):
        M = M[0][0] if isinstance(M[0], list) else M[0]
    M_tag = "M" + _sci_tag(M)

    # TDB version tag: HfNbTiV_BCC_PhaseForge_v3.tdb      → tdbv3
    #                  HfNbTiV_BCC_PhaseForge_v2_Tdep.tdb → tdbv2
    import re

    tdb_tag = "tdb"
    m = re.search(r"_v(\d+)", Path(db_path).stem, re.IGNORECASE)
    if m:
        tdb_tag = f"tdbv{m.group(1)}"

    elastic_tag = "_elastic" if sim.get("elastic", {}).get("enabled", False) else ""
    return f"pf_{ndim}d_{comp_tag}_{kappa_tag}_{M_tag}_{tdb_tag}{elastic_tag}"


def _sci_tag(value: float) -> str:
    """Format a number as a compact exponent tag without a decimal point.

    For example, ``1e-12`` gives ``"1e12"``, ``5e-21`` gives ``"5e21"`` and ``1.5e-10``
    gives ``"15e10"``. The sign of the exponent is dropped, so ``1e12`` and ``1e-12``
    give the same tag.
    """
    import math

    if value == 0:
        return "0"
    exp = int(math.floor(math.log10(abs(value))))
    mant = value / (10**exp)
    # round mantissa to avoid floating-point noise
    mant_r = round(mant, 6)
    mant_str = str(int(round(mant_r))) if abs(mant_r - round(mant_r)) < 1e-09 else f"{mant_r:.2g}".replace(".", "")
    return f"{mant_str}e{abs(exp)}"


def resolve_paths(cfg: dict, root: Path) -> dict:
    """Turn path strings in the ``output`` and ``system`` sections into paths under ``root``.

    A string counts as a path if it contains ``/`` or ends in ``.tdb``. When
    ``output.run_name`` is missing, empty or ``"auto"``, it is filled in by
    ``build_run_name``.

    Args:
        cfg: Nested config dict.
        root: Repository root that relative paths are joined to.

    Returns:
        A deep copy of ``cfg`` with those strings replaced by ``pathlib.Path`` objects.
    """
    cfg = copy.deepcopy(cfg)
    for section in ("output", "system"):
        if section not in cfg:
            continue
        for key, val in cfg[section].items():
            if isinstance(val, str) and ("/" in val or val.endswith(".tdb")):
                cfg[section][key] = root / val

    run_name = cfg.get("output", {}).get("run_name", "auto")
    if run_name in ("auto", None, ""):
        cfg["output"]["run_name"] = build_run_name(cfg)
        print(f"  [config] run_name (auto): {cfg['output']['run_name']!r}")

    return cfg


def summary(cfg: dict) -> str:
    """Format the config as an indented, human-readable listing.

    Args:
        cfg: Nested config dict.

    Returns:
        One line per section, followed by one line per key in that section.
    """
    lines = ["Active configuration:", "─" * 40]
    for section, values in cfg.items():
        lines.append(f"  [{section}]")
        if isinstance(values, dict):
            for k, v in values.items():
                lines.append(f"    {k:<22} {v}")
        else:
            lines.append(f"    {values}")
    return "\n".join(lines)
