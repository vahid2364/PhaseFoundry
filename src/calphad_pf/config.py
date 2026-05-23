"""
calphad_pf.config — Configuration loader and override engine
=============================================================

Loads a YAML config file and applies ``--set key.subkey=value``
command-line overrides.  Dot-notation keys map to nested dicts:

    simulation.kappa=1e-10   →   cfg["simulation"]["kappa"] = 1e-10
    thermo.n_points=100      →   cfg["thermo"]["n_points"]  = 100

Type coercion is automatic:
    "true"/"false"  → bool
    integers        → int
    floats          → float
    comma-separated → list   e.g.  thermo.temperatures_C=100,200,300
    everything else → str
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


# ── loader ────────────────────────────────────────────────────────────────────

def load_config(path: str | Path) -> dict:
    """Load a YAML config file and return as a nested dict."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path) as fh:
        cfg = yaml.safe_load(fh)
    if cfg is None:
        raise ValueError(f"Config file is empty: {path}")
    return cfg


# ── override engine ───────────────────────────────────────────────────────────

def _coerce(value: str) -> Any:
    """Coerce a string value to the most appropriate Python type."""
    # bool
    if value.lower() == "true":  return True
    if value.lower() == "false": return False
    if value.lower() in ("none", "null"): return None
    # list (comma-separated)
    if "," in value:
        return [_coerce(v.strip()) for v in value.split(",")]
    # int
    try:
        return int(value)
    except ValueError:
        pass
    # float (handles 1e-10, 1.5e3, etc.)
    try:
        return float(value)
    except ValueError:
        pass
    # string fallback
    return value


def apply_overrides(cfg: dict, overrides: list[str]) -> dict:
    """
    Apply a list of ``"key.subkey=value"`` strings to *cfg* (in-place copy).

    Parameters
    ----------
    cfg : dict
        Nested config dict (from ``load_config``).
    overrides : list of str
        Each string must be of the form ``"section.key=value"``
        or ``"section.subsection.key=value"``.

    Returns
    -------
    dict
        A deep copy of *cfg* with overrides applied.

    Raises
    ------
    ValueError
        If an override string is malformed or the key path doesn't exist.

    Examples
    --------
    >>> cfg = load_config("configs/nb_v.yaml")
    >>> cfg = apply_overrides(cfg, ["simulation.kappa=1e-10", "simulation.n_steps=5000"])
    """
    cfg = copy.deepcopy(cfg)
    for override in overrides:
        if "=" not in override:
            raise ValueError(
                f"Invalid --set override {override!r}: expected 'key.subkey=value'"
            )
        key_path, raw_value = override.split("=", 1)
        keys = key_path.strip().split(".")
        value = _coerce(raw_value.strip())

        # Walk to the parent node
        node = cfg
        for k in keys[:-1]:
            if k not in node:
                raise ValueError(
                    f"Override key {key_path!r}: section {k!r} not found in config."
                )
            node = node[k]
        leaf = keys[-1]
        if leaf not in node:
            raise ValueError(
                f"Override key {key_path!r}: key {leaf!r} not found in section."
            )
        old = node[leaf]
        node[leaf] = value
        print(f"  [config] {key_path}: {old!r} → {value!r}")

    return cfg


# ── convenience ───────────────────────────────────────────────────────────────

def resolve_paths(cfg: dict, root: Path) -> dict:
    """
    Resolve all path-like values in the ``output`` and ``system`` sections
    relative to *root* (the repository root).

    Returns a deep copy with string paths replaced by ``pathlib.Path`` objects.
    """
    cfg = copy.deepcopy(cfg)
    for section in ("output", "system"):
        if section not in cfg:
            continue
        for key, val in cfg[section].items():
            if isinstance(val, str) and ("/" in val or val.endswith(".tdb")):
                cfg[section][key] = root / val
    return cfg


def summary(cfg: dict) -> str:
    """Return a human-readable summary of the active configuration."""
    lines = ["Active configuration:", "─" * 40]
    for section, values in cfg.items():
        lines.append(f"  [{section}]")
        if isinstance(values, dict):
            for k, v in values.items():
                lines.append(f"    {k:<22} {v}")
        else:
            lines.append(f"    {values}")
    return "\n".join(lines)
