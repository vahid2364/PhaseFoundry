"""
phasefoundry.config — Configuration loader and override engine
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


# ── run-name builder ──────────────────────────────────────────────────────────

def build_run_name(cfg: dict) -> str:
    """
    Auto-construct a run_name from simulation parameters.

    Pattern:
        pf_{ndim}d_c{mol%}_kappa{kappa_str}_M{M_str}_{tdb_tag}

    Examples:
        pf_2d_c50_kappa1e12_M5e21_tdbv3
        pf_3d_c50Nb05Hf_kappa1e12_M5e21_tdbv3
    """
    sim      = cfg["simulation"]
    db_path  = str(cfg["system"].get("db_path", ""))
    elements = cfg["system"]["elements"]
    ref_el   = cfg["system"]["reference_element"]
    ind_els  = [e for e in elements if e != ref_el]

    ndim = int(sim.get("dimensions", 2))

    # composition tag: always includes element symbol to avoid ambiguity
    # binary: c60Nb  |  ternary: c60Nb05Hf
    c0 = sim["c0"]
    if not isinstance(c0, list):
        c0 = [c0]
    parts = []
    for xi, el in zip(c0, ind_els):
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
    """
    Convert a float to a compact scientific-notation tag (no sign, no dot).
    Examples: 1e-12 → '1e12',  5e-21 → '5e21',  1.5e-10 → '15e10'
    """
    import math
    if value == 0:
        return "0"
    exp  = int(math.floor(math.log10(abs(value))))
    mant = value / (10 ** exp)
    # round mantissa to avoid floating-point noise
    mant_r = round(mant, 6)
    if abs(mant_r - round(mant_r)) < 1e-9:
        mant_str = str(int(round(mant_r)))
    else:
        mant_str = f"{mant_r:.2g}".replace(".", "")
    return f"{mant_str}e{abs(exp)}"


# ── convenience ───────────────────────────────────────────────────────────────

def resolve_paths(cfg: dict, root: Path) -> dict:
    """
    Resolve all path-like values in the ``output`` and ``system`` sections
    relative to *root* (the repository root).

    Also auto-constructs ``output.run_name`` from simulation parameters when
    it is set to ``"auto"`` or is absent entirely.

    Returns a deep copy with string paths replaced by ``pathlib.Path`` objects.
    """
    cfg = copy.deepcopy(cfg)
    for section in ("output", "system"):
        if section not in cfg:
            continue
        for key, val in cfg[section].items():
            if isinstance(val, str) and ("/" in val or val.endswith(".tdb")):
                cfg[section][key] = root / val

    # auto run_name
    run_name = cfg.get("output", {}).get("run_name", "auto")
    if run_name in ("auto", None, ""):
        cfg["output"]["run_name"] = build_run_name(cfg)
        print(f"  [config] run_name (auto): {cfg['output']['run_name']!r}")

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
