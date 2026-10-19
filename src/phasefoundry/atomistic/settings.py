"""phasefoundry.atomistic.settings
===============================
Resolve MLIP calculator settings from the config and build MaterialsFramework
calculators.

Config
------
    mlip:
      calculator: orb            # any name in materialsframework.calculators.list_calculators()
      params:                    # forwarded to the calculator constructor
        model:  orb-v2
        device: cpu
        fmax:   0.01

A feature block (``simulation.elastic.sqs``, ``simulation.elastic.bond``,
``tdb_generation``) may carry its own ``calculator`` / ``params`` keys that
override the top-level section, e.g. to compute bond lengths with GRACE and
elastic constants with ORB.  If the override names a different calculator,
the top-level ``params`` are not inherited.

Nothing in this module imports MaterialsFramework at import time.
"""

from __future__ import annotations

import re
from typing import Any

INSTALL_HINT = (
    "MLIP calculations require MaterialsFramework (Python 3.12):\n"
    "    pip install -e '.[mlip]'\n"
    "plus the extra for the chosen potential, e.g.\n"
    "    pip install 'materialsframework[orb]'"
)


def require_materialsframework() -> None:
    """Raise an ImportError with install instructions if MaterialsFramework is missing."""
    try:
        import materialsframework  # noqa: F401
    except ImportError as exc:
        raise ImportError(INSTALL_HINT) from exc


def mlip_settings(cfg: dict, block: dict | None = None) -> dict:
    """Merge the top-level ``mlip`` section with a feature-block override.

    Returns ``{"calculator": str, "params": dict}``.
    """
    base = cfg.get("mlip") or {}
    block = block or {}
    name = block.get("calculator", base.get("calculator"))
    if not name:
        raise ValueError(
            "No MLIP calculator configured. Add a top-level 'mlip' section, e.g.\n"
            "  mlip:\n    calculator: orb\n    params: {model: orb-v2, device: cpu}"
        )
    if "calculator" in block and block["calculator"] != base.get("calculator"):
        params = dict(block.get("params") or {})
    else:
        params = {**(base.get("params") or {}), **(block.get("params") or {})}
    return {"calculator": str(name), "params": params}


def mlip_label(settings: dict) -> str:
    """Filesystem-safe tag for cache file names: the model name, else the calculator name."""
    raw = str(settings["params"].get("model") or settings["calculator"])
    return re.sub(r"[^A-Za-z0-9._-]+", "_", raw).strip("_")


def build_calculator(settings: dict) -> Any:
    """Instantiate a MaterialsFramework calculator from ``mlip_settings`` output."""
    require_materialsframework()
    from materialsframework.calculators import get_calculator

    return get_calculator(settings["calculator"], **settings["params"])
