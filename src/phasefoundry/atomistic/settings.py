"""MLIP calculator settings and MaterialsFramework calculator construction.

Calculator settings come from the top-level ``mlip`` section of the config:

```yaml
mlip:
  calculator: orb      # any name in materialsframework.calculators.list_calculators()
  params:              # passed to the calculator constructor
    model: orb-v2
    device: cpu
    fmax: 0.01
```

A feature block (``simulation.elastic.sqs``, ``simulation.elastic.bond`` or
``tdb_generation``) can set its own ``calculator`` and ``params`` to override that
section, for example to compute bond lengths with GRACE and elastic constants with ORB.
A block that names a different calculator does not inherit the top-level ``params``.

This module does not import MaterialsFramework at import time.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from phasefoundry.errors import ConfigError, MissingDependencyError

INSTALL_HINT = (
    "MLIP calculations require MaterialsFramework (Python 3.12):\n"
    "    pip install -e '.[mlip]'\n"
    "plus the extra for the chosen potential, e.g.\n"
    "    pip install 'materialsframework[orb]'"
)
"""Error message shown when MaterialsFramework is not installed."""


@dataclass(frozen=True)
class MLIPSettings:
    """MaterialsFramework calculator name and constructor parameters.

    Attributes:
        calculator: Calculator name, as accepted by ``materialsframework.calculators.get_calculator``.
        params: Keyword arguments for the calculator constructor.
    """

    calculator: str
    params: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Check that a calculator is named."""
        if not self.calculator:
            raise ConfigError("MLIPSettings.calculator must be a non-empty string")


def require_materialsframework() -> None:
    """Check that MaterialsFramework can be imported.

    Raises:
        MissingDependencyError: If MaterialsFramework is not installed. The message says how to install it.
    """
    try:
        import materialsframework  # noqa: F401
    except ImportError as exc:
        raise MissingDependencyError(INSTALL_HINT) from exc


def mlip_settings(cfg: dict, block: dict | None = None) -> MLIPSettings:
    """Merge the top-level ``mlip`` section with a feature block's overrides.

    Args:
        cfg: Nested config dict.
        block: Feature block that may set ``calculator`` and ``params``, e.g.
            ``cfg["simulation"]["elastic"]["sqs"]``. ``None`` uses the top-level section only.

    Returns:
        The merged settings. If ``block`` names a different calculator from the top-level
        one, ``params`` comes from ``block`` alone. Otherwise the block's ``params`` are
        merged over the top-level ones.

    Raises:
        ConfigError: If neither ``block`` nor the ``mlip`` section names a calculator.
    """
    base = cfg.get("mlip") or {}
    block = block or {}
    name = block.get("calculator", base.get("calculator"))
    if not name:
        raise ConfigError(
            "No MLIP calculator configured. Add a top-level 'mlip' section, e.g.\n"
            "  mlip:\n    calculator: orb\n    params: {model: orb-v2, device: cpu}"
        )
    if "calculator" in block and block["calculator"] != base.get("calculator"):
        params = dict(block.get("params") or {})
    else:
        params = {**(base.get("params") or {}), **(block.get("params") or {})}
    return MLIPSettings(str(name), params)


def mlip_label(settings: MLIPSettings) -> str:
    """Make a filesystem-safe tag for cache file names.

    Args:
        settings: Output of ``mlip_settings``.

    Returns:
        The model name from ``params["model"]``, or the calculator name if no model is set.
        Characters other than letters, digits, ``.``, ``_`` and ``-`` become ``_``.
    """
    raw = str(settings.params.get("model") or settings.calculator)
    return re.sub(r"[^A-Za-z0-9._-]+", "_", raw).strip("_")


def build_calculator(settings: MLIPSettings) -> Any:
    """Create a MaterialsFramework calculator.

    Args:
        settings: Output of ``mlip_settings``.

    Returns:
        The calculator returned by ``materialsframework.calculators.get_calculator``.

    Raises:
        MissingDependencyError: If MaterialsFramework is not installed.
    """
    require_materialsframework()
    from materialsframework.calculators import get_calculator

    return get_calculator(settings.calculator, **settings.params)
