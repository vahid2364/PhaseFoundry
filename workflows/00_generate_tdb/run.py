"""Step 0: generate the CALPHAD TDB from MLIP SQS energies, following PhaseForge.

The step fits a TDB with ATAT ``sqs2tdb``, computing every SQS energy with the MLIP from the
``mlip`` config section through MaterialsFramework's ``Sqs2tdb``. The TDB goes to
``tdb_generation.output`` if set and to ``system.db_path`` otherwise, where steps 1 to 5 read
it. The step is skipped when the config has no ``tdb_generation`` section, when
``tdb_generation.enabled`` is false, or when the TDB exists and ``tdb_generation.overwrite``
is not true.

It needs ATAT with ``sqs2tdb`` on PATH and ``~/.atat.rc`` set, plus the ``mlip`` extra and
the potential's extra, for example ``materialsframework[orb]``.

```yaml
mlip:
  calculator: orb
  params: {model: orb-v2, device: cpu}

tdb_generation:
  enabled: true
  lattices: [BCC_A2]   # default: every lattice in the ATAT SQS database
  level: 1             # SQS composition mesh level
  t_min: 298.15
  t_max: 10000
  fmax: 0.001          # relaxation force tolerance in eV/Å
```

``species`` (default ``system.elements``), ``output``, ``work_dir`` (default
``<output.results_dir>/sqs2tdb``) and ``overwrite`` are read by this step; any other key,
such as ``sro``, ``bv``, ``phonon``, ``open_calphad``, ``terms``, ``verbose`` or ``md_*``,
is passed to ``Sqs2tdb``.

```bash
python workflows/00_generate_tdb/run.py --config configs/nb_v_mlip_c50.yaml
```
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from phasefoundry.atomistic.settings import mlip_settings
from phasefoundry.config import apply_overrides, load_config, resolve_paths, summary

_ROOT = Path(__file__).resolve().parents[2]

_LOCAL_KEYS = {"enabled", "species", "output", "work_dir", "overwrite", "calculator", "params"}
"""Keys of ``tdb_generation`` used by this step instead of being passed to ``Sqs2tdb``."""


def _resolve(path, root: Path) -> Path:
    """Join a relative path to ``root`` and leave an absolute path unchanged."""
    path = Path(path)
    return path if path.is_absolute() else root / path


def run(cfg: dict) -> Path | None:
    """Generate the TDB if the config asks for it and it does not exist yet.

    Args:
        cfg: Resolved config dict.

    Returns:
        Path of the existing or newly fitted TDB, or ``None`` if ``tdb_generation`` is
        missing or disabled.
    """
    tdb_cfg = cfg.get("tdb_generation") or {}
    if not tdb_cfg or not tdb_cfg.get("enabled", True):
        print("  No tdb_generation section enabled — using system.db_path as given.")
        return None

    output = _resolve(tdb_cfg.get("output") or cfg["system"]["db_path"], _ROOT)
    if output.exists() and not tdb_cfg.get("overwrite", False):
        print(f"  TDB exists, skipping: {output}")
        return output

    from phasefoundry.atomistic import generate_tdb

    work_dir = _resolve(tdb_cfg.get("work_dir") or Path(cfg["output"]["results_dir"]) / "sqs2tdb", _ROOT)
    options = {k: v for k, v in tdb_cfg.items() if k not in _LOCAL_KEYS}
    return generate_tdb(
        species=tdb_cfg.get("species") or cfg["system"]["elements"],
        output=output,
        settings=mlip_settings(cfg, tdb_cfg),
        work_dir=work_dir,
        **options,
    )


if __name__ == "__main__":
    logging.basicConfig(format="%(message)s", stream=sys.stdout)
    logging.getLogger("phasefoundry").setLevel(logging.INFO)
    parser = argparse.ArgumentParser(description="Step 0 — Generate the CALPHAD TDB from MLIP SQS energies")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument(
        "--set", nargs="*", default=[], metavar="KEY=VALUE", help="Override config values, e.g. tdb_generation.level=2"
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.set:
        cfg = apply_overrides(cfg, args.set)
    cfg = resolve_paths(cfg, _ROOT)

    print("=" * 60)
    print("Step 0 — TDB generation (MLIP + sqs2tdb)")
    print(summary(cfg))
    print("=" * 60)

    run(cfg)
