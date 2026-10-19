"""Step 0 — Generate the CALPHAD TDB from MLIP SQS energies (PhaseForge)
======================================================================
Fits a TDB with ATAT's sqs2tdb, computing every SQS energy with the MLIP from
the ``mlip`` config section (MaterialsFramework ``Sqs2tdb``), and writes it
to ``system.db_path`` so steps 1–5 pick it up unchanged.

Skipped when the config has no ``tdb_generation`` section, when
``tdb_generation.enabled`` is false, or when the output TDB already exists
(set ``tdb_generation.overwrite: true`` to refit).

Requirements
------------
ATAT with ``sqs2tdb`` on PATH and ``~/.atat.rc`` set, and
``pip install -e '.[mlip]'`` plus the potential's extra
(e.g. ``materialsframework[orb]``).

Config
------
    mlip:
      calculator: orb
      params: {model: orb-v2, device: cpu}

    tdb_generation:
      enabled:  true
      lattices: [BCC_A2]         # default: every lattice in the ATAT SQS database
      level:    1                # SQS composition mesh level
      t_min:    298.15
      t_max:    10000
      fmax:     0.001            # relaxation force tolerance [eV/Å]
      # sro, bv, phonon, open_calphad, terms, verbose, md_* → Sqs2tdb
      # species:  [NB, V]        # default: system.elements
      # output:   data/tdb/...   # default: system.db_path
      # work_dir: ...            # default: <output.results_dir>/sqs2tdb
      # overwrite: false

Usage
-----
    python workflows/00_generate_tdb/run.py --config configs/nb_v_mlip.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from phasefoundry.atomistic.settings import mlip_settings
from phasefoundry.config import apply_overrides, load_config, resolve_paths, summary

# tdb_generation keys handled here rather than forwarded to Sqs2tdb
_LOCAL_KEYS = {"enabled", "species", "output", "work_dir", "overwrite", "calculator", "params"}


def _resolve(path, root: Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else root / path


def run(cfg: dict) -> Path | None:
    """Generate the TDB if configured and missing. Returns its path, or None if skipped."""
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
