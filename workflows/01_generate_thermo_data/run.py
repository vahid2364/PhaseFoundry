"""Step 1: compute the thermodynamic tables of an N-component system with pycalphad.

For each temperature in ``thermo.temperatures_C`` the step evaluates G_m, the chemical
potentials and their composition derivatives on the composition grid and writes one HDF5
file, ``<output.data_dir>/<system.name>_<phase>_<T>C.h5``, with the phase name in lower case
and T in °C. Tables that already exist are skipped.

```bash
python workflows/01_generate_thermo_data/run.py --config configs/nb_v_pipeline_test.yaml
python workflows/01_generate_thermo_data/run.py --config configs/nb_hf_v_pipeline_test.yaml
python workflows/01_generate_thermo_data/run.py --config configs/nb_v_pipeline_test.yaml --set thermo.n_points=50
```
"""

from __future__ import annotations

import argparse
import logging
import sys
import warnings
from pathlib import Path

import pycalphad  # noqa: F401

from phasefoundry.config import apply_overrides, load_config, resolve_paths, summary
from phasefoundry.thermodynamics import generate_table

_ROOT = Path(__file__).resolve().parents[2]


def _table_filename(cfg: dict, T_C: float) -> str:
    """Return the table file name for a temperature ``T_C`` in °C."""
    name = cfg["system"]["name"]
    phase = cfg["system"]["phase"].lower()
    return f"{name}_{phase}_{T_C:.0f}C.h5"


def run(cfg: dict) -> list[Path]:
    """Compute the missing tables for every configured temperature.

    Args:
        cfg: Resolved config dict.

    Returns:
        Paths of all tables, including the ones that already existed.
    """
    data_dir = Path(cfg["output"]["data_dir"])
    data_dir.mkdir(parents=True, exist_ok=True)

    paths = []
    for T_C in cfg["thermo"]["temperatures_C"]:
        T_K = T_C + 273.15
        out = data_dir / _table_filename(cfg, T_C)
        if out.exists():
            print(f"  {T_C:>6.1f} °C  →  exists, skipping")
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                generate_table(
                    db_path=str(cfg["system"]["db_path"]),
                    elements=cfg["system"]["elements"],
                    phase=cfg["system"]["phase"],
                    composition_ranges={k: tuple(v) for k, v in cfg["thermo"]["composition_ranges"].items()},
                    reference_element=cfg["system"]["reference_element"],
                    n_points=cfg["thermo"]["n_points"],
                    temperature=T_K,
                    output_path=str(out),
                    quantities=("GM", "MU", "dMU"),
                )
            print(f"  {T_C:>6.1f} °C  →  saved {out.name}")
        paths.append(out)
    return paths


if __name__ == "__main__":
    logging.basicConfig(format="%(message)s", stream=sys.stdout)
    logging.getLogger("phasefoundry").setLevel(logging.INFO)
    parser = argparse.ArgumentParser(description="Step 1 — Generate thermodynamic tables (general)")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg = apply_overrides(cfg, args.set)
    cfg = resolve_paths(cfg, _ROOT)

    n_el = len(cfg["system"]["elements"])
    n_ind = n_el - 1
    print("── Step 1: Generate thermodynamic tables ──")
    print(f"   System : {cfg['system']['name']}  ({n_el} components, {n_ind} independent field(s))")
    print(summary(cfg))

    paths = run(cfg)
    print(f"\nDone. {len(paths)} tables in {cfg['output']['data_dir']}")
