"""
Step 1 — Generate thermodynamic tables (general, any N-component system)
=========================================================================
Reads system definition from config, calls pycalphad to compute GM, MU, dMU
on the composition simplex grid at each requested temperature, and saves to HDF5.

Output
------
<output.data_dir>/<system.name>_<phase>_{T}C.h5   (one file per temperature)

Run standalone
--------------
    # Binary Nb-V
    python workflows/01_generate_thermo_data/run.py --config configs/nb_v.yaml

    # Ternary Nb-Hf-V
    python workflows/01_generate_thermo_data/run.py --config configs/nb_hf_v.yaml

    # Override parameters
    python workflows/01_generate_thermo_data/run.py \
        --config configs/nb_v.yaml \
        --set thermo.n_points=50
"""

from __future__ import annotations
import argparse, sys, warnings
from pathlib import Path

import pycalphad  # noqa: F401 — must be imported first

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from calphad_pf.config import load_config, apply_overrides, resolve_paths, summary
from calphad_pf.thermodynamics import generate_table


def _table_filename(cfg: dict, T_C: float) -> str:
    name  = cfg["system"]["name"]
    phase = cfg["system"]["phase"].lower()
    return f"{name}_{phase}_{T_C:.0f}C.h5"


def run(cfg: dict) -> list[Path]:
    """Generate thermodynamic tables; skip existing. Returns list of paths."""
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
                    db_path            = str(cfg["system"]["db_path"]),
                    elements           = cfg["system"]["elements"],
                    phase              = cfg["system"]["phase"],
                    composition_ranges = {k: tuple(v) for k, v in
                                          cfg["thermo"]["composition_ranges"].items()},
                    reference_element  = cfg["system"]["reference_element"],
                    n_points           = cfg["thermo"]["n_points"],
                    temperature        = T_K,
                    output_path        = str(out),
                    quantities         = ("GM", "MU", "dMU"),
                )
            print(f"  {T_C:>6.1f} °C  →  saved {out.name}")
        paths.append(out)
    return paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Step 1 — Generate thermodynamic tables (general)")
    parser.add_argument("--config", required=True,
                        help="Path to YAML config file")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg = apply_overrides(cfg, args.set)
    cfg = resolve_paths(cfg, _ROOT)

    n_el  = len(cfg["system"]["elements"])
    n_ind = n_el - 1
    print(f"── Step 1: Generate thermodynamic tables ──")
    print(f"   System : {cfg['system']['name']}  "
          f"({n_el} components, {n_ind} independent field(s))")
    print(summary(cfg))

    paths = run(cfg)
    print(f"\nDone. {len(paths)} tables in {cfg['output']['data_dir']}")
