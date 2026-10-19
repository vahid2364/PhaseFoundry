"""Integrated Phase-Field Pipeline
================================
Runs all workflow steps end-to-end for any alloy system, driven entirely
by a YAML config file.  No code changes are needed to switch systems —
only the config file changes.

Usage
-----
    # Full pipeline — binary Nb-V
    python workflows/run_all.py --config configs/nb_v.yaml

    # Full pipeline — ternary Nb-Hf-V
    python workflows/run_all.py --config configs/nb_hf_v.yaml

    # Resume from step 2
    python workflows/run_all.py --config configs/nb_v.yaml --from 2

    # Run only step 3
    python workflows/run_all.py --config configs/nb_v.yaml --only 3

    # Override config values on the fly (no file editing needed)
    python workflows/run_all.py --config configs/nb_v.yaml \
        --set simulation.kappa=1e-10 simulation.n_steps=10000 \
              output.run_name=pf_nb_v_500C_kappa1e10_test

    # New alloy system — just create a new config file
    cp configs/nb_v.yaml configs/ti_v.yaml
    # edit configs/ti_v.yaml, then:
    python workflows/run_all.py --config configs/ti_v.yaml
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import pycalphad  # noqa: F401 — must be imported first

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from phasefoundry.config import apply_overrides, load_config, resolve_paths, summary

STEP_SCRIPTS = {
    0: _ROOT / "workflows" / "00_generate_tdb" / "run.py",
    1: _ROOT / "workflows" / "01_generate_thermo_data" / "run.py",
    2: _ROOT / "workflows" / "02_fit_surrogate" / "run.py",
    3: _ROOT / "workflows" / "03_simulate" / "run.py",
    4: _ROOT / "workflows" / "04_postprocess" / "run.py",
    5: _ROOT / "workflows" / "05_validate" / "run.py",
}
STEP_LABELS = {
    0: "Generate TDB from MLIP (skipped unless tdb_generation is set)",
    1: "Generate thermodynamic tables",
    2: "Fit G_m spline surrogate",
    3: "Phase-field simulation",
    4: "Post-processing & figures",
    5: "Validate thermodynamic tables & derivative methods",
}


def _load_step(path: Path):
    spec = importlib.util.spec_from_file_location("step", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    parser = argparse.ArgumentParser(
        description="Integrated CALPHAD / phase-field pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--config", default="configs/nb_v.yaml", help="Path to YAML config file  (default: configs/nb_v.yaml)")
    parser.add_argument("--from", dest="from_step", type=int, default=0, metavar="N", help="Start from step N  (0–5)")
    parser.add_argument("--only", dest="only_step", type=int, default=None, metavar="N", help="Run only step N  (0–5)")
    parser.add_argument(
        "--set", nargs="*", default=[], metavar="KEY=VALUE", help="Override config values, e.g. --set simulation.kappa=1e-10"
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg = apply_overrides(cfg, args.set or [])
    cfg = resolve_paths(cfg, _ROOT)

    steps = [args.only_step] if args.only_step is not None else list(range(args.from_step, 6))

    print("=" * 60)
    print("  Integrated CALPHAD / Phase-Field Pipeline")
    print(f"  Config  : {args.config}")
    print(f"  System  : {cfg['system']['name'].upper()}")
    print(f"  Steps   : {steps}")
    print("=" * 60)
    print(summary(cfg))

    for s in steps:
        print(f"\n{'─' * 60}")
        print(f"  Step {s}: {STEP_LABELS[s]}")
        print(f"{'─' * 60}")
        mod = _load_step(STEP_SCRIPTS[s])
        mod.run(cfg)

    print("\n" + "=" * 60)
    print("  Pipeline complete.")
    print(f"  Results → {cfg['output']['results_dir']}/{cfg['output']['run_name']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
