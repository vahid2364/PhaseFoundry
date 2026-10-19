"""Run the CALPHAD and phase-field pipeline end to end from one YAML config.

Steps 0 to 5 run in order: TDB generation from MLIP energies (skipped unless the config has
a ``tdb_generation`` section), thermodynamic tables, surrogate fitting, the phase-field
simulation, post-processing and validation. Each step is a ``run.py`` under ``workflows/``,
and another alloy system only needs another config file. Steps 1b, 1c, 1d, 6 and 7 are
separate scripts that this driver does not run.

```bash
python workflows/run_all.py --config configs/nb_v_pipeline_test.yaml
python workflows/run_all.py --config configs/nb_hf_v_pipeline_test.yaml
python workflows/run_all.py --config configs/nb_v_pipeline_test.yaml --from 2
python workflows/run_all.py --config configs/nb_v_pipeline_test.yaml --only 3
python workflows/run_all.py --config configs/nb_v_pipeline_test.yaml --set simulation.kappa=1e-10 simulation.n_steps=10000
```
"""

from __future__ import annotations

import argparse
import importlib.util
import logging
import sys
from pathlib import Path

import pycalphad  # noqa: F401

from phasefoundry.config import apply_overrides, load_config, resolve_paths, summary

_ROOT = Path(__file__).resolve().parent.parent

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
    """Import a step script by path and return it as a module."""
    spec = importlib.util.spec_from_file_location("step", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    """Parse the command line, load and override the config, and run the selected steps."""
    parser = argparse.ArgumentParser(
        description="Integrated CALPHAD / phase-field pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--config", required=True, help="Path to YAML config file")
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
    logging.basicConfig(format="%(message)s", stream=sys.stdout)
    logging.getLogger("phasefoundry").setLevel(logging.INFO)
    main()
