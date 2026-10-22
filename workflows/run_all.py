"""
Integrated Phase-Field Pipeline
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
import time
from pathlib import Path

import pycalphad  # noqa: F401 — must be imported first

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from phasefoundry.config import load_config, apply_overrides, resolve_paths, summary
from phasefoundry.logger import get_logger, read_timings, PFLogger

STEP_SCRIPTS = {
    1:  _ROOT / "workflows" / "01_generate_thermo_data" / "run.py",
    2:  _ROOT / "workflows" / "01d_phase_map"           / "run.py",
    3:  _ROOT / "workflows" / "02_fit_surrogate"        / "run.py",
    4:  _ROOT / "workflows" / "03_simulate"             / "run.py",
    5:  _ROOT / "workflows" / "04_postprocess"          / "run.py",
    6:  _ROOT / "workflows" / "05_validate"             / "run.py",
}
STEP_LABELS = {
    1: "Generate thermodynamic tables",
    2: "Ternary phase map",
    3: "Fit G_m surrogate + spinodal analysis",
    4: "Phase-field simulation",
    5: "Post-processing & figures",
    6: "Validate thermodynamic tables & derivative methods",
}


def _load_step(path: Path):
    spec = importlib.util.spec_from_file_location("step", path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    parser = argparse.ArgumentParser(
        description="Integrated CALPHAD / phase-field pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--config", default="configs/nb_v.yaml",
                        help="Path to YAML config file  (default: configs/nb_v.yaml)")
    parser.add_argument("--from", dest="from_step", type=int, default=1,
                        metavar="N", help="Start from step N  (1–6)")
    parser.add_argument("--only", dest="only_step", type=int, default=None,
                        metavar="N", help="Run only step N  (1–6)")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                        help="Override config values, e.g. --set simulation.kappa=1e-10")
    args = parser.parse_args()

    # Load and resolve config
    cfg = load_config(args.config)
    cfg = apply_overrides(cfg, args.set or [])
    cfg = resolve_paths(cfg, _ROOT)

    steps = [args.only_step] if args.only_step else list(range(args.from_step, 7))

    print("=" * 60)
    print("  Integrated CALPHAD / Phase-Field Pipeline")
    print(f"  Config  : {args.config}")
    print(f"  System  : {cfg['system']['name'].upper()}")
    print(f"  Steps   : {steps}")
    print("=" * 60)
    print(summary(cfg))

    pipeline_records: list[dict] = []
    pipeline_t0 = time.perf_counter()
    lcfg = cfg.get("logging", {}) or {}

    for s in steps:
        label = STEP_LABELS[s]
        print(f"\n{'─' * 60}")
        print(f"  Step {s}: {label}")
        print(f"{'─' * 60}")
        t0 = time.perf_counter()
        status = "ok"
        try:
            mod = _load_step(STEP_SCRIPTS[s])
            mod.run(cfg)
        except Exception as exc:
            print(f"  ✗  Step {s} failed: {exc}")
            status = "failed"
        elapsed = time.perf_counter() - t0
        pipeline_records.append({"step": str(s), "label": label,
                                  "elapsed_s": round(elapsed, 2), "status": status})
        if status == "failed":
            break

    total_elapsed = time.perf_counter() - pipeline_t0
    pipeline_ok   = all(r["status"] == "ok" for r in pipeline_records)

    print("\n" + "=" * 60)
    if pipeline_ok:
        print("  Pipeline complete.")
        print(f"  Results → {cfg['output']['results_dir']}/{cfg['output']['run_name']}")
    else:
        failed_step = next(r for r in pipeline_records if r["status"] == "failed")
        print(f"  Pipeline FAILED at step {failed_step['step']}: {failed_step['label']}")
    print("=" * 60)

    # Pipeline summary table (always shown; also written to timings.jsonl if logging on)
    if lcfg.get("pipeline_summary", True) and pipeline_records:
        print(PFLogger.pipeline_summary(pipeline_records))

    if lcfg.get("enabled", False):
        results_dir = Path(cfg["output"]["results_dir"])
        pf_log = get_logger(cfg, step="run_all", results_dir=results_dir)
        pf_log.log_timing_record({
            "label":     "pipeline_total",
            "elapsed_s": round(total_elapsed, 2),
            "steps":     [r["step"] for r in pipeline_records],
            "status":    "ok" if pipeline_ok else "failed",
        })

    if not pipeline_ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
