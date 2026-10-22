"""
Console script entry points for PhaseFoundry.

Each function imports and calls the corresponding workflow script's main().
This indirection is necessary because workflow directory names contain digits
and are not valid Python package identifiers.
"""
import importlib.util, pathlib, sys

_ROOT = pathlib.Path(__file__).parent.parent.parent  # repo root


def _run(rel_path: str) -> None:
    script = _ROOT / rel_path
    spec = importlib.util.spec_from_file_location("_wf", script)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.main()


def generate():
    _run("workflows/01_generate_thermo_data/run.py")

def elastic():
    _run("workflows/01b_lookup_elastic/run.py")

def tieline():
    _run("workflows/01c_calphad_tieline/run.py")

def phase_map():
    _run("workflows/01d_phase_map/run.py")

def surrogate():
    _run("workflows/02_fit_surrogate/run.py")

def simulate():
    _run("workflows/03_simulate/run.py")

def postprocess():
    _run("workflows/04_postprocess/run.py")

def validate():
    _run("workflows/05_validate/run.py")

def compare():
    _run("workflows/06_compare/run.py")

def elastic_solve():
    from phasefoundry.elasticity.fft_solver import main
    main()

def run_all():
    _run("workflows/run_all.py")
