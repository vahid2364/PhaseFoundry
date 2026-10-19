"""phasefoundry.atomistic.tdb
==========================
Generate a CALPHAD TDB from MLIP SQS energies with the PhaseForge approach
(ATAT ``sqs2tdb`` driven by MaterialsFramework's ``Sqs2tdb`` wrapper).

ATAT must be installed with ``sqs2tdb`` on PATH and ``~/.atat.rc`` pointing at
the ATAT install (for the SQS database).  ATAT writes its lattice folders
into the current directory, so the fit runs inside a dedicated work
directory and only the final TDB is copied to the requested output path.

Public API
----------
generate_tdb(species, output, settings, …)  → Path
"""

from __future__ import annotations

import os
import shutil
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from .settings import build_calculator, require_materialsframework

SQS2TDB_INIT_KEYS = ("fmax", "verbose", "md_temperature", "md_pressure", "md_timestep", "md_scaling_matrix")
SQS2TDB_FIT_KEYS = ("lattices", "level", "t_min", "t_max", "sro", "bv", "phonon", "open_calphad", "terms")


@contextmanager
def _working_directory(path: Path):
    prev = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)


def generate_tdb(
    species: list[str],
    output: str | Path,
    settings: dict,
    *,
    work_dir: str | Path,
    calculator: Any = None,
    **options: Any,
) -> Path:
    """Fit a TDB from MLIP SQS energies and copy it to *output*.

    Parameters
    ----------
    species    : elements (any case), e.g. ['NB', 'V']
    output     : destination TDB path
    settings   : ``mlip_settings`` output
    work_dir   : directory for ATAT's lattice folders and intermediate files
    calculator : pre-built calculator (default: built from *settings*)
    **options  : ``Sqs2tdb`` options — fmax, verbose, md_* (constructor) and
                 lattices, level, t_min, t_max, sro, bv, phonon,
                 open_calphad, terms (``fit``)
    """
    require_materialsframework()
    from materialsframework.tools.sqs2tdb import Sqs2tdb

    unknown = set(options) - set(SQS2TDB_INIT_KEYS) - set(SQS2TDB_FIT_KEYS)
    if unknown:
        raise ValueError(f"Unknown tdb_generation option(s): {sorted(unknown)}")

    symbols = [s.capitalize() for s in species]
    output = Path(output).resolve()
    work_dir = Path(work_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    if calculator is None:
        calculator = build_calculator(settings)

    init_kw = {k: options[k] for k in SQS2TDB_INIT_KEYS if k in options}
    fit_kw = {k: options[k] for k in SQS2TDB_FIT_KEYS if k in options}
    print(f"  sqs2tdb: {'-'.join(symbols)}  lattices={fit_kw.get('lattices', 'all')}  level={fit_kw.get('level', 1)}")
    print(f"  Work dir: {work_dir}")

    with _working_directory(work_dir):
        fitter = Sqs2tdb(calculator=calculator, **init_kw)
        fitter.fit(species=symbols, **fit_kw)

    produced = work_dir / ("_".join(sorted(s.upper() for s in symbols)) + ".tdb")
    if not produced.exists():
        raise RuntimeError(
            f"sqs2tdb finished without writing {produced}; re-run with tdb_generation.verbose: true for ATAT output."
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(produced, output)
    print(f"  Written: {output}")
    return output
