"""CALPHAD TDB generation from MLIP SQS energies, following PhaseForge.

The fit is ATAT ``sqs2tdb`` driven through MaterialsFramework's ``Sqs2tdb`` wrapper.
ATAT must be installed, with ``sqs2tdb`` on PATH and ``~/.atat.rc`` pointing at the ATAT
installation so the SQS database can be found. ATAT writes its lattice folders into the
current directory, so the fit runs inside a separate work directory and only the final
TDB is copied to the output path.
"""

from __future__ import annotations

import logging
import os
import shutil
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from phasefoundry.atomistic.settings import MLIPSettings, build_calculator, require_materialsframework
from phasefoundry.errors import ConfigError

_logger = logging.getLogger(__name__)

SQS2TDB_INIT_KEYS = ("fmax", "verbose", "md_temperature", "md_pressure", "md_timestep", "md_scaling_matrix")
"""Options passed to the ``Sqs2tdb`` constructor."""

SQS2TDB_FIT_KEYS = ("lattices", "level", "t_min", "t_max", "sro", "bv", "phonon", "open_calphad", "terms")
"""Options passed to ``Sqs2tdb.fit``."""


@contextmanager
def _working_directory(path: Path):
    """Change into ``path`` for the duration of the ``with`` block."""
    prev = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)


def generate_tdb(
    species: list[str],
    output: str | Path,
    settings: MLIPSettings,
    *,
    work_dir: str | Path,
    calculator: Any = None,
    **options: Any,
) -> Path:
    """Fit a TDB from MLIP SQS energies and copy it to ``output``.

    Args:
        species: Elements, in any case, e.g. ``["NB", "V"]``.
        output: Destination TDB path.
        settings: Output of ``mlip_settings``.
        work_dir: Directory for ATAT's lattice folders and intermediate files.
        calculator: Calculator to use. Defaults to one built from ``settings``.
        **options: ``Sqs2tdb`` options. The keys in ``SQS2TDB_INIT_KEYS`` (``fmax``,
            ``verbose`` and the ``md_*`` settings) go to the constructor, and the keys in
            ``SQS2TDB_FIT_KEYS`` (``lattices``, ``level``, ``t_min``, ``t_max`` and so on)
            go to ``fit``.

    Returns:
        The resolved output path.

    Raises:
        MissingDependencyError: If MaterialsFramework is not installed.
        ConfigError: If ``options`` contains a key that ``Sqs2tdb`` does not accept.
        RuntimeError: If ``sqs2tdb`` finishes without writing a TDB.
    """
    require_materialsframework()
    from materialsframework.tools.sqs2tdb import Sqs2tdb

    unknown = set(options) - set(SQS2TDB_INIT_KEYS) - set(SQS2TDB_FIT_KEYS)
    if unknown:
        raise ConfigError(f"Unknown tdb_generation option(s): {sorted(unknown)}")

    symbols = [s.capitalize() for s in species]
    output = Path(output).resolve()
    work_dir = Path(work_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    if calculator is None:
        calculator = build_calculator(settings)

    init_kw = {k: options[k] for k in SQS2TDB_INIT_KEYS if k in options}
    fit_kw = {k: options[k] for k in SQS2TDB_FIT_KEYS if k in options}
    _logger.info("  sqs2tdb: %s  lattices=%s  level=%s", "-".join(symbols), fit_kw.get("lattices", "all"), fit_kw.get("level", 1))
    _logger.info("  Work dir: %s", work_dir)

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
    _logger.info("  Written: %s", output)
    return output
