"""SQS elastic-constant table computed with an MLIP through MaterialsFramework.

Each composition on a regular simplex grid over the system elements goes through five
steps:

1. Build a BCC or FCC SQS supercell with MaterialsFramework's ``SqsGenerator``.
2. Relax the cell and the atomic positions with the MLIP.
3. Take the volume-averaged cubic lattice parameter, a_relaxed = (V / n_cells)^(1/3).
4. Get C11, C12, C44 and the Voigt-Reuss-Hill moduli from
   ``CubicElasticConstantsAnalyzer`` (uniform, orthorhombic and monoclinic energy-strain
   fits).
5. Compute the biaxial modulus M = E / (1 − ν).

Rows are appended to the CSV as each composition finishes, so an interrupted run picks
up where it stopped. A ``<table>.meta.yaml`` file records the calculator and SQS
settings, and a table built with different settings is never extended. The columns
match the MaterialsFramework tables in ``data/``, so
``phasefoundry.elasticity.sqs_interpolant`` reads either kind.
"""

from __future__ import annotations

import itertools
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from .settings import build_calculator, mlip_settings, require_materialsframework

PROPERTY_COLUMNS = [
    "a_relaxed_A",
    "C11_GPa",
    "C12_GPa",
    "C44_GPa",
    "youngs_modulus_GPa",
    "bulk_modulus_vrh_GPa",
    "shear_modulus_vrh_GPa",
    "poisson_ratio",
    "biaxial_modulus_M_GPa",
    "sqs_objective",
]
"""Property columns of the table, written after the ``x_<El>`` and ``m_<El>`` columns."""

ATOMS_PER_CELL = {"bcc": 2, "fcc": 4}
"""Atoms per conventional cubic cell, used to count the unit cells in a relaxed SQS."""


def simplex_compositions(n_elements: int, step: float) -> np.ndarray:
    """List the compositions on the simplex whose mole fractions are multiples of ``step``.

    The grid is built from integer partitions, so each row sums to exactly 1 and no
    points on the simplex boundary are lost to rounding.

    Args:
        n_elements: Number of elements.
        step: Grid spacing in mole fraction. ``1 / step`` must be an integer.

    Returns:
        Array of shape ``(n_points, n_elements)``. With ``step=0.05`` there are 21, 231
        and 1771 points for 2, 3 and 4 elements.

    Raises:
        ValueError: If ``step`` does not divide 1 evenly.
    """
    n = round(1.0 / step)
    if n < 1 or abs(n * step - 1.0) > 1e-9:
        raise ValueError(f"step must divide 1 evenly, got {step}")
    rows = [c + (n - sum(c),) for c in itertools.product(range(n + 1), repeat=n_elements - 1) if sum(c) <= n]
    return np.array(rows, dtype=float) / n


def _row_key(fractions: np.ndarray) -> tuple[int, ...]:
    """Round mole fractions to integer millionths so rows can be compared as set keys."""
    return tuple(int(round(v * 1e6)) for v in fractions)


def _meta_path(path: Path) -> Path:
    """Return the metadata file path that belongs to a table."""
    return path.with_suffix(".meta.yaml")


def _check_meta(path: Path, meta: dict) -> None:
    """Make sure an existing table was built with the same settings.

    Args:
        path: Table CSV path. Nothing is checked if the file does not exist.
        meta: Settings of the current run.

    Raises:
        ValueError: If the table exists without a metadata file, or its metadata differs
            from ``meta``.
    """
    if not path.exists():
        return
    mpath = _meta_path(path)
    if not mpath.exists():
        raise ValueError(
            f"{path} exists but has no {mpath.name}, so it was not built by "
            "PhaseFoundry and cannot be extended.  Read it with "
            "elastic.source: csv, or set elastic.sqs.table to a new path."
        )
    with mpath.open() as fh:
        old = yaml.safe_load(fh) or {}
    diff = sorted(k for k in meta if old.get(k) != meta[k])
    if diff:
        details = "\n".join(f"    {k}: cached={old.get(k)!r}  config={meta[k]!r}" for k in diff)
        raise ValueError(
            f"{path} was built with different settings:\n{details}\nDelete it, or set elastic.sqs.table to a new path."
        )


def _compute_point(
    symbols: list[str],
    fractions: np.ndarray,
    *,
    calculator: Any,
    sqs_generator: Any,
    analyzer: Any,
    crystal_structure: str,
    supercell: tuple[int, int, int],
) -> dict:
    """Generate, relax and analyse one SQS and return its table row.

    Returns:
        The ``x_<El>`` mole fractions, the ``m_<El>`` atom counts in the relaxed cell, and
        the ``PROPERTY_COLUMNS`` values.
    """
    from pymatgen.core import Composition

    present = {el: x for el, x in zip(symbols, fractions, strict=True) if x > 0}
    sqs = sqs_generator.generate(Composition(present), crystal_structure=crystal_structure, supercell_size=supercell)
    relaxed = calculator.relax(sqs["structure"])["final_structure"]
    n_cells = len(relaxed) / ATOMS_PER_CELL[crystal_structure]
    a_relaxed = (relaxed.volume / n_cells) ** (1.0 / 3.0)

    props = analyzer.calculate(relaxed, is_relaxed=True)
    E, nu = props["youngs_modulus"], props["poisson_ratio"]
    counts = relaxed.composition.get_el_amt_dict()

    row = {f"x_{el}": float(x) for el, x in zip(symbols, fractions, strict=True)}
    row.update({f"m_{el}": int(round(counts.get(el, 0))) for el in symbols})
    row.update(
        {
            "a_relaxed_A": a_relaxed,
            "C11_GPa": props["C11"],
            "C12_GPa": props["C12"],
            "C44_GPa": props["C44"],
            "youngs_modulus_GPa": E,
            "bulk_modulus_vrh_GPa": props["voigt_reuss_hill_bulk_modulus"],
            "shear_modulus_vrh_GPa": props["voigt_reuss_hill_shear_modulus"],
            "poisson_ratio": nu,
            "biaxial_modulus_M_GPa": E / (1.0 - nu),
            "sqs_objective": float(sqs["objective"]),
        }
    )
    return row


def build_elastic_table(
    elements: list[str],
    path: str | Path,
    settings: dict,
    *,
    step: float = 0.05,
    crystal_structure: str = "bcc",
    supercell: tuple[int, int, int] = (4, 4, 4),
    sqs_iterations: int | None = None,
    eos: str = "birch_murnaghan",
    delta_max: float = 0.05,
    strain_step: float = 0.01,
    calculator: Any = None,
) -> pd.DataFrame:
    """Compute the SQS elastic-constant table at ``path``, or fill in its missing rows.

    A composition that fails is skipped with a warning, and running again retries it.

    Args:
        elements: System elements, in any case, e.g. ``["NB", "HF", "V"]``.
        path: Table CSV. Existing rows are kept and only missing grid points are computed.
        settings: Output of ``mlip_settings``, stored in the metadata file.
        step: Grid spacing in mole fraction.
        crystal_structure: ``"bcc"`` or ``"fcc"``. Only cubic elastic constants are supported.
        supercell: SQS supercell size in repeats of the conventional cell.
        sqs_iterations: Number of SQS search iterations. ``None`` uses the MaterialsFramework
            default.
        eos: Equation of state that ``CubicElasticConstantsAnalyzer`` fits.
        delta_max: Largest deformation that ``CubicElasticConstantsAnalyzer`` applies.
        strain_step: Deformation step of ``CubicElasticConstantsAnalyzer``.
        calculator: Calculator to use. Defaults to one built from ``settings``.

    Returns:
        Every row of the table, including rows computed in earlier runs.

    Raises:
        ImportError: If MaterialsFramework is not installed.
        ValueError: If ``crystal_structure`` is not cubic, or the existing table was built
            with different settings.
        RuntimeError: If no composition could be computed.
    """
    require_materialsframework()
    from materialsframework.analysis.cubic_elastic_constants import CubicElasticConstantsAnalyzer
    from materialsframework.tools.sqsgen import SqsGenerator
    from tqdm import tqdm

    crystal_structure = crystal_structure.lower()
    if crystal_structure not in ATOMS_PER_CELL:
        raise ValueError(
            f"crystal_structure must be one of {sorted(ATOMS_PER_CELL)} (cubic elastic constants), got {crystal_structure!r}"
        )

    path = Path(path)
    symbols = [e.capitalize() for e in elements]
    supercell = (int(supercell[0]), int(supercell[1]), int(supercell[2]))
    meta = {
        "elements": symbols,
        "calculator": settings,
        "crystal_structure": crystal_structure,
        "supercell": list(supercell),
        "sqs_iterations": sqs_iterations,
        "eos": eos,
        "delta_max": delta_max,
        "strain_step": strain_step,
    }
    _check_meta(path, meta)

    grid = simplex_compositions(len(symbols), step)
    done: set[tuple[int, ...]] = set()
    if path.exists():
        existing = pd.read_csv(path)
        done = {_row_key(r) for r in existing[[f"x_{el}" for el in symbols]].values}
    todo = [x for x in grid if _row_key(x) not in done]

    print(f"  SQS elastic table: {path}")
    print(f"    {len(grid)} grid points (step {step}), {len(grid) - len(todo)} cached, {len(todo)} to compute")
    if not todo:
        return pd.read_csv(path)

    path.parent.mkdir(parents=True, exist_ok=True)
    with _meta_path(path).open("w") as fh:
        yaml.safe_dump(meta, fh, sort_keys=False)

    if calculator is None:
        calculator = build_calculator(settings)
    sqs_generator = SqsGenerator() if sqs_iterations is None else SqsGenerator(iterations=int(sqs_iterations))
    analyzer = CubicElasticConstantsAnalyzer(
        eos_name=eos,  # ty: ignore[invalid-argument-type]
        delta_max=delta_max,
        step_size=strain_step,
        calculator=calculator,
    )

    failed = []
    for x in tqdm(todo, desc="  SQS + elastic", unit="comp"):
        try:
            row = _compute_point(
                symbols,
                x,
                calculator=calculator,
                sqs_generator=sqs_generator,
                analyzer=analyzer,
                crystal_structure=crystal_structure,
                supercell=supercell,
            )
        except Exception as exc:  # noqa: BLE001 — one bad composition must not stop the sweep
            failed.append(x)
            warnings.warn(f"Composition {dict(zip(symbols, x.round(4), strict=True))} failed: {exc}", stacklevel=1)
            continue
        pd.DataFrame([row]).to_csv(path, mode="a", header=not path.exists(), index=False)

    if failed:
        print(f"    {len(failed)} composition(s) failed; re-run step 1b to retry them.")
    if not path.exists():
        raise RuntimeError(f"No compositions could be computed for {path}")
    return pd.read_csv(path)


def ensure_elastic_table(cfg: dict, root: Path, calculator: Any = None) -> Path:
    """Build or complete the table configured in ``simulation.elastic.sqs``.

    Args:
        cfg: Nested config dict.
        root: Repository root used to resolve relative paths.
        calculator: Calculator to use. Defaults to one built from the config.

    Returns:
        Path to the table CSV.
    """
    from phasefoundry.elasticity.sources import elastic_table_path

    sqs_cfg = cfg["simulation"]["elastic"].get("sqs", {}) or {}
    path = elastic_table_path(cfg, root)
    build_elastic_table(
        elements=cfg["system"]["elements"],
        path=path,
        settings=mlip_settings(cfg, sqs_cfg),
        step=float(sqs_cfg.get("step", 0.05)),
        crystal_structure=sqs_cfg.get("crystal_structure", "bcc"),
        supercell=tuple(sqs_cfg.get("supercell", (4, 4, 4))),
        sqs_iterations=sqs_cfg.get("sqs_iterations"),
        eos=sqs_cfg.get("eos", "birch_murnaghan"),
        delta_max=float(sqs_cfg.get("delta_max", 0.05)),
        strain_step=float(sqs_cfg.get("strain_step", 0.01)),
        calculator=calculator,
    )
    return path
