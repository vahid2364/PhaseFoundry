"""HDF5 storage for thermodynamic tables.

A table file has this layout:

    /metadata             group whose attributes describe the run:
                          db_path, elements (comma-separated), phase,
                          temperature and temperature_units (K),
                          pressure and pressure_units (Pa), n_points_per_axis,
                          composition_ranges (JSON), pycalphad_version,
                          created_at (ISO 8601) and n_failed
    /compositions         float64, shape (n_valid, n_elements), mole fractions
    /GM                   float64, shape (n_valid,), J/mol
    /MU/{EL}              float64, shape (n_valid,), J/mol
    /dMU/{EL1}_{EL2}      float64, shape (n_valid,), dμ_EL1/dx_EL2 in J/mol

``GM``, ``MU`` and ``dMU`` are present only if they were computed. Every dataset is
gzip-compressed.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path

import h5py

try:
    import pycalphad

    _PYCALPHAD_VERSION = pycalphad.__version__
except Exception:
    _PYCALPHAD_VERSION = "unknown"

_logger = logging.getLogger(__name__)

_COMPRESS = {"compression": "gzip", "compression_opts": 4}
"""Compression settings for every dataset."""


def save_table(
    path: str | Path,
    data: dict,
    db_path: str,
    elements: list[str],
    phase: str,
    temperature: float,
    pressure: float,
    composition_ranges: dict[str, tuple[float, float]],
    n_points: int,
) -> None:
    """Write a thermodynamic table to an HDF5 file.

    Args:
        path: Output file path. An existing file is overwritten and missing parent
            directories are created.
        data: Output of ``compute_table``.
        db_path: TDB database the table was computed from.
        elements: Elements, in the column order of ``data["compositions"]``.
        phase: Phase name.
        temperature: Temperature in K.
        pressure: Pressure in Pa.
        composition_ranges: ``(min, max)`` mole fraction of each independent element used
            for the grid.
        n_points: Number of grid points along each independent axis.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(path, "w") as f:
        meta = f.create_group("metadata")
        meta.attrs["db_path"] = str(db_path)
        meta.attrs["elements"] = ",".join(elements)
        meta.attrs["phase"] = phase
        meta.attrs["temperature"] = temperature
        meta.attrs["temperature_units"] = "K"
        meta.attrs["pressure"] = pressure
        meta.attrs["pressure_units"] = "Pa"
        meta.attrs["n_points_per_axis"] = n_points
        meta.attrs["composition_ranges"] = json.dumps({k: list(v) for k, v in composition_ranges.items()})
        meta.attrs["pycalphad_version"] = _PYCALPHAD_VERSION
        meta.attrs["created_at"] = datetime.now(UTC).isoformat()
        meta.attrs["n_failed"] = data.get("n_failed", 0)

        f.create_dataset("compositions", data=data["compositions"], **_COMPRESS)
        f["compositions"].attrs["columns"] = ",".join(elements)
        f["compositions"].attrs["units"] = "mol fraction"

        if "GM" in data:
            f.create_dataset("GM", data=data["GM"], **_COMPRESS)
            f["GM"].attrs["units"] = "J/mol"

        if "MU" in data:
            mu_grp = f.create_group("MU")
            for el, arr in data["MU"].items():
                ds = mu_grp.create_dataset(el, data=arr, **_COMPRESS)
                ds.attrs["units"] = "J/mol"

        if "dMU" in data:
            dmu_grp = f.create_group("dMU")
            for key, arr in data["dMU"].items():
                ds = dmu_grp.create_dataset(key, data=arr, **_COMPRESS)
                ds.attrs["units"] = "J/mol"
                el1, el2 = key.split("_", 1)
                ds.attrs["description"] = f"d mu_{el1} / d x_{el2}  [J/mol per mol fraction]"

    _logger.info("Saved %s points to %s", len(data["compositions"]), path)


def load_table(path: str | Path) -> dict:
    """Load a thermodynamic table from HDF5.

    Args:
        path: Table file written by ``save_table``.

    Returns:
        A dict with ``"metadata"`` (the stored attributes, with ``composition_ranges``
        decoded from JSON), ``"elements"`` (list of element names in column order),
        ``"compositions"`` (shape ``(n_valid, n_elements)``) and, if stored, ``"GM"``
        (shape ``(n_valid,)``), ``"MU"`` and ``"dMU"`` (dicts of arrays of shape
        ``(n_valid,)``).

    Raises:
        FileNotFoundError: If ``path`` does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    out: dict = {}

    with h5py.File(path, "r") as f:
        meta = dict(f["metadata"].attrs)
        meta["composition_ranges"] = json.loads(meta["composition_ranges"])
        out["metadata"] = meta
        out["elements"] = meta["elements"].split(",")

        out["compositions"] = f["compositions"][:]

        if "GM" in f:
            out["GM"] = f["GM"][:]

        if "MU" in f:
            out["MU"] = {el: f["MU"][el][:] for el in f["MU"]}

        if "dMU" in f:
            out["dMU"] = {key: f["dMU"][key][:] for key in f["dMU"]}

    return out


def table_info(path: str | Path) -> str:
    """Summarize a stored table: phase, conditions, point counts and dataset ranges.

    Args:
        path: Table file written by ``save_table``.

    Returns:
        A multi-line summary.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
    """
    data = load_table(path)
    meta = data["metadata"]
    lines = [
        f"File    : {path}",
        f"Phase   : {meta['phase']}  ({meta['elements']})",
        f"T       : {meta['temperature']} K   P: {meta['pressure']} Pa",
        f"Points  : {len(data['compositions'])} valid  ({meta['n_failed']} failed)",
        f"Created : {meta['created_at']}",
        f"pycalphad: {meta['pycalphad_version']}",
        "Datasets:",
    ]
    if "GM" in data:
        lines.append(f"  GM    : shape {data['GM'].shape}  [{data['GM'].min():.1f}, {data['GM'].max():.1f}] J/mol")
    if "MU" in data:
        for el, arr in data["MU"].items():
            lines.append(f"  MU/{el} : shape {arr.shape}  [{arr.min():.1f}, {arr.max():.1f}] J/mol")
    if "dMU" in data:
        lines.append(f"  dMU   : {len(data['dMU'])} derivative arrays ({list(data['dMU'].keys())[:3]}...)")
    return "\n".join(lines)
