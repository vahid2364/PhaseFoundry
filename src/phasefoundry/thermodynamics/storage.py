"""HDF5 storage for thermodynamic tables.

File layout
-----------
/metadata/
    db_path            str
    elements           str  (comma-separated)
    phase              str
    temperature        float
    pressure           float
    n_points_per_axis  int
    pycalphad_version  str
    created_at         str  (ISO 8601)

/compositions          float64 (n_valid, N_elements)
/GM                    float64 (n_valid,)             [J/mol]
/MU/
    {EL}               float64 (n_valid,)             [J/mol]
/dMU/
    {EL1}_{EL2}        float64 (n_valid,)             [J/mol]
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import h5py

try:
    import pycalphad

    _PYCALPHAD_VERSION = pycalphad.__version__
except Exception:
    _PYCALPHAD_VERSION = "unknown"

_COMPRESS = {"compression": "gzip", "compression_opts": 4}


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
    """Write a thermodynamic data table to an HDF5 file.

    Parameters
    ----------
    path : str or Path
        Output file path (created or overwritten).
    data : dict
        Output of ``generate.compute_table()``.
    db_path, elements, phase, temperature, pressure : ...
        Parameters used to produce the data (stored as metadata).
    composition_ranges : dict
        Per-element (min, max) ranges used for the grid.
    n_points : int
        Number of grid points per element axis.
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

    print(f"Saved {len(data['compositions'])} points to {path}")


def load_table(path: str | Path) -> dict:
    """Load a thermodynamic table from HDF5.

    Returns:
    -------
    dict with keys:
        'metadata'      dict of stored attributes
        'compositions'  np.ndarray (n_valid, N_elements)
        'elements'      list[str]
        'GM'            np.ndarray (n_valid,)           if present
        'MU'            dict[str, np.ndarray(n_valid,)] if present
        'dMU'           dict[str, np.ndarray(n_valid,)] if present
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
    """Return a human-readable summary of a stored table."""
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
