"""
Step 1b — Lookup elastic constants from SQS/DFT CSV
=====================================================
Reads the SQS/DFT elastic constants CSV, filters to the active element
subspace, interpolates C11/C12/C44 and lattice parameter at the requested
matrix and precipitate compositions, validates Born stability criteria, and
writes ``elastic_params.yaml`` to the output directory.

Supports any system dimensionality:
    Binary   (e.g. NbV)    — 1-D interpolation in x_Nb
    Ternary  (e.g. NbHfV)  — 2-D interpolation in (x_Nb, x_Hf)
    Quaternary (NbHfVTi)   — 3-D interpolation in (x_Nb, x_Hf, x_Ti)

The written ``elastic_params.yaml`` is consumed automatically by
``workflows/03_simulate/run.py`` when ``elastic.stiffness_model: cubic_csv``.

Output
------
<output.results_dir>/elastic_params.yaml

Usage
-----
    python workflows/01b_lookup_elastic/run.py --config configs/nb_v_elastic.yaml
    python workflows/01b_lookup_elastic/run.py --config configs/nb_hf_v.yaml

    # Override endpoints on command line
    python workflows/01b_lookup_elastic/run.py \\
        --config configs/nb_v_elastic.yaml \\
        --set elastic.c_mat.NB=0.95 elastic.c_ppt.NB=0.15
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from calphad_pf.config import load_config, apply_overrides, resolve_paths, summary


# ── Born stability criteria (cubic crystal) ───────────────────────────────────

def born_check(C11: float, C12: float, C44: float) -> dict:
    return {
        "C44_positive":   bool(C44 > 0),
        "C11_gt_C12":     bool(C11 > C12),
        "bulk_positive":  bool(C11 + 2 * C12 > 0),
        "all_stable":     bool(C44 > 0 and C11 > C12 and C11 + 2 * C12 > 0),
    }


def born_summary(label: str, C11: float, C12: float, C44: float) -> str:
    chk = born_check(C11, C12, C44)
    lines = [f"  {label}:  C11={C11:.1f}  C12={C12:.1f}  C44={C44:.1f} GPa"]
    for crit, ok in chk.items():
        mark = "OK  " if ok else "FAIL"
        lines.append(f"    [{mark}] {crit}")
    return "\n".join(lines)


# ── CSV loader & N-D interpolator ─────────────────────────────────────────────

def _element_col(el: str) -> str:
    """Map element name (case-insensitive) to CSV column name x_Xx."""
    return "x_" + el.capitalize()


def load_and_filter(csv_path: Path, elements: list[str]) -> pd.DataFrame:
    """
    Load CSV and return rows belonging to the active element subspace.

    For a binary NbV system (elements=[NB, V]), rows where x_Hf=0 and
    x_Ti=0 are returned.  For ternary NbHfV, rows where x_Ti=0 are
    returned.  The constraint is: all CSV element columns NOT in `elements`
    must be zero (within 1e-6).
    """
    df = pd.read_csv(csv_path)
    all_csv_els = [c[2:].upper() for c in df.columns if c.startswith("x_")]
    active = {e.upper() for e in elements}
    inactive = [e for e in all_csv_els if e not in active]

    mask = np.ones(len(df), dtype=bool)
    for el in inactive:
        col = _element_col(el)
        if col in df.columns:
            mask &= df[col].abs() < 1e-6

    sub = df[mask].copy().reset_index(drop=True)
    if len(sub) == 0:
        raise ValueError(
            f"No CSV rows found for element subspace {sorted(active)}. "
            f"Check that the CSV contains data for this system."
        )
    return sub


def build_interpolators(df: pd.DataFrame, ind_elements: list[str]):
    """
    Build interpolators for each elastic property.

    1-D (binary)    → scipy.interpolate.interp1d (linear, clamp extrapolation)
    N-D (N >= 2)    → scipy.interpolate.LinearNDInterpolator + NearestND fallback

    Parameters
    ----------
    df : filtered DataFrame (rows belonging to the active subspace)
    ind_elements : independent elements (all elements except reference).
        For NbV with ref=V: ['NB'].
        For NbHfV with ref=V: ['NB', 'HF'].

    Returns
    -------
    dict mapping property name → callable interp(x_ind) → float
    """
    from scipy.interpolate import (interp1d, LinearNDInterpolator,
                                   NearestNDInterpolator)

    props = ["C11_GPa", "C12_GPa", "C44_GPa", "a_relaxed_A",
             "shear_modulus_vrh_GPa", "bulk_modulus_vrh_GPa", "poisson_ratio"]

    coords = np.column_stack([
        df[_element_col(el)].values for el in ind_elements
    ])   # shape (n_pts, n_ind)

    interps = {}

    if len(ind_elements) == 1:
        # 1-D binary: use interp1d along the single independent axis
        x1d = coords[:, 0]
        order = np.argsort(x1d)
        x1d_s = x1d[order]
        for prop in props:
            vals_s = df[prop].values[order]
            f = interp1d(x1d_s, vals_s, kind="linear", bounds_error=False,
                         fill_value=(vals_s[0], vals_s[-1]))  # clamp at edges
            def make_fn_1d(fn):
                def interp_fn(x):
                    return float(fn(np.atleast_1d(x)[0]))
                return interp_fn
            interps[prop] = make_fn_1d(f)
    else:
        # N-D (N >= 2): LinearND inside hull, NearestND outside
        for prop in props:
            vals = df[prop].values
            lin  = LinearNDInterpolator(coords, vals)
            near = NearestNDInterpolator(coords, vals)
            def make_fn_nd(l, n):
                def fn(x):
                    x = np.atleast_1d(np.asarray(x, dtype=float))
                    v = l(x)
                    if np.isnan(v):
                        warnings.warn(
                            f"Composition {x} outside convex hull — "
                            "falling back to nearest-neighbour.",
                            stacklevel=3,
                        )
                        v = n(x)
                    return float(v)
                return fn
            interps[prop] = make_fn_nd(lin, near)

    return interps


def query_composition(interps: dict, comp: dict,
                      ind_elements: list[str]) -> dict:
    """
    Query all elastic properties at a given composition.

    Parameters
    ----------
    comp : dict mapping element name (upper) → mole fraction
    ind_elements : independent elements in the same order used for interpolation
    """
    x_ind = np.array([comp.get(el.upper(), 0.0) for el in ind_elements],
                     dtype=float)
    return {prop: fn(x_ind) for prop, fn in interps.items()}


# ── main ──────────────────────────────────────────────────────────────────────

def run(cfg: dict) -> Path:
    """
    Perform elastic constant lookup and write elastic_params.yaml.
    Returns the path to the written file.
    """
    el_cfg      = cfg["simulation"].get("elastic", {})
    elements    = [e.upper() for e in cfg["system"]["elements"]]
    ref_el      = cfg["system"]["reference_element"].upper()
    ind_elements = [e for e in elements if e != ref_el]

    # ── validate config ───────────────────────────────────────────────────────
    if el_cfg.get("stiffness_model", "isotropic") != "cubic_csv":
        raise ValueError(
            "elastic.stiffness_model must be 'cubic_csv' to use this step. "
            f"Got: {el_cfg.get('stiffness_model')!r}"
        )

    csv_path = Path(cfg["elastic_csv_path"])
    if not csv_path.exists():
        raise FileNotFoundError(f"Elastic CSV not found: {csv_path}")

    # ── parse endpoint compositions ───────────────────────────────────────────
    def parse_comp(key: str) -> dict:
        """Parse c_mat or c_ppt from config as {ELEMENT: fraction}."""
        raw = el_cfg.get(key, {})
        if not raw:
            raise ValueError(
                f"elastic.{key} must be specified as a composition dict "
                f"when stiffness_model=cubic_csv. "
                f"Example: {{NB: 1.0, V: 0.0}}"
            )
        return {k.upper(): float(v) for k, v in raw.items()}

    comp_mat = parse_comp("c_mat")
    comp_ppt = parse_comp("c_ppt")

    print(f"\n  System:   {' – '.join(elements)}  (ref: {ref_el})")
    print(f"  CSV:      {csv_path.name}  ({len(pd.read_csv(csv_path))} rows total)")
    print(f"  c_mat:    {comp_mat}")
    print(f"  c_ppt:    {comp_ppt}")

    # ── load & filter CSV ─────────────────────────────────────────────────────
    df = load_and_filter(csv_path, elements)
    print(f"  Subspace: {len(df)} rows after filtering to {sorted(elements)}")

    # ── build N-D interpolators ───────────────────────────────────────────────
    print(f"  Interpolation: {len(ind_elements)}-D "
          f"({'×'.join(ind_elements)} space, LinearND + NearestND fallback)")
    interps = build_interpolators(df, ind_elements)

    # ── query at endpoints ────────────────────────────────────────────────────
    props_mat = query_composition(interps, comp_mat, ind_elements)
    props_ppt = query_composition(interps, comp_ppt, ind_elements)

    C11_mat, C12_mat, C44_mat = props_mat["C11_GPa"], props_mat["C12_GPa"], props_mat["C44_GPa"]
    C11_ppt, C12_ppt, C44_ppt = props_ppt["C11_GPa"], props_ppt["C12_GPa"], props_ppt["C44_GPa"]
    a_mat   = props_mat["a_relaxed_A"]
    a_ppt   = props_ppt["a_relaxed_A"]
    eps_T   = (a_ppt - a_mat) / a_mat

    # Zener anisotropy
    A_Z_mat = 2 * C44_mat / (C11_mat - C12_mat) if (C11_mat - C12_mat) != 0 else float("nan")
    A_Z_ppt = 2 * C44_ppt / (C11_ppt - C12_ppt) if (C11_ppt - C12_ppt) != 0 else float("nan")

    # ── Born stability check ──────────────────────────────────────────────────
    print("\n  Born stability check:")
    print(born_summary("C_mat", C11_mat, C12_mat, C44_mat))
    print(born_summary("C_ppt", C11_ppt, C12_ppt, C44_ppt))

    chk_mat = born_check(C11_mat, C12_mat, C44_mat)
    chk_ppt = born_check(C11_ppt, C12_ppt, C44_ppt)
    if not chk_mat["all_stable"]:
        warnings.warn("C_mat fails Born stability — elastic solver may diverge.", stacklevel=1)
    if not chk_ppt["all_stable"]:
        warnings.warn("C_ppt fails Born stability — elastic solver may diverge. "
                      "Consider increasing x_Nb in c_ppt.", stacklevel=1)

    # ── Vegard eigenstrain summary ────────────────────────────────────────────
    print(f"\n  Vegard eigenstrain:")
    print(f"    a_mat = {a_mat:.4f} Å  |  a_ppt = {a_ppt:.4f} Å")
    print(f"    eps_T = (a_ppt - a_mat) / a_mat = {eps_T:.5f} ({eps_T*100:.2f}%)")
    print(f"\n  Zener anisotropy:")
    print(f"    A_Z_mat = {A_Z_mat:.3f}  |  A_Z_ppt = {A_Z_ppt:.3f}")
    if A_Z_mat < 1:
        print(f"    A_Z < 1 → diamond morphology expected (⟨100⟩ corners)")
    else:
        print(f"    A_Z > 1 → square morphology expected (⟨100⟩ faces)")

    # ── assemble output dict ──────────────────────────────────────────────────
    out = {
        "meta": {
            "generated_by": "workflows/01b_lookup_elastic/run.py",
            "csv_source":   str(csv_path),
            "system":       elements,
            "reference_element": ref_el,
            "interpolation_dim": len(ind_elements),
        },
        "C_mat": {
            "composition": comp_mat,
            "C11_GPa":     round(C11_mat, 4),
            "C12_GPa":     round(C12_mat, 4),
            "C44_GPa":     round(C44_mat, 4),
            "a_relaxed_A": round(a_mat,   6),
            "G_VRH_GPa":   round(props_mat["shear_modulus_vrh_GPa"], 4),
            "B_VRH_GPa":   round(props_mat["bulk_modulus_vrh_GPa"],  4),
            "nu":          round(props_mat["poisson_ratio"],         6),
            "A_Z":         round(A_Z_mat, 4),
            "born_check":  chk_mat,
        },
        "C_ppt": {
            "composition": comp_ppt,
            "C11_GPa":     round(C11_ppt, 4),
            "C12_GPa":     round(C12_ppt, 4),
            "C44_GPa":     round(C44_ppt, 4),
            "a_relaxed_A": round(a_ppt,   6),
            "G_VRH_GPa":   round(props_ppt["shear_modulus_vrh_GPa"], 4),
            "B_VRH_GPa":   round(props_ppt["bulk_modulus_vrh_GPa"],  4),
            "nu":          round(props_ppt["poisson_ratio"],         6),
            "A_Z":         round(A_Z_ppt, 4),
            "born_check":  chk_ppt,
        },
        "eigenstrain": {
            "model":       "vegard",
            "a_matrix_A":  round(a_mat,  6),
            "a_ppt_A":     round(a_ppt,  6),
            "eps_T":       round(eps_T,  8),
            "eps_T_pct":   round(eps_T * 100, 4),
        },
    }

    # ── write elastic_params.yaml ─────────────────────────────────────────────
    results_dir = Path(cfg["output"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / "elastic_params.yaml"
    with open(out_path, "w") as f:
        yaml.dump(out, f, default_flow_style=False, sort_keys=False)

    print(f"\n  Written: {out_path}")
    return out_path


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Step 1b — Lookup elastic constants from SQS/DFT CSV"
    )
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                        help="Override config values, e.g. elastic.c_ppt.NB=0.15")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.set:
        cfg = apply_overrides(cfg, args.set)
    cfg = resolve_paths(cfg, _ROOT)

    # resolve elastic_csv path
    el_cfg = cfg["simulation"].get("elastic", {})
    csv_rel = el_cfg.get("elastic_csv", "")
    cfg["elastic_csv_path"] = str(_ROOT / csv_rel) if csv_rel else ""

    print("=" * 60)
    print("Step 1b — Elastic Constants Lookup")
    print(summary(cfg))
    print("=" * 60)

    run(cfg)
