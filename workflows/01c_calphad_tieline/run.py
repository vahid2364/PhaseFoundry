"""
Compute CALPHAD two-phase tie-line endpoints for a given composition
using pycalphad equilibrium.  Outputs a YAML file with the equilibrium
phase compositions that can be used as c_mat / c_ppt in the elastic config.

Usage:
    python workflows/01c_calphad_tieline/run.py \
        --config configs/nb_hf_ti_v_elastic_cNb25V25Hf25Ti25.yaml
"""

import argparse
import pathlib
import sys
import warnings

import numpy as np
import yaml

try:
    import pycalphad as pcp
    from pycalphad import equilibrium, variables as v
    from pycalphad.core.utils import filter_phases
except ImportError as e:
    sys.exit(f"pycalphad not found: {e}")


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def run_tieline(config_path: str) -> None:
    cfg = load_config(config_path)
    sys_cfg = cfg["system"]
    sim_cfg = cfg["simulation"]

    db_path = pathlib.Path(sys_cfg["db_path"])
    if not db_path.is_absolute():
        db_path = pathlib.Path(config_path).parent.parent / db_path
    db_path = db_path.resolve()

    elements = sys_cfg["elements"]           # e.g. [NB, HF, TI, V]
    ref_el = sys_cfg["reference_element"]    # e.g. V
    phase = sys_cfg["phase"]                 # e.g. BCC_A2
    temperature_C = sim_cfg["temperature_C"]
    temperature_K = temperature_C + 273.15
    c0 = sim_cfg["c0"]                       # independent fractions [x_NB, x_HF, x_TI]

    ind_els = [el for el in elements if el != ref_el]
    if len(ind_els) != len(c0):
        sys.exit(f"c0 length {len(c0)} != number of independent elements {len(ind_els)}")

    print(f"System   : {' - '.join(elements)}")
    print(f"Phase    : {phase}")
    print(f"TDB      : {db_path}")
    print(f"T        : {temperature_C} °C ({temperature_K} K)")
    print(f"c0       : {dict(zip(ind_els, c0))}")
    print()

    db = pcp.Database(str(db_path))

    conds = {v.T: temperature_K, v.P: 101325, v.N: 1}
    for el, xi in zip(ind_els, c0):
        conds[v.X(el)] = float(xi)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        eq = equilibrium(db, elements + ["VA"], [phase], conds)

    phase_frac = eq.NP.squeeze().values
    phase_names = eq.Phase.values.flatten()
    x_vals = eq.X.squeeze()   # shape: (n_phases, n_elements)

    print("=== Raw equilibrium output ===")
    for i, pn in enumerate(phase_names):
        if not pn:
            continue
        frac = float(phase_frac[i])
        print(f"  {pn}  fraction={frac:.4f}")

    # Find the active BCC_A2 vertices (two-phase region → two entries)
    active = []
    for i, pn in enumerate(phase_names):
        if pn and phase in pn and float(phase_frac[i]) > 1e-6:
            comp = {}
            for j, el in enumerate(x_vals.component.values):
                comp[el] = float(x_vals[i, j].values)
            active.append({"phase": pn, "fraction": float(phase_frac[i]), "X": comp})

    if len(active) == 0:
        sys.exit("No active BCC_A2 phase found in equilibrium output.")
    elif len(active) == 1:
        print("\nWARNING: single-phase equilibrium — alloy is inside the stable region.")
        print("Tip: check if the temperature is above the miscibility gap.")
        # Still write the single endpoint so the file exists
        endpoints = active
    else:
        print(f"\n{len(active)} BCC_A2 phases found (two-phase tie-line).")
        endpoints = active

    # Label matrix (V-poor) and precipitate (V-rich)
    if len(endpoints) == 2:
        endpoints.sort(key=lambda e: e["X"].get(ref_el, 0.0))
        label = ["mat", "ppt"]
    else:
        label = [f"phase_{i}" for i in range(len(endpoints))]

    print("\n=== Tie-line endpoints ===")
    result = {"temperature_C": temperature_C, "phase": phase, "reference_element": ref_el,
              "tie_line": {}}
    for lbl, ep in zip(label, endpoints):
        comp_ind = {el: round(ep["X"].get(el, 0.0), 6) for el in elements}
        result["tie_line"][lbl] = {
            "label": lbl,
            "phase_fraction": round(ep["fraction"], 6),
            "composition": comp_ind,
        }
        print(f"  {lbl}: " + "  ".join(f"x_{el}={comp_ind[el]:.4f}" for el in elements))

    results_dir = pathlib.Path(cfg["output"]["results_dir"])
    if not results_dir.is_absolute():
        results_dir = pathlib.Path(config_path).parent.parent / results_dir
    results_dir = results_dir.resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    out_path = results_dir / "tieline_calphad.yaml"
    with open(out_path, "w") as f:
        yaml.dump(result, f, default_flow_style=False, sort_keys=False)

    print(f"\nSaved → {out_path}")
    print()
    print("=== Suggested elastic config update ===")
    if "mat" in result["tie_line"] and "ppt" in result["tie_line"]:
        mat_c = result["tie_line"]["mat"]["composition"]
        ppt_c = result["tie_line"]["ppt"]["composition"]
        print(f"  c_mat: {{{', '.join(f'{el}: {mat_c[el]:.4f}' for el in elements)}}}")
        print(f"  c_ppt: {{{', '.join(f'{el}: {ppt_c[el]:.4f}' for el in elements)}}}")


def main():
    parser = argparse.ArgumentParser(description="CALPHAD tie-line endpoint calculator")
    parser.add_argument("--config", required=True, help="Path to simulation config YAML")
    args = parser.parse_args()
    run_tieline(args.config)


if __name__ == "__main__":
    main()
