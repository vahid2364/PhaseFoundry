"""
Step 1e — Isotropic coherent-Hessian stability sweep
======================================================
Computes chemical and coherent (isotropic Cahn 1961) Hessian eigenvalues
over a temperature sweep, for one or more compositions, and plots them.

This is the config-driven, reproducible replacement for the notebook
calculation that generated the submitted manuscript's Figure 4
(``notebooks/stability_analysis_nonisotropic.ipynb``): that notebook queried
a TDB and an elastic-constants CSV that are not tracked in this repository,
with eta/M/lattice-parameter values hand-transcribed into the notebook
rather than read from any file here. This workflow instead reads everything
from the config, through ``phasefoundry.thermodynamics.coherent_hessian``.

See that module's docstring for the isotropic-screening-only scope of this
calculation, and its documented difference from the full anisotropic,
composition-dependent elasticity actually solved by the elasto-chemical
phase-field coupling (``phasefoundry.elasticity``).

Usage
-----
    python workflows/01e_coherent_hessian/run.py \\
        --config configs/hf_nb_ti_v_coherent_hessian.yaml

    # Override on the command line
    python workflows/01e_coherent_hessian/run.py \\
        --config configs/hf_nb_ti_v_coherent_hessian.yaml \\
        --set temperature.step_K=25
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from phasefoundry.config import load_config, apply_overrides, resolve_paths, summary
from phasefoundry.thermodynamics.coherent_hessian import (
    chemical_hessian,
    isotropic_coherent_correction,
    hessian_eigenvalues,
    KADIRVEL_HF_NB_TI_O,
)
from phasefoundry.elasticity.sqs_interpolant import (
    load_and_filter, build_interpolators, query_composition,
    vegard_eta_coefficients,
)

_PROJECTIONS = {
    "kadirvel_hf_nb_ti": KADIRVEL_HF_NB_TI_O,
    "none": None,
}


def _resolve_projection(name: str) -> np.ndarray | None:
    key = (name or "none").strip().lower()
    if key not in _PROJECTIONS:
        raise ValueError(
            f"Unknown gauge_invariant_projection {name!r}. "
            f"Choose one of {sorted(_PROJECTIONS)}."
        )
    return _PROJECTIONS[key]


def run(cfg: dict) -> Path:
    sys_cfg  = cfg["system"]
    el_cfg   = cfg["elastic"]
    T_cfg    = cfg["temperature"]
    comps    = cfg["compositions"]

    elements          = [e.upper() for e in sys_cfg["elements"]]
    reference_element = sys_cfg["reference_element"].upper()
    phase             = sys_cfg["phase"]
    db_path           = sys_cfg["db_path"]
    independent_elements = [e for e in elements if e != reference_element]

    csv_rel = el_cfg["elastic_csv"]
    elastic_csv_path = _ROOT / csv_rel if not Path(csv_rel).is_absolute() else Path(csv_rel)
    if not elastic_csv_path.exists():
        raise FileNotFoundError(f"Elastic-constants CSV not found: {elastic_csv_path}")

    O = _resolve_projection(cfg.get("gauge_invariant_projection", "none"))
    delta_fd       = float(cfg.get("delta_fd", 0.02))
    atoms_per_cell = float(cfg.get("atoms_per_cell", 2.0))

    T_values = np.arange(T_cfg["start_K"], T_cfg["stop_K"], T_cfg["step_K"], dtype=float)

    print(f"\n  System:   {' - '.join(elements)}  (ref: {reference_element})")
    print(f"  TDB:      {db_path}")
    print(f"  Elastic:  {elastic_csv_path.name}")
    print(f"  T sweep:  {T_values[0]:.0f}-{T_values[-1]:.0f} K, step {T_cfg['step_K']:.0f} K "
          f"({len(T_values)} points)")
    print(f"  Projection: {cfg.get('gauge_invariant_projection', 'none')}")
    print(f"  Compositions: {len(comps)}")

    # ── per-composition elastic lookup (T-independent: done once per composition) ──
    df = load_and_filter(elastic_csv_path, elements)
    interps = build_interpolators(df, independent_elements)

    results_dir = Path(cfg["output"]["results_dir"])
    run_name    = cfg["output"]["run_name"]
    out_dir     = results_dir / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    provenance = {
        "db_path": str(db_path),
        "elastic_csv": str(elastic_csv_path),
        "reference_element": reference_element,
        "independent_elements": independent_elements,
        "gauge_invariant_projection": cfg.get("gauge_invariant_projection", "none"),
        "compositions": {},
    }

    n_comp = len(comps)
    fig, axes = plt.subplots(1, n_comp, figsize=(5.2 * n_comp, 4.2), squeeze=False)
    axes = axes[0]

    for ax, entry in zip(axes, comps):
        label = entry["label"]
        comp  = {k.upper(): float(v) for k, v in entry["composition"].items()}
        print(f"\n  --- {label}: {comp} ---")

        props = query_composition(interps, comp, independent_elements)
        eta   = vegard_eta_coefficients(interps, comp, independent_elements, delta_fd=delta_fd)
        M_GPa = props["biaxial_modulus_M_GPa"]
        a_ref = props["a_relaxed_A"]
        print(f"    a_ref = {a_ref:.4f} A   M = {M_GPa:.4f} GPa")
        for el in independent_elements:
            print(f"    eta_{el} = {eta[el]:.6f}")

        correction = isotropic_coherent_correction(
            eta, M_GPa * 1e9, a_ref,
            independent_elements=independent_elements,
            atoms_per_cell=atoms_per_cell,
        )

        n_ind = len(independent_elements)
        chem_eigs = np.full((len(T_values), n_ind), np.nan)
        coh_eigs  = np.full((len(T_values), n_ind), np.nan)

        for i, T in enumerate(T_values):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    H_chem = chemical_hessian(
                        db_path, elements, phase, comp, float(T), reference_element,
                    )
            except Exception as exc:
                print(f"    [skip] T={T:.0f} K: {exc}")
                continue
            H_coh = H_chem + correction
            chem_eigs[i] = hessian_eigenvalues(H_chem, O)
            coh_eigs[i]  = hessian_eigenvalues(H_coh, O)

        out = pd.DataFrame({"temperature_K": T_values})
        for k in range(n_ind):
            out[f"chem_eig_{k + 1}"] = chem_eigs[:, k]
            out[f"coh_eig_{k + 1}"]  = coh_eigs[:, k]
        csv_path = out_dir / f"eigenvalues_{label}.csv"
        out.to_csv(csv_path, index=False)
        print(f"    Written: {csv_path}")

        provenance["compositions"][label] = {
            "composition": comp,
            "a_ref_A": round(float(a_ref), 6),
            "M_GPa": round(float(M_GPa), 6),
            "eta": {el: round(float(v), 8) for el, v in eta.items()},
        }

        ax.plot(T_values, chem_eigs[:, 0], linewidth=3, label=r"Chem $\lambda_1$")
        ax.plot(T_values, coh_eigs[:, 0], linewidth=3, label=r"Coherent $\lambda_1$")
        ax.axhline(0.0, color="black", linestyle="--", linewidth=1)
        ax.set_xlabel("Temperature (K)")
        ax.set_ylabel("Eigenvalues (J/mol)")
        ax.set_title(label)
        ax.legend()

    fig.tight_layout()
    fig_path_png = out_dir / "fig_04_eigenvalues.png"
    fig_path_pdf = out_dir / "fig_04_eigenvalues.pdf"
    fig.savefig(fig_path_png, dpi=200, bbox_inches="tight")
    fig.savefig(fig_path_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Written: {fig_path_png}")
    print(f"  Written: {fig_path_pdf}")

    provenance_path = out_dir / "provenance.yaml"
    with open(provenance_path, "w") as f:
        yaml.dump(provenance, f, default_flow_style=False, sort_keys=False)
    print(f"  Written: {provenance_path}")

    return out_dir


def main():
    parser = argparse.ArgumentParser(
        description="Step 1e — Isotropic coherent-Hessian stability sweep"
    )
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                        help="Override config values, e.g. temperature.step_K=25")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.set:
        cfg = apply_overrides(cfg, args.set)
    cfg = resolve_paths(cfg, _ROOT)

    print("=" * 60)
    print("Step 1e — Coherent-Hessian Stability Sweep")
    print(summary(cfg))
    print("=" * 60)

    run(cfg)


if __name__ == "__main__":
    main()
