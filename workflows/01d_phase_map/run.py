"""
Step 1d — Phase map (thermodynamic, any N-component system)
===========================================================
Reads HDF5 tables from step 1 and generates phase diagrams for each
temperature.  Two visualization styles are supported and controlled by the
``thermo.phase_map_style`` config key:

``ternary`` (default for n=3)
    Classic ternary diagram using python-ternary.  For ternary systems one
    diagram per temperature is produced.  For quaternary and higher-order
    systems one diagram per unique element pair is produced (all others held
    at equimolar fractions), giving C(n-1, 2) projected ternary maps.

``umap``
    Alloy-Space UMAP (AS-UMAP) following Vela et al. (arXiv:2408.07681).
    The full composition simplex is grid-sampled, embedded in 2D by UMAP
    (t-SNE fallback), and data points are overlaid colored by phase region.
    For ternary systems the embedding yields a deltoid (3-cusped hypocycloid);
    for quaternary an astroid (4-cusped hypocycloid), and so on.

``both`` (default for n>=4)
    Produces both the ternary projections and the AS-UMAP figure.

Output
------
<results_dir>/<T>C/thermodynamics/
    phase_map_<el0>_<el1>_<T>C.pdf    (ternary style, one per pair)
    as_umap_<T>C.pdf                   (umap style)

Prerequisites
-------------
Run step 1 first::

    pf-generate --config <config.yaml>

Run standalone
--------------
    pf-phase-map --config configs/nb_hf_v_pipeline_test.yaml

    # AS-UMAP only for a ternary system
    pf-phase-map --config configs/nb_hf_v.yaml --set thermo.phase_map_style=umap

    # Both styles for a quaternary system
    pf-phase-map --config configs/nb_hf_ti_v.yaml --set thermo.phase_map_style=both
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from phasefoundry.config import load_config, apply_overrides, resolve_paths, summary
from phasefoundry.thermodynamics import plot_ternary_spinodal, plot_as_umap


def _glob_tables(cfg: dict) -> list[Path]:
    """Return HDF5 tables from step 1 matching this config's system name."""
    data_dir = Path(cfg["output"]["data_dir"])
    name     = cfg["system"]["name"]
    paths    = sorted(data_dir.glob(f"{name}_*C.h5"))
    if not paths:
        raise FileNotFoundError(
            f"No HDF5 tables matching '{name}_*C.h5' in {data_dir}.\n"
            "Run  pf-generate --config <config>  first."
        )
    return paths


def _sim_composition(cfg: dict) -> dict[str, float] | None:
    """
    Build a {ELEMENT: x} dict from ``simulation.c0`` so the simulated
    composition is shown as a star on the phase map.

    Returns None when c0 is not set or the system is binary.
    """
    c0 = cfg.get("simulation", {}).get("c0")
    if c0 is None:
        return None
    elements  = cfg["system"]["elements"]
    ref_el    = cfg["system"]["reference_element"]
    ind_els   = [e for e in elements if e != ref_el]
    # c0 may be a scalar (binary) or a list (ternary+)
    if isinstance(c0, (int, float)):
        c0 = [c0]
    if len(c0) != len(ind_els):
        return None
    return {el: float(x) for el, x in zip(ind_els, c0)}


def _element_pairs(cfg: dict) -> list[tuple[str, str]]:
    """
    Return the list of independent-element pairs to plot.

    For ternary (2 independent elements): one pair.
    For quaternary+ (3+ independent elements): all C(n,2) combinations.
    """
    elements = cfg["system"]["elements"]
    ref_el   = cfg["system"]["reference_element"]
    ind_els  = [e for e in elements if e != ref_el]
    if len(ind_els) < 2:
        return []
    return list(itertools.combinations(ind_els, 2))


def run(cfg: dict) -> list[Path]:
    """
    Generate phase maps for every temperature in the config.
    Returns paths to all saved figures.
    """
    elements  = cfg["system"]["elements"]
    ref_el    = cfg["system"]["reference_element"]
    n_el      = len(elements)

    if n_el < 3:
        print("  Binary system — phase map not applicable. Skipping.")
        return []

    style_cfg  = cfg.get("thermo", {}).get("phase_map_style", None)
    if style_cfg is None:
        style = "both" if n_el >= 4 else "ternary"
    else:
        style = str(style_cfg).lower()

    do_ternary = style in ("ternary", "both")
    do_umap    = style in ("umap", "both")

    pairs       = _element_pairs(cfg)
    sim_comp    = _sim_composition(cfg)
    tables      = _glob_tables(cfg)
    results_dir = Path(cfg["output"]["results_dir"])
    ind_els     = [e for e in elements if e != ref_el]
    saved: list[Path] = []

    for table_path in tables:
        try:
            T_C = int(table_path.stem.split("_")[-1].replace("C", ""))
        except ValueError:
            T_C = cfg["simulation"]["temperature_C"]

        thermo_dir = results_dir / f"{T_C}C" / "thermodynamics"
        thermo_dir.mkdir(parents=True, exist_ok=True)

        if do_ternary:
            for el0, el1 in pairs:
                out = thermo_dir / f"phase_map_{el0.lower()}_{el1.lower()}_{T_C}C.pdf"
                print(f"  [ternary]  {el0}–{el1}–{ref_el}  @  {T_C} °C"
                      f"  →  {out.relative_to(_ROOT)}")

                pair_sim = None
                if sim_comp is not None:
                    pair_sim = {k: v for k, v in sim_comp.items()
                                if k in (el0, el1)}
                    if len(pair_sim) != 2:
                        pair_sim = None

                plot_ternary_spinodal(
                    table_path=table_path,
                    independent_elements=[el0, el1],
                    corner_labels=[el0.title(), el1.title(), ref_el.title()],
                    sim_composition=pair_sim,
                    save_path=out,
                    show=False,
                )
                saved.append(out)

        # AS-UMAP (works for any n >= 3)
        if do_umap:
            out_umap = thermo_dir / f"as_umap_{T_C}C.pdf"
            print(f"  [AS-UMAP]  {n_el} components  @  {T_C} °C"
                  f"  →  {out_umap.relative_to(_ROOT)}")
            plot_as_umap(
                table_path=table_path,
                independent_elements=ind_els,
                reference_element=ref_el,
                corner_labels=[e.title() for e in elements],
                sim_composition=sim_comp,
                save_path=out_umap,
                show=False,
            )
            saved.append(out_umap)

    return saved


def main():
    parser = argparse.ArgumentParser(
        description="Step 1d — Ternary phase map (general, any N-component system)"
    )
    parser.add_argument(
        "--config", required=True,
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--set", nargs="*", default=[], metavar="KEY=VALUE",
        help="Override config values, e.g. --set thermo.temperatures_C=[500]",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg = apply_overrides(cfg, args.set)
    cfg = resolve_paths(cfg, _ROOT)

    elements = cfg["system"]["elements"]
    n_el     = len(elements)
    n_ind    = n_el - 1
    print("── Step 1d: Ternary phase map ──")
    print(f"   System : {cfg['system']['name']}  "
          f"({n_el} components, {n_ind} independent field(s))")
    print(summary(cfg))

    saved = run(cfg)
    if saved:
        print(f"\nDone. {len(saved)} figure(s) saved.")
    else:
        print("\nDone (nothing to plot).")


if __name__ == "__main__":
    main()
