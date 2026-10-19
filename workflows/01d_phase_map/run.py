"""Workflow step 1d: draw phase maps from the step-1 tables.

For systems with three or more components, the step draws maps for every temperature
table in ``output.data_dir``. The style comes from ``thermo.phase_map_style``:

- ``ternary`` (the default for three components): a python-ternary diagram for each pair
  of independent elements, saved as ``phase_map_<el0>_<el1>_<T>C.pdf``.
- ``umap``: an AS-UMAP map of the whole composition simplex (Vela et al.,
  arXiv:2408.07681), saved as ``as_umap_<T>C.pdf``.
- ``both`` (the default for four or more components): both of the above.

The figures go to ``<results_dir>/<T>C/thermodynamics/``. Binary systems are skipped.

```bash
python workflows/01d_phase_map/run.py --config configs/nb_hf_v_pipeline_test.yaml
python workflows/01d_phase_map/run.py --config configs/nb_hf_v.yaml --set thermo.phase_map_style=umap
python workflows/01d_phase_map/run.py --config configs/nb_hf_ti_v.yaml --set thermo.phase_map_style=both
```
"""

from __future__ import annotations

import argparse
import itertools
import logging
import sys
from pathlib import Path

from phasefoundry.config import apply_overrides, load_config, resolve_paths, summary
from phasefoundry.thermodynamics import plot_as_umap, plot_ternary_spinodal

_ROOT = Path(__file__).resolve().parents[2]


def _glob_tables(cfg: dict) -> list[Path]:
    """Return the step-1 tables of this config's system, sorted by file name.

    Args:
        cfg: Resolved config.

    Returns:
        Paths of ``<data_dir>/<system name>_*C.h5``.

    Raises:
        FileNotFoundError: If no table matches.
    """
    data_dir = Path(cfg["output"]["data_dir"])
    name = cfg["system"]["name"]
    paths = sorted(data_dir.glob(f"{name}_*C.h5"))
    if not paths:
        raise FileNotFoundError(
            f"No HDF5 tables matching '{name}_*C.h5' in {data_dir}.\nRun  pf-generate --config <config>  first."
        )
    return paths


def _sim_composition(cfg: dict) -> dict[str, float] | None:
    """Return the simulated composition, marked with a star on the phase maps.

    Args:
        cfg: Resolved config.

    Returns:
        Mole fraction of each independent element from ``simulation.c0``, or ``None``
        if ``c0`` is missing or its length does not match the independent elements.
    """
    c0 = cfg.get("simulation", {}).get("c0")
    if c0 is None:
        return None
    elements = cfg["system"]["elements"]
    ref_el = cfg["system"]["reference_element"]
    ind_els = [e for e in elements if e != ref_el]
    if isinstance(c0, (int, float)):
        c0 = [c0]
    if len(c0) != len(ind_els):
        return None
    return {el: float(x) for el, x in zip(ind_els, c0, strict=True)}


def _element_pairs(cfg: dict) -> list[tuple[str, str]]:
    """Return every pair of independent elements, one ternary map per pair.

    Args:
        cfg: Resolved config.

    Returns:
        The pairs, or an empty list for a binary system.
    """
    elements = cfg["system"]["elements"]
    ref_el = cfg["system"]["reference_element"]
    ind_els = [e for e in elements if e != ref_el]
    if len(ind_els) < 2:
        return []
    return list(itertools.combinations(ind_els, 2))


def run(cfg: dict) -> list[Path]:
    """Draw the phase maps for every step-1 table.

    Args:
        cfg: Resolved config.

    Returns:
        Paths of the saved figures, empty for a binary system.

    Raises:
        FileNotFoundError: If the step-1 tables are missing.
    """
    elements = cfg["system"]["elements"]
    ref_el = cfg["system"]["reference_element"]
    n_el = len(elements)

    if n_el < 3:
        print("  Binary system — phase map not applicable. Skipping.")
        return []

    style_cfg = cfg.get("thermo", {}).get("phase_map_style", None)
    default_style = "both" if n_el >= 4 else "ternary"
    style = default_style if style_cfg is None else str(style_cfg).lower()

    do_ternary = style in ("ternary", "both")
    do_umap = style in ("umap", "both")

    pairs = _element_pairs(cfg)
    sim_comp = _sim_composition(cfg)
    tables = _glob_tables(cfg)
    results_dir = Path(cfg["output"]["results_dir"])
    ind_els = [e for e in elements if e != ref_el]
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
                print(f"  [ternary]  {el0}–{el1}–{ref_el}  @  {T_C} °C  →  {out.relative_to(_ROOT)}")

                pair_sim = None
                if sim_comp is not None:
                    pair_sim = {k: v for k, v in sim_comp.items() if k in (el0, el1)}
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

        if do_umap:
            out_umap = thermo_dir / f"as_umap_{T_C}C.pdf"
            print(f"  [AS-UMAP]  {n_el} components  @  {T_C} °C  →  {out_umap.relative_to(_ROOT)}")
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
    """Parse the command line, load the config and draw the phase maps."""
    parser = argparse.ArgumentParser(description="Step 1d — Ternary phase map (general, any N-component system)")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--set",
        nargs="*",
        default=[],
        metavar="KEY=VALUE",
        help="Override config values, e.g. --set thermo.temperatures_C=[500]",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg = apply_overrides(cfg, args.set)
    cfg = resolve_paths(cfg, _ROOT)

    elements = cfg["system"]["elements"]
    n_el = len(elements)
    n_ind = n_el - 1
    print("── Step 1d: Ternary phase map ──")
    print(f"   System : {cfg['system']['name']}  ({n_el} components, {n_ind} independent field(s))")
    print(summary(cfg))

    saved = run(cfg)
    if saved:
        print(f"\nDone. {len(saved)} figure(s) saved.")
    else:
        print("\nDone (nothing to plot).")


if __name__ == "__main__":
    logging.basicConfig(format="%(message)s", stream=sys.stdout)
    logging.getLogger("phasefoundry").setLevel(logging.INFO)
    main()
