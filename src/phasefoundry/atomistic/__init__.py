"""phasefoundry.atomistic — MLIP calculations via MaterialsFramework
================================================================

Produces the atomistic inputs that the rest of the pipeline consumes, so
they are computed inside PhaseFoundry rather than imported as CSV files:

  SQS elastic-constant table   C11, C12, C44, a_relaxed over the composition
                               simplex  (step 1b, elastic.source: mlip)
  Bond-length matrix           pairwise FNN bond lengths for the analytic
                               a(x) model  (step 1b, elastic.lattice_model: bond)
  CALPHAD TDB                  PhaseForge-style sqs2tdb fit  (step 0)

All MaterialsFramework imports are deferred to call time; install with
``pip install -e '.[mlip]'`` plus the extra for the chosen potential
(e.g. ``materialsframework[orb]``).  MaterialsFramework requires Python 3.12.

Public API
----------
mlip_settings(cfg, block)                → {"calculator", "params"}
build_calculator(settings)               → MaterialsFramework calculator
build_elastic_table(elements, path, …)   → pd.DataFrame
ensure_elastic_table(cfg, root)          → Path
compute_bond_table(elements, path, …)    → BondLatticeModel
ensure_bond_table(cfg, root)             → Path
generate_tdb(species, output, …)         → Path
"""

from .bond import compute_bond_table, ensure_bond_table
from .elastic import build_elastic_table, ensure_elastic_table, simplex_compositions
from .settings import build_calculator, mlip_label, mlip_settings
from .tdb import generate_tdb

__all__ = [
    "mlip_settings",
    "mlip_label",
    "build_calculator",
    "simplex_compositions",
    "build_elastic_table",
    "ensure_elastic_table",
    "compute_bond_table",
    "ensure_bond_table",
    "generate_tdb",
]
