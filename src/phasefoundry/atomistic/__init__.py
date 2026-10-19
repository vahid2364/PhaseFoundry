"""MLIP calculations through MaterialsFramework.

Computes the atomistic inputs that the rest of the pipeline reads:

- the SQS elastic-constant table, with C11, C12, C44 and the relaxed lattice parameter
  over the composition simplex (step 1b, ``elastic.source: mlip``);
- the bond-length matrix behind the analytic a(x) model (step 1b,
  ``elastic.lattice_model: bond``);
- a CALPHAD TDB fitted with ATAT ``sqs2tdb`` in the PhaseForge style (step 0).

MaterialsFramework is imported only when a calculation runs. Install it with
``pip install -e '.[mlip]'`` plus the extra for the chosen potential, for example
``materialsframework[orb]``. MaterialsFramework requires Python 3.12.
"""

from .bond import compute_bond_table, ensure_bond_table
from .elastic import build_elastic_table, ensure_elastic_table, simplex_compositions
from .settings import build_calculator, mlip_label, mlip_settings
from .tdb import generate_tdb

__all__ = [
    "build_calculator",
    "build_elastic_table",
    "compute_bond_table",
    "ensure_bond_table",
    "ensure_elastic_table",
    "generate_tdb",
    "mlip_label",
    "mlip_settings",
    "simplex_compositions",
]
