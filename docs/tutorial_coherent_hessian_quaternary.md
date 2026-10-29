# Tutorial: Coherent-Hessian Stability Analysis (Quaternary Hf-Nb-Ti-V)

The [binary Hessian stability tutorial](tutorial_hessian_stability_binary.md)
covers `N_ind = 1`, where the Hessian is a bare scalar, and leaves the
coherency (elastic) correction out entirely. This tutorial covers both
extensions at once, for the quaternary Hf-Nb-Ti-V system used throughout the
[manuscript](https://github.com/vahid2364/PhaseFoundry)'s Section 3.4: a
genuine `N_ind = 3` Hessian, plus the classical isotropic (Cahn, *Acta
Metall.* 9 (1961) 795) coherency correction added on top of it.

## Why this tutorial exists

The manuscript's Figure 4 — chemical and coherent Hessian eigenvalues vs.
temperature, at three Hf-Nb-Ti-V compositions — was originally produced by
`notebooks/stability_analysis_nonisotropic.ipynb`. That notebook queries a
TDB named `HfNbVTi.tdb` and reads per-composition eigenstrain/modulus values
from `sqs_elastic_constants_HfNbVTi.csv` ("pulled from excel sheet," per the
notebook's own comment). **Neither file exists anywhere in this
repository.** The production TDB and elastic-constants CSV used for every
other quaternary result in the paper — including the Fig. 7(b)
elasto-chemical simulation itself — are different files:
`data/tdb/HfNbTiV_BCC_PhaseForge_v2_Tdep.tdb` and
`data/HfNbTiV_BCC_PhaseForge_v2/sqs_elastic_constants_HfNbVTi-orb-v2.csv`.

Substituting the production files in turned out to matter quantitatively,
not just for provenance hygiene. At the equiatomic composition, 500°C
(773.15 K, the same temperature as Fig. 7(b)):

| Inputs | λ_min(chemical) | λ_min(coherent) |
|---|---|---|
| Original notebook (untraceable TDB + CSV) | — | **positive** (reviewer's reading of the published figure) |
| Production TDB, original notebook's η/M | −39,096 J/mol | **−18,786 J/mol** (negative) |
| Production TDB **and** production η/M | −39,096 J/mol | **+11,214 J/mol** (positive) |

Swapping only the TDB flips the sign one way; also correcting η/M (in
particular η_Hf, which turns out to be 69% larger from the production data
than the notebook's hand-transcribed value) flips it back. The
fully-production-consistent result reproduces the reviewer's reading of the
published figure — so the apparent contradiction between Fig. 4(a) and
Fig. 7(b) is **not** a data-provenance artifact, and the real explanation
has to come from the modeling gap described in [Limitations](#limitations-what-this-does-not-settle)
below. This tutorial exists so that calculation is reproducible from files
actually tracked in this repo, instead of living only in a notebook cell.

## The formula

```
H^coh_ij = H^chem_ij + 2 * V_m * M * eta_i * eta_j
```

- `H^chem` is the Gibbs-Duhem-reduced chemical Hessian,
  `d(mu_i - mu_ref)/dx_j`, exactly as in the
  [binary tutorial](tutorial_hessian_stability_binary.md#the-hessian-api-and-what-it-reduces-to-for-binary)
  — just with `N_ind = 3` (Hf, Nb, Ti) instead of 1.
- `eta_k = (1/a) * da/dx_k` is the per-element Vegard eigenstrain
  coefficient, evaluated at one composition via central finite difference
  on the SQS/MLIP-interpolated lattice parameter.
- `M` is a single scalar "biaxial modulus," read directly from the elastic
  CSV's own `biaxial_modulus_M_GPa` column.
- `V_m` is the molar volume, `a³·N_A/2` (BCC, two atoms per cell).

This is implemented in
`src/phasefoundry/thermodynamics/coherent_hessian.py`:

```python
from phasefoundry.thermodynamics.coherent_hessian import (
    evaluate_coherent_stability, KADIRVEL_HF_NB_TI_O,
)

result = evaluate_coherent_stability(
    db_path="data/tdb/HfNbTiV_BCC_PhaseForge_v2_Tdep.tdb",
    elastic_csv_path="data/HfNbTiV_BCC_PhaseForge_v2/sqs_elastic_constants_HfNbVTi-orb-v2.csv",
    elements=["HF", "NB", "TI", "V"],
    phase="BCC_A2",
    composition={"HF": 0.25, "NB": 0.25, "TI": 0.25},
    temperature_K=773.15,
    reference_element="V",
    O=KADIRVEL_HF_NB_TI_O,
)
result["lambda_min_chemical"]   # -39095.6
result["lambda_min_coherent"]   # +11214.0
```

`O` is the gauge-invariant orthogonal projection (Table 1, Kadirvel et al.,
*Scripta Materialia* 212 (2022) 114521) that makes the eigenvalues
independent of which element is chosen as the dependent (reference)
component — the same role the eigen-decomposition machinery plays in the
binary tutorial, just nontrivial here because `N_ind = 3` makes the Hessian
a genuine matrix rather than a bare scalar.

!!! warning "This is an isotropic, single-composition, linear approximation"
    `2·V_m·M·(eta ⊗ eta)` uses **one scalar modulus** applied as a rank-1
    correction, evaluated **once**, at a single fixed composition. It is
    not the same calculation as the full anisotropic, composition-dependent
    `C_ijkl(x)` elasticity actually solved by the elasto-chemical
    coupling in `phasefoundry.elasticity`
    (see the [ternary elasto-chemical tutorial](tutorial_2d_ternary_elastic.md)),
    which the Fig. 7(b) simulation uses. Treat this as a fast linear-stability
    screening tool, not a substitute for running the phase-field
    simulation.

## Running it as a workflow

```bash
python workflows/01e_coherent_hessian/run.py \
    --config configs/hf_nb_ti_v_coherent_hessian.yaml
```

The config lists the TDB, the elastic CSV, one or more compositions, and a
temperature range — no hardcoded file paths or per-composition literals
inside the script itself, following the same `--config`/`--set` convention
as every other `workflows/*/run.py`:

```yaml
system:
  db_path: data/tdb/HfNbTiV_BCC_PhaseForge_v2_Tdep.tdb
  elements: [HF, NB, TI, V]
  phase: BCC_A2
  reference_element: V

elastic:
  elastic_csv: data/HfNbTiV_BCC_PhaseForge_v2/sqs_elastic_constants_HfNbVTi-orb-v2.csv

gauge_invariant_projection: kadirvel_hf_nb_ti

compositions:
  - label: equiatomic_Hf25Nb25Ti25V25
    composition: {HF: 0.25, NB: 0.25, TI: 0.25}
  # ... additional compositions, e.g. Figure 4's other two panels
```

This writes, per composition, a CSV of the full eigenvalue sweep (not just
`λ_min` — useful for reproducing Fig. 4's second-eigenvalue panel too), a
combined multi-panel plot, and a `provenance.yaml` recording the exact
`a_ref`/`M`/`eta` used — so the next person who needs these numbers reads
them from a file instead of reverse-engineering a notebook cell, which is
exactly the problem that motivated this tutorial in the first place.

## Limitations: what this does not settle

A positive `λ_min(H^coh)` at one composition only rules out *spontaneous,
barrierless* decomposition from that exact point under the isotropic
approximation above. It does not, by itself, explain why Fig. 7(b)'s
finite-amplitude, anisotropic, spatially-resolved phase-field simulation
decomposes anyway. Three candidate explanations remain open, in roughly
decreasing order of how directly testable they are with code already in
this repo:

1. **Isotropic vs. anisotropic elasticity** — the real solver's
   `C_ijkl(x)` is strongly anisotropic at this composition (`A_Z = 1.52`);
   the isotropic worst-case estimate here may simply be too stabilizing.
2. **Fixed point vs. composition-dependent field** — `eta` and `M` are
   evaluated once, at exactly the mean composition; the real simulation
   starts from finite-amplitude noise and re-evaluates composition-dependent
   elastic properties pointwise as the field evolves.
3. **Local vs. global stability** — a positive `λ_min` rules out linear
   (spinodal) decomposition, not a lower-energy two-phase state reached via
   nucleation past a finite barrier, which discretized noise over a long
   (700,000 s) simulation can still seed.

None of these is resolved by the calculation in this tutorial; testing them
would mean evaluating the anisotropic correction along multiple modulation
directions instead of the isotropic worst case, or tracking local stability
along the real matrix/precipitate composition trajectory rather than only
at the symmetric starting point — natural follow-on work for this module.

## Where this leads

- **The `N_ind = 1` case, without coherency** — the
  [binary Hessian stability tutorial](tutorial_hessian_stability_binary.md)
  this one extends.
- **The full anisotropic elasticity this isotropic correction approximates** —
  the [2D ternary elasto-chemical tutorial](tutorial_2d_ternary_elastic.md).
- **The SQS/MLIP elastic-constants interpolation `eta`/`M` are read from** —
  `src/phasefoundry/elasticity/sqs_interpolant.py`, also used by
  `workflows/01b_lookup_elastic` for every simulation's eigenstrain.
