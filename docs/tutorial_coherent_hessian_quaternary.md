# Tutorial: Coherent-Hessian Stability Analysis (Quaternary Hf-Nb-Ti-V)

The [binary Hessian stability tutorial](tutorial_hessian_stability_binary.md)
covers `N_ind = 1`, where the Hessian is a bare scalar, and leaves the
coherency (elastic) correction out entirely. This tutorial covers both
extensions at once, for the equiatomic quaternary Hf-Nb-Ti-V system: a
genuine `N_ind = 3` Hessian, plus the classical isotropic (Cahn, *Acta
Metall.* 9 (1961) 795) coherency correction added on top of it. By the end
you will have built the calculation by hand, piece by piece, then run the
same thing as a one-command, config-driven workflow.

!!! note "Prerequisites"
    No new config file is needed for the step-by-step part below — it only
    reads two files already on disk:

    - `data/tdb/HfNbTiV_BCC_PhaseForge_v2_Tdep.tdb` — the CALPHAD
      thermodynamic database
    - `data/HfNbTiV_BCC_PhaseForge_v2/sqs_elastic_constants_HfNbVTi-orb-v2.csv`
      — the SQS/MLIP elastic-constants table

    Running the full sweep later *as a workflow* does need a config file —
    see ["Running the full temperature sweep as a
    workflow"](#running-the-full-temperature-sweep-as-a-workflow) below.

---

## Step 1 — the chemical Hessian

`chemical_hessian()` evaluates the Gibbs-Duhem-reduced Hessian
\(H_{ij} = \partial(\mu_i - \mu_{\rm ref})/\partial x_j\) at one composition
and temperature, via pycalphad's Jansson derivatives — the same quantity the
[binary tutorial](tutorial_hessian_stability_binary.md#the-hessian-api-and-what-it-reduces-to-for-binary)
introduces, just a genuine 3×3 matrix here instead of a 1×1 one (independent
elements Hf, Nb, Ti; V is the reference).

What this matrix *is*: the curvature of the Gibbs free energy surface,

$$
H_{ij} = \frac{\partial^2 G}{\partial x_i\, \partial x_j}
$$

Nudge the composition in direction `i` and watch how the thermodynamic
driving force in direction `j` responds — a positive eigenvalue means the
free-energy surface curves upward there (a bowl: any fluctuation raises the
energy, so it decays back), a negative eigenvalue means it curves downward
(a dome: some fluctuation *lowers* the energy, so the system spontaneously
grows it). That spontaneous growth of a composition fluctuation is exactly
spinodal decomposition, which is why a negative eigenvalue is the
instability signal everywhere in this codebase.

pycalphad doesn't hand you \(\partial^2 G/\partial x_i\,\partial x_j\)
directly — it gives \(\partial \mu_i/\partial x_j\), a first derivative of
chemical potential. The two are related through the definition of chemical
potential for a dependent (reference) component,
\(\mu_i - \mu_{\rm ref} = \partial G/\partial x_i\). Differentiating both
sides by \(x_j\):

$$
\frac{\partial \mu_i}{\partial x_j} - \frac{\partial \mu_{\rm ref}}{\partial x_j}
= \frac{\partial^2 G}{\partial x_i\, \partial x_j}
$$

That subtraction — `dmu(eli, elj) - dmu(reference_element, elj)` in the
code — is this relation applied directly: it converts "how element `i`'s
chemical potential reacts to an `x_j` change" into the actual curvature, by
removing the part of that reaction that's just bookkeeping for whichever
element was designated dependent (V here). This is the same Gibbs-Duhem
step already used in `phasefoundry.thermodynamics.spinodal`, just written
out here for the full 3x3 case instead of a single binary entry.

```python
from phasefoundry.thermodynamics.coherent_hessian import chemical_hessian

comp = {"HF": 0.25, "NB": 0.25, "TI": 0.25}   # equiatomic
H_chem = chemical_hessian(
    db_path="data/tdb/HfNbTiV_BCC_PhaseForge_v2_Tdep.tdb",
    elements=["HF", "NB", "TI", "V"],
    phase="BCC_A2",
    composition=comp,
    temperature_K=773.15,                      # 500 C
    reference_element="V",
)
```

```
array([[-38709.4, -17265.5, -22225.9],
       [-17265.5,   3582.7,  -1897.2],
       [-22225.9,  -1897.2,  17867.1]])
```

This matrix alone already has a negative eigenvalue (checked in
[Step 5](#step-5-gauge-invariant-eigenvalues) below) — the purely chemical
free energy predicts this alloy is spinodally unstable at 500°C. Coherency
is what might change that conclusion.

## Step 2 — the elastic inputs

The eigenstrain and modulus come from the SQS/MLIP elastic-constants CSV,
through `phasefoundry.elasticity.sqs_interpolant` — the same interpolation
code `workflows/01b_lookup_elastic` uses for every simulation's eigenstrain.

```python
from pathlib import Path
from phasefoundry.elasticity.sqs_interpolant import (
    load_and_filter, build_interpolators, query_composition,
)

csv_path = Path("data/HfNbTiV_BCC_PhaseForge_v2/sqs_elastic_constants_HfNbVTi-orb-v2.csv")
df = load_and_filter(csv_path, ["HF", "NB", "TI", "V"])
interps = build_interpolators(df, ["HF", "NB", "TI"])
props = query_composition(interps, comp, ["HF", "NB", "TI"])
```

```
props["a_relaxed_A"]           # 3.2918   (lattice parameter, angstrom)
props["biaxial_modulus_M_GPa"] # 144.5950
```

`M` is a stiffness number: the elastic energy, per unit volume, it costs to
force a region to keep matching its neighbors' lattice spacing ("stay
coherent") instead of relaxing freely to its own preferred size. It's what
Step 4 below uses to convert a lattice mismatch into an energy penalty.

## Step 3 — the Vegard eigenstrain coefficients

$$
\eta_k = \frac{1}{a}\frac{\partial a}{\partial x_k}
$$

one per independent element, via central finite difference on the
interpolated lattice parameter — same method as the alpha-coefficients
computed inside `workflows/01b_lookup_elastic`, just callable directly at a
single composition here. Physically, \(\eta_k\) is the fractional stretch
of the lattice per unit change in element `k`'s composition: Vegard's law
treats \(a(x)\) as roughly linear in composition, and \(\eta\) is that
line's slope, normalized by \(a\) itself so it's a strain (dimensionless)
rather than an absolute length.

```python
from phasefoundry.elasticity.sqs_interpolant import vegard_eta_coefficients

eta = vegard_eta_coefficients(interps, comp, ["HF", "NB", "TI"])
```

```
{'HF': 0.1455, 'NB': 0.0734, 'TI': 0.0608}
```

Hf's coefficient is roughly twice Nb's or Ti's — Hf has by far the biggest
effect on the lattice parameter of the three.

## Step 4 — the isotropic coherency correction

Where \(2 V_m M (\eta \otimes \eta)\) comes from (Cahn, 1961): if the
composition fluctuates by \(\delta x_k\) in some direction, the local
lattice *wants* to change size by \(\sum_k \eta_k\, \delta x_k\) (that's
just Step 3's definition of \(\eta\), applied to a fluctuation instead of
the whole composition). If that region is mechanically constrained to stay
coherent with its surroundings — forced to keep the same lattice spacing
rather than relaxing to its own preferred size — the elastic energy it
costs is proportional to the *square* of that mismatch:

$$
\left(\sum_k \eta_k\, \delta x_k\right)^2 = \delta x^T (\eta \eta^T) \delta x
$$

\(\eta \eta^T\) (the outer product) is exactly the matrix form of that
quadratic penalty; \(2 V_m M\) is the constant converting squared lattice
mismatch into energy per mole. The code below builds exactly this matrix.

```python
from phasefoundry.thermodynamics.coherent_hessian import isotropic_coherent_correction

correction = isotropic_coherent_correction(
    eta, props["biaxial_modulus_M_GPa"] * 1e9, props["a_relaxed_A"],
    independent_elements=["HF", "NB", "TI"],
)
```

```
array([[65744.2, 33181.6, 27483.9],
       [33181.6, 16747. , 13871.3],
       [27483.9, 13871.3, 11489.4]])
```

Two structural properties follow directly from "a vector times itself":
the matrix is **positive semi-definite** (built from a square, it can never
be negative in any direction), so adding it to `H_chem` can only push
eigenvalues *up*, toward stability — the mathematical statement of
"coherency strain always opposes decomposition, never promotes it." And it
is **rank 1** — only the single direction `eta` points along gets
stiffened; the two composition directions orthogonal to it are completely
unaffected by this correction. (`tests/test_coherent_hessian.py`'s
`test_isotropic_correction_is_rank_one_and_psd` checks exactly this: two of
the three eigenvalues of the correction matrix are zero, the third equals
\(2 V_m M |\eta|^2\).)

```python
H_coh = H_chem + correction
```

```
array([[27034.8, 15916.1,  5258. ],
       [15916.1, 20329.7, 11974.1],
       [ 5258. , 11974.1, 29356.5]])
```

## Step 5 — gauge-invariant eigenvalues

Why this projection is needed: `H_chem` as built in Step 1 lives in a basis
that secretly depends on which element got picked as the reference (V,
here) — Hf, Nb, Ti are treated as independent coordinates and V is
whatever's left over, and that asymmetry is baked into the raw matrix
entries. If "is this alloy stable" (the sign of the smallest eigenvalue)
changed depending on whether V, Hf, or Nb happened to be labeled the
dependent component, that would be an artifact of notation, not physics.
`hessian_eigenvalues()` removes that dependence by projecting through the
orthogonal matrix `KADIRVEL_HF_NB_TI_O` (Table 1, Kadirvel et al., *Scripta
Materialia* 212 (2022) 114521) first — it rotates into a basis where all
four elements are treated symmetrically (genuinely orthonormal composition
directions, not "three independents plus a leftover"), so the eigenvalues
computed afterward are the same no matter which element you'd chosen as
reference. That's what "gauge-invariant" means here:

```python
from phasefoundry.thermodynamics.coherent_hessian import (
    hessian_eigenvalues, KADIRVEL_HF_NB_TI_O,
)

hessian_eigenvalues(H_chem, KADIRVEL_HF_NB_TI_O)
hessian_eigenvalues(H_coh, KADIRVEL_HF_NB_TI_O)
```

```
array([-39095.6,  13585.2,  41009.5])   # chemical: smallest eigenvalue negative -> unstable
array([ 11214. ,  23819.8,  46899.5])   # coherent: all positive -> stable
```

**Reading the result:** the chemical-only Hessian has one negative
eigenvalue (−39,095.6 J/mol) — unstable. Adding the coherency correction
pushes every eigenvalue positive, smallest now +11,214 J/mol — stable. At
exactly this composition and temperature, the isotropic coherency
correction is large enough to flip the stability conclusion.

That conclusion is sensitive to \(\eta\) specifically, because the
correction scales as \(\eta^2\) (Step 4): a coefficient change of a few
tens of percent in \(\eta_{\rm Hf}\) alone is enough, on its own, to flip
the sign of \(\lambda_{\min}({\rm coherent})\) at this composition and
temperature. Getting \(M\) and \(\eta\) from a different source than the
one \(H_{\rm chem}\) is evaluated against — rather than the same production
elastic-constants CSV used in Steps 2–3 — can silently change the answer.

All five steps above are also available as the one convenience call
`evaluate_coherent_stability(...)`, which returns a dict with every
intermediate quantity (`chemical_hessian`, `coherent_hessian`, `eta`,
`M_GPa`, `a_ref_A`, and both eigenvalue arrays) — useful once you already
know what each piece means and just want the numbers.

---

## Step 6 — generalizing to a direction-dependent coherency modulus Y(n)

Step 4's scalar \(M\) is applied the same way regardless of which direction
a composition fluctuation modulates along. That's only correct if the
crystal is elastically isotropic. This alloy isn't: at the equiatomic
composition \(A_Z = 2C_{44}/(C_{11}-C_{12}) = 1.29 \ne 1\), so the true
coherency modulus depends on the wavevector direction \(\mathbf{n}\):

$$
H^{\rm coh}_{ij}(\mathbf{n}) = H^{\rm chem}_{ij} + 2 V_m\, Y(\mathbf{n})\, \eta_i \eta_j
$$

with

$$
Y(\mathbf{n}) = \frac{1}{2}\left[
  C_{ijkl}\,\delta_{ij}\delta_{kl}
  - n_i\,\sigma^0_{ij}\,\Omega_{jk}(\mathbf{n})\,\sigma^0_{kl}\,n_k
\right],
\qquad
\sigma^0 = C\!:\!I,
\qquad
\Omega(\mathbf{n}) = \left[C_{ijkl}\,n_j n_l\right]^{-1}
$$

$\sigma^0$ is the "eigenstress" a unit dilatational eigenstrain would produce
if the material were held rigidly fixed, and \(\Omega(\mathbf{n})\) is the
same acoustic-tensor Green's function
`phasefoundry.elasticity.fft_solver._build_green_operator()` builds for the
real phase-field simulation — evaluated here at one direction instead of a
full Fourier grid. `coherency_modulus()` implements this directly:

```python
from phasefoundry.thermodynamics.coherent_hessian import coherency_modulus
from phasefoundry.elasticity.fft_solver import build_cubic_stiffness

C = build_cubic_stiffness(
    props["C11_GPa"] * 1e9, props["C12_GPa"] * 1e9, props["C44_GPa"] * 1e9,
)
coherency_modulus(C, n=[1, 0, 0])
```

```
128565186632.9  # Pa  ->  128.5652 GPa
```

Two checks validate this before trusting it for anything else. First, for
an isotropic stiffness tensor (\(A_Z=1\)) \(Y(\mathbf{n})\) must come out
identical for every direction — it does, to 9 significant figures, for any
\(\mathbf{n}\) tested. Second, \(Y([1,0,0])\) must reduce to the textbook
cubic \(\langle100\rangle\) result \(C_{11}+C_{12}-2C_{12}^2/C_{11}\) — it
does, exactly. (Both checks are
`tests/test_coherent_hessian.py::test_coherency_modulus_isotropic_is_direction_independent`
and `::test_coherency_modulus_cubic_100_matches_closed_form`.) This also
settles something from Step 2: the CSV's own `biaxial_modulus_M_GPa`
column (144.60 GPa) is *not* this quantity — it's some other modulus,
computed by whichever external script built that CSV column, not tracked
in this repository. `coherency_modulus()` is the one that's actually
validated against known closed-form elasticity results.

**The direction that matters is the softest one** — the \(\mathbf{n}\) that
minimizes \(Y(\mathbf{n})\), since that's the fluctuation direction least
resisted by coherency and therefore the one that actually governs onset of
instability:

```python
from phasefoundry.thermodynamics.coherent_hessian import (
    softest_direction, anisotropic_coherent_correction,
)

scan = softest_direction(C)   # dense scan over the unit sphere
scan["Y_min"] / 1e9, scan["n_min"]
```

```
(128.5652, array([0., 0., 1.]))
```

For this composition (\(A_Z=1.29>1\)), the softest direction is exactly
\(\langle100\rangle\) — the classical cubic-elasticity result ($A_Z>1 \Rightarrow
\langle100\rangle$ soft, $A_Z<1 \Rightarrow \langle110\rangle$ soft),
confirmed numerically here rather than assumed. There's no hidden softer
direction being missed by using \(\langle100\rangle\); it already is the
worst case.

Using this validated, worst-case \(Y(\mathbf{n}_{\min})=128.57\)~GPa in
place of the CSV's \(M=144.60\)~GPa:

```python
from phasefoundry.thermodynamics.coherent_hessian import (
    evaluate_anisotropic_coherent_stability,
)

result = evaluate_anisotropic_coherent_stability(
    db_path="data/tdb/HfNbTiV_BCC_PhaseForge_v2_Tdep.tdb",
    elastic_csv_path="data/HfNbTiV_BCC_PhaseForge_v2/sqs_elastic_constants_HfNbVTi-orb-v2.csv",
    elements=["HF", "NB", "TI", "V"],
    phase="BCC_A2",
    composition={"HF": 0.25, "NB": 0.25, "TI": 0.25},
    temperature_K=773.15,
    reference_element="V",
    O=KADIRVEL_HF_NB_TI_O,
)
result["lambda_min_coherent"]
```

```
9839.0   # J/mol -- still positive (stable), vs. +11214 with the CSV's M
```

**Reading the result:** the weaker, properly-derived worst-case correction
lowers \(\lambda_{\min}(\rm coherent)\) from +11,214 to +9,839 J/mol — about
12% smaller, as expected — but it stays solidly positive. Generalizing
away from the isotropic single-\(M\) approximation to the actual
direction-dependent, worst-case coherency modulus does **not** flip the
stability conclusion at this composition and temperature. This rules out
one specific candidate explanation for why a full phase-field simulation
of this composition can still show decomposition despite this screening
tool predicting stability: it is not that the isotropic approximation was
hiding a softer, more destabilizing direction.

`workflows/01e_coherent_hessian/run.py` runs Steps 1–6 together over the
full temperature sweep and all three compositions, plotting both
corrections side by side:

![Chemical and coherent eigenvalues, both the isotropic scalar-M correction and the anisotropic worst-case Y(n) correction, for all three compositions](images/tutorial_coherent_hessian_quaternary/01_fig4_reproduction.png)

The anisotropic (dashed red) curve sits consistently below the isotropic
(solid green) one in all three panels — a weaker, more physically
defensible correction — but never crosses back to negative anywhere the
isotropic curve doesn't already predict instability. Each composition's
`provenance.yaml` entry now also records `A_Z`, `Y_min_GPa`, and `n_min`
alongside the existing `a_ref`/`M`/`eta` fields.

---

## Running the full temperature sweep as a workflow

Steps 1–5 are one composition, one temperature. The config-driven workflow
repeats them over a temperature range and over several compositions in one
command, writing results to disk instead of a Python session:

```bash
python workflows/01e_coherent_hessian/run.py \
    --config configs/hf_nb_ti_v_coherent_hessian.yaml
```

```
  System:   HF - NB - TI - V  (ref: V)
  TDB:      .../data/tdb/HfNbTiV_BCC_PhaseForge_v2_Tdep.tdb
  Elastic:  sqs_elastic_constants_HfNbVTi-orb-v2.csv
  T sweep:  300-1950 K, step 50 K (34 points)
  Projection: kadirvel_hf_nb_ti
  Compositions: 3

  --- equiatomic_Hf25Nb25Ti25V25: {'HF': 0.25, 'NB': 0.25, 'TI': 0.25} ---
    a_ref = 3.2918 A   M = 144.5950 GPa
    eta_HF = 0.145488
    eta_NB = 0.073429
    eta_TI = 0.060820
    Written: .../coherent_hessian_fig4_reproduction/eigenvalues_equiatomic_Hf25Nb25Ti25V25.csv

  --- Hf10Nb40Ti40V10: {'HF': 0.1, 'NB': 0.4, 'TI': 0.4} ---
    a_ref = 3.2879 A   M = 129.0303 GPa
    eta_HF = 0.170015
    eta_NB = 0.085021
    eta_TI = 0.066252
    Written: .../coherent_hessian_fig4_reproduction/eigenvalues_Hf10Nb40Ti40V10.csv

  --- Hf60Nb10Ti10V20: {'HF': 0.6, 'NB': 0.1, 'TI': 0.1} ---
    a_ref = 3.4009 A   M = 150.3315 GPa
    eta_HF = 0.139535
    eta_NB = 0.086307
    eta_TI = 0.053103
    Written: .../coherent_hessian_fig4_reproduction/eigenvalues_Hf60Nb10Ti10V20.csv

  Written: .../coherent_hessian_fig4_reproduction/fig_04_eigenvalues.png
  Written: .../coherent_hessian_fig4_reproduction/fig_04_eigenvalues.pdf
  Written: .../coherent_hessian_fig4_reproduction/provenance.yaml
```

![All three composition panels, regenerated from the production TDB and elastic CSV](images/tutorial_coherent_hessian_quaternary/01_fig4_reproduction.png)

The original notebook's plotting cells only ever plot chemical λ₁ and
coherent λ₁ — two lines — even though the full, sorted eigenvalue array it
computes at every temperature already contains a second chemical eigenvalue
(λ₂) that's simply never passed to `plt.plot()`. Plotting it too (dotted
line above) exposes something the two-line version hides: for the
Hf10Nb40Ti40V10 panel, chemical λ₂ rises far more steeply with temperature
than chemical λ₁ does, crossing zero around 575–600 K versus λ₁'s ~700 K —
a large difference in the slopes of the two principal curvature directions
for that composition, invisible if you only ever look at the smallest
eigenvalue.

Three files land in `results/HfNbTiV_BCC_PhaseForge_v2/NbHfTiV/coherent_hessian_fig4_reproduction/`
per composition and overall:

- `eigenvalues_<label>.csv` — the full eigenvalue sweep (all three
  eigenvalues at every temperature, not just \(\lambda_{\min}\) — enough to
  plot \(\lambda_2\) as above for any composition), e.g.:

  | temperature_K | chem_eig_1 | coh_eig_1 |
  |---|---|---|
  | 300.0 | −46,877.4 | −16,411.5 |
  | 750.0 | −39,990.2 | 9,788.9 |
  | 900.0 | −33,662.1 | 19,027.8 |

- `fig_04_eigenvalues.png`/`.pdf` — a 3-panel plot, one per composition,
  chemical and coherent \(\lambda_1\) vs. temperature.
- `provenance.yaml` — the exact `a_ref`/`M`/`eta` used per composition, so
  reproducing a specific number later doesn't mean re-deriving it.

---

## Limitations: what this does not settle

A positive \(\lambda_{\min}(H^{\rm coh})\) at one composition only rules out spontaneous,
barrierless decomposition from that exact point. It does not, by itself,
explain why a full elasto-chemical phase-field simulation of this same
composition (see the
[2D ternary elasto-chemical tutorial](tutorial_2d_ternary_elastic.md) for
the anisotropic machinery such a simulation actually uses) can still
decompose, starting from finite-amplitude initial noise rather than an
infinitesimal perturbation at the exact mean composition. Two candidate
explanations remain open; a third has been tested and ruled out:

1. ~~**Isotropic vs. anisotropic elasticity**~~ — **ruled out (Step 6).**
   Generalizing Step 4's single scalar \(M\) to the direction-dependent,
   worst-case \(Y(\mathbf{n}_{\min})\) for this composition's actual
   \(A_Z=1.29\) anisotropy weakens the correction (+9,839 vs. +11,214 J/mol)
   but does not flip its sign. (Note: \(A_Z=1.29\) here is evaluated at the
   exact equiatomic composition tested in Steps 1–6; the simulation's
   matrix endpoint composition, a different point in composition space, has
   its own separate \(A_Z\approx1.52\) — the two should not be conflated.)
2. **Fixed point vs. composition-dependent field** — \(\eta\) and \(M\) (or
   \(Y(\mathbf{n})\)) are evaluated once, at exactly the mean composition
   (Steps 2–3, 6). The real simulation starts from finite-amplitude noise
   and re-evaluates composition-dependent elastic properties pointwise as
   the field evolves.
3. **Local vs. global stability** — a positive \(\lambda_{\min}\) rules out
   linear (spinodal) decomposition, not a lower-energy two-phase state
   reached via nucleation past a finite barrier, which discretized noise
   over a long (700,000 s) simulation can still seed.

Neither remaining item is resolved by the calculation in this tutorial;
testing them would mean tracking local stability along the real
matrix/precipitate composition trajectory rather than only
at the symmetric starting point — natural follow-on work for this module.

---

## Where this leads

- **The `N_ind = 1` case, without coherency** — the
  [binary Hessian stability tutorial](tutorial_hessian_stability_binary.md)
  this one extends.
- **The full anisotropic elasticity the Step 4 correction approximates** —
  the [2D ternary elasto-chemical tutorial](tutorial_2d_ternary_elastic.md).
- **The SQS/MLIP elastic-constants interpolation Steps 2–3 use** —
  `src/phasefoundry/elasticity/sqs_interpolant.py`, also used by
  `workflows/01b_lookup_elastic` for every simulation's eigenstrain.
