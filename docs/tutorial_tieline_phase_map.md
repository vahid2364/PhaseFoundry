# Tutorial: tie-lines and phase maps (ternary Nb-Hf-V)

A two-phase equilibrium in a multicomponent alloy is defined by a
**tie-line**: the pair of compositions at which every component's chemical
potential is equal between the two phases. It's the quantity that actually
answers "if this alloy splits into two phases, what are they made of?" —
and it's not something you can guess or estimate by eye in a ternary or
higher-order system, because it depends on the full shape of the
multicomponent Gibbs energy surface, not just the overall composition.

That matters here specifically because PhaseFoundry's elasto-chemical model
needs a real tie-line as input, not a free parameter: `c_mat`/`c_ppt` set
the two reference compositions the Vegard eigenstrain model measures
coherency strain between, so if they don't correspond to an actual
equilibrium, the elastic driving force computed from them doesn't
correspond to anything physical either. `pf-tieline` exists to close
exactly that gap — it calls `pycalphad`'s equilibrium solver directly, at
your exact parent composition and temperature, and reports the real
phase split instead of a hand-picked estimate. This tutorial runs it for
real, alongside `pf-phase-map` for a wider composition-space view, and the
result turns out to be more interesting than simply replacing an imprecise
placeholder with a precise number.

## What these two commands actually compute

- **`pf-tieline`** calls `pycalphad.equilibrium()` once, at a single
  composition and temperature, and reports every stable phase instance it
  finds there.
- **`pf-phase-map`** scans the *entire* composition simplex and classifies
  every point by how many phase regions the lower convex hull of `G_m`
  resolves into there (single-phase / two-phase / three-phase), drawing the
  result as a ternary diagram with the spinodal boundary and your
  simulation's mean composition overlaid.

They're independent implementations of related ideas — one answers "what's
stable *here*," the other answers "what's stable *everywhere*." Run
together, they cross-check each other, which is exactly what happens below.

---

## Step 1c — what does CALPHAD actually say about Nb50Hf25V25 at 500 °C?

```bash
pf-tieline --config configs/nb_hf_v_elastic_cNb50V25Hf25.yaml
```

This is the real output — nothing trimmed from the equilibrium section:

```
System   : NB - HF - V
Phase    : BCC_A2
T        : 500 °C (773.15 K)
c0       : {'NB': 0.5, 'HF': 0.25}

=== Raw equilibrium output ===
  BCC_A2  fraction=0.2648
  BCC_A2  fraction=0.4544
  BCC_A2  fraction=0.2807

3 BCC_A2 phases found (two-phase tie-line).

=== Tie-line endpoints ===
  phase_0: x_NB=0.2316  x_HF=0.7631  x_V=0.0054
  phase_1: x_NB=0.8757  x_HF=0.1051  x_V=0.0192
  phase_2: x_NB=0.1451  x_HF=0.0006  x_V=0.8543

=== Suggested elastic config update ===
```

Two things worth stopping on immediately:

1. **Three phase instances, not two.** The script's own log line says
   "two-phase tie-line" (that's just its default message — it was written
   assuming the common case), but it actually found three simultaneously
   stable `BCC_A2` compositions, with comparable phase fractions
   (26%/45%/28%, not one dominant phase plus two numerical slivers).
2. **The "suggested elastic config update" section is empty.** The
   script only knows how to turn its result into `c_mat`/`c_ppt` text when
   exactly two phases are found (it sorts two endpoints into `mat`/`ppt` by
   reference-element content); given three, it has nothing to suggest.

This isn't a bug in the tie-line script — it's telling you something true
and specific: **at this exact composition and temperature, the real
equilibrium state isn't a simple two-phase matrix/precipitate split at
all.** No pair of endpoints you choose from a two-phase framework can be
"the" correct tie-line here, because the real answer has three vertices,
not two.

---

## Step 1d — seeing why, across the whole diagram

```bash
pf-phase-map --config configs/nb_hf_v_elastic_cNb50V25Hf25.yaml
```

![Ternary phase map at 500 °C, showing the three-phase region](images/tutorial_tieline_phase_map/phase_map_500C.png)

The green triangle is the three-phase region `compute_ternary_phase_regions()`
finds by taking the lower convex hull of the `G_m(x_Nb, x_Hf)` surface — and
the black star (the `simulation.c0` composition every ternary-system
tutorial on this site uses, Nb50Hf25V25) sits **inside it**, not in the
orange two-phase region surrounding it.

**The cross-check:** read the green triangle's three corners directly off
this figure and compare them to `pf-tieline`'s three endpoints above — they
match, to plotting resolution: the top vertex (~x_Hf=0.76, x_Nb=0.23) is
`phase_0`; the right vertex (~x_Nb=0.88, x_Hf=0.11) is `phase_1`; the
bottom-left vertex (near the V corner) is `phase_2` (x_V=0.854). Two
independently-implemented tools — a single-point `pycalphad.equilibrium()`
call and a whole-simplex convex-hull scan — agree on the same answer. This
is a real three-phase field in this TDB's assessed Nb-Hf-V BCC_A2 system at
500 °C, not a fluke of either tool.

### How it changes with temperature

The same command, run against the cached 300 °C, 700 °C, and 900 °C tables
in this `data_dir`, shows the three-phase field isn't a fixed feature — it
closes as temperature rises:

<div class="grid cards" markdown>

-   **300 °C**

    ---

    ![300 C phase map, three-phase region covers most of the diagram](images/tutorial_tieline_phase_map/phase_map_300C.png)

    The three-phase region is *larger* here — it covers most of the
    diagram, and the star composition sits deep inside it.

-   **500 °C**

    ---

    ![500 C phase map, three-phase region present, star inside it](images/tutorial_tieline_phase_map/phase_map_500C.png)

    The case worked through above — three-phase field still present, star
    just inside it.

-   **700 °C**

    ---

    ![700 C phase map, three-phase region gone](images/tutorial_tieline_phase_map/phase_map_700C.png)

    The three-phase field has closed entirely — only single-phase and
    two-phase regions remain, and the star is now in an ordinary two-phase
    region.

-   **900 °C**

    ---

    ![900 C phase map, no three-phase region, small spinodal dome](images/tutorial_tieline_phase_map/phase_map_900C.png)

    Further above the closure temperature — the spinodal dome itself has
    also shrunk substantially.

</div>

**Observation:** the three-phase field shrinks monotonically between 300 °C
and 700 °C and is gone by 700 °C (somewhere between 500 °C and 700 °C it
closes — these four cached temperatures don't pin down exactly where,
the same sampling-density point made in the
[binodal/spinodal tutorial](tutorial_binodal_spinodal_3T.md#common-gotchas)).

**Interpretation:** this is consistent with a three-phase field bounded
above by an invariant-reaction temperature in this system's assessment —
below it, three `BCC_A2` composition sets coexist; above it, the system
reduces to the ordinary two-phase (or single-phase) behavior the rest of
this site's tutorials assume. None of this is a claim about the real
physical Nb-Hf-V system — it's a property of this specific TDB
(`HfNbTiV_BCC_PhaseForge_v2_Tdep.tdb`) at this specific composition range,
exactly the same caveat every other tutorial on this site makes about its
own TDB-derived numbers.

---

## What this means for the elastic config

The [ternary tutorial](tutorial_2d_ternary_elastic.md)'s
`elastic.c_mat`/`elastic.c_ppt` values
(`{NB: 0.70, HF: 0.10, V: 0.20}` / `{NB: 0.15, HF: 0.35, V: 0.50}`) aren't
just numerically off from the real tie-line — they're the wrong *shape* of
answer. The eigenstrain/Vegard framework that config feeds into is built
for a two-phase matrix/precipitate microstructure; the real equilibrium at
that composition is a three-phase one. Fixing this properly means either
choosing a parent composition that sits outside the three-phase field (any
composition in the orange region of the 500 °C map above), or extending the
elastic model to handle three coexisting phases — not just plugging
`pf-tieline`'s numbers in, since for this exact composition it doesn't
produce usable `mat`/`ppt` values at all.

---

## Common gotchas

| Symptom | Likely cause |
|---|---|
| `pf-tieline` raises `FileNotFoundError` for `db_path` when run from outside the repo root, or with a config file located somewhere other than directly under `configs/` | Unlike every other workflow (which resolves paths relative to the repo root via `resolve_paths()`), `workflows/01c_calphad_tieline/run.py` resolves `db_path` as `Path(config_path).parent.parent / db_path` — relative to *your config file's own location*, not the repo root. It only works correctly for configs sitting directly in `configs/`. Keep tie-line configs there, or use an absolute `db_path`. |
| `pf-tieline`'s "Suggested elastic config update" section is empty | More than 2 stable phases were found at that composition/temperature — see above. Check the raw equilibrium output's phase count before assuming the command is broken. |
| `pf-phase-map` shows more temperatures than your config lists | Same shared-`data_dir` caching behavior as `pf-surrogate` — see the [binodal/spinodal tutorial's Step 2 note](tutorial_binodal_spinodal_3T.md#step-2-fit-the-surrogate-and-compute-both-boundaries). |
| Three-phase region visible on the phase map but `pf-tieline` at a composition inside it only reports 1-2 phases | Possible if that exact composition sits exactly on a region boundary, or a different temperature than the map shown — re-run `pf-tieline` at the exact `(T, x)` you're checking against the map. |

## Where to go next

- **See the config this caveat applies to** — the [2D ternary elasto-chemical tutorial](tutorial_2d_ternary_elastic.md) runs the actual simulation with the placeholder `c_mat`/`c_ppt` values this tutorial shows are the wrong shape of answer for this composition.
- **The Hessian behind the spinodal boundary drawn on these maps** — the [Hessian stability analysis tutorial](tutorial_hessian_stability_binary.md) (binary case; the same `min_eigenvalue()` machinery scales to this ternary map).
- **Full command reference** — [CLI Reference](cli_reference.md) for `pf-tieline` and `pf-phase-map` alongside every other `pf-*` command.
