# Tutorial: binodal vs. spinodal boundaries at three temperatures (binary Nb-V)

This tutorial is thermodynamics-only — no phase-field simulation (Step 3) is
run. It computes and compares the **binodal** (equilibrium two-phase/solvus
boundary) and **spinodal** (local-instability boundary) for Nb-V BCC_A2 at
three temperatures chosen to show qualitatively different regimes: deep
inside the miscibility gap, near the critical point, and above it. Every
number and figure below came from an actual run on this machine.

## The physical distinction

For a binary system, the molar Gibbs energy curve `G_m(x)` at a fixed
temperature defines two different boundaries, and it's easy to conflate them:

- **Binodal** (equilibrium/solvus boundary): the two compositions connected
  by a **common tangent line** to `G_m(x)`. A homogeneous alloy between these
  two compositions can lower its total free energy by separating into two
  phases at exactly these two compositions — this is the thermodynamic
  equilibrium condition (equal chemical potentials in both phases).
- **Spinodal** (instability boundary): the two compositions where
  `d²G_m/dx² = 0` — i.e. where the curve's local curvature changes sign.
  Between them, `d²G_m/dx² < 0`: the homogeneous alloy is *locally*
  unstable to infinitesimal composition fluctuations, with no nucleation
  barrier. This is the boundary that actually matters for the phase-field
  tutorials ([binary](tutorial_2d_binary.md), [ternary](tutorial_2d_ternary_elastic.md)) — spinodal decomposition only happens inside it.

The spinodal region is always nested **inside** the binodal region (binodal
⊇ spinodal), and the two coincide at the critical temperature `T_c`, where
the two-phase gap closes entirely. All three facts show up directly in the
numbers below.

---

## The config file

Full contents of `configs/nb_v_binodal_spinodal_3T.yaml`:

```yaml
system:
  name:              nb_v
  db_path:           data/tdb/HfNbTiV_BCC_PhaseForge_v2_Tdep.tdb
  elements:          [NB, V]
  phase:             BCC_A2
  reference_element: V

thermo:
  composition_ranges:
    NB: [0.001, 0.999]
  n_points:          200
  temperatures_C:    [500, 900, 1200]   # deep in gap / near T_c / above T_c

simulation:
  temperature_C: 500

output:
  data_dir:      data/HfNbTiV_BCC_PhaseForge_v2/NbV
  results_dir:   results/HfNbTiV_BCC_PhaseForge_v2/NbV
  run_name:      binodal_spinodal_3T
```

This reuses the same `data_dir` as the [2D binary tutorial](tutorial_2d_binary.md)
(same TDB, same system) — see that tutorial's
[path-resolution explainer](tutorial_2d_binary.md#the-config-file) for how
`db_path`/`data_dir`/`results_dir` resolve relative to the repo root
regardless of your working directory.

**Why the `simulation:` block is this short:** the other tutorials' configs
carry a full phase-field parameter set (`N`, `kappa`, `M`, `dt`, `n_steps`,
...) because Step 3 actually integrates the PDE. This config never calls
Step 3, and Steps 1-2 only ever read one key out of `simulation:` —
`temperature_C`, used purely as an output-folder label
(`workflows/02_fit_surrogate/run.py:76,83`). Everything else in a typical
`simulation:` block is simply unused here.

The one thing to watch: `output.run_name` is set to an explicit string
(`binodal_spinodal_3T`) rather than `auto`. `auto` triggers
`build_run_name()` (`src/phasefoundry/config.py:126`), which *does* read
`simulation.c0`, `simulation.kappa`, and `simulation.M` to build a
descriptive filename — so leaving `run_name: auto` with a trimmed
`simulation:` block would raise a `KeyError`. Set `run_name` explicitly (as
here) for any config that skips Step 3 entirely.

---

## Step 1 — generate thermodynamic tables

```bash
pf-generate --config configs/nb_v_binodal_spinodal_3T.yaml
```

```
500.0 °C  →  exists, skipping
900.0 °C  →  exists, skipping
1200.0 °C  →  exists, skipping

Done. 3 tables in .../data/HfNbTiV_BCC_PhaseForge_v2/NbV
```

All three were already cached from earlier work in this same `data_dir`
(shared across every Nb-V v2 tutorial/config), so this just confirms the
files exist rather than recomputing anything.

## Step 2 — fit the surrogate and compute both boundaries

```bash
pf-surrogate --config configs/nb_v_binodal_spinodal_3T.yaml
```

```
Found 9 temperature table(s).
  Fitting G_m spline surrogate ...
T = 673.1 K (400.0 °C)  |  Gm spline built on 200 pts  x_V=[0.0010, 0.9990]
T = 773.1 K (500.0 °C)  |  Gm spline built on 200 pts  x_V=[0.0010, 0.9990]
  ...
T = 1673.2 K (1400.0 °C) |  Gm spline built on 200 pts  x_V=[0.0010, 0.9990]
  Computing spinodal/binodal boundaries ...
Saved figure to .../phase_diagram.png
Saved figure to .../GM_all.png
Saved figure to .../500C/thermodynamics/spinodal.png
```

**Worth noticing:** this says `Found 9 temperature table(s)`, not 3. Step 2
globs *every* `nb_v_bcc_a2_*C.h5` file already sitting in `data_dir` — not
just the three temperatures listed in this config's `thermo.temperatures_C`
(that list only controls what Step 1 *ensures exists*). Since this
`data_dir` is shared with other Nb-V tutorials/configs, tables already
existed at 400, 600, 700, 800, 1000, and 1400 °C too, so the resulting
phase diagram below is actually built from all 9 — a nicer result than the
3 we asked for, not a bug. If you want a phase diagram built from *only*
your configured temperatures, point `data_dir` at an empty directory
instead.

![Phase diagram: binodal, spinodal, and critical point](images/tutorial_binodal_spinodal_3T/02_phase_diagram_dome.png)

This is the real output — binodal (blue) and spinodal (red) domes for Nb-V
BCC_A2, with the critical point located at **T_c ≈ 1000 °C**. Notice the
spinodal dome sits strictly inside the binodal dome at every temperature
below `T_c`, and both envelopes converge to the same point at `T_c` — both
are direct consequences of the two boundaries' definitions above, not
something imposed by the plotting code.

---

## The construction, explicitly: three temperatures side by side

The figure above is convenient but hides *how* each point was computed. This
one makes it explicit — `G_m(x)` at each of our three chosen temperatures,
with the common tangent (binodal) and the `d²G/dx² = 0` crossings (spinodal)
drawn directly on the curve:

![Common tangent and spinodal construction at 500, 900, and 1200 °C](images/tutorial_binodal_spinodal_3T/01_common_tangent_construction.png)

Numeric boundaries (x_V), read directly off these constructions:

| T | Binodal (x_V) | Spinodal (x_V) | Regime |
|---|---|---|---|
| 500 °C | [0.025, 0.831] | [0.099, 0.610] | deep inside the two-phase gap |
| 900 °C | [0.161, 0.514] | [0.213, 0.435] | close to T_c — both envelopes narrow sharply |
| 1200 °C | — (none) | — (none) | above T_c — `G_m(x)` is globally convex, single phase stable everywhere |

At 1200 °C, `G_m(x)` (rightmost panel) has a single minimum and positive
curvature everywhere — there's no common tangent to draw and no sign change
in `d²G/dx²`, which is exactly what "above the critical temperature" means
physically, not a numerical failure.

### How this is actually computed (Python API)

The CLI runs this for you, but this is the exact call — useful if you want
boundaries at temperatures other than your `thermo.temperatures_C` list, or
want to script a boundary sweep yourself:

```python
from pathlib import Path
from phasefoundry.thermodynamics.spinodal import (
    spinodal_boundary_binary, binodal_boundary_binary,
)

data_dir = Path("data/HfNbTiV_BCC_PhaseForge_v2/NbV")
temps = [500.0, 900.0, 1200.0]
paths = [data_dir / f"nb_v_bcc_a2_{int(t)}C.h5" for t in temps]

spinodal = spinodal_boundary_binary(paths, "NB", temperatures_celsius=temps,
                                     reference_element="V")

# method="hull": lower convex hull of G_m(x) — the common-tangent
# construction, geometrically. No initial guess needed, can't converge to a
# spurious root. method="fsolve" is the legacy equal-chemical-potential
# solver, kept for cross-checking.
binodal = binodal_boundary_binary(paths, "NB", temperatures_celsius=temps,
                                   method="hull", spinodal_data=spinodal)
```

`spinodal_data=spinodal` is passed into `binodal_boundary_binary()` as a
fallback: very close to `T_c` the two-phase gap can be too narrow for either
boundary-finding method to resolve reliably, and since binodal ⊇ spinodal
always holds, falling back to the (numerically better-behaved) spinodal
result in that regime is more defensible than returning nothing.

---

## Interpreting the result

**Observations** (directly from the numbers above):

- Both the binodal and spinodal composition windows narrow monotonically as
  T increases from 500 → 900 °C.
- At 1200 °C, neither boundary exists — `G_m(x)` is convex over the entire
  composition range.
- The spinodal window is strictly narrower than the binodal window at both
  500 °C and 900 °C, consistent with binodal ⊇ spinodal.

**Interpretation:** these three temperatures bracket a critical temperature
for this TDB's Nb-V BCC_A2 assessment at `T_c ≈ 1000 °C` (read directly off
the phase-diagram figure, where the binodal and spinodal curves meet). This
is a property of the specific CALPHAD database being queried
(`HfNbTiV_BCC_PhaseForge_v2_Tdep.tdb`), not a universal fact about Nb-V —
a different assessment could place `T_c` elsewhere.

---

## Common gotchas

| Symptom | Likely cause |
|---|---|
| `binodal_boundary_binary()` / `spinodal_boundary_binary()` return `None` for a temperature | Expected above `T_c` — `G_m(x)` has no two-phase region there, not a bug. Check against the phase-diagram figure before assuming something's broken. |
| Phase diagram shows more temperatures than your config lists | `data_dir` is shared and Step 2 globs everything cached there (see the Step 2 note above) — use a dedicated empty `data_dir` if you want it limited to exactly your configured temperatures. |
| Binodal and spinodal boundaries look identical very close to `T_c` | Expected — they coincide exactly at `T_c`, so near it the two methods can legitimately agree to within numerical tolerance. |
| `method="hull"` and `method="fsolve"` disagree noticeably | `"fsolve"` depends on hardcoded initial guesses and can miss a root the hull method finds (or vice versa, rarely); `"hull"` is the default/more robust choice for exactly this reason — cross-check with `"fsolve"` only if you have a specific reason to distrust the hull result. |

## Where to go next

- **See the spinodal region actually decompose** — the [2D binary tutorial](tutorial_2d_binary.md) runs a full Cahn-Hilliard simulation at 500 °C, inside the spinodal window computed here.
- **Ternary boundaries** — for 2+ independent fields there's no single `d²G/dx²=0` crossing; the [ternary tutorial](tutorial_2d_ternary_elastic.md)'s Step 2 spinodal map instead scans the composition simplex and tests the sign of the Hessian's minimum eigenvalue at each point (`GmInterpolant.min_eigenvalue()`), and `compute_ternary_phase_regions()` in `src/phasefoundry/thermodynamics/spinodal.py` provides a related region-classification approach for ternary systems.
- **Recommended phase-field parameters from this curvature** — `recommend_pfm_params()` in the same module takes the `G_m` spline fit (not the boundary dicts above) and derives suggested `kappa`/`dt`/`Lx`/`N` starting points for Step 3 directly from `d²G/dx²` at your chosen `c0`.
