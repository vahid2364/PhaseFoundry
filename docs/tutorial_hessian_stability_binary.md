# Tutorial: Hessian stability analysis (binary Nb-V)

Every tutorial on this site has used the Gibbs-energy Hessian without
explaining the machinery behind it: the [binodal/spinodal tutorial](tutorial_binodal_spinodal_3T.md)
found *where* `d²G_m/dx² = 0`, and the [2D binary tutorial](tutorial_2d_binary.md)'s
Step 3 printed `H_NBNB`, `λ_min`, `δ`, `λ*`, and `dt_stable` without deriving
any of them. This tutorial is that derivation — the actual API
(`GmInterpolant.d2g()` / `.min_eigenvalue()`), validated against finite
differences, with every number traced back to where it was already shown to
you. No new config file is needed; everything below reuses
`configs/nb_v_chem_c50.yaml` and the Nb-V v2 thermodynamic tables already on
disk from the earlier tutorials.

---

## The Hessian API, and what it reduces to for binary

`src/phasefoundry/thermodynamics/interpolant.py` implements the Hessian
generally, for any number of independent components:

```python
def d2g(self, c):
    """Hᵢⱼ = ∂²G/∂cᵢ∂cⱼ, shape (N_ind, N_ind, *spatial)"""
    ...

def min_eigenvalue(self, c):
    """
    Minimum eigenvalue of the Hessian at each spatial point.
    Negative → spinodally unstable.
    """
    H = self.d2g(c)                      # (N_ind, N_ind, *spatial)
    ...
    lmin = np.linalg.eigvalsh(H[:, :, k]).min()   # per point
    ...
```

For a binary system there's only one independent composition field
(`N_ind = 1`), so `d2g()` returns a **1×1 matrix** at every point — its only
"eigenvalue" is just that single entry, `d²G_m/dx²`. The eigen-decomposition
machinery isn't doing anything interesting yet for `N_ind = 1`; it's only
once you get to 2+ independent fields (ternary and beyond) that `min_eigenvalue()`
is actually extracting the worst direction out of a real matrix rather than
reporting a scalar back to you. That's deliberate: the same function call
works unchanged whether you're running binary, ternary, or quaternary —
this tutorial establishes what it computes using the simplest case, where
you can sanity-check every number by hand.

---

## Reproducing a number you've already seen

```python
from pathlib import Path
from phasefoundry.thermodynamics.interpolant import build_interpolant

paths = [Path("data/HfNbTiV_BCC_PhaseForge_v2/NbV/nb_v_bcc_a2_500C.h5")]
interp = build_interpolant(paths, ["NB", "V"], "V", temperature_K=773.15,
                            derivative_method="jansson")

interp.min_eigenvalue([[0.5]])   # x_Nb = 0.5, the binary tutorial's c0
```

```
-7239.28
```

This is the exact number the [2D binary tutorial](tutorial_2d_binary.md#step-3-run-the-simulation)
printed as `H_NBNB (diagonal) = -7.239e+03 J/mol` and
`λ_min (spinodal) λ_min = -7.239e+03 J/mol (UNSTABLE)` — not a coincidence,
the same `GmInterpolant` machinery computes both. Negative means unstable,
consistent with x_Nb = 0.5 sitting well inside the spinodal window
`[0.390, 0.901]` found in the binodal/spinodal tutorial.

---

## Validating against finite differences

```bash
pf-validate --config configs/nb_v_chem_c50.yaml
```

```
Method          RMSE G_m        RMSE μ        RMSE H  J/mol
fd                   0.0        1452.5    23846737.7
jansson              0.0           0.0           0.0
hybrid               0.0           0.0    23846737.7
```

!!! warning "That RMSE_H number looks alarming. It's misleading, not wrong."
    23.8 million J/mol sounds like finite differences are badly broken. They
    aren't — in the region that actually matters, they agree with the
    analytic (Jansson) Hessian to **4 significant figures**. The RMSE above
    is computed over the *entire* composition range (200 points,
    x_V ∈ [0.001, 0.999]), and is completely dominated by a handful of
    points near the pure-element edges. Scroll down for the breakdown.

![CALPHAD data vs. Jansson/FD/Hybrid derivatives, with the spinodal region shaded](images/tutorial_hessian_stability_binary/01_derivative_comparison.png)

This is `pf-validate`'s own figure. Visually, the three derivative methods
are indistinguishable in all three panels, including panel (c), the
Hessian — which is exactly why the huge RMSE number is surprising at first.

### Where the disagreement actually lives

```python
import numpy as np

x = np.linspace(0.005, 0.995, 400)
Hj  = interp.d2g(x.reshape(1, -1))[0, 0]               # Jansson
Hfd = build_interpolant(paths, ["NB","V"], "V", 773.15,
                         derivative_method="fd").d2g(x.reshape(1, -1))[0, 0]
rel_err = np.abs(Hfd - Hj) / np.abs(Hj) * 100
```

![Jansson vs. finite-difference Hessian, and where the relative error actually blows up](images/tutorial_hessian_stability_binary/02_jansson_vs_fd_relative_error.png)

| x_Nb | Jansson H (J/mol) | FD H (J/mol) | Relative error |
|---|---|---|---|
| 0.500 (binary tutorial's c0) | -7160.8 | -7160.8 | 0.0001% |
| 0.099 (spinodal edge) | 50497.3 | 50509.7 | 0.025% |
| 0.050 | 114333.4 | 114403.7 | 0.062% |
| 0.020 | 300973.2 | 308698.4 | 2.57% |
| 0.390 (spinodal boundary, H≈0) | 15.7 | 15.8 | 0.68% |
| 0.901 (spinodal boundary, H≈0) | -185.3 | -172.9 | 6.70% |

Two **different** things are going on here, and conflating them would be
wrong:

1. **Near the pure-element limits (x_Nb → 0 or → 1), the disagreement is
   real and grows large in both absolute and relative terms** (2.6% by
   x_Nb = 0.02, tens of percent closer to the edge). This is the ideal
   configurational-entropy term in the CALPHAD model
   (`RT·Σ y ln y`, mentioned in the
   [binodal/spinodal tutorial's mathematics section](tutorial_binodal_spinodal_3T.md#the-mathematics-what-equilibrium-means-at-each-of-the-198-points))
   — its second derivative diverges as `x → 0`, and a finite central
   difference's truncation error grows right where the function it's
   differencing is blowing up. This is also precisely the region the
   degenerate mobility `M(c) = M̊·c·(1-c)` (from the
   [binary tutorial's equation section](tutorial_2d_binary.md#the-equation-being-solved))
   is designed to suppress — the two features of the codebase are solving
   the same underlying problem from different directions.
2. **Right at the spinodal boundary crossings (x_Nb ≈ 0.390, 0.901), the
   large *relative* error is an artifact, not a real disagreement.** H ≈ 0
   there by definition (that's what "boundary" means), so dividing a tiny,
   numerically-noisy absolute difference (~0.1-12 J/mol) by a tiny
   denominator produces a misleadingly large percentage. The *absolute*
   agreement is still excellent.

Only #1 is a genuine accuracy limitation of finite differences; #2 is just
relative error behaving badly near a zero crossing, the same way it would
for any two numerically close methods. In the region any of this
tutorial's configs actually simulate — `c0` chosen well inside the spinodal
window, away from the pure-element corners — Jansson and FD agree to
4-5 significant figures, and `derivative_method: jansson` is used throughout
mainly to avoid the FD step-size tuning problem and the cost of 2-4 extra
`G_m` evaluations per point, not because FD is unusable here.

---

## From eigenvalue to solver parameters

Step 3's diagnostics (`sim_params.txt`) derive several numbers directly
from this same Hessian — but not always at the same composition. There are
two distinct evaluations happening, and it's worth being precise about
which is which (`src/phasefoundry/phasefield/solver.py`):

**At exactly `c0`** (`_print_sim_summary`, around line 1040) — the Hessian
is evaluated once, at the mean composition, and its most-negative eigenvalue
`λ_min` feeds:

```
δ       = sqrt(κ / |λ_min|)                     interface width
λ*      = 4π · sqrt(κ / |λ_min|)                most-unstable wavelength
λ_c     = 2π · sqrt(2κ / |λ_min|)               critical wavelength
```

With κ = 1×10⁻¹² J·m²/mol and λ_min = -7239.28 J/mol:
`δ = sqrt(1e-12/7239.28) = 1.175×10⁻⁸ m` — exactly the `δ_NB = 1.175e-08 m`
printed in the binary tutorial.

**Over a scan window around `c0`** (`stable_dt()`, `src/phasefoundry/phasefield/solver.py:194`) —
`dt_stable` instead uses the *largest* `|eigenvalue|` found anywhere in a
composition window of ±0.3 around `c0` (20-point coarse scan), not the
value at `c0` itself:

```python
dt_stable = safety * dx**2 / (M_max * eig_max)
```

```python
>>> from phasefoundry.phasefield.solver import stable_dt
>>> stable_dt(M_diag=[5e-21], dx=1e-6/128, interpolant=interp, c0=[0.5])
0.18325325038772003
```

Matches the binary tutorial's `dt_stable = 1.8325e-01 s` exactly. The
window-scanned `|eigenvalue|` that actually set this value is **19983.9
J/mol** — almost 3× larger than `|λ_min(c0)| = 7239.3 J/mol` — because the
scan reaches compositions closer to the spinodal edge, where the curvature
is steeper, even though the simulation's actual mean composition is at the
much gentler x_Nb = 0.5. This is a genuinely more conservative (safer)
estimate than using `λ_min(c0)` alone would give.

!!! note "A small, confirmed mismatch worth knowing about"
    `sim_params.txt` prints this line as `Stable dt (safety=0.4) dt_stable=
    ...` — but `stable_dt()` is called without an explicit `safety`
    argument (`solver.py:484`), so it actually uses the function's own
    default, `safety=0.3` (`solver.py:200`), which is what the `0.18325`
    value above was computed with. The `"(safety=0.4)"` in the printed
    label doesn't match the number actually used — a small, pre-existing
    documentation/display inconsistency, not a correctness bug in the
    stability calculation itself.

---

## A practical API note

`min_eigenvalue()` only returns a bare Python `float` when the input
composition is truly 0-dimensional. Pass anything with an explicit spatial
axis — including a single point like `[[0.5]]` — and you get back a NumPy
array instead:

```python
>>> interp.min_eigenvalue([[0.5]])
array([-7239.28])          # NOT a bare float
>>> float(interp.min_eigenvalue([[0.5]])[0])
-7239.28                   # what you probably want
```

Worth knowing before you write `if interp.min_eigenvalue(c) < 0:` and get a
confusing array-truthiness error instead of the comparison you expected.

---

## Where this goes next

For `N_ind = 1`, the Hessian's "eigenvector" is trivial — there's only one
composition direction, so instability always points along it. That stops
being true at `N_ind = 2` (ternary): the Hessian becomes a genuine 2×2
matrix, and its eigenvector tells you *which combination* of composition
fluctuations is most unstable — not just whether one exists. That's the
natural follow-up to this tutorial, using the same Nb-Hf-V system as the
[ternary elasto-chemical tutorial](tutorial_2d_ternary_elastic.md), and
it's where the anisotropic raft orientation seen in that tutorial's final
microstructure actually traces back to.

- **See this Hessian define the spinodal boundary** — the [binodal/spinodal tutorial](tutorial_binodal_spinodal_3T.md) uses exactly `spinodal_boundary_binary()`, built on this same `d2g()` call, to find where it crosses zero.
- **See it drive an actual simulation** — the [2D binary tutorial](tutorial_2d_binary.md) is where every number reproduced above first appeared.
