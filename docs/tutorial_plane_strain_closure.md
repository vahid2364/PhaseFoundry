# Tutorial: The Plane-Strain Eigenstrain Closure for the 2D Elasticity Solver

The [coherent-Hessian tutorial](tutorial_coherent_hessian_quaternary.md) works
entirely with a single wavevector direction \(\mathbf{n}\) in an infinite 3D
solid — it never has to decide what to do about a simulation that only
spans two spatial dimensions. The actual phase-field solver
(`phasefoundry.elasticity.fft_solver.solve_elasticity`) does have to decide
that, every time it runs a 2D simulation of what is physically a 3D crystal.
This tutorial builds that decision up from first principles, validates one
particular choice against the Y(n) machinery from the previous tutorial, and
then walks through a real mistake made while applying it to real simulation
data — kept in because it's exactly the kind of mistake this closure invites.

!!! note "Prerequisites"
    Steps 0–5 need nothing but the solver itself and a handful of made-up or
    production cubic elastic constants (no TDB, no CSV). Step 6 reads two
    files written by an already-completed simulation:

    - `results/HfNbTiV_BCC_PhaseForge_v2/NbHfTiV/500C/kinetics/pf_2d_c25Nb25Hf25Ti_kappa1e12_M5e21_tdbv2_elastic/fields.npz`
    - `.../elastic_params.yaml` (same directory)

    It does not re-run the kinetics — only re-solves the mechanical
    equilibrium problem once per saved snapshot.

---

## Step 0 — why a 2D simulation of a 3D crystal needs a decision at all

A composition field `c(x, y)` only varies in two directions, but the crystal
it represents still has a third one. Classical continuum mechanics has two
standard ways to collapse a 3D elasticity problem into 2D, and they are not
the same thing:

- **Plane stress** (\(\sigma_{33} = 0\)): appropriate for a thin, free-standing
  sheet — the top and bottom surfaces are free, so no stress can build up
  normal to them, and the sheet is free to thin or thicken (\(\varepsilon_{33}\)
  floats).
- **Plane strain** (\(\varepsilon_{33} = 0\)): appropriate for a slice through
  a *thick*, bulk solid, far from any free surface — the material above and
  below the slice clamps it, so it cannot move in the third direction at
  all, but it is perfectly free to develop whatever \(\sigma_{33}\) that
  clamping requires.

A 2D phase-field simulation of a precipitate growing inside a bulk alloy is
the second case, not the first — there's no free surface a few nanometers
away; the material in `z` is just more of the same bulk crystal.

`solve_elasticity`'s existing default, `eigenstrain_closure="truncate"`, is
neither of these. It simply never computes an out-of-plane eigenstrain or
stiffness component at all — the 3×3 tensors get sliced down to 2×2 before
anything happens with them. That's not a modeling error introduced by this
codebase; it's the literal published algorithm this solver is based on
(Gururajan & Abinandanan, *Acta Materialia* 55 (2007) 5015, which works in
2D throughout and never discusses a closure by name). But it does mean the
solver has been silently making a dimensional-reduction choice that is
neither textbook option above, and this tutorial's job is to add the
missing option and show it is the correct one for "a slice through bulk
material."

## Step 1 — what an eigenstrain is, from Vegard's law up

Swap some Nb atoms for Hf in a BCC lattice and the equilibrium lattice
parameter changes — Vegard's law treats this change as roughly linear in
composition. If a region's local composition "wants" a lattice spacing
different from its neighbors', but the crystal has to stay coherent (same
lattice across the interface, no dislocations), the region is forced away
from its own preferred size. The *stress-free strain* it would relax to if
cut free and allowed to relax — one number for how much bigger or smaller
it wants to be, applied equally in every direction for a cubic lattice —
is the eigenstrain \(\varepsilon^{*}_{ij} = \eta(c)\,\delta_{ij}\), the same
quantity Step 1 of the coherent-Hessian tutorial calls \(\eta\). It is a
genuine 3×3 tensor (`\varepsilon^*_{11} = \varepsilon^*_{22} = \varepsilon^*_{33} = \eta`
for an isotropic Vegard assumption), even though a 2D simulation only ever
asks about its `x` and `y` entries directly.

## Step 2 — where does the eigenstrain's z-component go in a 2D code?

Pose the question concretely. At every grid point, the composition field
implies a definite \(\varepsilon^*_{33}(x,y)\) — it's exactly the same
number as \(\varepsilon^*_{11}(x,y)\) and \(\varepsilon^*_{22}(x,y)\), just
along an axis the simulation doesn't track a displacement for. Three things
could happen to it:

1. **Ignore it** (`"truncate"`): never compute it, never let it affect
   anything. This is the current default.
2. **Plane stress**: let \(\varepsilon_{33}\) float freely so
   \(\sigma_{33}=0\) — wrong physics here (Step 0), but worth naming so it's
   not confused with option 3.
3. **Plane strain**: fix \(\varepsilon_{33}=0\) exactly (no `z` displacement
   at all, consistent with "more bulk crystal above and below"), and let
   \(\sigma_{33}\) be whatever that constraint, combined with the real
   eigenstrain, requires — including feeding back into \(\sigma_{11}\) and
   \(\sigma_{22}\) through the material's Poisson-type coupling.

Option 1 is not option 2 or option 3. It discards information a correct
treatment of either textbook case would keep.

## Step 3 — deriving the plane-strain closure

Hooke's law for a cubic crystal, one component at a time, with
\(\varepsilon^{\rm el} = \varepsilon - \varepsilon^{*}\):

$$
\sigma_{33} = C_{12}\,\varepsilon^{\rm el}_{11} + C_{12}\,\varepsilon^{\rm el}_{22}
            + C_{11}\,\varepsilon^{\rm el}_{33}
$$

Plane strain imposes \(\varepsilon_{33}=0\) exactly, so
\(\varepsilon^{\rm el}_{33} = 0 - \varepsilon^{*}_{33} = -\varepsilon^{*}_{33}\).
That's enough to eliminate \(\sigma_{33}\) entirely — there is no need to
solve for a `z` displacement field, because the constraint already *is* the
answer for \(\varepsilon_{33}\). Substituting into the in-plane stress
components:

$$
\sigma_{11} = C_{11}\,\varepsilon^{\rm el}_{11} + C_{12}\,\varepsilon^{\rm el}_{22}
            + \underbrace{C_{12}\,\varepsilon^{\rm el}_{33}}_{=\,-C_{12}\,\varepsilon^{*}_{33}}
$$

The underbraced term is exactly what `"truncate"` throws away. It is not a
small correction in general — it is the entire Poisson-type coupling that
makes a locked-in out-of-plane eigenstrain push back on the in-plane stress
at all. `solve_elasticity`'s `eigenstrain_closure="plane_strain"` option
adds precisely this term, as a constant correction to the in-plane stress
(`sigma_oop_correction` in the code — constant because it never depends on
the in-plane strain being solved for, so it's computed once, outside the
iteration loop, not an extra degree of freedom):

```python
C_io_full = C_field[..., :d, :d, :, :]          # (*spatial, d, d, 3, 3)
eps_el_oop = -eps_star_src.copy()
eps_el_oop[..., :d, :d] = 0.0                    # keep only the 33 (and 13, 23) entries
sigma_oop_correction = np.einsum("...ijkl,...kl->...ij", C_io_full, eps_el_oop)
```

## Step 4 — connecting it to Y(n)

This isn't a new theory bolted onto the coherent-Hessian tutorial's
machinery — it's a special case of it. For a cubic crystal with its axes
aligned to the grid and a wavevector confined to the `xy` plane,
\(\mathbf{n} = (n_1, n_2, 0)\), two facts about cubic symmetry combine to
make the full 3D \(Y(\mathbf{n})\) formula *already equal* the plane-strain
answer, exactly — not approximately:

1. **The acoustic tensor decouples.** \(A_{ik} = C_{ijkl}\,n_j n_l\) has no
   `13` or `23` entries for an axis-aligned cubic crystal (no coupling
   between a normal strain and a shear strain), so with \(n_3=0\) the
   `z`-row and `z`-column of \(A\) depend only on \(C_{44}\) and nothing
   else — they decouple completely from the in-plane \(2\times2\) block.
2. **The eigenstress has no `z`-component either.** \(\sigma^0 = C\!:\!I\)
   is diagonal for a cubic crystal, so \(v_i = \sigma^0_{ij} n_j\) has
   \(v_3 = \sigma^0_{33}\,n_3 = 0\) whenever \(n_3=0\).

Since `term2` in \(Y(\mathbf{n})\) is \(v \cdot \Omega(\mathbf{n}) \cdot v\)
and \(v_3=0\), the decoupled `z`-block of \(\Omega\) never enters the
calculation at all — only the in-plane \(2\times2\) sub-block does, and that
sub-block is identical to what you'd get solving the 2D plane-strain problem
directly. `term1` (\(C_{ijkl}\delta_{ij}\delta_{kl}\)) doesn't depend on
\(\mathbf{n}\) or on any solve at all — it's the same material constant
either way. So:

$$
Y_{\rm 3D}(\mathbf{n})\Big|_{n_3=0} \;=\; Y_{\rm plane\;strain}(\mathbf{n})
$$

exactly, for any in-plane direction and any cubic anisotropy. Step 5 checks
this two different ways.

## Step 5 — validating the closure

**Problem A — pure algebra, no solver.** Re-derive \(Y(\mathbf{n})\) using
*only* the in-plane \(2\times2\) acoustic-tensor block (no `z`-components
anywhere in the calculation), and compare against the full 3D formula from
the coherent-Hessian tutorial:

```python
def Y_planestrain_2x2(C, n2):
    n3 = np.array([n2[0], n2[1], 0.0]) / np.linalg.norm(n2)
    I3 = np.eye(3)
    sigma0 = np.einsum("ijkl,kl->ij", C, I3)
    term1  = np.einsum("ijkl,ij,kl->", C, I3, I3)
    A2     = np.einsum("abcd,b,d->ac", C[:2, :, :2, :], n3, n3)   # 2x2 only
    v2     = (sigma0 @ n3)[:2]
    term2  = v2 @ np.linalg.inv(A2) @ v2
    return (term1 - term2) / 2.0
```

Checked against `coherency_modulus(C, n)` for three materials (\(A_Z=0.7\),
\(A_Z=1.0\), and the production equiatomic value \(A_Z=1.52\)) at seven
in-plane angles each — every pair agrees to machine precision
(`rel.diff < 1e-12`), confirming Step 4's claim is exact, not approximate.

**Problem B — does the actual FFT solver agree?** Build an eigenstrain
field that's a single, exactly-representable Fourier mode,
\(\varepsilon^*(x,y) = \eta_0 \cos(\mathbf{k}\cdot\mathbf{r})\,\delta_{ij}\),
run `solve_elasticity(..., eigenstrain_closure="plane_strain")` on a
homogeneous medium, and compare its total elastic energy to Problem A's
prediction. A single mode (rather than a general microstructure) sidesteps
the catastrophic cancellation that made pointwise stress comparisons
unreliable earlier in this closure's development — the energies compared
here are directly the quantities of interest, not a near-zero difference of
two much larger numbers.

Converting energy to \(Y\) needs the amplitude convention spelled out, not
assumed — getting this wrong is an easy, silent mistake. In the
zero-relaxation limit (hypothetically no elastic relaxation at all,
\(\varepsilon^{\rm el} = -\varepsilon^*\) everywhere), \(Y_{\rm unrelaxed} =
{\rm term1}/2\) by definition, while direct substitution gives
\(\langle u_{\rm el}\rangle = \tfrac{1}{2}\eta_0^2\,{\rm term1}\,\langle\cos^2\rangle
= \tfrac{1}{4}\eta_0^2\,{\rm term1} = \tfrac{1}{2} Y_{\rm unrelaxed}\,\eta_0^2\).
So the general relation is \(Y = 2\langle u_{\rm el}\rangle / \eta_0^2\) —
*not* \(4\langle u_{\rm el}\rangle/\eta_0^2\), which is what a naive
"energy = \(\tfrac12 Y \eta^2\), then divide out \(\langle\cos^2\rangle=\tfrac12\)
again" guess gives, because that \(\langle\cos^2\rangle\) factor is already
absorbed into how \(Y=({\rm term1}-{\rm term2})/2\) is defined. Getting this
wrong produces a reproducible, exact factor of 2 in the comparison — if a
validation like this one is ever off by exactly a clean integer factor
across every single test case, suspect the amplitude convention before the
solver.

With the relation correct, the FFT solver's energy matches Problem A's
\(Y(\mathbf{n})\) to \(\sim 10^{-15}\) relative error, for 3 materials × 3
directions (9/9). The default `"truncate"` closure, computed the same way
but without the out-of-plane correction, is consistently *lower* — e.g.
86.3 vs. 120.8 GPa along \(\langle100\rangle\) for the production
composition — quantifying exactly how much softer the published algorithm's
omission makes the coherency barrier, rather than just asserting that it
does. All of this is captured as a permanent regression test in
`tests/test_plane_strain_closure.py` (`TestPlaneStrainAnalytical`,
`TestPlaneStrainFFTEnergy`, `TestTruncateDiffersFromPlaneStrain`).

## Step 6 — a worked pitfall on real simulation data

Applying this to an actual saved simulation (the −1.6% quaternary run whose
`fields.npz` holds 61 composition snapshots) at first gave a nonsensical
result: `plane_strain`'s energy came out **100,000× to 240,000× larger**
than `truncate`'s, varying with time in a way the physics gives no reason
to expect.

```
     t [s]   std(x_Hf)   U_truncate   U_plane_strain (wrong)     ratio
      25000   5.96e-04      762.5          1.845e+08           241,991
     100000   8.37e-04     1739.6          1.846e+08           106,096
     200000   1.62e-03     6701.9          1.846e+08            27,540
     325000   4.07e-03    42909.5          1.846e+08             4,303
```

The cause: `FixedVegardModel.eps_star_field()` computes
\(\varepsilon^*(x) = \sum_k \alpha_k\, c_k(x)\) using the **absolute**
composition fields, not their deviation from the simulation's mean
composition \(c_0 = (0.25, 0.25, 0.25)\). Evaluated at `c0` itself, this
gives \(\varepsilon^*_{33} \approx \sum_k \alpha_k \cdot 0.25 \approx 0.065\)
— a 6.5% "eigenstrain," present even where the composition is exactly at
its mean and there is, physically, nothing to be strained about.

For `"truncate"` this mistake is invisible: a spatially uniform offset to
\(\varepsilon^*\) just shifts the macroscopic strain \(E\) by the same
constant and costs zero elastic energy, regardless of whether that offset
is "real" or an artifact of where the reference point was chosen. For
`"plane_strain"` it is not invisible — \(\varepsilon_{33}\) is pinned at
exactly 0 with no macroscopic degree of freedom to absorb *any* uniform
offset, so that spurious 6.5% gets locked in as if it were real, physically
meaningful strain energy, at every single grid point, independent of the
actual composition fluctuation the simulation is supposed to be measuring.

The fix is the same idea Step 1's Vegard's-law picture already implies: the
eigenstrain should be measured relative to the reference state the lattice
parameter \(a_{\rm ref}\) was defined against, i.e.
\(\varepsilon^*(x) = \sum_k \alpha_k\,(c_k(x) - c_{0,k})\), which is exactly
zero at the mean composition:

```python
c_dev = c_all - c0[:, None, None]
eps_star_full = model_3d.eps_star_field(c_dev, ndim=3)
```

With that correction, the same four snapshots give a stable, physically
reasonable result:

```
     t [s]   std(x_Hf)   U_truncate   U_plane_strain      ratio
      25000   5.96e-04      761.7         1272.0           1.670
     100000   8.37e-04     1738.7         2926.4           1.683
     200000   1.62e-03     6701.9        11332.1           1.691
     325000   4.07e-03    42909.5        72869.4           1.698
```

A consistent ~1.67–1.70× across a 13× range of fluctuation amplitude — a
robust, amplitude-independent material effect, not noise. The lesson for
anyone else extending the `"truncate"`-only production pipeline to use
`"plane_strain"`: check what your eigenstrain model does *at the mean
composition* before trusting it under this closure. A convention that is
completely harmless under `"truncate"` is not automatically harmless under
`"plane_strain"`.

## Step 7 — result, and which closure to use

Converting that ~1.7× ratio to absolute terms puts it in context. At
500 °C, equiatomic, the purely chemical Hessian's smallest eigenvalue is
−39,095.6 J/mol (from the coherent-Hessian tutorial) — about
\(-3.67\times10^{9}\ {\rm J/m^3}\) once converted through this run's molar
volume. Even at the latest snapshot checked here (\(t=325{,}000\ {\rm s}\)),
the full plane-strain elastic energy is \(\sim 1.18\times10^{5}\ {\rm
J/m^3}\) — about five orders of magnitude smaller. At this stage of
decomposition the correction is real and reproducible, but numerically
negligible next to the chemical driving force; it only grows as
decomposition proceeds toward finite amplitude.

More importantly, the *direction* of the effect settles a question this
closure was originally built to answer: `plane_strain` is *more*
stabilizing than `truncate`, not less. A phase-field simulation using the
weaker `truncate` closure that still shows decomposition would decompose
*even more reluctantly* under the physically more complete `plane_strain`
treatment — so this closure cannot be the explanation for a simulation that
decomposes despite a linear-stability screen predicting it shouldn't.
Ruling that out is itself a useful, if negative, result.

**Practical rule:** use `"truncate"` when reproducing a published algorithm
exactly as described (it matches Gururajan & Abinandanan (2007) verbatim,
and is the production default for that reason). Use `"plane_strain"` when
the 2D simulation is meant to represent a slice through bulk 3D material
and the out-of-plane eigenstrain is physically present and non-negligible
— and when doing so, make sure the eigenstrain model is zeroed at the
simulation's actual mean composition first (Step 6).

---

## Where this leads

- **The Y(n) formula this closure turns out to be a special case of** — the
  [coherent-Hessian tutorial](tutorial_coherent_hessian_quaternary.md),
  Step 6.
- **The full anisotropic, composition-dependent elasticity this closure is
  a simplification of** — the
  [2D ternary elasto-chemical tutorial](tutorial_2d_ternary_elastic.md).
- **The solver itself** — `src/phasefoundry/elasticity/fft_solver.py`,
  `solve_elasticity(..., eigenstrain_closure=...)`.
