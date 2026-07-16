---
title: 'PhaseFoundry: An Open-Source CALPHAD-Coupled Phase-Field Framework for Spinodal Microstructure Design in Multicomponent Alloys'

tags:
  - Python
  - phase-field modeling
  - CALPHAD
  - spinodal decomposition
  - high-entropy alloys
  - elasto-chemical coupling
  - materials science

authors:
  - name: Vahid Attari
    orcid: 0000-0003-3507-8847
    affiliation: 1
  - name: Raymundo Arróyave
    orcid: 0000-0001-7548-8686
    affiliation: "1, 2, 3"

affiliations:
  - name: Department of Materials Science and Engineering, Texas A&M University, College Station, TX 77843, USA
    index: 1
  - name: J. Mike Walker '66 Department of Mechanical Engineering, Texas A&M University, College Station, TX 77843, USA
    index: 2
  - name: Wm Michael Barnes '64 Department of Industrial and Systems Engineering, Texas A&M University, College Station, TX 77843, USA
    index: 3

date: 2026
bibliography: paper.bib
---

# Abstract

Predicting how a multicomponent alloy decomposes into a nanoscale two-phase microstructure requires connecting thermodynamic databases, which encode where a solid solution is unstable, to phase-field simulations, which resolve how that instability evolves in space and time. In practice this connection is the weakest link in the integrated computational materials engineering (ICME) toolchain: free energy models are hand-transcribed into solver source code, chemical potentials are approximated by finite differences, and coherency strain effects are either neglected or parameterized ad hoc. We present `PhaseFoundry`, an open-source Python framework that closes this gap for spinodal microstructure design. Starting from a CALPHAD thermodynamic database, including databases generated from machine-learned interatomic potentials (MLIPs) via `PhaseForge`, the framework evaluates exact Gibbs energy Hessians and chemical potentials using analytic Jansson derivatives in `pycalphad`, maps spinodal instability regions through gauge-invariant eigenanalysis, and evolves the resulting instabilities with an unconditionally stable pseudo-spectral Cahn-Hilliard solver that optionally includes coherency strain via a Moulinec-Suquet spectral micromechanics solver with MLIP-derived, composition-dependent elastic constants. Binary, ternary, and quaternary alloy systems are specified entirely through YAML configuration files, with no code changes. We describe the architecture, numerical methods, and validation tooling, and demonstrate the framework on the Hf-Nb-Ti-V refractory high-entropy alloy system, for which a full scientific study appears in a companion paper. `PhaseFoundry` is available under the MIT license at https://github.com/vahid2364/PhaseFoundry.

# 1 Introduction

Spinodal decomposition, the barrierless unmixing of a supersaturated solid solution into compositionally modulated domains by uphill diffusion, has been understood thermodynamically since Gibbs [@gibbs1878equilibrium] and given its modern continuum formulation by Cahn and Hilliard [@cahn1958free; @cahn1961spinodal]. It has recently reemerged as a deliberate microstructure design strategy. Depending on the alloy, the transformation can be harmful, embrittling otherwise ductile solid solutions [@wang2023spinodal] or providing a pathway for deleterious ordered phases [@dasari2023crystallographic], or it can be exploited: in refractory and 3d-transition-metal high-entropy alloys (HEAs), spinodally modulated solid solutions have been shown to overcome the strength-ductility trade-off [@an2021spinodal; @zhang2022refinement; @lee2025spinodal], tailor magnetic properties [@rao2021beyond; @dong2024cooperative], and improve thermoelectric performance [@duan2025spinodal]. Either way, the designer must first locate the spinodal region within a vast, high-dimensional composition space, and then predict the morphology, length scale, and phase compositions of the resulting microstructure. That task spans two traditionally separate computational communities: CALPHAD-based computational thermodynamics and phase-field modeling (PFM), the standard mesoscale approach for simulating microstructure evolution [@chen2002phase].

The thermodynamic half of the problem is well served by mature tools. CALPHAD databases encode assessed Gibbs energy models for multicomponent phases, and both commercial engines such as Pandat [@cao2009pandat] and open-source engines such as `pycalphad` [@otis2017pycalphad] make these models programmatically accessible, with ESPEI providing automated database development and uncertainty quantification [@bocklund2019espei]. Where classical assessments do not exist, MLIP-based workflows such as `PhaseForge` [@sariturk2025phaseforge; @zhu2025accelerating; @zhu2025machine] can now generate CALPHAD-compatible thermodynamic database (TDB) files for arbitrary alloy systems at a small fraction of the cost of density functional theory, and these machine-learned models can be further refined against experimental data [@kunselman2025construction]. Building on this thermodynamic infrastructure, the stability analysis needed to locate spinodal regions has a long lineage, from de Fontaine's stability criteria for coherent multicomponent solids [@de1967computer; @de1972analysis] through the recent extension of Gibbs energy Hessian analysis to HEAs [@morral2021stability] and high-throughput CALPHAD-based spinodal screening frameworks for multi-principal element alloy design [@koneru2022high; @kadirvel2022exploration]; first-principles routes to binodal and spinodal boundaries have also been demonstrated for binary systems [@usanmaz2016first; @divilov2024priori]. The kinetic half is likewise well served: general-purpose phase-field frameworks such as MOOSE [@permann2020moose], PRISMS-PF [@dewitt2020prisms], and FiPy [@guyer2009fipy] provide robust solvers for Cahn-Hilliard and Allen-Cahn type equations.

The difficulty lies in the coupling. A quantitative spinodal simulation needs the derivatives of the CALPHAD molar Gibbs energy, the relative chemical potentials $\partial G_m^\alpha / \partial x_i$ and the Hessian $H_{ij} = \partial^2 G_m^\alpha / \partial x_i \partial x_j$, evaluated millions of times over the course of a simulation. General-purpose PFM frameworks require the user to supply these free energy functions explicitly, which for multicomponent phases with internal degrees of freedom means either hand-fitting polynomial surrogates that drift from the underlying database, or numerically differentiating equilibrium calculations, an approach that is both expensive and sensitive to stencil size, particularly in high-dimensional composition spaces [@kunselman2024analytically]. Coherency strain, which shrinks the effective spinodal region [@cahn1961spinodal; @de1972analysis] and selects decomposition morphology, adds a further layer: composition-dependent elastic constants and lattice parameters must be obtained from atomistic calculations and interpolated consistently with the thermodynamic description. To our knowledge, no open-source tool provides an end-to-end pipeline from a CALPHAD TDB to a running elasto-chemical phase-field simulation for multicomponent alloys.

`PhaseFoundry` provides that pipeline. Its central design decisions are:

1. **Analytic thermodynamic derivatives.** Chemical potentials and Gibbs energy Hessians are computed with the Jansson derivative method [@kunselman2024analytically], an adjoint-like technique that differentiates the outputs of `pycalphad` equilibrium calculations exactly, eliminating finite-difference error and stencil-size sensitivity.

2. **Gauge-invariant stability analysis.** Spinodal regions are identified from the eigenvalues of the Gibbs energy Hessian after a change of basis to Cartesian coordinates [@kadirvel2022exploration; @morral2021stability], so that the predicted instability does not depend on the arbitrary choice of dependent component, and are visualized in high-dimensional composition spaces using affine projections [@vela2025visualizing].

3. **An unconditionally stable spectral solver.** The multicomponent Cahn-Hilliard equations are advanced with a semi-implicit pseudo-spectral scheme stabilized by Eyre convex splitting [@eyre1998unconditionally] with per-component stabilization constants, allowing large time steps deep inside the spinodal region where the diagonal Hessian entries are strongly negative.

4. **MLIP-informed elasticity.** Coherency strain enters through a composition-dependent Vegard eigenstrain model solved with the Moulinec-Suquet spectral method [@moulinec1998numerical], with elastic constants and lattice parameters interpolated across the full composition space from special quasirandom structure (SQS) [@zunger1990special] calculations performed with MLIPs.

5. **Configuration over code.** A single YAML file specifies the alloy system, thermodynamic database, composition grid, solver parameters, and elastic model. Binary, ternary, and quaternary systems, in two or three dimensions, chemical-only or elasto-chemical, are all handled by the same code path.

This paper describes the software itself: its architecture, numerical methods, validation tooling, and usage. A companion paper [@kunselman2025spinodal] presents the scientific application of the combined `PhaseForge`-`PhaseFoundry` workflow to the design of spinodal microstructures in the Hf-Nb-Ti-V refractory HEA system, from MLIP-generated thermodynamics through microstructure prediction.

# 2 Framework overview

## 2.1 Pipeline

`PhaseFoundry` is organized as an installable Python package, `calphad_pf`, driven by a set of numbered workflow scripts that share one YAML configuration file per run (Figure 1):

- **Step 1** samples the CALPHAD TDB over a composition grid at the temperatures of interest, evaluating the molar Gibbs energy, relative chemical potentials, and their Jansson derivatives with `pycalphad`, and stores the results in HDF5 tables.
- **Step 1b** (elasto-chemical runs) tabulates SQS/MLIP elastic constants $C_{ij}$ and lattice parameters across the composition space from CSV inputs generated by `PhaseForge`.
- **Step 1c** (elasto-chemical runs) computes CALPHAD tie-line endpoint compositions at the simulation temperature, which parameterize the classical eigenstrain model.
- **Step 2** fits a scattered-data interpolant surrogate to the HDF5 tables and computes spinodal boundaries from the gauge-invariant Hessian eigenvalues.
- **Step 3** runs the pseudo-spectral Cahn-Hilliard simulation, with optional elasto-chemical coupling and optional GPU execution.
- **Step 4** post-processes simulation snapshots and extracts composition fields.
- **Steps 5 and 6** are verification stages: Step 5 validates the thermodynamic tables and derivative methods against direct `pycalphad` calculations, and Step 6 compares chemical-only and elasto-chemical simulations of the same alloy.

The entire pipeline can be executed with a single command:

```bash
python workflows/run_all.py --config configs/nb_hf_ti_v_elastic_cNb25V25Hf25Ti25.yaml
```

![Workflow of `PhaseFoundry`. From a CALPHAD TDB (assessed or MLIP-generated via `PhaseForge`), SQS/MLIP elastic data, and a single YAML configuration file, `PhaseFoundry` samples the database with analytic Jansson derivatives, fits a free-energy surrogate, and maps spinodal regions through gauge-invariant Hessian eigenanalysis. The pseudo-spectral Cahn--Hilliard solver evolves the instability with an Eyre-stabilized spectral update, producing 2D and 3D spinodal microstructures, composition fields, and stress/strain fields when elasto-chemical coupling is enabled. Verification stages (Steps 5--6) validate thermodynamic tables and derivatives against direct `pycalphad` calculations and compare chemical-only against elasto-chemical simulations.](figs/PhaseFoundry.pdf)

## 2.2 Package architecture

The package separates concerns into three subpackages. `calphad_pf.thermodynamics` handles everything upstream of the solver: TDB sampling, HDF5 storage, Hessian eigenanalysis, spinodal boundary extraction, and the interpolant surrogate. `calphad_pf.phasefield` contains the Cahn-Hilliard solver, the Eyre-stabilized spectral update, FFT operations (with a NumPy/CuPy backend switch), and visualization utilities. `calphad_pf.elasticity` implements the eigenstrain models, the SQS elastic-constant interpolant, and the Lippmann-Schwinger spectral solver for mechanical equilibrium. The dependency stack is deliberately shallow: NumPy, SciPy, h5py, scikit-learn, Matplotlib, and `pycalphad`, with CuPy as an optional extra for GPU acceleration.

Because the solver consumes only the surrogate interface, not the TDB directly, the thermodynamic sampling (minutes to hours, embarrassingly parallel over grid points) and the simulation (which evaluates the surrogate millions of times) are decoupled. A new alloy system requires only a new TDB and a new YAML file.

# 3 Numerical methods and implementation

## 3.1 CALPHAD thermodynamics with analytic derivatives

For an $n$-component solid solution phase $\alpha$ with the $n$-th component chosen as the dependent variable, local stability against composition fluctuations is governed by the Hessian of the molar Gibbs energy, $H_{ij} = \partial^2 G_m^\alpha / \partial x_i \partial x_j$: the phase is spinodally unstable wherever at least one eigenvalue of $H$ is negative [@cahn1961spinodal; @morral2021stability]. The first derivatives are simple combinations of chemical potentials, $\partial G_m^\alpha / \partial x_i = \mu_i - \mu_n$, but the second derivatives require differentiating the outputs of constrained equilibrium calculations, which for sublattice models involve internal degrees of freedom. `PhaseFoundry` computes these derivatives analytically with the Jansson derivative method [@kunselman2024analytically] as implemented in `pycalphad`, avoiding numerical differentiation entirely. The user can also select `finite_diff` or `hybrid` derivative modes through the configuration file, which is useful both for older `pycalphad` versions and for the validation studies in Step 5.

Raw Hessian eigenvalues depend on the arbitrary choice of dependent component because composition axes on the Gibbs simplex are not orthogonal. Following @kadirvel2022exploration, the Hessian is transformed to a Cartesian basis before eigenanalysis, yielding gauge-invariant eigenvalues and unstable-fluctuation directions. Spinodal boundaries are then extracted as zero-crossing contours of the smallest eigenvalue, and for quaternary and higher systems the resulting instability maps are displayed with the affine projection technique of @vela2025visualizing.

Because a phase-field simulation requires thermodynamic quantities at arbitrary compositions, not just grid points, Step 2 wraps the HDF5 tables in a scattered-data interpolant surrogate providing $G_m^\alpha$, the relative chemical potentials, and the Hessian as smooth functions of composition. The surrogate is fit once per system and temperature and is the only thermodynamic object the solver touches at run time.

## 3.2 Phase-field solver

The solver evolves $n-1$ independent composition fields $x_i(\mathbf{r}, t)$ under the free energy functional

$$\mathcal{F}[\mathbf{x}] = \int_\Omega \left[ G_m^\alpha(\mathbf{x}) + \sum_{i=1}^{n-1} \frac{\kappa_i}{2} |\nabla x_i|^2 \right] dV,$$

where $\kappa_i$ is the gradient energy coefficient of field $i$, via the coupled Cahn-Hilliard equations

$$\frac{\partial x_i}{\partial t} = \nabla \cdot \sum_{j=1}^{n-1} M_{ij}(\mathbf{x}) \, \nabla \mu_j^{\mathrm{tot}}, \qquad \mu_j^{\mathrm{tot}} = \frac{\partial G_m^\alpha}{\partial x_j} - \kappa_j \nabla^2 x_j + \mu_j^{\mathrm{el}},$$

with $\mu_j^{\mathrm{el}}$ the optional elastic contribution described below. The mobility is degenerate, $M_{ij}(\mathbf{x}) = \mathring{M}_{ij} \prod_{k=1}^{n} x_k$, so that diffusion vanishes as any component approaches a pure-element limit, and the bare mobility matrix $\mathring{M}_{ij}$ is supplied in the configuration file, allowing strongly asymmetric diffusivities (for example, slow Hf diffusion relative to Nb at 500 °C) to be represented directly.

The equations are discretized on a uniform periodic grid and advanced with a semi-implicit pseudo-spectral scheme [@chen2002phase]. Inside the spinodal region the diagonal Hessian entries are negative, and the corresponding explicit bulk term is destabilizing at large time steps. Following @eyre1998unconditionally, the molar Gibbs energy is split into convex and concave parts, with the convex quadratic part $\frac{A_i}{2} x_i^2$ treated implicitly. The Fourier-space update for field $i$ is

$$\hat{x}_i^{\,n+1} = \frac{\hat{x}_i^{\,n} + \Delta t\,k^2 \sum_j M^{\mathrm{eff}}_{ij}\,\hat{\mu}_{j,\mathrm{bulk}}^{\,n} - \Delta t\,k^2 M^{\mathrm{eff}}_{ii} A_i \hat{x}_i^{\,n} - \Delta t\,k^4 \sum_{j \neq i} M^{\mathrm{eff}}_{ij}\,\kappa_j \hat{x}_j^{\,n}}{1 + \Delta t\,k^2 M^{\mathrm{eff}}_{ii} A_i + \Delta t\,k^4 M^{\mathrm{eff}}_{ii}\,\kappa_i},$$

where $k$ is the wavenumber magnitude, $M^{\mathrm{eff}}_{ij}$ is the spatially averaged effective mobility, and $A_i \geq \max(0, -H_{ii}(\mathbf{x}_0))$ is the per-component stabilization constant evaluated from the surrogate Hessian at the mean composition $\mathbf{x}_0$. The denominator is strictly positive for any $\Delta t > 0$, making the scheme unconditionally stable; the per-component choice of $A_i$ matters in multicomponent systems where the diagonal curvatures of different fields can differ by orders of magnitude, since over-stabilizing a weakly unstable field would artificially retard its decomposition kinetics. When `dt: null` is specified, the solver estimates a stable, accuracy-limited time step from the largest Hessian eigenvalue at $\mathbf{x}_0$.

The same code path runs in two and three dimensions. All FFT and array operations dispatch through a backend layer, so setting `use_gpu: true` moves the entire simulation to a CUDA device via CuPy with no solver code changes. Long runs write incremental NPZ checkpoints, allowing simulations of several million steps to be interrupted and resumed.

## 3.3 Elasto-chemical coupling

Coherency strain arising from composition-dependent lattice mismatch penalizes decomposition and biases morphology. When elastic coupling is enabled, mechanical equilibrium is solved on the current composition field with the fixed-point spectral scheme of @moulinec1998numerical, which treats the heterogeneous stiffness field as a perturbation around a homogeneous reference medium and iterates a Lippmann-Schwinger equation in Fourier space. The resulting stress field defines the elastic chemical potential $\mu_k^{\mathrm{el}}$ added to the driving force, following the formulation of @gururajan2007phase.

Two eigenstrain models are implemented. The classical model (`fixed_vegard`) uses Vegard coefficients and endpoint stiffness tensors fixed at the CALPHAD tie-line endpoints computed in Step 1c. The composition-dependent model (`local_vegard`) instead evaluates the eigenstrain pointwise as the lattice mismatch relative to the parent composition, $\varepsilon^*(\mathbf{r}) = [a(\mathbf{x}(\mathbf{r})) - a_0]/a_0$, with the lattice parameter $a(\mathbf{x})$ and the local stiffness tensor $\mathbf{C}(\mathbf{x}(\mathbf{r}))$ interpolated across the full composition space from SQS/MLIP tables. The local model has the physically correct property that the eigenstrain vanishes identically in the initial homogeneous alloy and grows only as decomposition proceeds. Elastic constants from several MLIPs (ORB-v2, GRACE-2L) are shipped as data tables, and the CSV interface makes substituting other potentials straightforward.

Because the elastic solve is the most expensive step, it is performed every $n_{\mathrm{el}}$ time steps (configurable, `solve_every`) with warm starting from the previous solution, and the elastic potential is cached between solves. This is a controlled approximation: composition fields evolve slowly relative to the elastic relaxation implied by the quasi-static formulation, and the interval can be reduced to check convergence.

## 3.4 Verification and validation tooling

Two properties of the framework make its predictions auditable. First, Step 5 regenerates thermodynamic quantities directly from `pycalphad` at randomly sampled compositions and compares them against the surrogate, quantifying both interpolation error and the discrepancy between Jansson, finite-difference, and hybrid derivative modes; Figure 2 shows this comparison for the Nb-V system at 500 °C, where the three derivative modes agree closely on the molar Gibbs energy, relative chemical potential, and Hessian across the composition range, including the spinodal region. Second, Step 6 runs matched chemical-only and elasto-chemical simulations from identical initial noise realizations (the random seed is part of the configuration), isolating the effect of coherency strain on the decomposition path. The repository also includes a pytest suite covering the thermodynamic sampling, surrogate fitting, spectral operators, and elasticity solver.

![Step 5 validation output for the Nb-V BCC_A2 system at 500 °C: (a) molar Gibbs energy, (b) relative chemical potential $\mu_{\mathrm{Nb}} - \mu_{\mathrm{V}}$, and (c) Gibbs energy Hessian, each computed with finite-difference, direct Jansson, and hybrid derivative modes and overlaid on the sampled CALPHAD data. The shaded band marks the spinodal region where the Hessian is negative.](figs/derivative_method_comparison.png)

# 4 Illustrative applications

The configurations shipped with the repository span the four supported modes and reproduce the workflow of the companion study.

**Binary Nb-V.** The Nb-V BCC_A2 system, which exhibits a well-characterized miscibility gap, serves as the reference case. Chemical-only simulations at 500 °C across mean compositions $x_{\mathrm{Nb}} = 0.3$ to $0.8$ show the expected progression from isolated precipitation-like morphologies near the spinodal boundary to bicontinuous interconnected structures at the center of the gap, in both 2D ($128^2$) and 3D ($128^3$) domains. Figure 3 shows a representative equiatomic microstructure after decomposition and initial coarsening. Adding elastic coupling visibly aligns and refines the modulations and delays coarsening, consistent with classical expectations for coherent spinodals.

![Spinodally decomposed microstructure of equiatomic Nb$_{50}$V$_{50}$ at 500 °C after $t = 800$ s (Step 4 output of the chemical-only pipeline): Nb (left) and V (right) composition fields on a 1 µm domain, showing Nb-rich domains in a V-rich matrix with near-pure-element phase compositions consistent with the CALPHAD tie-line at this temperature.](figs/nbv_500C_micrograph.pdf)

**Ternary Nb-Hf-V.** With two independent composition fields, anisotropic mobilities (Hf roughly two orders of magnitude slower than Nb at 500 °C), and distinct gradient energy coefficients per field, the ternary configurations demonstrate the multicomponent capabilities: the two fields decompose on different time scales, and the gauge-invariant stability analysis of Step 2 maps the composition-dependent instability directions across the Gibbs triangle.

**Quaternary Hf-Nb-Ti-V.** The equiatomic Nb25V25Hf25Ti25 elasto-chemical configuration is the flagship case, running three million time steps with three coupled fields, MLIP-derived composition-dependent stiffness, and tie-line endpoints informed by the experimental XRD phase compositions of @an2021spinodal. The predicted decomposition into a Nb/Ti-rich matrix and a Hf/V-rich second phase, and the influence of coherency strain on the resulting length scale and morphology, are analyzed in detail in the companion paper [@kunselman2025spinodal].

Across these cases the practical point is uniformity: the same package, solver, and post-processing scripts handle every system, and moving from a binary chemical 2D run to a quaternary elasto-chemical run changes only the YAML file.

# 5 Relation to existing software

`PhaseFoundry` is complementary to, not a replacement for, general-purpose phase-field frameworks. MOOSE [@permann2020moose] and PRISMS-PF [@dewitt2020prisms] offer massively parallel finite element infrastructure, adaptive meshing, and broad multiphysics coverage, but leave the construction of multicomponent CALPHAD free energy models, their derivatives, and consistent elastic parameterizations to the user. FiPy [@guyer2009fipy] provides flexible PDE specification in Python at a lower performance ceiling. Commercial couplings between CALPHAD engines and phase-field codes exist (for example, MICRESS with Thermo-Calc via the TQ interface), but are closed-source and not readily extensible to MLIP-generated databases. `PhaseFoundry` occupies the specific niche of open, reproducible, TDB-to-microstructure spinodal simulation: it trades geometric and multiphysics generality (periodic spectral grids, a single solid solution phase, Cahn-Hilliard dynamics) for a fully automated and thermodynamically exact pipeline. Users needing conserved-order-parameter spinodal predictions grounded in a CALPHAD database get a complete workflow; users needing complex geometries, polycrystals, or coupled non-conserved fields are better served by the general frameworks, potentially reusing the surrogate tables `PhaseFoundry` generates.

# 6 Limitations and future directions

Several limitations delimit the current scope. The solver assumes a single solid solution phase decomposing coherently; nucleation of distinct crystal structures and interfacial defects are outside the model. The spectral discretization requires periodic domains and uniform grids. Thermodynamic sampling and surrogate fitting are performed at fixed temperature, so simulations are isothermal; non-isothermal treatments would require temperature-parameterized surrogates, which the HDF5 schema already anticipates by storing multiple temperatures. The elastic solver neglects the stiffness-contrast energy correction, standard practice at moderate contrast, and the `local_vegard` model evaluates Vegard coefficients at the parent composition as a first-order approximation. Mobility matrices are user-supplied constants scaled by the degenerate prefactor; coupling to CALPHAD kinetic (mobility) databases is a natural next step. Planned developments include temperature-stepped annealing schedules, DICTRA-style mobility database ingestion, and extension of the stability analysis to quinary systems, where the affine projection visualization already applies.

# 7 Conclusions

`PhaseFoundry` automates the path from a CALPHAD thermodynamic database to quantitative spinodal microstructure predictions in multicomponent alloys, combining analytic Jansson derivatives, gauge-invariant stability analysis, an unconditionally stable spectral Cahn-Hilliard solver, and MLIP-informed elasto-chemical coupling behind a single YAML configuration interface. Together with `PhaseForge`, it completes an open-source chain from machine-learned interatomic potentials to microstructure, demonstrated on the Hf-Nb-Ti-V refractory HEA system in the companion paper [@kunselman2025spinodal]. We expect the framework to be useful both as a production tool for spinodal alloy design and as a reference implementation for the broader CALPHAD-coupled phase-field community.

# Data and code availability

`PhaseFoundry` is available under the MIT license at https://github.com/vahid2364/PhaseFoundry, including all YAML configurations, workflow scripts, SQS/MLIP elastic constant tables, and post-processing scripts used in this paper and the companion study. Thermodynamic databases for the Hf-Nb-Ti-V system were generated with `PhaseForge` [@sariturk2025phaseforge], archived at DOI 10.5281/zenodo.15730911.

# Author contributions

V.A.: conceptualization, software, methodology, validation, visualization, writing (original draft). R.A.: conceptualization, funding acquisition, supervision, writing (review and editing).

# Conflicts of interest

There are no conflicts of interest to declare.

# Acknowledgements

The authors acknowledge support from the Army Research Office (Grant No. W911NF-22-2-0117) and the National Science Foundation. Simulations were carried out at the Texas A&M High-Performance Research Computing (HPRC) Facility. The authors thank Courtney Kunselman, Doğuhan Sarıtürk, and Siya Zhu for discussions and for their contributions to the companion scientific study.

# References
