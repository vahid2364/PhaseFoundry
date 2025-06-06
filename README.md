PROOJECT PhaseFoundry: 
====================================
- A “foundry” for phase-field, CALPHAD, and ML interatomic potentials 

Description:
-------------
This notebook implements a thermodynamically consistent phase-field model for simulating spinodal decomposition and microstructural evolution in multicomponent alloy systems. The framework is coupled with pyCALPHAD to incorporate CALPHAD-based free energies directly from thermodynamic databases, enabling accurate thermodynamic representation. First-order derivatives of the free energy are computed using a semi-analytical approach to improve numerical precision. The thermodynamic descriptions are derived from databases constructed using machine learning–based interatomic potentials, allowing flexible and data-driven thermodynamic integration.

Features:
- Multicomponent Cahn–Hilliard equation using pseudo-spectral method
- Periodic boundary conditions
- Supports both idealized and CALPHAD-based free energy descriptions

Contributors:
-------------
- From Thermodynamic Potentials to Microstructure Evolution: Vahid Attari, Ph.D., TAMU (attari.v@tamu.edu)
  - Semi-implicit solver for Cahn-Hillaird Model coupled to semi-analytic derivatives of thermodynamic properties
- From Thermodynamic description to Chemical Potentials: Courtney Kunselman, TAMU (cjkunselman18@tamu.edu)
  - Semi-analytic derivatives of thermodynamic potentials
- From Phase to Thermodynamic Description: Doguhan Sariturk, TAMU (sariturk@tamu.edu)
  - Accelerating CALPHAD-based Phase Diagram Predictions Using machine learning interatomic potentials (MLIPs)
  
Model Name:
------------
CALPHAD-Reinforced Microstructure Modeling using Cahn–Hilliard Phase-Field Model

License:
---------
MIT License (feel free to reuse and modify with attribution)
