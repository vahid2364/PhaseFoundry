# Thermodynamic Database (TDB) Files

All TDB files follow the naming convention:

```
{elements_alphabetical}_{phase}_{source}_{version}.tdb
```

---

## Files

### `HfNbTiV_BCC_PhaseForge_v1_Tindep.tdb`

| Field | Value |
|---|---|
| System | Hf-Nb-Ti-V |
| Phase | BCC_A2 |
| Source | PhaseForge (MLIP-derived CALPHAD) |
| Version | v1 |
| R-K parameters | **Temperature-independent** — constant L(i,j;k) values |
| Notes | Earlier fit. Simpler model; interaction parameters do not vary with T. |

### `HfNbTiV_BCC_PhaseForge_v2_Tdep.tdb`

| Field | Value |
|---|---|
| System | Hf-Nb-Ti-V |
| Phase | BCC_A2 |
| Source | PhaseForge (MLIP-derived CALPHAD) |
| Version | v2 |
| R-K parameters | **Temperature-dependent** — L parameters contain exponential T terms |
| Notes | Newer fit. Includes T-dependent interaction parameters fitted via PhaseForge. Unary FUNC0008–0011 also include a linear T term. This is the **recommended** TDB for new simulations. |

---

## Key differences between v1 and v2

| | v1 (T-independent) | v2 (T-dependent) |
|---|---|---|
| Unary terms FUNC0008–0011 | Constants | Linear in T |
| Binary L(i,j;0) parameters | Scalar constants | Exponential-T series |
| Higher-order L(i,j;2) terms | Absent | Present |
| Temperature sensitivity of G_m | Weaker | Stronger, more physical |

---

## Usage in configs

Reference TDB files by relative path from the repository root:

```yaml
system:
  db_path: data/tdb/HfNbTiV_BCC_PhaseForge_v2_Tdep.tdb
```
