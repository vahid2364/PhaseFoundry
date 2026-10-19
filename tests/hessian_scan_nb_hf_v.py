"""
Hessian scan: Nb50+xHf, V=0.50-xHf  over temperatures 500–1000°C
==================================================================
For each (T, xHf) point computes:
  - H_NB  = ∂²G/∂x_Nb²
  - H_HF  = ∂²G/∂x_Hf²
  - H_NbHf = ∂²G/∂x_Nb∂x_Hf  (off-diagonal)
  - lambda_min = minimum eigenvalue of Hessian matrix
  - spinodal   = True if lambda_min < 0

Output saved to:  tests/results_hessian_scan_nb_hf_v/
  - hessian_scan.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from phasefoundry.thermodynamics.interpolant import build_interpolant
from phasefoundry.thermodynamics.storage import load_table

OUT_DIR = _ROOT / "tests" / "results_hessian_scan_nb_hf_v"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ELEMENTS  = ["NB", "HF", "V"]
REF_EL    = "V"
X_NB      = 0.50
X_HF_VALS = [0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.15, 0.20]
TEMPS_C   = [500, 600, 700, 800, 900, 1000]

table_dir = _ROOT / "data" / "nb_hf_v"
tables    = sorted(table_dir.glob("*.h5"))

rows = []

for T_target in TEMPS_C:
    T_K_target = T_target + 273.15
    best = min(tables, key=lambda p: abs(
        float(load_table(p)["metadata"]["temperature"]) - T_K_target))
    T_K = float(load_table(best)["metadata"]["temperature"])
    T_C = T_K - 273.15

    print(f"\nT = {T_C:.0f} °C  ({best.name})")

    interp = build_interpolant(
        [best], elements=ELEMENTS, reference_element=REF_EL,
        temperature_K=T_K, derivative_method="jansson")

    for xhf in X_HF_VALS:
        xv = 1.0 - X_NB - xhf
        if xv <= 0:
            continue

        c = np.array([[[X_NB]], [[xhf]]])
        try:
            H = interp.d2g(c)[:, :, 0, 0]   # (2, 2)
            H_nb   = float(H[0, 0])
            H_hf   = float(H[1, 1])
            H_nbhf = float(H[0, 1])
            eigvals = np.linalg.eigvalsh(H)
            lam_min = float(eigvals.min())
            lam_max = float(eigvals.max())
            spinodal = lam_min < 0
        except Exception as e:
            print(f"  xHf={xhf:.2f} ERROR: {e}")
            H_nb = H_hf = H_nbhf = lam_min = lam_max = float("nan")
            spinodal = None

        rows.append(dict(
            T_C      = T_C,
            T_K      = T_K,
            x_Nb     = X_NB,
            x_Hf     = xhf,
            x_V      = round(xv, 4),
            H_NB     = round(H_nb,   1),
            H_HF     = round(H_hf,   1),
            H_NbHf   = round(H_nbhf, 1),
            lam_min  = round(lam_min, 1),
            lam_max  = round(lam_max, 1),
            spinodal = spinodal,
        ))
        flag = "SPINODAL" if spinodal else "stable"
        print(f"  xHf={xhf:.2f}  H_NB={H_nb:10.0f}  H_HF={H_hf:10.0f}"
              f"  lam_min={lam_min:10.0f}  [{flag}]")

df = pd.DataFrame(rows)
csv_path = OUT_DIR / "hessian_scan.csv"
df.to_csv(csv_path, index=False)
print(f"\nSaved: {csv_path}  ({len(df)} rows)")
print(df.to_string(index=False))
