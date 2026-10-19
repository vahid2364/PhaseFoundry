"""Zener anisotropy A_Z against x_V along three composition paths, from the ORB-v2 SQS table.

The paths are Nb(50-x)V(x), Nb50Hf(50-x)V(x) and Nb25Hf25V(x)Ti(50-x). The Nb-V path is a
smoothing spline through the binary edge of the table; the other two use linear RBF
interpolants of C11, C12 and C44 and mark where A_Z crosses 1 with a dotted line. The
figure is built when the module is imported, and running the script saves it to
``FIGS/az_vs_ti.pdf`` and ``FIGS/az_vs_ti.png``.

```bash
python workflows/07_ternary_figure/plot_az_vs_ti.py
```
"""

import pathlib

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import RBFInterpolator, UnivariateSpline

mpl.rcParams.update({"font.family": "sans-serif", "font.size": 9})

ROOT = pathlib.Path(__file__).parent.parent.parent
FIGS = ROOT / "paper" / "2025_Kunselman_Design_of_Spinodal_Microstructures_in_HEAs" / "figs"


def load_rbf(csv_path):
    """Build linear RBF interpolants of C11, C12 and C44 from an SQS elastic-constant table.

    Args:
        csv_path: Table CSV with Hf, Nb, V and Ti composition columns.

    Returns:
        Interpolants keyed by ``c11_gpa``, ``c12_gpa`` and ``c44_gpa``, taking compositions
        ordered as x_Hf, x_Nb, x_V, x_Ti.
    """
    d = pd.read_csv(csv_path)
    d.columns = [c.lower() for c in d.columns]
    coord_cols = ["x_hf", "x_nb", "x_v", "x_ti"] if "x_hf" in d.columns else ["hf", "nb", "v", "ti"]
    p = d[coord_cols].values
    return {k: RBFInterpolator(p, d[k].values, kernel="linear") for k in ("c11_gpa", "c12_gpa", "c44_gpa")}


def az_along_path(rbf, c_tern, c_quat, n=300):
    """Return x_Ti and A_Z along the straight line from ``c_tern`` to ``c_quat``.

    Args:
        rbf: Interpolants from ``load_rbf``.
        c_tern: Start composition, ordered as x_Hf, x_Nb, x_V, x_Ti.
        c_quat: End composition, in the same order.
        n: Number of points along the line.

    Returns:
        x_Ti and A_Z = C44 / ((C11 - C12) / 2) at each point.
    """
    f = np.linspace(0, 1, n)
    comp = np.outer(1 - f, c_tern) + np.outer(f, c_quat)
    C11 = rbf["c11_gpa"](comp)
    C12 = rbf["c12_gpa"](comp)
    C44 = rbf["c44_gpa"](comp)
    AZ = C44 / ((C11 - C12) / 2)
    return comp[:, 3], AZ


# composition columns follow the RBF coordinates: x_Hf, x_Nb, x_V, x_Ti
xti_range = np.linspace(0, 0.50, 300)
xv_quat = np.linspace(0, 0.50, 300)
comps_a = np.column_stack(
    [
        np.full(300, 0.25),
        np.full(300, 0.25),
        xv_quat,
        0.50 - xv_quat,
    ]
)

xv_range = np.linspace(0, 0.50, 300)
comps_b = np.column_stack(
    [
        0.50 - xv_range,
        np.full(300, 0.50),
        xv_range,
        np.zeros(300),
    ]
)

xv_bin = np.linspace(0, 0.50, 300)
xv_at = xv_bin * 100.0
comps_c = np.column_stack(
    [
        np.zeros(300),
        (50 - xv_at) / 100.0,
        xv_bin,
        np.zeros(300),
    ]
)

VERSIONS = {
    "ORB-v2": {
        "csv": ROOT / "data/HfNbTiV_BCC_PhaseForge_v2/sqs_elastic_constants_HfNbVTi-orb-v2.csv",
        "color": "#1f77b4",
        "ls": "-",
    },
}


def load_binary_points(csv_path):
    """Return x_V and A_Z on the Nb-V edge of an SQS table, where x_Hf = x_Ti = 0 and x_V ≤ 0.5.

    Args:
        csv_path: Table CSV with Hf, Nb, V and Ti composition columns.

    Returns:
        x_V sorted in ascending order and A_Z = C44 / ((C11 - C12) / 2) at those points.
    """
    d = pd.read_csv(csv_path)
    d.columns = [c.lower() for c in d.columns]
    hf_col = "x_hf" if "x_hf" in d.columns else "hf"
    ti_col = "x_ti" if "x_ti" in d.columns else "ti"
    v_col = "x_v" if "x_v" in d.columns else "v"
    b = d[(d[hf_col] == 0) & (d[ti_col] == 0)].sort_values(v_col)
    b = b[b[v_col] <= 0.50]
    AZ = b["c44_gpa"].values / ((b["c11_gpa"].values - b["c12_gpa"].values) / 2)
    return b[v_col].values, AZ


CURVES = [
    (comps_c, xv_bin, r"Nb$_{50-x}$V$_{x}$  ($x_{\rm V}$)", "#2ca02c", "--", True),
    (comps_b, xv_range, r"Nb$_{50}$Hf$_{50-x}$V$_{x}$  ($x_{\rm V}$)", "#d62728", "-.", False),
    (comps_a, xv_quat, r"Nb$_{25}$Hf$_{25}$V$_{x}$Ti$_{50-x}$  ($x_{\rm V}$)", "#1f77b4", "-", False),
]

fig, ax = plt.subplots(figsize=(5.5, 4.0))

for _label, cfg in VERSIONS.items():
    for comps, x_range, curve_label, color, ls, is_binary in CURVES:
        if is_binary:
            xv, AZ = load_binary_points(cfg["csv"])
            mask = AZ > 0
            spl = UnivariateSpline(xv[mask], AZ[mask], s=0.05, k=3)
            xv_dense = np.linspace(xv[mask].min(), xv[mask].max(), 300)
            ax.plot(xv_dense, spl(xv_dense), color=color, lw=1.8, ls=ls, label=curve_label)
        else:
            rbf = load_rbf(cfg["csv"])
            C11 = rbf["c11_gpa"](comps)
            C12 = rbf["c12_gpa"](comps)
            C44 = rbf["c44_gpa"](comps)
            AZ = C44 / ((C11 - C12) / 2)
            ax.plot(x_range, AZ, color=color, lw=1.8, ls=ls, label=curve_label)
            cross = np.where(np.diff(np.sign(AZ - 1)))[0]
            if len(cross):
                xc = x_range[cross[0]]
                ax.axvline(xc, color=color, lw=0.7, ls=":", alpha=0.6)

ax.axhline(1.0, color="black", lw=0.8, ls="--")
ax.text(0.99, 1.02, r"$\langle 100\rangle$ favored", fontsize=7.5, transform=ax.get_yaxis_transform(), va="bottom", ha="right")
ax.text(0.99, 0.97, r"$\langle 110\rangle$ favored", fontsize=7.5, transform=ax.get_yaxis_transform(), va="top", ha="right")

ax.set_xlabel(r"$x_{\rm V}$ (mole fraction)", fontsize=9)
ax.set_ylabel(r"Zener anisotropy $A_Z = C_{44}/C\,^{\prime}$", fontsize=9)
ax.set_xlim(-0.01, 0.51)
ax.set_ylim(0.2, 2.2)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(fontsize=7.5, frameon=False, loc="upper left")

fig.tight_layout()

if __name__ == "__main__":
    for ext in ("pdf", "png"):
        out = FIGS / f"az_vs_ti.{ext}"
        fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
        print("saved", out)
    plt.close(fig)
