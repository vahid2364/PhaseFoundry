"""
Plot A_Z vs. x_V along three compositional paths, Nb(50-x)Vx,
Nb50Hf(50-x)Vx and Nb25Hf25VxTi(50-x), for the ORB-v2 MLIP database.
"""
import pathlib
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from scipy.interpolate import RBFInterpolator, UnivariateSpline

matplotlib.rcParams.update({'font.family': 'sans-serif', 'font.size': 9})

ROOT = pathlib.Path(__file__).parent.parent.parent
FIGS = ROOT / 'paper' / '2025_Kunselman_Design_of_Spinodal_Microstructures_in_HEAs' / 'figs'

def load_rbf(csv_path):
    d = pd.read_csv(csv_path)
    d.columns = [c.lower() for c in d.columns]
    coord_cols = ['x_hf','x_nb','x_v','x_ti'] if 'x_hf' in d.columns else ['hf','nb','v','ti']
    p = d[coord_cols].values
    return {k: RBFInterpolator(p, d[k].values, kernel='linear')
            for k in ('c11_gpa','c12_gpa','c44_gpa')}

def az_along_path(rbf, c_tern, c_quat, n=300):
    f    = np.linspace(0, 1, n)
    comp = np.outer(1-f, c_tern) + np.outer(f, c_quat)   # [xHf, xNb, xV, xTi]
    C11  = rbf['c11_gpa'](comp)
    C12  = rbf['c12_gpa'](comp)
    C44  = rbf['c44_gpa'](comp)
    AZ   = C44 / ((C11 - C12) / 2)
    return comp[:, 3], AZ   # xTi, AZ

# Path A: Nb25Hf25V(x)Ti(50-x), x_V from 0 to 0.50  [xHf, xNb, xV, xTi]
xti_range  = np.linspace(0, 0.50, 300)          # kept for compatibility
xv_quat    = np.linspace(0, 0.50, 300)           # x_V increases, x_Ti decreases
comps_A    = np.column_stack([
    np.full(300, 0.25),
    np.full(300, 0.25),
    xv_quat,
    0.50 - xv_quat,
])

# Path B: Nb50Hf(50-x)Vx, x from 0 to 0.50  [xHf, xNb, xV, xTi]
xv_range   = np.linspace(0, 0.50, 300)
comps_B    = np.column_stack([
    0.50 - xv_range,
    np.full(300, 0.50),
    xv_range,
    np.zeros(300),
])

# Path C: Nb(50-x)Vx, x from 0 to 0.50 mole fraction  [xHf, xNb, xV, xTi]
xv_bin     = np.linspace(0, 0.50, 300)        # mole fraction
xv_at      = xv_bin * 100.0                   # at.% (kept for compatibility)
comps_C    = np.column_stack([
    np.zeros(300),
    (50 - xv_at) / 100.0,
    xv_bin,
    np.zeros(300),
])

VERSIONS = {
    'ORB-v2': {
        'csv':   ROOT / 'data/HfNbTiV_BCC_PhaseForge_v2/sqs_elastic_constants_HfNbVTi-orb-v2.csv',
        'color': '#1f77b4',
        'ls':    '-',
    },
}

def load_binary_points(csv_path):
    """Return (xV mole fraction, A_Z) for NbV binary edge (Hf=Ti=0), xV<=0.50."""
    d = pd.read_csv(csv_path)
    d.columns = [c.lower() for c in d.columns]
    hf_col = 'x_hf' if 'x_hf' in d.columns else 'hf'
    ti_col = 'x_ti' if 'x_ti' in d.columns else 'ti'
    v_col  = 'x_v'  if 'x_v'  in d.columns else 'v'
    b = d[(d[hf_col] == 0) & (d[ti_col] == 0)].sort_values(v_col)
    b = b[b[v_col] <= 0.50]
    AZ = b['c44_gpa'].values / ((b['c11_gpa'].values - b['c12_gpa'].values) / 2)
    return b[v_col].values, AZ   # mole fraction

CURVES = [
    # (comps, x_param, label, color, ls, is_binary)
    (comps_C, xv_bin,     r'Nb$_{50-x}$V$_{x}$  ($x_{\rm V}$)',             '#2ca02c', '--',  True),
    (comps_B, xv_range,   r'Nb$_{50}$Hf$_{50-x}$V$_{x}$  ($x_{\rm V}$)',   '#d62728', '-.',  False),
    (comps_A, xv_quat,    r'Nb$_{25}$Hf$_{25}$V$_{x}$Ti$_{50-x}$  ($x_{\rm V}$)',  '#1f77b4', '-', False),
]

fig, ax = plt.subplots(figsize=(5.5, 4.0))

for label, cfg in VERSIONS.items():
    for comps, x_range, curve_label, color, ls, is_binary in CURVES:
        if is_binary:
            xv, AZ = load_binary_points(cfg['csv'])
            mask = AZ > 0
            spl = UnivariateSpline(xv[mask], AZ[mask], s=0.05, k=3)
            xv_dense = np.linspace(xv[mask].min(), xv[mask].max(), 300)
            ax.plot(xv_dense, spl(xv_dense), color=color, lw=1.8, ls=ls, label=curve_label)
        else:
            rbf = load_rbf(cfg['csv'])
            C11 = rbf['c11_gpa'](comps)
            C12 = rbf['c12_gpa'](comps)
            C44 = rbf['c44_gpa'](comps)
            AZ  = C44 / ((C11 - C12) / 2)
            ax.plot(x_range, AZ, color=color, lw=1.8, ls=ls, label=curve_label)
            cross = np.where(np.diff(np.sign(AZ - 1)))[0]
            if len(cross):
                xc = x_range[cross[0]]
                ax.axvline(xc, color=color, lw=0.7, ls=':', alpha=0.6)

ax.axhline(1.0, color='black', lw=0.8, ls='--')
ax.text(0.99, 1.02, r'$\langle 100\rangle$ favored', fontsize=7.5,
        transform=ax.get_yaxis_transform(), va='bottom', ha='right')
ax.text(0.99, 0.97, r'$\langle 110\rangle$ favored', fontsize=7.5,
        transform=ax.get_yaxis_transform(), va='top', ha='right')

ax.set_xlabel(r'$x_{\rm V}$ (mole fraction)', fontsize=9)
ax.set_ylabel(r'Zener anisotropy $A_Z = C_{44}/C\,^{\prime}$', fontsize=9)
ax.set_xlim(-0.01, 0.51)
ax.set_ylim(0.2, 2.2)
ax.spines[['top', 'right']].set_visible(False)
ax.legend(fontsize=7.5, frameon=False, loc='upper left')

fig.tight_layout()

if __name__ == '__main__':
    for ext in ('pdf', 'png'):
        out = FIGS / f'az_vs_ti.{ext}'
        fig.savefig(out, dpi=200, bbox_inches='tight', facecolor='white')
        print('saved', out)
    plt.close(fig)
