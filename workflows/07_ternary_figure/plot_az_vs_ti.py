"""
Plot A_Z vs. Ti fraction along the compositional path from
ternary Nb70Hf10V20 matrix to the quaternary matrix composition,
for three ORB-MLIP database versions.
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

# Path A: Nb25Hf25V(50-x)Ti(x), x from 0 to 0.50  [xHf, xNb, xV, xTi]
xti_range  = np.linspace(0, 0.50, 300)
comps_A    = np.column_stack([
    np.full(300, 0.25),
    np.full(300, 0.25),
    0.50 - xti_range,
    xti_range,
])

# Path B: Nb50Hf(50-x)Vx, x from 0 to 0.50  [xHf, xNb, xV, xTi]
xv_range   = np.linspace(0, 0.50, 300)
comps_B    = np.column_stack([
    0.50 - xv_range,
    np.full(300, 0.50),
    xv_range,
    np.zeros(300),
])

# Path C: Nb(50-x)Vx, x in at.% from 0 to 50  [xHf, xNb, xV, xTi]
xv_at      = np.linspace(0, 50, 300)          # at.%
xv_bin     = xv_at / 100.0                    # mole fractions for interpolator
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

fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.0))

panels = [
    (comps_C, xv_at,     r'$x_{\rm V}$  (at.\%)  in  Nb$_{50-x}$V$_{x}$',            '(a)', (-0.5,  51.0),  (0.2, 0.7)),
    (comps_B, xv_range,  r'$x_{\rm V}$  in  Nb$_{50}$Hf$_{50-x}$V$_{x}$',           '(b)', (-0.01, 0.51), (0.2, 2.2)),
    (comps_A, xti_range, r'$x_{\rm Ti}$  in  Nb$_{25}$Hf$_{25}$V$_{50-x}$Ti$_{x}$', '(c)', (-0.01, 0.51), (0.2, 2.2)),
]

def load_binary_points(csv_path):
    """Return (xV_at%, A_Z) for NbV binary edge (Hf=Ti=0), limited to xV<=50 at%."""
    d = pd.read_csv(csv_path)
    d.columns = [c.lower() for c in d.columns]
    hf_col = 'x_hf' if 'x_hf' in d.columns else 'hf'
    ti_col = 'x_ti' if 'x_ti' in d.columns else 'ti'
    v_col  = 'x_v'  if 'x_v'  in d.columns else 'v'
    b = d[(d[hf_col] == 0) & (d[ti_col] == 0)].sort_values(v_col)
    b = b[b[v_col] <= 0.50]   # Nb-rich half only
    AZ = b['c44_gpa'].values / ((b['c11_gpa'].values - b['c12_gpa'].values) / 2)
    return b[v_col].values * 100, AZ   # return at.%

for ax, (comps, x_range, xlabel, panel, xlim, ylim) in zip(axes, panels):
    is_binary = panel == '(a)'
    for label, cfg in VERSIONS.items():
        if is_binary:
            xv, AZ = load_binary_points(cfg['csv'])
            # only use physically valid points (A_Z > 0)
            mask = AZ > 0
            spl  = UnivariateSpline(xv[mask], AZ[mask], s=0.05, k=3)
            xv_dense = np.linspace(xv[mask].min(), xv[mask].max(), 300)
            ax.plot(xv_dense, spl(xv_dense), color=cfg['color'], lw=1.8,
                    ls=cfg['ls'], label=label)
        else:
            rbf = load_rbf(cfg['csv'])
            C11 = rbf['c11_gpa'](comps)
            C12 = rbf['c12_gpa'](comps)
            C44 = rbf['c44_gpa'](comps)
            AZ  = C44 / ((C11 - C12) / 2)
            ax.plot(x_range, AZ, color=cfg['color'], lw=1.8, ls=cfg['ls'], label=label)
            cross = np.where(np.diff(np.sign(AZ - 1)))[0]
            if len(cross):
                xc = x_range[cross[0]]
                ax.axvline(xc, color=cfg['color'], lw=0.7, ls=':', alpha=0.5)

    if is_binary:
        ax.text(0.5, 0.92, r'$\langle 110\rangle$ favored throughout',
                fontsize=7.5, ha='center', transform=ax.transAxes,
                color='gray', style='italic')
    else:
        ax.axhline(1.0, color='black', lw=0.8, ls='--')
        ax.text(0.02, 1.03, r'$\langle 100\rangle$ favored', fontsize=7.5,
                transform=ax.get_yaxis_transform(), va='bottom')
        ax.text(0.02, 0.96, r'$\langle 110\rangle$ favored', fontsize=7.5,
                transform=ax.get_yaxis_transform(), va='top')
    ax.set_xlabel(xlabel)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.text(0.04, 0.94, panel, transform=ax.transAxes,
            fontsize=9, fontweight='bold', va='top')
    ax.spines[['top', 'right']].set_visible(False)

for ax in axes:
    ax.set_ylabel(r'Zener anisotropy $A_Z$')

fig.tight_layout()

for ext in ('pdf', 'png'):
    out = FIGS / f'az_vs_ti.{ext}'
    fig.savefig(out, dpi=200, bbox_inches='tight', facecolor='white')
    print('saved', out)

plt.close(fig)
