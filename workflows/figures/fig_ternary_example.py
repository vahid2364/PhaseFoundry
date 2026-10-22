"""
Two-panel figure for Section 5 (Examples) of the PhaseFoundry paper
(Computer Physics Communications).

Panel (a): Ternary phase map for Nb-Hf-V BCC_A2 at 500 C — single-phase,
           two-phase, three-phase tie-triangle, and spinodal region — using
           phasefoundry.thermodynamics.plot_ternary_spinodal().
Panel (b): STEM-style composition microstructure from the ternary CH simulation
           (main panel x_V, insets x_Nb and x_Hf) using
           phasefoundry.phasefield.plot_microstructure_composition().
"""

import io
import pathlib
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

matplotlib.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 8,
    'axes.linewidth': 0.6,
})

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))

H5  = ROOT / 'data/HfNbTiV_BCC_PhaseForge_v3/NbHfV/nb_hf_v_bcc_a2_500C_n70_v2_full.h5'
NPZ = (ROOT / 'results/HfNbTiV_BCC_PhaseForge_v2/NbHfV/500C/kinetics'
       / 'pf_2d_c50Nb25Hf_kappa1e12_M5e21_tdbv2/fields.npz')
OUT = ROOT / 'paper/2025_Attari_PhaseFoundry/figs/fig_ternary_example.pdf'

SIM_XNB, SIM_XHF = 0.50, 0.25

from phasefoundry.thermodynamics import plot_ternary_spinodal
from phasefoundry.phasefield import plot_microstructure_composition

# ── load microstructure ───────────────────────────────────────────────────────
d     = np.load(NPZ, allow_pickle=True)
t_s   = float(d['time'][-1])
Lx_nm = float(d['Lx']) * 1e9
xnb   = d['x_nb'][-1].astype(float)
xhf   = d['x_hf'][-1].astype(float)
xv    = 1.0 - xnb - xhf
N     = xnb.shape[0]

# ── build microstructure figure, render to buffer ─────────────────────────────
r0, r1 = int(0.35 * N), int(0.90 * N)
c0, c1 = int(0.50 * N), int(0.82 * N)
micro_fig = plot_microstructure_composition(
    fields={'Nb': xnb, 'Hf': xhf, 'V': xv},
    Lx_nm=Lx_nm,
    main_element='V',
    crop=(r0, r1, c0, c1),
    t_s=t_s,
    scalebar_nm=200,
    cmap='coolwarm',
    dpi=200,
    show=False,
)
buf = io.BytesIO()
micro_fig.savefig(buf, format='png', dpi=200, bbox_inches='tight', facecolor='k')
plt.close(micro_fig)
buf.seek(0)
micro_img = np.array(Image.open(buf))

# ── figure layout ─────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(6.8, 3.6))

# ── panel (a): ternary phase map ──────────────────────────────────────────────
ax_l = fig.add_axes([0.01, 0.12, 0.46, 0.86])
plot_ternary_spinodal(
    table_path=H5,
    independent_elements=['NB', 'HF'],
    corner_labels=['Nb', 'Hf', 'V'],
    sim_composition={'NB': SIM_XNB, 'HF': SIM_XHF},
    ax=ax_l,
    marker_size=6,
    show=False,
)
ax_l.set_title('')
for child in ax_l.get_children():
    if hasattr(child, 'get_text') and '500' in str(getattr(child, 'get_text', lambda: '')() or ''):
        try:
            child.set_text('(a) Phase map — 500 °C')
            child.set_fontsize(8)
        except Exception:
            pass

# ── panel (b): microstructure (embed rendered image) ─────────────────────────
ax2 = fig.add_axes([0.52, 0.02, 0.47, 0.96])
ax2.imshow(micro_img, aspect='equal')
ax2.axis('off')

fig.savefig(OUT, dpi=300, bbox_inches='tight')
print('saved', OUT)
