"""
Generate quaternary composition figures at t=60000 s:
  quaternary_chem_step600k     — chemical-only PFM (panel a)
  quaternary_elastic_step600k  — elasto-chemical PFM (panel b)

Each figure: main panel x_V + three insets (Nb, Hf, Ti) + crop rectangle.
"""
import pathlib
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import ConnectionPatch

matplotlib.rcParams.update({
    'font.family': 'sans-serif', 'font.size': 8,
    'axes.linewidth': 0.5,
})

CMAP = 'coolwarm'

ROOT = pathlib.Path(__file__).parent.parent.parent
KINETICS = ROOT / 'results' / 'HfNbTiV_BCC_PhaseForge_v2' / 'NbHfTiV' / '500C' / 'kinetics'
FIGS = ROOT / 'paper' / '2025_Kunselman_Design_of_Spinodal_Microstructures_in_HEAs' / 'figs'

_elastic_dir = KINETICS / 'pf_2d_c25Nb25Hf25Ti_kappa1e12_M5e21_tdbv2_elastic'
_elastic_npz = (_elastic_dir / 'fields_checkpoint.npz'
                if (_elastic_dir / 'fields_checkpoint.npz').exists()
                else _elastic_dir / 'fields.npz')

RUNS = {
    'quaternary_chem_step600k':    KINETICS / 'pf_2d_c25Nb25Hf25Ti_kappa1e12_M5e21_tdbv2' / 'fields.npz',
    'quaternary_elastic_step600k': _elastic_npz,
}

# crop region — portrait strip capturing Hf halos and 4-5 precipitates
row0, row1 = 62, 123
col0, col1 = 74, 106


def make_figure(npz_path: pathlib.Path) -> plt.Figure:
    d   = np.load(npz_path, allow_pickle=True)
    t   = d['time']
    Lx  = float(d['Lx'])
    N   = int(d['N'])
    idx = len(t) - 1   # latest available snapshot
    print(f"  {npz_path.parent.name}: idx={idx}, t={t[idx]:.0f} s")

    xnb = d['x_nb'][idx]
    xhf = d['x_hf'][idx]
    xti = d['x_ti'][idx]
    xv  = 1.0 - xnb - xhf - xti

    px_to_nm = Lx * 1e9 / N

    fig = plt.figure(figsize=(4.0, 4.0), facecolor='white')

    ax_main = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax_main.set_aspect('equal')
    ax_main.axis('off')
    ax_main.set_facecolor('white')

    extent = [0, Lx*1e9, 0, Lx*1e9]
    ax_main.imshow(xv, origin='lower', cmap=CMAP,
                   vmin=xv.min(), vmax=xv.max(),
                   extent=extent, interpolation='bilinear')

    ax_main.text(0.02, 0.87, 'V', transform=ax_main.transAxes,
                 color='black', fontsize=18.6, fontweight='bold',
                 va='top', ha='left', zorder=6,
                 bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=1.5))

    rect_x = col0 * px_to_nm
    rect_y = row0 * px_to_nm
    rect_w = (col1 - col0) * px_to_nm
    rect_h = (row1 - row0) * px_to_nm
    ax_main.add_patch(mpatches.Rectangle(
        (rect_x, rect_y), rect_w, rect_h,
        linewidth=1.8, edgecolor='yellow', facecolor='none', zorder=5))

    cb_ax = fig.add_axes([0.962, 0.744, 0.03, 0.216])
    sm = plt.cm.ScalarMappable(cmap=CMAP,
                                norm=plt.Normalize(vmin=xv.min(), vmax=xv.max()))
    cb = fig.colorbar(sm, cax=cb_ax)
    cb.set_label(r'$x_{\rm V}$', fontsize=9.1, labelpad=3)
    cb.ax.tick_params(labelsize=10.1)
    cb.ax.yaxis.set_label_position('left')
    cb.ax.yaxis.set_ticks_position('left')

    sb_x0 = Lx*1e9 * 0.72
    sb_y0 = Lx*1e9 * 0.04
    sb_len = 200
    ax_main.plot([sb_x0, sb_x0 + sb_len], [sb_y0, sb_y0],
                 color='black', lw=4.9, zorder=5)
    ax_main.text(sb_x0 + sb_len/2, sb_y0 + 18, '200 nm',
                 color='black', fontsize=13.0, ha='center', va='bottom',
                 fontweight='bold', zorder=6)

    inset_fields = [('Nb', xnb), ('Hf', xhf), ('Ti', xti)]
    n_insets   = 3
    inset_w    = 0.155
    inset_h    = 0.38
    inset_gap  = 0.005
    inset_left = 0.01
    inset_bot  = 0.01

    for i, (label, field) in enumerate(inset_fields):
        left   = inset_left + i * (inset_w + inset_gap)
        ax_ins = fig.add_axes([left, inset_bot, inset_w, inset_h])
        ax_ins.imshow(field[row0:row1, col0:col1], origin='lower', cmap=CMAP,
                      vmin=field.min(), vmax=field.max(),
                      interpolation='bilinear', aspect='auto')
        ax_ins.axis('off')
        ax_ins.text(0.08, 0.97, label, transform=ax_ins.transAxes,
                    color='black', fontsize=11.4, fontweight='bold',
                    va='top', ha='left',
                    bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=1.5))

    inset_right = inset_left + n_insets * inset_w + (n_insets - 1) * inset_gap
    inset_top   = inset_bot + inset_h

    lw_outer = 1.5
    lw_div   = 0.8
    # outer border — four explicit lines so linewidth is exact
    for xs, ys in [
        ([inset_left, inset_right], [inset_bot,  inset_bot ]),   # bottom
        ([inset_left, inset_right], [inset_top,  inset_top ]),   # top
        ([inset_left, inset_left ], [inset_bot,  inset_top ]),   # left
        ([inset_right,inset_right], [inset_bot,  inset_top ]),   # right
    ]:
        fig.add_artist(plt.Line2D(xs, ys, transform=fig.transFigure,
                                  color='black', lw=lw_outer,
                                  clip_on=False, zorder=10))
    # thin vertical dividers between insets
    for j in range(1, n_insets):
        x_div = inset_left + j * inset_w + (j - 1) * inset_gap
        fig.add_artist(plt.Line2D(
            [x_div, x_div], [inset_bot, inset_top],
            transform=fig.transFigure, color='black', lw=lw_div,
            clip_on=False, zorder=10))

    fig.add_artist(ConnectionPatch(
        xyA=(rect_x, rect_y),          coordsA=ax_main.transData,
        xyB=(inset_left, inset_top),    coordsB=fig.transFigure,
        color='yellow', lw=0.9, zorder=7, clip_on=False))
    fig.add_artist(ConnectionPatch(
        xyA=(rect_x + rect_w, rect_y), coordsA=ax_main.transData,
        xyB=(inset_right, inset_top),  coordsB=fig.transFigure,
        color='yellow', lw=0.9, zorder=7, clip_on=False))

    return fig


for stem, npz in RUNS.items():
    fig = make_figure(npz)
    for ext in ('pdf', 'png'):
        out = FIGS / f'{stem}.{ext}'
        fig.savefig(out, bbox_inches='tight', dpi=200, facecolor='white')
        print('saved', out)
    plt.close(fig)
