"""
Quaternary NbHfTiV composition microstructure figures for the Kunselman paper.

Generates:
  quaternary_chem_step600k     — chemical-only PFM at t ≈ 60 000 s
  quaternary_elastic_step600k  — elasto-chemical PFM at latest snapshot

Delegates all layout logic to phasefoundry.phasefield.plot_microstructure_from_npz.
"""
import pathlib, sys
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from phasefoundry.phasefield import plot_microstructure_from_npz

KINETICS = ROOT / "results/HfNbTiV_BCC_PhaseForge_v2/NbHfTiV/500C/kinetics"
FIGS     = ROOT / "paper/2025_Kunselman_Design_of_Spinodal_Microstructures_in_HEAs/figs"

_elastic_dir = KINETICS / "pf_2d_c25Nb25Hf25Ti_kappa1e12_M5e21_tdbv2_elastic"
_elastic_npz = (
    _elastic_dir / "fields_checkpoint.npz"
    if (_elastic_dir / "fields_checkpoint.npz").exists()
    else _elastic_dir / "fields.npz"
)

RUNS = {
    "quaternary_chem_step600k":    (KINETICS / "pf_2d_c25Nb25Hf25Ti_kappa1e12_M5e21_tdbv2" / "fields.npz",    -1),
    "quaternary_elastic_step600k": (_elastic_npz, -1),
}

# Crop region: portrait strip capturing Hf halos and 4-5 clear precipitates
CROP = (62, 123, 74, 106)

DISPLAY = {"NB": "Nb", "HF": "Hf", "TI": "Ti", "V": "V"}

for stem, (npz, tidx) in RUNS.items():
    fig = plot_microstructure_from_npz(
        npz_path=npz,
        elements=["NB", "HF", "TI", "V"],
        ref_element="V",
        time_index=tidx,
        display_names=DISPLAY,
        crop=CROP,
        scalebar_nm=200,
        show=False,
    )
    for ext in ("pdf", "png"):
        out = FIGS / f"{stem}.{ext}"
        fig.savefig(out, bbox_inches="tight", dpi=200, facecolor="white")
        print("saved", out)
    plt.close(fig)
