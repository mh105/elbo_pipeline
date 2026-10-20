"""Render native-reference mixing-map weights from the saved sub-03 ICA."""

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PathCollection
import mne
import numpy as np
from PIL import Image


HERE = Path(__file__).resolve().parent
ICA_PATH = HERE.parent / "sub03_step3_ica_assessment_20261016/sub-03-baseline-ica.fif"
OUT = HERE / "topomaps"
SENSOR_FACE_COLOR = "#000000"
SENSOR_EDGE_COLOR = "#39ff14"
SENSOR_SIZE_PT2 = 10.0
SENSOR_LINEWIDTH_PT = 0.65


def main():
    input_sha = hashlib.sha256(ICA_PATH.read_bytes()).hexdigest()
    ica = mne.preprocessing.read_ica(ICA_PATH, verbose=False)
    assert ica.info.ch_names == ica.ch_names
    maps = ica.get_components()
    assert maps.shape == (119, 119)
    positions = np.array([ch["loc"][:3] for ch in ica.info["chs"]])
    assert np.isfinite(positions).all()
    scales = np.max(np.abs(maps), axis=0)
    assert (scales > 0).all()
    OUT.mkdir(exist_ok=True)
    components = []
    for component in range(ica.n_components_):
        name = f"ICA{component:03d}"
        fig, ax = plt.subplots(figsize=(2.4, 2.4), dpi=150)
        fig.subplots_adjust(left=0.03, right=0.97, bottom=0.03, top=0.97)
        mne.viz.plot_topomap(
            maps[:, component] / scales[component], ica.info,
            axes=ax, vlim=(-1, 1), cmap="RdBu_r", sensors=True,
            contours=6, outlines="head", res=128, show=False,
        )
        sensors = [item for item in ax.collections if isinstance(item, PathCollection)]
        assert len(sensors) == 1
        assert sensors[0].get_offsets().shape == (119, 2)
        sensors[0].set_edgecolor(SENSOR_EDGE_COLOR)
        sensors[0].set_facecolor(SENSOR_FACE_COLOR)
        sensors[0].set_sizes([SENSOR_SIZE_PT2])
        sensors[0].set_linewidths([SENSOR_LINEWIDTH_PT])
        path = OUT / f"{name}.png"
        fig.savefig(path, dpi=150, facecolor="white")
        plt.close(fig)
        with Image.open(path) as image:
            assert image.size == (360, 360)
            assert np.asarray(image.convert("RGB")).std() > 1
        components.append(dict(
            component=component, name=name, file=f"topomaps/{name}.png",
            scale_max_abs=float(scales[component]),
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        ))
    assert hashlib.sha256(ICA_PATH.read_bytes()).hexdigest() == input_sha
    provenance = dict(
        subject="sub-03", input=str(ICA_PATH.relative_to(HERE.parent)),
        input_sha256=input_sha, n_components=ica.n_components_,
        mne_version=mne.__version__, matplotlib_version=matplotlib.__version__,
        map_source="ica.get_components()[:, component], in saved ICA.ch_names order",
        map_formula="(mixing_matrix_[:, :n_components_].T @ pca_components_[:n_components_]).T",
        reference="Saved native-reference maps; no centering, reference row restoration, or rereferencing",
        scale="Each map divided by its own maximum absolute electrode weight; symmetric color limits [-1, 1]",
        polarity="Saved signs preserved: blue negative, red positive; ICA sign itself is arbitrary",
        orientation="MNE head outline with nose at top and ears at left/right; saved EEG sensor locations",
        amplitude_comparison="Colors describe relative within-map weights, not absolute amplitude across components",
        sensor_style=dict(facecolor=SENSOR_FACE_COLOR, edgecolor=SENSOR_EDGE_COLOR,
                          size_pt2=SENSOR_SIZE_PT2, linewidth_pt=SENSOR_LINEWIDTH_PT),
        channel_mapping=[dict(row=index, name=name, position_head_m=positions[index].tolist())
                         for index, name in enumerate(ica.ch_names)],
        components=components,
        images_verified=119,
    )
    (HERE / "topomaps_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps(dict(n_components=len(components), input_sha256=input_sha,
                          images_verified=len(components), output=str(OUT))))


if __name__ == "__main__":
    main()
