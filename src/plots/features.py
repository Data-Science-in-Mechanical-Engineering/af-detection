from pathlib import Path

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

from src.data.dataset import SPHDataset
from src.method.features import extract_normalized_rri
from src.plots.util import Style
from src.results import RESULTS_FOLDER


def plot_plain_rri_kde(
    qrs_complexes: list[np.ndarray],
    poly_scale: int = 2,
    size: tuple[float, float] | None = None,
    path: Path | None = None
):
    n_patients = len(qrs_complexes)
    rris = extract_normalized_rri(qrs_complexes)
    rri_data = dict(enumerate(rris))

    if size is not None:
        plt.figure(figsize=size)

    ax = sns.kdeplot(
        data=rri_data,
        legend=False,
        fill=True,
        palette=Style.vibrant_colors_discrete(n_patients),
        linewidth=2
    )

    plt.yscale("function", functions=[lambda x: x ** (1 / poly_scale), lambda x: x ** poly_scale])
    plt.xlabel("RRI", size=Style.LABEL_FONT_SIZE)
    plt.ylabel("")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)

    ax.set_xticks([])
    ax.set_yticks([])

    if path is None:
        plt.show()
    else:
        plt.savefig(path, dpi=300, bbox_inches="tight")

    plt.close()


if __name__ == "__main__":
    # stereotypical examples:
    # SPH: AF: MUSE_20180113_073230_90000
    # SPH: noAF: MUSE_20180210_120332_11000

    SPH = SPHDataset.load_train() | SPHDataset.load_validate() | SPHDataset.load_test()

    example_af = SPH.get_by_identifier("MUSE_20180113_073230_90000")
    example_no_af = SPH.get_by_identifier("MUSE_20180210_120332_11000")

    plot_plain_rri_kde(
        [example_af.qrs_complexes, example_no_af.qrs_complexes],
        size=(5, 4),
        # path=RESULTS_FOLDER / "example-plain-kde.pdf"
    )
