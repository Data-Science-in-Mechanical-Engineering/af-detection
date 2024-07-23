from collections import defaultdict
from pathlib import Path
from typing import Callable, NamedTuple, Iterable

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle

from src.plots.util import Style
from src.results import Outcome, Result, RESULTS_FOLDER


class ConfusionData(NamedTuple):
    annotations: list[list[str]]
    row_normalized_confusion_matrix: np.ndarray
    x_labels: list[str]
    y_labels: list[str]


def get_confusion_data(outcome: Outcome, label_mapping: Callable[[str], str]) -> ConfusionData:
    confusion_matrix, x_labels, y_labels = get_confusion_matrix(outcome, label_mapping)
    row_normalized_confusion_matrix = confusion_matrix / confusion_matrix.sum(axis=1, keepdims=True)

    x_labels.sort()
    y_labels.sort()

    annotations = [
        [f"{n_examples} ({100 * n_examples / n:.2f}%)" for n_examples in row]
        for row, n in zip(confusion_matrix, confusion_matrix.sum(axis=1))
    ]

    return ConfusionData(annotations, row_normalized_confusion_matrix, x_labels, y_labels)


def get_confusion_matrix(outcome: Outcome, label_mapping: Callable[[str], str]) -> tuple[np.ndarray, list, list]:
    x_indices = [label_mapping(label) for label in outcome.confusion.keys()]
    all_y_indices = set().union(key for row in outcome.confusion.values() for key in row)
    y_indices = [label_mapping(label) for label in all_y_indices]

    x_indices.sort()
    y_indices.sort()

    confusion = {
        label_mapping(key): defaultdict(lambda: 0, {
            label_mapping(other_key): entry
            for other_key, entry in row.items()
        })
        for key, row in outcome.confusion.items()
    }

    confusion_matrix = np.array([
        [confusion[x_index][y_index] for y_index in y_indices]
        for x_index in x_indices
    ]).T

    return confusion_matrix, x_indices, y_indices


def plot_confusion_heatmap(data: ConfusionData, **kwargs):
    annot_kws = dict(
        size=Style.LABEL_FONT_SIZE,
        color="white",
        fontweight="bold"
    )

    if "annot_kws" in kwargs:
        annot_kws |= kwargs["annot_kws"]

    arguments = dict(
        data=data.row_normalized_confusion_matrix,
        annot=data.annotations,
        xticklabels=data.x_labels,
        yticklabels=data.y_labels,
        fmt="",
        cmap=Style.VIBRANT_COLOR_PALETTE.reversed(),
    ) | kwargs | dict(annot_kws=annot_kws)

    ax = sns.heatmap(**arguments)
    ax.add_patch(Rectangle((0, 0), 1, 1, fill=False, edgecolor=Style.GREEN_COLOR, lw=3, clip_on=False))
    ax.add_patch(Rectangle((1, 1), 1, 10, fill=False, edgecolor=Style.GREEN_COLOR, lw=3, clip_on=False))

    return ax

def plot_confusions_jointly(outcomes: Iterable[Outcome], titles: Iterable[str], label_mapping: Callable[[str], str], path: Path | None = None):
    confusion_data = [get_confusion_data(outcome, label_mapping) for outcome in outcomes]
    n_plots = len(confusion_data)
    fig, axs = plt.subplots(1, n_plots, layout='constrained', figsize=(3*n_plots+1,4))
    
    for data, title, ax in zip(confusion_data[:-1], titles[:-1], axs[:-1]):
        ax = plot_confusion_heatmap(data, cbar=False, vmin=0., vmax=1., ax=ax)
        ax.tick_params(labelsize=Style.LABEL_FONT_SIZE)
        ax.tick_params(size=Style.LABEL_FONT_SIZE, axis='x')
        ax.tick_params(rotation=0, size=Style.LABEL_FONT_SIZE, axis='y')
        ax.set_xlabel("Predicted class", size=Style.LABEL_FONT_SIZE)
        ax.set_title(title, size=Style.LABEL_FONT_SIZE_LARGE)
    plot_confusion_heatmap(confusion_data[-1], cbar=True, vmin=0., vmax=1., ax=axs[-1])
    axs[-1].tick_params(labelsize=Style.LABEL_FONT_SIZE)
    axs[-1].tick_params(size=Style.LABEL_FONT_SIZE, axis='x')
    axs[-1].tick_params(rotation=0, size=Style.LABEL_FONT_SIZE, axis='y')
    axs[-1].set_xlabel("Predicted class", size=Style.LABEL_FONT_SIZE)
    axs[0].set_ylabel("True class", size=Style.LABEL_FONT_SIZE)
    axs[-1].set_title(titles[-1], size=Style.LABEL_FONT_SIZE_LARGE)

    if path is None:
        plt.show()
    else:
        plt.savefig(path, dpi=300)

    plt.clf()



def plot_confusion(outcome: Outcome, label_mapping: Callable[[str], str], path: Path | None = None):
    confusion_data = get_confusion_data(outcome, label_mapping)
    ax = plot_confusion_heatmap(confusion_data)

    plt.yticks(rotation=0, size=Style.LABEL_FONT_SIZE)
    plt.xticks(size=Style.LABEL_FONT_SIZE)
    plt.ylabel("True class", size=Style.LABEL_FONT_SIZE)
    plt.xlabel("Predicted class", size=Style.LABEL_FONT_SIZE)
    ax.figure.axes[-1].tick_params(labelsize=Style.LABEL_FONT_SIZE)

    if path is None:
        plt.show()
    else:
        plt.savefig(path, dpi=300)

    plt.clf()


if __name__ == "__main__":
    results_file = RESULTS_FOLDER / "svm rri" / "test_imbalanced_cross.json"
    save_path = RESULTS_FOLDER / "svm rri" / "test_imbalanced_cross" / "confusion_matrices.pdf"
    results_sph_validate = Result.from_json(results_file) \
        .filter(lambda snapshot: snapshot.setup["dataset_validate"]["name"] == "SPHDataset") \
        .partition(lambda snapshot: snapshot.setup["dataset_train"]["name"])


    def label_renaming(label: str) -> str:
        if label == "AFIB":
            return "AF"
        elif label == "AF":
            return "AFlut"
        elif label == "noAFIB":
            return "noAF"
        else:
            return label


    sph_to_sph_outcome = results_sph_validate["SPHDataset"].snapshots[0].outcomes[0]
    coat_to_sph_outcome = results_sph_validate["COATDataset"].snapshots[0].outcomes[0]

    # plot_confusion(sph_to_sph_outcome, label_renaming)
    plot_confusions_jointly(
        [coat_to_sph_outcome, sph_to_sph_outcome], 
        ["DiagnoStick", "SPH"], 
        label_renaming,
        path=save_path
    )