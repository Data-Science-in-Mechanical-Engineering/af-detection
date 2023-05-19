import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import seaborn as sb
from matplotlib import pyplot as plt

from src.experiments.util import RESULTS_FOLDER


def plot_confusion_matrix(
        confusion: dict[Any, dict[Any, int]],
        title: str,
        folder: Path | None = None,
        normalize: str | None = None
):
    x_indices = list(confusion.keys())
    y_indices = list(set().union(key for row in confusion.values() for key in row))
    x_indices.sort()
    y_indices.sort()

    confusion = {
        key: defaultdict(lambda: 0, row)
        for key, row in confusion.items()
    }

    confusion_matrix = np.array([
        [confusion[x_index][y_index] for y_index in y_indices]
        for x_index in x_indices
    ])

    if normalize == "class":
        confusion_matrix = confusion_matrix / confusion_matrix.sum(axis=0, keepdims=True)
    elif normalize == "total":
        confusion_matrix = confusion_matrix / confusion_matrix.sum()

    show_annotation = normalize is not None

    if show_annotation:
        confusion_matrix = confusion_matrix.round(2)

    plt.suptitle(title)
    sb.heatmap(confusion_matrix, xticklabels=y_indices, yticklabels=x_indices, annot=show_annotation)

    if folder is None:
        plt.show()
    else:
        plt.savefig(folder / f"{re.sub(r'[^a-zA-Z0-9_ ]', '', title)}.pdf")


if __name__ == "__main__":
    con = {
        "AFIB": {
            "AFIB": 336,
            "SA": 9,
            "AF": 11,
            "ST": 4,
            "SB": 11,
            "SR": 1,
            "AT": 2
        },
        "noAFIB": {
            "AFIB": 14,
            "SA": 5,
            "SB": 147,
            "SR": 70,
            "SVT": 27,
            "AT": 3,
            "ST": 49,
            "AF": 9,
            "AVRT": 1,
            "AVNRT": 1
        }
    }

    plot_confusion_matrix(
        con,
        "Confusion: SPH / SPH (normalized by class)",
        RESULTS_FOLDER,
        normalize="class"
    )
