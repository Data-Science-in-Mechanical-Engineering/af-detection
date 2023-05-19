import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import seaborn as sb
from matplotlib import pyplot as plt


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
        filename = re.sub(r'[^a-zA-Z0-9_ ]', '', title)
        filename = re.sub(' +', ' ', filename)
        plt.savefig(folder / f"{filename}.pdf")
