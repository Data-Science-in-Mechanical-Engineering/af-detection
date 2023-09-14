from collections import defaultdict
from pathlib import Path
from typing import Callable

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

from .util import Style
from ..results import Result, Outcome, RESULTS_FOLDER


def plot_dataset_logsize_experiment(
    result: Result,
    metric: Callable[[Outcome], float],
    metric_label: str,
    dataset_label: Callable[[str, str], str],
    common_sizes_only: bool = False,
    path: Path | None = None
):
    labels, sizes, scores = [], [], []
    sizes_by_label = defaultdict(set)

    for snapshot in result:
        for outcome in snapshot:
            label = dataset_label(snapshot.setup["dataset_train"]["name"], snapshot.setup["dataset_validate"]["name"])
            score = metric(outcome)
            sizes_by_label[label].add(snapshot.size_train)

            labels.append(label)
            sizes.append(snapshot.size_train)
            scores.append(score)

    labels, sizes, scores = np.array(labels), np.array(sizes), np.array(scores)

    if common_sizes_only:
        common_sizes = set.intersection(*sizes_by_label.values())
        use_indices = np.array([i for i, size in enumerate(sizes) if size in common_sizes])
        labels, sizes, scores = labels[use_indices], sizes[use_indices], scores[use_indices]

    data = {"size": sizes, "score": scores, "label": labels}

    plot = sns.boxplot(
        data=data,
        x="size",
        y="score",
        hue="label",
        palette=Style.vibrant_colors_discrete(2),
        showfliers=False
    )

    plt.ylabel(metric_label, size=Style.LABEL_FONT_SIZE)
    plt.xlabel("Dataset size", size=Style.LABEL_FONT_SIZE)
    plot.tick_params(labelsize=Style.LABEL_FONT_SIZE)
    plt.legend(fontsize=Style.LABEL_FONT_SIZE)

    if path is None:
        plt.show()
    else:
        plt.savefig(path, dpi=300)

    plt.clf()


if __name__ == "__main__":
    RESULT = Result.from_json(RESULTS_FOLDER / "svm rri/test-logsize-cross-database.json") \
        .filter(lambda snapshot: snapshot.setup["dataset_train"]["name"] == snapshot.setup["dataset_validate"]["name"])

    LABELS = {
        ("SPHDataset", "SPHDataset"): "SPH",
        ("COATDataset", "COATDataset"): "DiagnoStick"
    }

    plot_dataset_logsize_experiment(
        RESULT,
        lambda outcome: outcome.accuracy,
        "Accuracy",
        lambda a, b: LABELS[a, b],
        common_sizes_only=True,
        path=RESULTS_FOLDER / "accuracy-data-size.pdf"
    )
