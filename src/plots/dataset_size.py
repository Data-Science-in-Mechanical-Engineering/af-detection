from collections import defaultdict
from pathlib import Path
from typing import Callable

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.patches import PathPatch

from src.plots.util import Style
from src.results import Result, Outcome, RESULTS_FOLDER


def get_logsize_data(
        result: Result,
        metric: Callable[[Outcome], float],
        dataset_label: Callable[[str, str], str],
        common_sizes_only: bool
) -> dict[str, np.ndarray]:
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

    return {"size": sizes, "score": scores, "label": labels}


# taken from: https://stackoverflow.com/questions/31498850/set-space-between-boxplots-in-python-graphs-generated-nested-box-plots-with-seab
def adjust_box_widths(ax, fac):
    for c in ax.get_children():
        if isinstance(c, PathPatch):
            p = c.get_path()
            verts = p.vertices
            verts_sub = verts[:-1]
            xmin = np.min(verts_sub[:, 0])
            xmax = np.max(verts_sub[:, 0])
            xmid = 0.5 * (xmin + xmax)
            xhalf = 0.5 * (xmax - xmin)

            xmin_new = xmid - fac * xhalf
            xmax_new = xmid + fac * xhalf
            verts_sub[verts_sub[:, 0] == xmin, 0] = xmin_new
            verts_sub[verts_sub[:, 0] == xmax, 0] = xmax_new

            for l in ax.lines:
                if np.all(l.get_xdata() == [xmin, xmax]):
                    l.set_xdata([xmin_new, xmax_new])


def plot_logsize_boxplot(data: dict[str, np.ndarray], **kwargs):
    arguments = dict(
        data=data,
        x="size",
        y="score",
        hue="label",
        palette=Style.vibrant_colors_discrete(2),
        showfliers=False
    ) | kwargs

    ax = sns.boxplot(**arguments)
    adjust_box_widths(ax, 0.8)
    return ax


def plot_dataset_logsize_experiment(
        result: Result,
        metric: Callable[[Outcome], float],
        metric_label: str,
        dataset_label: Callable[[str, str], str],
        common_sizes_only: bool = False,
        path: Path | None = None
):
    data = get_logsize_data(result, metric, dataset_label, common_sizes_only)

    ax = plot_logsize_boxplot(data)

    plt.ylabel(metric_label, size=Style.LABEL_FONT_SIZE)
    plt.xlabel("Dataset size", size=Style.LABEL_FONT_SIZE)
    ax.tick_params(labelsize=Style.LABEL_FONT_SIZE)
    plt.legend(fontsize=Style.LABEL_FONT_SIZE)

    if path is None:
        plt.show()
    else:
        plt.savefig(path, dpi=300)

    plt.clf()


if __name__ == "__main__":
    RESULT = Result.from_json(RESULTS_FOLDER / "svm rri/dataset_size_imbalanced.json") \
        .filter(lambda snapshot: snapshot.setup["dataset_train"]["name"] == snapshot.setup["dataset_validate"]["name"])

    LABELS = {
        ("SPHDataset", "SPHDataset"): "SPH",
        ("COATDataset", "COATDataset"): "DiagnoStick"
    }

    plot_dataset_logsize_experiment(
        RESULT,
        lambda outcome: outcome.f1,
        "f1",
        lambda a, b: LABELS[a, b],
        common_sizes_only=True,
        path=RESULTS_FOLDER / "f1-data-size.pdf"
    )
