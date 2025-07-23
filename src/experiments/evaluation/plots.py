import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import Normalize, LinearSegmentedColormap
from matplotlib.patches import Rectangle

from src.experiments.evaluation.config import Result, Config
from src.metrics import Evaluation
from src.util import get_label_names, TEXT_FONT_SIZE


def plot_confusion_matrix(
        evaluation: Evaluation,
        positive_labels: list[str],
        label_names: dict[str, str],
        ax: plt.Axes,
        font_size: float = TEXT_FONT_SIZE
):
    positive_labels = {
        label for label in positive_labels
        if evaluation.attribution_positive[label] + evaluation.attribution_negative[label] > 0
    }

    attribution_positive = {
        label: count
        for label, count in evaluation.attribution_positive.items()
        if evaluation.attribution_positive[label] + evaluation.attribution_negative[label] > 0
    }

    attribution_negative = {
        label: count
        for label, count in evaluation.attribution_negative.items()
        if evaluation.attribution_positive[label] + evaluation.attribution_negative[label] > 0
    }

    assert attribution_positive.keys() == attribution_negative.keys()
    assert attribution_positive.keys() <= label_names.keys()

    attribution_labels = [label for label in label_names.keys() if label in attribution_positive]

    attribution_positive = np.array([attribution_positive[label] for label in attribution_labels])
    attribution_negative = np.array([attribution_negative[label] for label in attribution_labels])

    attributions = np.stack([attribution_positive, attribution_negative])
    relative_attributions = attributions / attributions.sum(axis=0)

    annotations = np.array([
        [f"{attributions[i, j]} ({100 * relative_attributions[i, j]:.2f}%)" for j, _ in enumerate(attribution_labels)]
        for i in [0, 1]
    ])

    mask_correct = np.array([
        [label in positive_labels for label in attribution_labels],
        [label not in positive_labels for label in attribution_labels],
    ])

    def plot(mask: np.ndarray, cmap_):
        sns.heatmap(
            data=relative_attributions.T,
            annot=annotations.T,
            annot_kws=dict(
                color="white",
                fontweight="bold",
                fontsize=font_size,
            ),
            cmap=cmap_,
            cbar=False,
            norm=Normalize(vmin=0, vmax=1),
            xticklabels=["AF", "not AF"],
            yticklabels=[label_names[label] for label in attribution_labels],
            fmt="",
            mask=mask.T,
            ax=ax
        )

    cmap = LinearSegmentedColormap.from_list("green-red", [(0.0, "green"), (0.5, "tab:red"), (1.0, "black")])

    plot(mask_correct, cmap)
    plot(~mask_correct, cmap.reversed())

    ax.tick_params(axis="both", which="both", length=0)

    i = 0
    while i < len(attribution_labels):
        is_positive = attribution_labels[i] in positive_labels

        j = i
        while j < len(attribution_labels) and (attribution_labels[j] in positive_labels) == is_positive:
            j += 1

        x = 0 if is_positive else 1
        ax.add_patch(Rectangle(
            xy=(x, i),
            width=1,
            height=j - i,
            edgecolor="gray", lw=1.5, linestyle="--",
            capstyle="round", joinstyle="round",
            fill=False, clip_on=False
        ))

        i = j


def _plot_confusion_matrix(
        evaluation: Evaluation,
        positive_labels: list[str],
        label_names: dict[str, str]
) -> plt.Figure:
    fig = plt.figure(constrained_layout=True)
    ax = fig.subplots()

    plot_confusion_matrix(evaluation, positive_labels, label_names, ax)

    ax.set_xlabel("Predicted class", fontweight="bold")
    ax.set_ylabel("True class", fontweight="bold")

    return fig


def plot_confusion_matrices_evaluation(config: Config, result: Result) -> dict[str, plt.Figure]:
    return {
        f"{name_evaluation}": _plot_confusion_matrix(
            evaluation=evaluation.run.evaluation_validation,
            positive_labels=config.evaluation[name_evaluation].positive_labels,
            label_names=get_label_names(config.evaluation[name_evaluation].name)
        )
        for name_evaluation, evaluation in result.items()
    }
