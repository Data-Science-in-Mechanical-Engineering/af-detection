from typing import Callable, Any

import matplotlib.pyplot as plt
import seaborn as sns

from src.experiments.evaluation.__main__ import experiment
from src.experiments.evaluation.config import Config, Result
from src.expyro.experiment import Run
from src.metrics import Evaluation
from src.util import set_plot_style, TEXT_FONT_SIZE, get_dataset_label, make_color_palette, FIGURE_FULL_WIDTH, \
    DIR_FIGURES

DATASETS = ["coat", "sph", "cinc"]
SIZES = [10, 25, 50, 100, 200]

BASE_PATH = experiment.directory / experiment.name
BASE_PATH_SIZED = BASE_PATH / "distributional_sized"
BASE_PATH_FULL = BASE_PATH / "distributional"


def boxplot(
        runs: dict[str, dict[int, list[Run[Config, Result]]]],
        runs_full: dict[str, Run[Config, Result]],
        metric: Callable[[Evaluation], float],
        palette: Any,
        label: str,
        min_y: float,
        max_y: float,
        ax: plt.Axes
):
    sizes_index = {size: i for i, size in enumerate(sorted(SIZES))}

    assert all(
        all(run in sizes_index for run in runs[dataset])
        for dataset in runs
    )

    xs = [
        sizes_index[size]
        for dataset, runs_by_size in runs.items()
        for size, runs in runs_by_size.items()
        for _ in runs
    ]

    ys = [
        metric(run.result[dataset].run.evaluation_validation)
        for dataset, runs_by_size in runs.items()
        for size, runs in runs_by_size.items()
        for run in runs
    ]

    hue = [
        dataset
        for dataset, runs_by_size in runs.items()
        for size, runs in runs_by_size.items()
        for _ in runs
    ]

    sns.boxplot(
        x=xs, y=ys, hue=hue,
        linewidth=1,
        flierprops=dict(
            marker="o",
            markerfacecolor="black",
            linewidth=0,
            markersize=2,
            alpha=0.25,
        ),
        boxprops=dict(linewidth=0.5),
        medianprops=dict(linewidth=0.5),
        whiskerprops=dict(linewidth=0.5),
        capprops=dict(linewidth=0.5),
        palette=palette,
        legend=False,
        showfliers=False,
        ax=ax
    )

    sns.scatterplot(
        x=[len(sizes_index)] * len(DATASETS),
        y=[metric(runs_full[dataset].result[dataset].run.evaluation_validation) for dataset in DATASETS],
        hue=[dataset for dataset in DATASETS],
        legend=False,
        marker="X",
        s=50,
        ax=ax
    )

    ax.set_xticks(
        ticks=[i for i in range(len(sizes_index) + 1)],
        labels=[f"{size}" for size in sizes_index] + ["all"]
    )

    ax.set_ylim(min_y, max_y)

    ax.tick_params(which="both", direction="in", top=True, right=True, bottom=True, left=True)
    ax.minorticks_on()

    ax.set_title(label, fontweight="bold", fontsize=TEXT_FONT_SIZE)


def main():
    runs = {
        dataset: {
            size: [
                experiment[path]
                for path in (BASE_PATH_SIZED / dataset / "xqrs" / f"n={size}").iterdir()
            ]
            for size in SIZES
        }
        for dataset in DATASETS
    }

    runs_full = {
        dataset: experiment[BASE_PATH_FULL / dataset / "xqrs" / "seed-000"]
        for dataset in DATASETS
    }

    palette = make_color_palette("bright", DATASETS)

    fig = plt.figure(figsize=(FIGURE_FULL_WIDTH, 4))
    ax = fig.subplots(nrows=2, ncols=3, sharex=True)

    boxplot(runs, runs_full, lambda x: x.f1, palette=palette, label="F1-Score", min_y=-0.02, max_y=1.02, ax=ax[0, 0])
    boxplot(runs, runs_full, lambda x: x.sensitivity, palette=palette, label="Sensitivity", min_y=-0.02, max_y=1.02,
            ax=ax[0, 1])
    boxplot(runs, runs_full, lambda x: x.precision, palette=palette, label="Precision", min_y=-0.02, max_y=1.02,
            ax=ax[0, 2])
    boxplot(runs, runs_full, lambda x: x.specificity, palette=palette, label="Specificity", min_y=0.8, max_y=1.005,
            ax=ax[1, 0])
    boxplot(runs, runs_full, lambda x: x.roc_auc, palette=palette, label="AUROC", min_y=0.8, max_y=1.005, ax=ax[1, 1])
    boxplot(runs, runs_full, lambda x: x.accuracy, palette=palette, label="Accuracy", min_y=0.8, max_y=1.005,
            ax=ax[1, 2])

    for i in range(1, 3):
        ax[0, i].set_yticklabels([])
        ax[1, i].set_yticklabels([])

    ax[1, 1].set_xlabel("Number of AF examples")

    fig.legend(
        handles=[plt.Line2D([], [], color="tab:gray", linestyle="-", linewidth=0)] + [
            plt.Line2D([], [], color=palette[dataset], linestyle="-", linewidth=2)
            for dataset in DATASETS
        ],
        labels=[r"$\bf{Data}$ $\bf{set:}$"] + [
            get_dataset_label(dataset)
            for dataset in DATASETS
        ],
        loc="upper center",
        ncol=len(DATASETS) + 1,
        frameon=False,
        fontsize=TEXT_FONT_SIZE,
        handlelength=1.5,
        handletextpad=1,
        columnspacing=2,
        bbox_to_anchor=(0, 1.02, 1, 0),
    )

    fig.tight_layout(pad=0.25, w_pad=1, rect=(0, 0, 1, 0.925))

    DIR_FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(DIR_FIGURES / "dataset-size.pdf", dpi=500)


if __name__ == "__main__":
    set_plot_style("whitegrid")
    main()
