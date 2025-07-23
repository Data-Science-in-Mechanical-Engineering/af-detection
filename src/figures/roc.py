import matplotlib.pyplot as plt
import seaborn as sns

from src.experiments.evaluation.__main__ import experiment
from src.experiments.evaluation.config import Result, Config
from src.expyro.experiment import Run
from src.util import FIGURE_FULL_WIDTH, make_color_palette, set_plot_style, get_dataset_label, TEXT_FONT_SIZE, \
    DIR_FIGURES

FIGURE_HEIGHT = 2

DATASETS = ["coat", "sph", "cinc"]
PEAK_EXTRACTION = "xqrs"


def main():
    fig, axes = plt.subplots(
        ncols=len(DATASETS),
        figsize=(FIGURE_FULL_WIDTH, FIGURE_HEIGHT),
        sharey=True, sharex=True,
    )

    axes = {
        dataset: ax
        for dataset, ax in zip(DATASETS, axes, strict=True)
    }

    colors = make_color_palette("bright", DATASETS)

    for dataset_evaluation in DATASETS:
        for dataset_train in DATASETS:
            run: Run[Config, Result] = experiment[f"distributional/{dataset_train}/{PEAK_EXTRACTION}/seed-000"]
            curve = run.result[dataset_evaluation].roc_curve

            sns.lineplot(
                x=curve.fpr, y=curve.tpr,
                linewidth=2, color=colors[dataset_train],
                ax=axes[dataset_evaluation],
            )

        axes[dataset_evaluation].set_title(
            f"Evaluated on {get_dataset_label(dataset_evaluation)}",
            fontsize=TEXT_FONT_SIZE, fontweight="bold"
        )

        axes[dataset_evaluation].tick_params(which="both", direction="in", top=True, right=True, bottom=True, left=True)
        axes[dataset_evaluation].minorticks_on()

    axes["sph"].set_xlabel("False Positive Rate", fontweight="bold")
    axes["coat"].set_ylabel("True Positive Rate", fontweight="bold")

    fig.legend(
        handles=[
                    plt.Line2D([], [], color="tab:gray", linestyle="-", linewidth=0),
                ] + [
                    plt.Line2D([], [], color=colors[dataset], linestyle="-", linewidth=2)
                    for dataset in DATASETS
                ],
        labels=[r"$\bf{Training}$ $\bf{set:}$"] + [
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

    fig.tight_layout(pad=0.25, w_pad=1, rect=(0, 0, 1, 0.85))

    DIR_FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(DIR_FIGURES / "roc.pdf", dpi=500)


if __name__ == "__main__":
    set_plot_style("whitegrid")
    main()
