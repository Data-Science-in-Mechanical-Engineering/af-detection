import matplotlib.pyplot as plt

from src.experiments.evaluation.__main__ import experiment
from src.experiments.evaluation.plots import plot_confusion_matrix
from src.util import FIGURE_FULL_WIDTH, set_plot_style, get_label_names, get_dataset_label, TEXT_FONT_SIZE, DIR_FIGURES

FIGURE_HEIGHT = 3
TRAIN_DATASETS = ["sph", "coat", "cinc"]
PEAK_EXTRACTION = "xqrs"


def main():
    fig, axes = plt.subplots(
        ncols=len(TRAIN_DATASETS),
        figsize=(FIGURE_FULL_WIDTH, FIGURE_HEIGHT),
        sharey=True, sharex=True,
    )

    for dataset, ax in zip(TRAIN_DATASETS, axes, strict=True):
        run = experiment[f"distributional/{dataset}/{PEAK_EXTRACTION}/seed-000"]

        assert "sph" in run.result
        assert "sph" in run.config.evaluation

        plot_confusion_matrix(
            evaluation=run.result["sph"].run.evaluation_validation,
            positive_labels=run.config.evaluation["sph"].positive_labels,
            label_names=get_label_names("sph"),
            ax=ax,
            font_size=8
        )

        ax.set_title(
            f"Trained on {get_dataset_label(dataset)}",
            fontsize=TEXT_FONT_SIZE, fontweight="bold"
        )

        ax.tick_params(labelsize=8)

    axes[0].set_ylabel("True class", fontweight="bold")
    axes[len(TRAIN_DATASETS) // 2].set_xlabel("Predicted class", fontweight="bold")

    fig.tight_layout(pad=0.25, w_pad=1)

    DIR_FIGURES.mkdir(parents=True, exist_ok=True)
    plt.savefig(DIR_FIGURES / "confusion.pdf", dpi=500)


if __name__ == "__main__":
    set_plot_style()
    main()
