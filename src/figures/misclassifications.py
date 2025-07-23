import jax.numpy as jnp
import matplotlib.pyplot as plt
import seaborn as sns
import tqdm

from src.data import Dataset
from src.experiments.evaluation.__main__ import experiment
from src.experiments.evaluation.config import Config, Result
from src.util import set_plot_style, DIR_FIGURES


DATASETS = ["coat", "cinc", "sph"]
PEAK_EXTRACTION = "xqrs"


def plot_ecg(dataset: Dataset, identifiers: list[str]) -> plt.Figure:
    indices = dataset.identifier_indices(identifiers)

    height = 3 * len(identifiers)
    width = 0.67 * (dataset.ecg_lengths[indices] / dataset.frequency).max().item()

    fig, axes = plt.subplots(
        figsize=(width, height),
        nrows=len(identifiers), ncols=1,
        sharex=True
    )

    for identifier, index, ax in tqdm.tqdm(zip(identifiers, indices, axes, strict=True), total=len(identifiers)):
        ecg = dataset.ecgs[index][:dataset.ecg_lengths[index]]
        peaks = dataset.peak_indices[index]
        length_seconds = len(ecg) / dataset.frequency

        time = jnp.linspace(0, length_seconds, num=len(ecg))

        sns.lineplot(
            x=time, y=ecg,
            linewidth=1,
            ax=ax
        )

        for peak in peaks:
            t_seconds = time[peak].item()
            ax.axvline(x=t_seconds, color="red", linestyle="--", linewidth=0.5)

        ax.set_xlabel("Time (s)")
        ax.set_xmargin(0)

        ax.set_title(identifier, fontweight="bold")

    fig.tight_layout(h_pad=2)

    return fig


def main():
    for dataset_name_train in tqdm.tqdm(DATASETS):
        run = experiment[f"distributional/{dataset_name_train}/{PEAK_EXTRACTION}/seed-000"]
        config: Config = run.config
        results: Result = run.result

        dir_misclassifications = DIR_FIGURES / "misclassifications"
        dir_misclassifications.mkdir(parents=True, exist_ok=True)

        for dataset_name_evaluation, result in results.items():
            dataset_evaluation = config.evaluation[dataset_name_evaluation].load("test")

            plot_ecg(dataset_evaluation, result.run.evaluation_validation.false_positives).savefig(
                dir_misclassifications / f"{dataset_name_train}_{dataset_name_evaluation}__false_positives.pdf"
            )

            plot_ecg(dataset_evaluation, result.run.evaluation_validation.false_negatives).savefig(
                dir_misclassifications / f"{dataset_name_train}_{dataset_name_evaluation}__false_negatives.pdf"
            )


if __name__ == "__main__":
    set_plot_style("white")
    main()
