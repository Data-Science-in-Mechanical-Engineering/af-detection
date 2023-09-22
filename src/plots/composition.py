import math
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import matplotlib.ticker as tkr
import numpy as np
import seaborn as sns
from matplotlib.colors import Normalize
from matplotlib.patches import FancyArrowPatch

from src.method.features import extract_normalized_rri
from .confusion import get_confusion_data, plot_confusion_heatmap
from .dataset_size import get_logsize_data, plot_logsize_boxplot
from .ecg import extract_ecg_information
from .roc import get_roc_data, plot_roc_scatterplot, plot_roc_lineplot
from .util import Style, Metric
from ..data.dataset import ECGEntry, SPHDataset
from ..results import Outcome, RESULTS_FOLDER, Result, Snapshot

DISTRIBUTION_COLORS = ["#d62728", "#9467bd"]
SAMPLE_COLORS = ["#ff7f0e", "#e377c2"]


def plot_result_composition(
        confusion_left: Outcome,
        confusion_right: Outcome,
        logsize_result: Result,
        roc_snapshot: Snapshot,
        roc_line_cs: set[float],
        roc_highlighted_cs: set[float],
        roc_metric_x: Metric,
        roc_metric_y: Metric,
        roc_metric_x_label: str,
        roc_metric_y_label: str,
        logsize_metric: Metric,
        validation_dataset_label: Callable[[str, str], str],
        logsize_metric_label: str,
        label_mapping: Callable[[str], str],
        training_dataset_label: Callable[[Outcome], str],
        path: Path | None = None,
        size: tuple[float, float] | None = None
):
    confusion_data_left = get_confusion_data(confusion_left, label_mapping)
    confusion_data_right = get_confusion_data(confusion_right, label_mapping)
    logsize_data = get_logsize_data(logsize_result, logsize_metric, validation_dataset_label, True)
    roc_data = get_roc_data(roc_snapshot, roc_line_cs, roc_highlighted_cs, roc_metric_x, roc_metric_y)

    figure = plt.figure(figsize=size)
    plt.subplots_adjust(left=0.075, bottom=0.075, right=0.925, top=0.925, wspace=None, hspace=None)

    grid = figure.add_gridspec(2, 3, width_ratios=[1, 1, 1], wspace=0.2, hspace=0.5)
    ax1 = figure.add_subplot(grid[:, 0])
    ax2 = figure.add_subplot(grid[:, 1])
    ax3 = figure.add_subplot(grid[0, 2])
    ax4 = figure.add_subplot(grid[1, 2])

    plot_confusion_heatmap(confusion_data_left, cbar=False, ax=ax1, annot_kws={"size": Style.LABEL_FONT_SIZE_LARGE})

    ax1.set_ylabel("True class", size=Style.LABEL_FONT_SIZE_LARGE)
    ax1.set_xlabel(training_dataset_label(confusion_left), size=Style.LABEL_FONT_SIZE_LARGE)
    ax1.tick_params(labelrotation=0, labelsize=Style.LABEL_FONT_SIZE)

    plot_confusion_heatmap(
        confusion_data_right,
        cbar=False,
        yticklabels=False,
        ax=ax2,
        annot_kws={"size": Style.LABEL_FONT_SIZE_LARGE}
    )

    ax2.set_xlabel(training_dataset_label(confusion_right), size=Style.LABEL_FONT_SIZE_LARGE)
    ax2.tick_params(labelsize=Style.LABEL_FONT_SIZE)

    plot_logsize_boxplot(logsize_data, ax=ax3)
    ax3.set_ylabel(logsize_metric_label, size=Style.LABEL_FONT_SIZE_LARGE)
    ax3.set_xlabel("Dataset size", size=Style.LABEL_FONT_SIZE_LARGE)
    ax3.tick_params(labelsize=Style.LABEL_FONT_SIZE)
    ax3.legend(fontsize=Style.LABEL_FONT_SIZE)
    ax3.yaxis.tick_right()

    plot_roc_scatterplot(roc_data, roc_metric_x, roc_metric_y, ax=ax4)
    plot_roc_lineplot(roc_data, ax=ax4)

    formatter = tkr.ScalarFormatter(useMathText=True)
    formatter.set_scientific(False)
    sm = plt.cm.ScalarMappable(cmap=Style.VIBRANT_COLOR_PALETTE, norm=Normalize(roc_data.min_rho, roc_data.max_rho))
    cb = ax4.figure.colorbar(sm, ax=ax4, location="top", format=formatter)

    cb.outline.set_visible(False)
    cb.set_label(r"Relative penalization parameter $\rho$", size=Style.LABEL_FONT_SIZE)
    ax4.figure.axes[-1].tick_params(labelsize=Style.LABEL_FONT_SIZE)

    plt.rcParams["pgf.texsystem"] = "pdflatex"
    ax4.set_xlabel(roc_metric_x_label, size=Style.LABEL_FONT_SIZE_LARGE)
    ax4.set_ylabel(roc_metric_y_label, size=Style.LABEL_FONT_SIZE_LARGE)
    ax4.tick_params(labelsize=Style.LABEL_FONT_SIZE)
    ax4.yaxis.tick_right()

    if path is None:
        plt.show()
    else:
        plt.savefig(path, dpi=300, bbox_inches="tight")

    plt.clf()


def main_plot_results(path: Path | None = None):
    confusion_result = Result.from_json(RESULTS_FOLDER / "svm rri/test-performance-cross-database.json") \
        .filter(lambda snapshot: snapshot.setup["dataset_validate"]["name"] == "SPHDataset") \
        .partition(lambda snapshot: snapshot.setup["dataset_train"]["name"])

    confusion_sph = confusion_result["SPHDataset"].snapshots[0].outcomes[0]
    confusion_coat = confusion_result["COATDataset"].snapshots[0].outcomes[0]

    logsize_result = Result.from_json(RESULTS_FOLDER / "svm rri/test-logsize-cross-database.json") \
        .filter(lambda snapshot: snapshot.setup["dataset_train"]["name"] == snapshot.setup["dataset_validate"]["name"])

    roc_result = Result.from_json(RESULTS_FOLDER / "svm rri/validate-roc-cross-database.json") \
        .partition_by_datasets()

    roc_sph_result = roc_result["SPHDataset", "SPHDataset"]
    roc_highlighted_cs = {10.531052631578946}
    roc_line_cs = roc_highlighted_cs | {1.0621052631578947, 20}

    def label_renaming(label: str) -> str:
        if label == "AFIB":
            return "AF"
        elif label == "AF":
            return "AFlut"
        elif label == "noAFIB":
            return "noAF"
        else:
            return label

    def label_dataset(outcome: Outcome):
        if outcome == confusion_sph:
            return "Predictions on SPH, trained on SPH"
        elif outcome == confusion_coat:
            return "Predictions on SPH, trained on DiagnoStick"

    validation_labels = {
        ("SPHDataset", "SPHDataset"): "SPH",
        ("COATDataset", "COATDataset"): "DiagnoStick"
    }

    plot_result_composition(
        confusion_sph,
        confusion_coat,
        logsize_result,
        roc_sph_result.snapshots[0],
        roc_line_cs,
        roc_highlighted_cs,
        lambda outcome: 1 - outcome.specificity,
        lambda outcome: outcome.recall,
        "False positive rate",
        "True positive rate",
        lambda outcome: outcome.accuracy,
        lambda label_train, label_validate: validation_labels[label_train, label_validate],
        "Accuracy",
        label_renaming,
        label_dataset,
        size=(11, 6),
        path=path
    )


def plot_double_ecg(
        entry_left: ECGEntry,
        entry_right: ECGEntry,
        n_peaks_left: int,
        n_peaks_right: int,
        size: tuple[float, float] | None = None,
        path: Path | None = None
):
    ecg_info_left = extract_ecg_information(entry_left.ecg_signal, 250, 1, n_peaks_left, entry_left.qrs_complexes)
    ecg_info_right = extract_ecg_information(entry_right.ecg_signal, 250, 1, n_peaks_right, entry_right.qrs_complexes)

    def transform_ecg_signal(ecg_signal: np.ndarray) -> np.ndarray:
        ecg_signal = ecg_signal / ecg_signal.max()
        ecg_signal[ecg_signal > 0.5] = 0.5 + (ecg_signal[ecg_signal > 0.5] - 0.5) ** 5
        return ecg_signal

    figure, axes = plt.subplots(1, 2, figsize=size, sharey=True)

    x_left = ecg_info_left.time_indices
    x_right = ecg_info_right.time_indices
    y_left = transform_ecg_signal(ecg_info_left.ecg_signal)
    y_right = transform_ecg_signal(ecg_info_right.ecg_signal)

    sns.lineplot(x=x_left, y=y_left, color=Style.VIBRANT_COLOR_LIGHT, ax=axes[0])
    sns.lineplot(x=x_right, y=y_right, color=Style.VIBRANT_COLOR_DARK, ax=axes[1])

    axes[0].fill_between(x_left, -0.05, y_left, color=Style.VIBRANT_COLOR_LIGHT, alpha=0.1)
    axes[1].fill_between(x_right, -0.05, y_right, color=Style.VIBRANT_COLOR_DARK, alpha=0.1)

    axes[0].axis("off")
    axes[1].axis("off")

    plt.ylim(-0.1, y_left.max())

    if path is None:
        plt.show()
    else:
        plt.savefig(path, dpi=300, bbox_inches="tight")

    plt.clf()


def main_plot_ecg_examples(path: Path | None = None):
    sph_dataset = SPHDataset.load_train() | SPHDataset.load_test() | SPHDataset.load_validate()
    entry_afib = sph_dataset.get_by_identifier("MUSE_20180113_073230_90000")
    entry_no_afib = sph_dataset.get_by_identifier("MUSE_20180210_120332_11000")

    plot_double_ecg(
        entry_no_afib,
        entry_afib,
        8,
        8,
        size=(11, 6),
        path=path
    )


def plot_kme_sketch(
        entry_afib: ECGEntry,
        entry_healthy: ECGEntry,
        n_peaks: int,
        size: tuple[float, float] | None = None,
        path: Path | None = None
):
    ecg_info_healthy = extract_ecg_information(entry_healthy.ecg_signal, 250, 1, n_peaks, entry_healthy.qrs_complexes)
    ecg_info_afib = extract_ecg_information(entry_afib.ecg_signal, 250, 1, n_peaks, entry_afib.qrs_complexes)

    if size is not None:
        plt.figure(figsize=size)

    figure = plt.figure(figsize=size)
    grid = figure.add_gridspec(2, 3, width_ratios=[1, 1, 1], wspace=0.4, hspace=0.2)

    ax_healthy = figure.add_subplot(grid[0, 0])
    ax_afib = figure.add_subplot(grid[1, 0])
    ax_kde_healthy = figure.add_subplot(grid[0, 1])
    ax_kde_afib = figure.add_subplot(grid[1, 1], sharex=ax_kde_healthy, sharey=ax_kde_healthy)
    ax_kme = figure.add_subplot(grid[:, 2])

    sns.lineplot(
        x=ecg_info_healthy.time_indices,
        y=ecg_info_healthy.ecg_signal,
        linewidth=2,
        color=Style.VIBRANT_COLOR_LIGHT,
        ax=ax_healthy,
    )

    sns.lineplot(
        x=ecg_info_afib.time_indices,
        y=ecg_info_afib.ecg_signal,
        linewidth=2,
        color=Style.VIBRANT_COLOR_DARK,
        ax=ax_afib,
    )

    for ax in [ax_healthy, ax_afib]:
        ax.set_xlabel("Time", size=Style.LABEL_FONT_SIZE_LARGE)

    rris = extract_normalized_rri([entry_healthy.qrs_complexes, entry_afib.qrs_complexes])
    kme_plot_data = zip([ax_kde_healthy, ax_kde_afib], DISTRIBUTION_COLORS, SAMPLE_COLORS, rris)

    for ax, color, color_scatter, rri in kme_plot_data:
        sns.kdeplot(
            data=rri,
            legend=False,
            color=color,
            linewidth=2,
            ax=ax
        )

        x = ax.lines[0].get_xdata()
        y = ax.lines[0].get_ydata()
        ax.fill_between(x, y, color=color, alpha=0.3)

        sns.scatterplot(
            x=rri,
            y=[-1] * rri.size,
            color=color_scatter,
            legend=False,
            marker="x",
            s=100,
            ax=ax
        )

        ax.set_ylim(bottom=-2)
        ax.set_xlabel("RRI", size=Style.LABEL_FONT_SIZE_LARGE)

    for ax_left, ax_right in zip([ax_healthy, ax_afib], [ax_kde_healthy, ax_kde_afib]):
        ax_left_bounds = ax_left.get_position()
        y_center = ax_left_bounds.y0 + (ax_left_bounds.y1 - ax_left_bounds.y0) / 2
        ax_left_east = (ax_left_bounds.x1, y_center)

        ax_right_bounds = ax_right.get_position()
        ax_right_west = (ax_right_bounds.x0, y_center)

        arrow = FancyArrowPatch(
            ax_left_east,
            ax_right_west,
            shrinkB=0,
            transform=figure.transFigure,
            arrowstyle="->",
            mutation_scale=20,
            color=Style.GREEN_COLOR,
            linewidth=2
        )

        figure.patches.append(arrow)
        plt.annotate("distribution", xy=(0.5, 2), xycoords=arrow, horizontalalignment="center")

    vector_healthy = np.array([0, 0, 3, 2])
    vector_afib = np.array([0, 0, 6, -2])
    vector_healthy = vector_healthy / np.linalg.norm(vector_healthy)
    vector_afib = vector_afib / np.linalg.norm(vector_afib)

    x, y, u, v = zip(vector_healthy, vector_afib)

    ax_kme.quiver(
        x, y, u, v,
        angles="xy",
        scale_units="xy",
        scale=1,
        headwidth=3,
        headlength=5,
        width=0.015,
        color=DISTRIBUTION_COLORS
    )

    ax_kme.set_xlim([-0.5, 1.5])
    ax_kme.set_ylim([-0.75, 1])

    for rri, center, color in zip(rris, [vector_healthy, vector_afib], SAMPLE_COLORS):
        thetas = rri - rri.mean()
        thetas = thetas / np.abs(thetas).max() * math.pi / 8

        for rri_value, theta in zip(rri, thetas):
            rotation = np.array([
                [math.cos(theta), -math.sin(theta)],
                [math.sin(theta), math.cos(theta)]
            ])

            vector = np.dot(rotation, center[-2:])
            vector = vector / np.linalg.norm(vector)

            sns.scatterplot(
                x=[vector[0]], y=[vector[1]],
                color=color,
                marker="x",
                s=100,
                ax=ax_kme
            )

            ax_kme.quiver(
                [0], [0], [vector[0]], [vector[1]],
                angles="xy",
                scale_units="xy",
                scale=1,
                headwidth=3,
                headlength=5,
                width=0.015,
                color=color,
                alpha=0.15
            )

    ax_kme.set_xlabel("Reproducing Kernel Hilbert Space", size=Style.LABEL_FONT_SIZE_LARGE)

    for ax_left in [ax_kde_healthy, ax_kde_afib]:
        ax_left_bounds = ax_left.get_position()
        y_center = ax_left_bounds.y0 + (ax_left_bounds.y1 - ax_left_bounds.y0) / 2
        ax_left_east = (ax_left_bounds.x1, y_center)

        ax_right_bounds = ax_kme.get_position()
        ax_right_west = (ax_right_bounds.x0, y_center)

        arrow = FancyArrowPatch(
            ax_left_east,
            ax_right_west,
            shrinkB=0,
            transform=figure.transFigure,
            arrowstyle="->",
            mutation_scale=20,
            color=Style.GREEN_COLOR,
            linewidth=2
        )

        figure.patches.append(arrow)
        plt.annotate("KME", xy=(0.5, 2), xycoords=arrow, horizontalalignment="center")

    for ax in [ax_healthy, ax_afib, ax_kde_healthy, ax_kde_afib, ax_kme]:
        ax.set_ylabel("")
        ax.set_xticks([])
        ax.set_yticks([])

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)

    if path is None:
        plt.show()
    else:
        plt.savefig(path, dpi=300, bbox_inches="tight")

    plt.clf()


def main_plot_kme_sketch(path: Path | None = None):
    sph_dataset = SPHDataset.load_train() | SPHDataset.load_test() | SPHDataset.load_validate()
    entry_afib = sph_dataset.get_by_identifier("MUSE_20180113_073230_90000")
    entry_no_afib = sph_dataset.get_by_identifier("MUSE_20180210_120332_11000")

    plot_kme_sketch(
        entry_afib,
        entry_no_afib,
        n_peaks=8,
        size=(11, 4),
        path=path
    )


if __name__ == "__main__":
    main_plot_kme_sketch(RESULTS_FOLDER / "kme-sketch.pdf")
