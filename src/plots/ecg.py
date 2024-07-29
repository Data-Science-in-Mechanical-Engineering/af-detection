import math
from pathlib import Path
from typing import Callable, Any, NamedTuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.collections import LineCollection
from matplotlib.colors import LogNorm
from matplotlib.patches import Circle, Rectangle

from src.data.qrs import XQRSPeakDetectionAlgorithm
from .util import Style, merge_pdfs
from ..data.dataset import ECGDataset, SPHDataset
from ..results import Result, Snapshot, RESULTS_FOLDER

DEFAULT_LINE_WIDTH = 2


class ECGTemporalInformation(NamedTuple):
    ecg_signal: np.ndarray
    qrs_complexes: np.ndarray
    qrs_complexes_seconds: np.ndarray
    time_indices: np.ndarray
    time_seconds: np.ndarray


def extract_ecg_information(
        ecg_signal: np.ndarray,
        first_peak_offset: int,
        frequency: float,
        n_peaks: int | None,
        qrs_complexes: np.ndarray
) -> ECGTemporalInformation:
    if n_peaks is not None:
        qrs_complexes = qrs_complexes[-n_peaks:]

    start_index = max(0, qrs_complexes.min() - first_peak_offset)
    ecg_signal = ecg_signal[start_index:]
    ecg_signal = ecg_signal - ecg_signal.min()

    time_indices = np.arange(ecg_signal.size) + start_index
    time_seconds = time_indices / frequency
    qrs_complexes_seconds = qrs_complexes / frequency

    return ECGTemporalInformation(ecg_signal, qrs_complexes, qrs_complexes_seconds, time_indices, time_seconds)


def plot_ecg(
        ecg_signal: np.ndarray,
        qrs_complexes: np.ndarray,
        frequency: float,
        n_peaks: int | None = None,
        first_peak_offset: int = 100,
        mark_peaks: bool = True,
        size: tuple[float, float] | None = None,
        line_width: int = DEFAULT_LINE_WIDTH,
        decision_distance: float | None = None,
        plain: bool = False,
        title: str | None = None,
        path: Path | None = None
):
    info = extract_ecg_information(ecg_signal, first_peak_offset, frequency, n_peaks, qrs_complexes)

    if size is not None:
        plt.figure(figsize=size)

    ax = sns.lineplot(x=info.time_seconds, y=info.ecg_signal)
    x, y = ax.get_lines()[0].get_data()
    segments = np.array([x[:-1], y[:-1], x[1:], y[1:]]).T.reshape(-1, 2, 2)

    peak_distance = np.abs(info.qrs_complexes[:, None] - info.time_indices[None, :]).min(axis=0) + 1
    cmap = Style.VIBRANT_COLOR_PALETTE.reversed()
    norm = LogNorm(peak_distance.min(), peak_distance.max())

    lc = LineCollection(segments, cmap=cmap, norm=norm)
    lc.set_array(peak_distance[:-1])
    lc.set_linewidth(line_width)
    ax.add_collection(lc)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    plt.tick_params(labelsize=Style.LABEL_FONT_SIZE)
    plt.yticks([])

    if plain:
        plt.axis("off")
    else:
        plt.xlabel("Time (in seconds)", size=Style.LABEL_FONT_SIZE)

    if mark_peaks:
        marker_size = line_width / DEFAULT_LINE_WIDTH

        sns.scatterplot(
            x=info.qrs_complexes_seconds,
            y=np.zeros_like(info.qrs_complexes_seconds) - 75 * marker_size,
            color=Style.VIBRANT_COLOR_DARK,
            marker="^",
            s=100 * marker_size
        )

    if title is not None:
        plt.suptitle(title)

    if decision_distance is not None:
        confidence = 1 / (1 + math.exp(-abs(decision_distance)))
        plt.title(f"Confidence: {(100 * confidence):.2f}%")

    if path is None:
        plt.show()
    else:
        plt.savefig(path, dpi=300, bbox_inches="tight")

    plt.close()


def plot_plain_ecg(
        ecg_signal: np.ndarray,
        qrs_complexes: np.ndarray,
        color: str,
        n_peaks: int | None = None,
        first_peak_offset: int = 100,
        mark_rris: set[int] | None = None,
        mark_p_waves: set[int] | None = None,
        plain: bool = False,
        size: tuple[float, float] | None = None,
        path: Path | None = None,
):
    mark_rris = set() if mark_rris is None else mark_rris
    mark_p_waves = set() if mark_p_waves is None else mark_p_waves
    info = extract_ecg_information(ecg_signal, first_peak_offset, 1, n_peaks, qrs_complexes)

    if size is not None:
        plt.figure(figsize=size)

    ax = sns.lineplot(x=info.time_indices, y=info.ecg_signal, color=color, linewidth=2)

    for qrs_index in mark_p_waves:
        time_index = int(info.qrs_complexes_seconds[qrs_index] - 60)
        time_ecg_signal = int(info.ecg_signal[int(time_index - info.time_indices[0])])

        ax.add_patch(Circle(
            xy=(time_index, time_ecg_signal),
            radius=100,
            color=Style.RED_COLOR,
            fill=False,
            linewidth=2,
            zorder=100
        ))

    y_connection_patch = 0.75 * float(info.ecg_signal.max())

    for qrs_index in mark_rris:
        time_index = int(info.qrs_complexes_seconds[qrs_index] + 40)
        next_time_index = int(info.qrs_complexes_seconds[qrs_index + 1] - 40)

        ax.add_patch(Rectangle(
            xy=(time_index, y_connection_patch),
            width=next_time_index - time_index,
            height=20,
            color=Style.RED_COLOR,
            zorder=100
        ))

        ax.add_patch(Rectangle(
            xy=(time_index, y_connection_patch - 20),
            width=5,
            height=60,
            color=Style.RED_COLOR,
            zorder=100
        ))

        ax.add_patch(Rectangle(
            xy=(next_time_index - 5, y_connection_patch - 20),
            width=5,
            height=60,
            color=Style.RED_COLOR,
            zorder=100
        ))

    label_color = "white" if plain else "black"
    plt.xlabel("Time", size=Style.LABEL_FONT_SIZE, color=label_color)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)

    ax.set_xticks([])
    ax.set_yticks([])

    if path is None:
        plt.show()
    else:
        plt.savefig(path, dpi=300, bbox_inches="tight")

    plt.close()


def plot_misclassifications(
        result: Result,
        datasets: dict[str, ECGDataset],
        n_peaks_by_dataset: dict[str, int],
        root_folder: Path,
        setup_folder: Callable[[Snapshot], str],
        plot_confidence=False
):
    def plot_many(distances_by_identifier: dict[str, float], folder: Path, dataset: ECGDataset, n_peaks: int):
        entries = {}

        for entry in dataset:
            identifier = str(entry.identifier)
            if identifier in distances_by_identifier:
                entries[identifier] = entry

        size = (max([n_peaks, 11.7]), 8.27)

        for identifier, entry in entries.items():
            path = folder / f"{identifier}.pdf"
            print("plot", path)

            plot_ecg(
                entry.ecg_signal,
                entry.qrs_complexes,
                dataset.FREQUENCY,
                size=size,
                line_width=3,
                title=identifier,
                decision_distance=distances_by_identifier[identifier] if plot_confidence else None,
                path=path,
                n_peaks=n_peaks,
            )

    for snapshot in result:
        dataset_validate_name = snapshot.setup["dataset_validate"]["name"]
        dataset_validate = datasets[dataset_validate_name]
        n_peaks_validate = n_peaks_by_dataset[dataset_validate_name]

        snapshot_folder = root_folder / setup_folder(snapshot)
        false_positive_folder = snapshot_folder / "false positives"
        false_negatives_folder = snapshot_folder / "false negatives"

        false_positive_folder.mkdir(parents=True, exist_ok=True)
        false_negatives_folder.mkdir(parents=True, exist_ok=True)

        for outcome in snapshot:
            false_positives = outcome.false_positive_distance_by_identifier()
            false_negatives = outcome.false_negative_distance_by_identifier()

            plot_many(false_positives, false_positive_folder, dataset_validate, n_peaks_validate)
            plot_many(false_negatives, false_negatives_folder, dataset_validate, n_peaks_validate)


def plot_sampled_ecgs(dataset: ECGDataset, chunks: dict[Any, int], n_peaks: int, path: Path):
    dataset = dataset.subsample(chunks)

    path.mkdir(parents=True, exist_ok=False)

    for entry in dataset:
        plot_ecg(
            entry.ecg_signal,
            entry.qrs_complexes,
            dataset.FREQUENCY,
            n_peaks=n_peaks,
            path=path / f"{entry.identifier}.pdf",
            title=str(entry.identifier)
        )

    merge_pdfs(path, target_pdf_name="subsampled")


if __name__ == "__main__":
    # stereotypical examples:
    # SPH: AF: MUSE_20180113_073230_90000
    # SPH: noAF: MUSE_20180210_120332_11000
    # SPH: False Positive due to faulty peak extraction: MUSE_20180113_122732_19000
    # COAT: Dataset Example AF: DEV52-PAT53
    # SPH: Dataset Example AF: MUSE_20180113_132000_94000

    ALGORITHM = XQRSPeakDetectionAlgorithm()  # Christov2004()
    SPH = SPHDataset.load_train(qrs_algorithm=ALGORITHM) | SPHDataset.load_validate(
        qrs_algorithm=ALGORITHM) | SPHDataset.load_test(qrs_algorithm=ALGORITHM)
    # COAT = COATDataset.load_train() | COATDataset.load_validate() | COATDataset.load_test()

    example = SPH.get_by_identifier("MUSE_20180113_122732_19000")

    plot_ecg(
        example.ecg_signal,
        example.qrs_complexes,
        SPH.FREQUENCY,
        first_peak_offset=250,
        size=(11, 2),
        plain=True,
        path=RESULTS_FOLDER / "example-false-positive-peak-extraction-xqrs.pdf"
    )
