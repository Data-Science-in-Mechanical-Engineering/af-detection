import math
from pathlib import Path
from typing import Callable, Any, NamedTuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.collections import LineCollection
from matplotlib.colors import LogNorm

from .util import Style, merge_pdfs
from ..data.dataset import ECGDataset, SPHDataset, COATDataset
from ..results import Result, Snapshot, RESULTS_FOLDER
from ..scripts.util import STANDARD_SPH_MINIMUM_RRIS

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
        title: str | None = None,
        path: Path | None = None
):
    info = extract_ecg_information(ecg_signal, first_peak_offset, frequency, n_peaks, qrs_complexes)

    if size is not None:
        plt.figure(figsize=size)

    ax = sns.lineplot(x=info.time_seconds, y=ecg_signal)
    x, y = ax.get_lines()[0].get_data()
    segments = np.array([x[:-1], y[:-1], x[1:], y[1:]]).T.reshape(-1, 2, 2)

    peak_distance = np.abs(qrs_complexes[:, None] - info.time_indices[None, :]).min(axis=0) + 1
    cmap = Style.VIBRANT_COLOR_PALETTE.reversed()
    norm = LogNorm(peak_distance.min(), peak_distance.max())

    lc = LineCollection(segments, cmap=cmap, norm=norm)
    lc.set_array(peak_distance[:-1])
    lc.set_linewidth(line_width)
    ax.get_lines()[0].remove()
    ax.add_collection(lc)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    plt.tick_params(labelsize=Style.LABEL_FONT_SIZE)
    plt.yticks([])
    plt.xlabel("Time (in seconds)", size=Style.LABEL_FONT_SIZE)

    if mark_peaks:
        marker_size = line_width / DEFAULT_LINE_WIDTH

        sns.scatterplot(
            x=info.qrs_complexes_seconds,
            y=np.zeros_like(info.qrs_complexes_seconds) - 10 * marker_size,
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
        plt.savefig(path, dpi=300)

    plt.close()


def plot_misclassifications(
        result: Result,
        datasets: dict[str, ECGDataset],
        n_peaks_by_dataset: dict[str, int],
        root_folder: Path,
        setup_folder: Callable[[Snapshot], str]
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
                decision_distance=distances_by_identifier[identifier],
                path=path,
                n_peaks=n_peaks
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

    SPH = SPHDataset.load_train() | SPHDataset.load_validate() | SPHDataset.load_test()
    COAT = COATDataset.load_train() | COATDataset.load_validate() | COATDataset.load_test()

    plot_sampled_ecgs(SPH, {SPH.SR: 40}, STANDARD_SPH_MINIMUM_RRIS + 1, RESULTS_FOLDER / "tmp_noAF")
