from pathlib import Path

import numpy as np
import seaborn as sb
from matplotlib import pyplot as plt

from src.data.dataset import ECGDataset
from src.method.features import extract_rri


def plot_rri_kde(dataset: ECGDataset, title: str, path: Path | None = None):
    vanilla_rris = extract_rri(dataset.qrs_complexes)
    mean = vanilla_rris.mean(axis=1, keepdims=True)
    std = vanilla_rris.std(axis=1, keepdims=True)

    rri_setups = {
        "RRI": vanilla_rris,
        "RRI - mean": vanilla_rris - mean,
        "(RRI - mean) / frequency": (vanilla_rris - mean) / dataset.FREQUENCY,
        "RRI / mean": vanilla_rris / mean,
        "(RRI - mean) / std": (vanilla_rris - mean) / std
    }

    figure, axes = plt.subplots(nrows=5, figsize=(8, 6))
    figure.suptitle(title)

    for axis, (name, rris) in zip(axes, rri_setups.items()):
        sb.kdeplot(dict(enumerate(rris)), ax=axis, legend=False)
        axis.set(xlabel=name)

    plt.subplots_adjust(hspace=1)

    if path is None:
        plt.show()
    else:
        plt.savefig(path / f"{title}.pdf")


def plot_r_peaks(dataset: ECGDataset, title: str, path: Path | None = None):
    figure, axes = plt.subplots(nrows=dataset.n, figsize=(8, 1.5 * dataset.n))
    figure.suptitle(title)
    plt.subplots_adjust(hspace=1)

    for axis, ecg, r_peaks in zip(axes, dataset.ecg_signals, dataset.qrs_complexes):
        x = np.arange(1, ecg.size + 1)
        axis.plot(x, ecg)
        axis.scatter(r_peaks, ecg[r_peaks], color="r")

    if path is None:
        plt.show()
    else:
        plt.savefig(path / f"{title}.pdf")
