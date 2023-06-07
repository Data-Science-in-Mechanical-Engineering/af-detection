from pathlib import Path

import seaborn as sb
from matplotlib import pyplot as plt

from src.data.dataset import ECGDataset, COATDataset
from src.data.util import COATPath
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


if __name__ == "__main__":
    data = COATDataset.load_from_folder(COATPath.TRAIN_DATA) \
        .filter(lambda entry: len(entry.qrs_complexes) > 50) \
        .subsample({0: 10})

    plot_rri_kde(data, "KDE Healthy")
