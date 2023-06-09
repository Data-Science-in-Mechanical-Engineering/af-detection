from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from src.data.util import DATA_PATH
from ..data.dataset import ECGDataset, Identifier, COATDataset, COATIdentifier
from ..experiments.util import ExperimentTracker, make_binary_labels, METRICS, compute_confusion


def track(
        name: str,
        dataset: ECGDataset,
        labels: dict[Identifier, str | int],
        af_labels: set,
        source_name: str
) -> ExperimentTracker:
    assert af_labels <= dataset.label_domain()
    assert af_labels <= set(labels.values())

    dataset = dataset.filter(lambda entry: entry.identifier in labels)

    setup = {"dataset": repr(dataset), "source": source_name}
    tracker = ExperimentTracker(name, setup)

    predicted_labels = np.array([labels[identifier] for identifier in dataset.identifiers])
    binary_predicted_labels = make_binary_labels(predicted_labels, af_labels)
    binary_dataset_labels = make_binary_labels(dataset.labels, af_labels)

    scores = {
        name: metric(binary_dataset_labels, binary_predicted_labels)
        for name, metric in METRICS.items()
    }

    scores["confusion"] = compute_confusion(
        binary_predicted_labels,
        binary_dataset_labels,
        dataset.labels,
        {0: "noAFIB", 1: "AFIB"}
    )

    tracker[{}] = scores

    return tracker


def parse_my_diagnostic_predictions(xlsx_file: Path) -> dict[COATIdentifier, int]:
    assert xlsx_file.is_file()

    data = pd.read_excel(xlsx_file)
    data = data[~data["screenresult_af"].isnull()]

    assert "basic_studyid" in data
    assert "screenresult_af" in data

    label_mapping = defaultdict(lambda: COATDataset.UNKNOWN, {
        0: COATDataset.noAF,
        1: COATDataset.AF
    })

    return {
        COATIdentifier.from_string_patient_id(identifier): label_mapping[int(label)]
        for identifier, label in zip(data["basic_studyid"], data["screenresult_af"])
    }
