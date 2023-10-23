from argparse import ArgumentParser, Namespace
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.dataset import COATDataset, COATIdentifier, Identifier
from src.experiments import make_binary_labels
from src.results import Outcome, Result, Snapshot
from src.scripts.util import COATSetup, finish_experiment


def parse_proprietary_predictions(xlsx_file: Path) -> dict[Identifier, int]:
    assert xlsx_file.is_file()

    data = pd.read_excel(xlsx_file)
    data = data[~data["screenresult_af"].isnull()]

    assert "basic_studyid" in data
    assert "screenresult_af" in data

    label_mapping = defaultdict(lambda: COATDataset.UNKNOWN, {
        1: COATDataset.AFIB,
        0: COATDataset.noAFIB
    })

    return defaultdict(lambda: COATDataset.UNKNOWN, {
        COATIdentifier.from_string_patient_id(identifier): label_mapping[int(label)]
        for identifier, label in zip(data["basic_studyid"], data["screenresult_af"])
    })


def main_proprietary_performance(arguments: Namespace):
    predictions = parse_proprietary_predictions(arguments.file)

    dataset = COATSetup.standard_preprocessing(COATDataset.load_test())

    if not arguments.imbalanced:
        dataset = dataset.balanced_binary_partition({dataset.AFIB}, dataset.count(dataset.AFIB))

    labels_binary = make_binary_labels(dataset.labels, {dataset.AFIB})

    predicted_labels = np.array([predictions[identifier] for identifier in dataset.identifiers])
    predicted_labels[predicted_labels == COATDataset.AFIB] = 1
    predicted_labels[predicted_labels == COATDataset.noAFIB] = 0

    unknown_mask = predicted_labels == COATDataset.UNKNOWN
    predicted_labels[unknown_mask] = 1 - labels_binary[unknown_mask]

    outcome = Outcome.evaluate({"algorithm": "MyDiagnoStick"}, dataset, predicted_labels, labels_binary)
    result = Result("proprietary predictions", snapshots=[Snapshot([outcome], {
        "dataset_validate": dataset.description()
    })])

    finish_experiment(result)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-f", dest="file", type=Path, required=True, help="XLSX file with MyDiagnoStick predictions.")

    parser.add_argument("--imbalanced", dest="imbalanced", default=False, action="store_true",
                        help="Whether or not to train and evaluate on the full datasets.")

    args = parser.parse_args()
    main_proprietary_performance(args)
