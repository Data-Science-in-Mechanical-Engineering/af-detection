import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Final, Any

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

Setup = Parametrization = dict[str, Any]
Outcome = dict[str, float]

if "__file__" in globals():
    RESULTS_FOLDER = Path(__file__).parent.parent.parent / "results"
else:
    RESULTS_FOLDER = Path().resolve().parent.parent / "results"

METRICS = {
    "accuracy": accuracy_score,
    "f1-score": f1_score,
    "precision": precision_score,
    "recall": recall_score,
}


def make_binary_labels(labels: np.ndarray, values_1: set) -> np.ndarray:
    values_1 = list(values_1)
    return np.isin(labels, values_1).astype(int)


def compute_confusion(
        grouped_predictions: np.ndarray,
        grouped_labels: np.ndarray,
        actual_labels: np.ndarray,
        prediction_mapping: dict[Any, str] | None = None,
        label_mapping: dict[Any, str] | None = None
):
    assert grouped_predictions.ndim == 1
    assert grouped_labels.ndim == 1
    assert actual_labels.ndim == 1
    assert grouped_predictions.size == grouped_labels.size == actual_labels.size

    confusion = defaultdict(lambda: defaultdict(lambda: 0))

    for group_prediction, group_label, actual_label in zip(grouped_predictions, grouped_labels, actual_labels):
        if prediction_mapping is not None:
            group_prediction = prediction_mapping[group_prediction]
        if label_mapping is not None:
            actual_label = label_mapping[actual_label]

        confusion[str(group_prediction)][str(actual_label)] += 1

    return confusion


class ExperimentTracker:
    created_time: Final[datetime]
    experiment_name: Final[str]
    setup: Final[Setup]
    outcomes: Final[list[tuple[Parametrization, Outcome]]]
    description: Final[dict[str, Any]]

    def __init__(self, experiment_name: str, setup: dict[str, Any], description: dict[str, Any] | None = None):
        self.created_time = datetime.now()
        self.experiment_name = experiment_name
        self.setup = setup
        self.outcomes = []
        self.description = {} if description is None else description

    def as_dict(self):
        return {
            "created": self.created_time.strftime("%A %d, %b %Y at %H:%M:%S.%f"),
            **self.description,
            "setup": self.setup,
            "runs": [
                {
                    "parameters": parameters,
                    "outcome": outcome
                } for parameters, outcome in self.outcomes
            ]
        }

    def save(self, filename: str | None = None):
        if filename is None:
            now = datetime.now()
            filename = f"{now.strftime('%Y-%m-%d %H-%M-%S.%f')[:-3]}.json"

        experiment_folder = ExperimentTracker.ROOT_FOLDER / self.experiment_name
        experiment_folder.mkdir(parents=True, exist_ok=True)

        with open(experiment_folder / filename, "w+") as file:
            json.dump(self.as_dict(), file, indent=4)

    def __setitem__(self, parametrization: Parametrization, result: Outcome):
        outcome = parametrization, result
        self.outcomes.append(outcome)

    def __repr__(self):
        return repr(self.as_dict())
