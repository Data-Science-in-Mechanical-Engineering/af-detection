import json
from datetime import datetime
from pathlib import Path
from typing import Final, Any

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

Setup = Parametrization = dict[str, Any]
Outcome = dict[str, float]

METRICS = {
    "accuracy": accuracy_score,
    "f1-score": f1_score,
    "precision": precision_score,
    "recall": recall_score,
}


def make_binary_labels(labels: np.ndarray, values_1: set) -> np.ndarray:
    values_1 = list(values_1)
    return np.isin(labels, values_1).astype(int)


class ExperimentTracker:
    ROOT_FOLDER = Path().resolve().parent.parent / "results"
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
