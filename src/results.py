from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import NamedTuple, final, Final, Any, Callable, Hashable, TypeVar
from uuid import uuid4, UUID

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score

from .data.dataset import ECGDataset

Description = dict[str]

if "__file__" in globals():
    RESULTS_FOLDER = Path(__file__).parent.parent / "results"
else:
    RESULTS_FOLDER = Path().resolve().parent / "results"


def specificity_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return recall_score(y_true, y_pred, pos_label=0)


def distance_by_identifier(identifiers: list[str], distances: list[float]) -> dict[str, float]:
    if not distances:
        return {identifier: math.inf for identifier in identifiers}

    assert len(identifiers) == len(distances)

    return {
        identifier: distance
        for identifier, distance in zip(identifiers, distances)
    }


def compute_confusion(
        predictions_binary: np.ndarray,
        labels_binary: np.ndarray,
        actual_labels: np.ndarray,
        prediction_mapping: dict[Any, str] | None = None,
        label_mapping: dict[Any, str] | None = None
) -> dict[str, dict[str, int]]:
    assert predictions_binary.ndim == 1
    assert labels_binary.ndim == 1
    assert actual_labels.ndim == 1
    assert predictions_binary.size == labels_binary.size == actual_labels.size

    confusion = defaultdict(lambda: defaultdict(lambda: 0))

    for group_prediction, group_label, actual_label in zip(predictions_binary, labels_binary, actual_labels):
        if prediction_mapping is not None:
            group_prediction = prediction_mapping[group_prediction]
        if label_mapping is not None:
            actual_label = label_mapping[actual_label]

        confusion[str(group_prediction)][str(actual_label)] += 1

    return confusion


@final
class Snapshot(NamedTuple):
    outcomes: list[Outcome]
    setup: Description

    @staticmethod
    def from_dict(data: dict[str]) -> Snapshot:
        outcomes = [Outcome.from_dict(outcome) for outcome in data["outcomes"]]
        return Snapshot(outcomes, data["setup"])

    @property
    def size_train(self) -> int:
        return sum(self.setup["dataset_train"]["composition"].values())

    @property
    def size_validate(self) -> int:
        return sum(self.setup["dataset_validate"]["composition"].values())

    def __iter__(self):
        yield from self.outcomes

    def max(self, key: Callable[[Outcome], float | int]) -> Outcome:
        return max(self.outcomes, key=key)

    def filter(self, condition: Callable[[Outcome], bool]) -> Snapshot:
        filtered_outcomes = filter(condition, self.outcomes)
        return Snapshot(list(filtered_outcomes), self.setup)

    def partition(self, key: Callable[[Outcome], KeyT]) -> dict[KeyT, Snapshot]:
        partition = {}

        for outcome in self.outcomes:
            outcome_key = key(outcome)

            if outcome_key not in partition:
                partition[outcome_key] = Snapshot([], self.setup)

            partition[outcome_key].outcomes.append(outcome)

        return partition


@final
@dataclass(frozen=True)
class Outcome:
    idx: UUID
    parametrization: dict[str]
    accuracy: float
    f1: float
    precision: float
    recall: float
    specificity: float
    confusion: dict[str, dict[str]]
    false_positives: list[str]
    false_negatives: list[str]
    false_positive_distances: list[float]
    false_negative_distances: list[float]

    @staticmethod
    def evaluate(
            parametrization: dict[str],
            dataset_validate: ECGDataset,
            predictions_binary: np.ndarray,
            labels_binary: np.ndarray,
            distances: np.ndarray | None = None
    ) -> Outcome:
        assert predictions_binary.ndim == 1
        assert labels_binary.ndim == 1
        assert predictions_binary.size == labels_binary.size == dataset_validate.labels.size

        accuracy = accuracy_score(labels_binary, predictions_binary)
        f1 = f1_score(labels_binary, predictions_binary)
        precision = precision_score(labels_binary, predictions_binary)
        recall = recall_score(labels_binary, predictions_binary)
        specificity = specificity_score(labels_binary, predictions_binary)

        confusion = compute_confusion(
            predictions_binary,
            labels_binary,
            dataset_validate.labels,
            {0: "noAFIB", 1: "AFIB"}
        )

        false_positives = np.argwhere(labels_binary < predictions_binary).T[0]
        false_negatives = np.argwhere(labels_binary > predictions_binary).T[0]

        false_positive_identifiers = [str(dataset_validate.identifiers[index]) for index in false_positives]
        false_negative_identifiers = [str(dataset_validate.identifiers[index]) for index in false_negatives]
        false_positive_distances = distances[false_positives] if distances is not None else []
        false_negative_distances = distances[false_negatives] if distances is not None else []

        return Outcome(
            uuid4(),
            parametrization,
            accuracy,
            f1,
            precision,
            recall,
            specificity,
            confusion,
            false_positive_identifiers,
            false_negative_identifiers,
            list(false_positive_distances),
            list(false_negative_distances)
        )

    @staticmethod
    def from_dict(data: dict[str]) -> Outcome:
        return Outcome(
            data["id"],
            data["parametrization"],
            data["scores"]["accuracy"],
            data["scores"]["f1"],
            data["scores"]["precision"],
            data["scores"]["recall"],
            data["scores"]["specificity"],
            data["confusion"],
            data["false_positives"],
            data["false_negatives"],
            data["false_positive_distances"] if "false_positive_distances" in data else [],
            data["false_negative_distances"] if "false_negative_distances" in data else []
        )

    def false_positive_distance_by_identifier(self) -> dict[str, float]:
        return distance_by_identifier(self.false_positives, self.false_positive_distances)

    def false_negative_distance_by_identifier(self) -> dict[str, float]:
        return distance_by_identifier(self.false_negatives, self.false_negative_distances)

    def as_dict(self) -> dict[str]:
        return {
            "id": str(self.idx),
            "parametrization": self.parametrization,
            "scores": {
                "accuracy": self.accuracy,
                "f1": self.f1,
                "precision": self.precision,
                "recall": self.recall,
                "specificity": self.specificity
            },
            "confusion": self.confusion,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "false_positive_distances": self.false_positive_distances,
            "false_negative_distances": self.false_negative_distances
        }

    def __repr__(self):
        return f"acc: {100 * self.accuracy:.2f}% f1: {100 * self.f1:.2f}% prec: {100 * self.precision:.2f}% " \
               f"rec: {100 * self.recall:.2f}%"


KeyT = TypeVar("KeyT", bound=Hashable)


@final
class Result:
    name: Final[str]
    created_time: Final[datetime]
    snapshots: list[Snapshot]

    @staticmethod
    def from_json(path: str | Path) -> Result:
        with open(path) as file:
            data = json.load(file)

        created_time = datetime.fromisoformat(data["created"])
        snapshots = [Snapshot.from_dict(snapshot) for snapshot in data["runs"]]
        return Result(data["name"], created_time, snapshots)

    def __init__(self, name: str, created_time: datetime | None = None, snapshots: list[Snapshot] | None = None):
        self.name = name
        self.created_time = datetime.now() if created_time is None else created_time
        self.snapshots = [] if snapshots is None else snapshots

    def __iter__(self):
        yield from self.snapshots

    def add(self, setup: Description) -> list[Outcome]:
        outcomes = []
        snapshot = Snapshot(outcomes, setup)
        self.snapshots.append(snapshot)
        return outcomes

    def as_dict(self) -> dict[str]:
        return {
            "name": self.name,
            "created": self.created_time.isoformat(),
            "runs": [
                {
                    "setup": snapshot.setup,
                    "outcomes": [outcome.as_dict() for outcome in snapshot.outcomes]
                }
                for snapshot in self.snapshots
            ]
        }

    def save(self, filename: str | None = None) -> Path:
        if filename is None:
            now = datetime.now()
            filename = now.strftime('%Y-%m-%d %H-%M-%S.%f')[:-3]

        experiment_folder = RESULTS_FOLDER / self.name
        experiment_folder.mkdir(parents=True, exist_ok=True)
        path = experiment_folder / f"{filename}.json"

        with open(path, "w+") as file:
            json.dump(self.as_dict(), file, indent=4)

        return path

    def filter(self, condition: Callable[[Snapshot], bool]) -> Result:
        snapshots = filter(condition, self.snapshots)
        return Result(self.name, self.created_time, list(snapshots))

    def partition(self, key: Callable[[Snapshot], KeyT]) -> dict[KeyT, Result]:
        results = {}

        for snapshot in self:
            snapshot_key = key(snapshot)

            if snapshot_key not in results:
                results[snapshot_key] = Result(self.name, self.created_time)

            results[snapshot_key].snapshots.append(snapshot)

        return results

    def partition_by_datasets(self) -> dict[tuple[str, str], Result]:
        def datasets(snapshot: Snapshot) -> tuple[str, str]:
            return snapshot.setup["dataset_train"]["name"], snapshot.setup["dataset_validate"]["name"]

        return self.partition(datasets)

    def max(self, key: Callable[[Outcome], float | int]):
        return max((snapshot.max(key) for snapshot in self.snapshots), key=key)
