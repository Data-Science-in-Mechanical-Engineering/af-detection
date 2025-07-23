from typing import NamedTuple, Self

import jax.numpy as jnp
from jax import Array
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.svm import SVC
from torch import Tensor

from src.data import Dataset


def specificity_score(y_true: Array, y_pred: Array) -> float:
    return recall_score(y_true, y_pred, pos_label=0)


class Evaluation(NamedTuple):
    accuracy: float
    f1: float
    precision: float
    sensitivity: float
    specificity: float
    roc_auc: float
    false_positives: list[str]
    false_negatives: list[str]
    attribution_positive: dict[str, int]
    attribution_negative: dict[str, int]

    @classmethod
    def from_predictions(cls, labels_binary: Array, predictions: Array, scores: Array, dataset: Dataset) -> Self:
        assert predictions.ndim == 1
        assert len(predictions) == len(dataset)

        accuracy = accuracy_score(labels_binary, predictions)
        f1 = f1_score(labels_binary, predictions, zero_division=0)
        precision = precision_score(labels_binary, predictions, zero_division=0)
        specificity = specificity_score(labels_binary, predictions)
        sensitivity = recall_score(labels_binary, predictions, zero_division=0)
        roc_auc = roc_auc_score(labels_binary, scores)

        false_positives = [
            dataset.identifiers[i]
            for i, (label, pred) in enumerate(zip(labels_binary, predictions))
            if label == 0 and pred == 1
        ]

        false_negatives = [
            dataset.identifiers[i]
            for i, (label, pred) in enumerate(zip(labels_binary, predictions))
            if label == 1 and pred == 0
        ]

        attribution_positive = {
            label: int(jnp.sum((predictions == 1) & (dataset.labels == dataset.label_name_mapping[label])))
            for label in dataset.label_names
        }

        attribution_negative = {
            label: int(jnp.sum((predictions == 0) & (dataset.labels == dataset.label_name_mapping[label])))
            for label in dataset.label_names
        }

        return cls(
            accuracy=accuracy,
            f1=f1,
            precision=precision,
            sensitivity=sensitivity,
            specificity=specificity,
            roc_auc=roc_auc,
            false_positives=false_positives,
            false_negatives=false_negatives,
            attribution_positive=attribution_positive,
            attribution_negative=attribution_negative
        )

    @classmethod
    def from_torch_predictions(
            cls, labels_binary: Tensor, predictions: Tensor, scores: Tensor, dataset: Dataset
    ) -> Self:
        labels_binary = jnp.asarray(labels_binary.cpu().detach())
        predictions = jnp.asarray(predictions.cpu().detach())
        scores = jnp.asarray(scores.cpu().detach())
        return cls.from_predictions(labels_binary, predictions, scores, dataset)

    @classmethod
    def empty(cls) -> Self:
        return cls(
            accuracy=0.0,
            f1=0.0,
            precision=0.0,
            sensitivity=0.0,
            specificity=0.0,
            roc_auc=0.0,
            false_positives=[],
            false_negatives=[],
            attribution_positive={},
            attribution_negative={}
        )


def biased_svm_prediction(svc: SVC, scores: Array, bias: float) -> Array:
    return jnp.where(
        scores + bias > 0,
        svc.classes_[1],
        svc.classes_[0]
    )


class Run(NamedTuple):
    evaluation_train: Evaluation
    evaluation_validation: Evaluation

    @classmethod
    def svm(
            cls,
            kernel_matrix_train: Array, kernel_matrix_evaluation: Array,
            dataset_train: Dataset, dataset_evaluation: Dataset,
            positive_labels_train: list[str], positive_labels_evaluation: list[str],
            c: float, rho: float,
            return_scores: bool = False
    ) -> Self | tuple[Self, Array, Array]:
        binary_labels_train = dataset_train.binarize_labels(positive_labels_train)
        binary_labels_evaluation = dataset_evaluation.binarize_labels(positive_labels_evaluation)

        svc = SVC(
            C=c,
            kernel="precomputed",
            class_weight={0: 1.0, 1: rho},
            shrinking=False,
            random_state=0,
            max_iter=1_000_000,
            verbose=False
        )

        svc.fit(X=kernel_matrix_train, y=binary_labels_train)

        scores_train = svc.decision_function(kernel_matrix_train)
        scores_validation = svc.decision_function(kernel_matrix_evaluation)

        predictions_train = biased_svm_prediction(svc, scores_train, bias=0)
        predictions_validation = biased_svm_prediction(svc, scores_validation, bias=0)

        run = cls(
            evaluation_train=Evaluation.from_predictions(
                binary_labels_train, predictions_train, scores_train, dataset_train
            ),
            evaluation_validation=Evaluation.from_predictions(
                binary_labels_evaluation, predictions_validation, scores_validation, dataset_evaluation
            )
        )

        if return_scores:
            return run, scores_train, scores_validation
        else:
            return run

    @classmethod
    def empty(cls) -> Self:
        return cls(
            evaluation_train=Evaluation.empty(),
            evaluation_validation=Evaluation.empty()
        )
