from typing import Iterable, Any, Callable

import numpy as np

from src.data.qrs import XQRSPeakDetectionAlgorithm
from ..data.dataset import ECGDataset, SPHDataset
from ..experiments.util import ExperimentTracker, make_binary_labels, METRICS, compute_confusion
from ..method.features import extract_normalized_rri, extract_smooth_pre_peak_trajectories
from ..method.kernels import RBFKernel, ProductKernel, BaseKernel
from ..method.svm_classifier import SVMKMEClassifier, SVClassifier

FeatureExtractor = Callable[[ECGDataset], np.ndarray | list[np.ndarray]]
Classifier = Callable[[BaseKernel, float], SVClassifier]


def experiment(
        name: str,
        description: dict[str, Any],
        train_ds: ECGDataset,
        validate_ds: ECGDataset,
        train_af_labels: set,
        validate_af_labels: set,
        features: FeatureExtractor,
        cs: Iterable[float],
        kernels: Iterable[tuple[BaseKernel, dict]],
        classifier_cls: Classifier
) -> ExperimentTracker:
    assert train_af_labels <= train_ds.label_domain()
    assert validate_af_labels <= validate_ds.label_domain()

    # make sure labels are binary (1 = AF, 0 = noAF)
    labels_train = make_binary_labels(train_ds.labels, train_af_labels)
    labels_validate = make_binary_labels(validate_ds.labels, validate_af_labels)

    train_features = features(train_ds)
    validate_features = features(validate_ds)

    setup = {"train": repr(train_ds), "validate": repr(validate_ds)}
    tracker = ExperimentTracker(name, setup, description)

    for c in cs:
        for kernel, kernel_info in kernels:
            classifier = classifier_cls(kernel, c)

            classifier.fit(train_features, labels_train)
            predictions_validate = classifier.predict(validate_features)

            scores = {
                name: metric(labels_validate, predictions_validate)
                for name, metric in METRICS.items()
            }

            scores["confusion"] = compute_confusion(
                predictions_validate,
                labels_validate,
                validate_ds.labels,
                {0: "noAFIB", 1: "AFIB"}
            )

            tracker[kernel_info] = scores

            print(f"c={c}, {kernel_info}: accuracy: {scores['accuracy']}, precision: {scores['precision']}, "
                  f"recall: {scores['recall']}")

    return tracker


def svc_rri(
        description: dict[str, Any],
        train_ds: ECGDataset,
        validate_ds: ECGDataset,
        train_af_labels: set,
        validate_af_labels: set,
        cs: Iterable[float],
        bandwidths: Iterable[float],
        classifier_cls: Classifier = SVMKMEClassifier
) -> ExperimentTracker:
    def features(dataset: ECGDataset) -> np.ndarray:
        return extract_normalized_rri(dataset.qrs_complexes)[:, :, None]

    kernels = ((RBFKernel(bandwidth), {"bandwidth": bandwidth}) for bandwidth in bandwidths)

    return experiment(
        "SVM RRI",
        description,
        train_ds,
        validate_ds,
        train_af_labels,
        validate_af_labels,
        features,
        cs,
        kernels,
        classifier_cls
    )


def svc_rri_trajectory(
        description: dict[str, Any],
        train_ds: ECGDataset,
        validate_ds: ECGDataset,
        train_af_labels: set,
        validate_af_labels: set,
        cs: Iterable[float],
        bandwidths_rri: Iterable[float],
        bandwidths_ecg: Iterable[float],
        pre_peak_trajectory_time: float,
        pre_peak_trajectory_encoding_dim: int,
        classifier_cls: Classifier
) -> ExperimentTracker:
    def features(dataset: ECGDataset) -> list[np.ndarray]:
        rris = extract_normalized_rri(dataset.qrs_complexes)[:, :, None]

        trajectories = extract_smooth_pre_peak_trajectories(
            dataset.ecg_signals,
            dataset.qrs_complexes,
            pre_peak_trajectory_time,
            dataset.FREQUENCY,
            pre_peak_trajectory_encoding_dim
        )

        return [rris, trajectories]

    kernels = (
        (
            ProductKernel([RBFKernel(bandwidth_rri), RBFKernel(bandwidth_ecg)]),
            {"rri bandwidth": bandwidths_rri, "trajectory bandwidth": bandwidth_ecg}
        )
        for bandwidth_rri in bandwidths_rri
        for bandwidth_ecg in bandwidths_ecg
    )

    return experiment(
        "SVM RRI TRAJ",
        description,
        train_ds,
        validate_ds,
        train_af_labels,
        validate_af_labels,
        features,
        cs,
        kernels,
        classifier_cls
    )


if __name__ == "__main__":
    qrs_algorithm = XQRSPeakDetectionAlgorithm()

    train_data = SPHDataset.load_train(qrs_algorithm=qrs_algorithm) \
        .filter(lambda entry: len(entry.qrs_complexes) > 7) \
        .balanced_binary_partition({SPHDataset.AFIB}, 1000)

    validate_data = SPHDataset.load_validate(qrs_algorithm=qrs_algorithm) \
        .filter(lambda entry: len(entry.qrs_complexes) > 7) \
        .balanced_binary_partition({SPHDataset.AFIB}, 350)

    result = svc_rri(
        {"r peak detection algorithm": qrs_algorithm.name},
        train_data,
        validate_data,
        {SPHDataset.AFIB},
        {SPHDataset.AFIB},
        [10],
        [0.05],
        SVMKMEClassifier
    )

    result.save()
