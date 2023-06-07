from typing import Iterable, Any

from src.data.qrs import ALL_PEAK_DETECTION_ALGORITHMS
from ..data.dataset import ECGDataset, SPHDataset
from ..experiments.util import ExperimentTracker, make_binary_labels, METRICS, compute_confusion
from ..method.features import extract_normalized_rri
from ..method.kernels import RBFKernel
from ..method.svm_classifier import SVMVarianceClassifier


def svc_rri(
        name: str,
        description: dict[str, Any],
        train_ds: ECGDataset,
        validate_ds: ECGDataset,
        train_af_labels: set,
        validate_af_labels: set,
        c: float,
        bandwidths: Iterable[float],
        verbose_metrics: bool = True
) -> ExperimentTracker:
    assert train_af_labels <= train_ds.label_domain()
    assert validate_af_labels <= validate_ds.label_domain()

    # extract normalized RR intervals as feature
    rri_train = extract_normalized_rri(train_ds.qrs_complexes)
    rri_validate = extract_normalized_rri(validate_ds.qrs_complexes)

    # make sure labels are binary (1 = AF, 0 = noAF)
    labels_train = make_binary_labels(train_ds.labels, train_af_labels)
    labels_validate = make_binary_labels(validate_ds.labels, validate_af_labels)

    # make sure data has correct shape (n_instances, n_trajectories, length_trajectory)
    # every RRI is considered a 1D trajectory
    rri_train = rri_train[:, :, None]
    rri_validate = rri_validate[:, :, None]

    setup = {"train": repr(train_ds), "validate": repr(validate_ds)}
    tracker = ExperimentTracker(name, setup, description)

    for bandwidth in bandwidths:
        kernel = RBFKernel(bandwidth)
        classifier = SVMVarianceClassifier(kernel, c)

        classifier.fit(rri_train, labels_train)
        predictions_validate = classifier.predict(rri_validate)

        scores = {
            name: metric(labels_validate, predictions_validate)
            for name, metric in METRICS.items()
        }

        if verbose_metrics:
            scores["confusion"] = compute_confusion(
                predictions_validate,
                labels_validate,
                validate_ds.labels,
                {0: "noAFIB", 1: "AFIB"}
            )

        tracker[{"c": c, "bandwidth": bandwidth}] = scores

        print(f"c={c}, sigma={bandwidth}: accuracy: {scores['accuracy']}, precision: {scores['precision']}, "
              f"recall: {scores['recall']}")

    return tracker


if __name__ == "__main__":
    for algorithm_cls in ALL_PEAK_DETECTION_ALGORITHMS:
        algorithm = algorithm_cls()

        train_data = SPHDataset.load_train(qrs_algorithm=algorithm) \
            .filter(lambda entry: len(entry.qrs_complexes) > 7) \
            .balanced_binary_partition({SPHDataset.AFIB}, 1000)

        validate_data = SPHDataset.load_validate(qrs_algorithm=algorithm) \
            .filter(lambda entry: len(entry.qrs_complexes) > 7) \
            .balanced_binary_partition({SPHDataset.AFIB}, 350)

        result = svc_rri(
            "SVM RRI",
            {"r peak detection algorithm": algorithm.name},
            train_data,
            validate_data,
            {SPHDataset.AFIB},
            {SPHDataset.AFIB},
            10,
            [0.05]
        )

        result.save()
