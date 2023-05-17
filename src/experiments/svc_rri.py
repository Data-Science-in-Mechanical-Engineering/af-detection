from typing import Iterable, Any

import numpy as np

from ..data.dataset import ECGDataset, COATDataset
from ..experiments.util import ExperimentTracker, METRICS, make_binary_labels
from ..method.features import extract_normalized_rri
from ..method.kernels import RBFKernel
from ..method.svm_classifier import SVCClassifier


def svc_rri(
        name: str,
        description: dict[str, Any],
        train_ds: ECGDataset,
        validate_ds: ECGDataset,
        train_af_labels: set,
        validate_af_labels: set,
        c: float,
        bandwidths: Iterable[float]
) -> ExperimentTracker:
    assert train_af_labels <= train_ds.label_domain()
    assert validate_af_labels <= validate_ds.label_domain()

    # extract normalized RR intervals as feature
    rri_train = extract_normalized_rri(train_ds.qrs_complexes)
    rri_validate = extract_normalized_rri(validate_ds.qrs_complexes)

    # make sure labels are binary (1 = AF, 0 = noAF)
    labels_train = make_binary_labels(train_ds.labels, train_af_labels)
    labels_validate = make_binary_labels(validate_ds.labels, validate_af_labels)

    # make sure data has correct shape (n_instances, n_trajectories, length_trajectory, dim_trajectory)
    # every RRI is considered a 1D trajectory
    rri_train = rri_train[:, :, None, None]
    rri_validate = rri_validate[:, :, None, None]

    tracker = ExperimentTracker(name, {
        "train": repr(train_ds),
        "validate": repr(validate_ds)
    }, description)

    for bandwidth in bandwidths:
        kernel = RBFKernel(bandwidth)
        classifier = SVCClassifier(kernel, c)
        classifier.fit(rri_train, labels_train)

        predictions_validate = classifier.predict(rri_validate)

        scores = {
            name: metric(labels_validate, predictions_validate)
            for name, metric in METRICS.items()
        }

        tracker[{"c": c, "bandwidth": bandwidth}] = scores

        print(f"c={c}, sigma={bandwidth}: {scores}")

    return tracker


DESCRIPTION = {}

if __name__ == "__main__":
    train_data = COATDataset.load_train() \
        .filter(lambda entry: len(entry.qrs_complexes) > 50) \
        .balanced_binary_partition({COATDataset.AF}, 200)

    validate_data = COATDataset.load_validate() \
        .filter(lambda entry: len(entry.qrs_complexes) > 50) \
        .balanced_binary_partition({COATDataset.AF}, 70)

    print(train_data)
    print(validate_data)

    result = svc_rri(
        "SVM RRI",
        DESCRIPTION,
        train_data,
        validate_data,
        {COATDataset.AF},
        {COATDataset.AF},
        10,
        np.logspace(-2, 0, 30)
    )

    result.save()
