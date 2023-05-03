import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    pairwise_distances,
    precision_score,
    recall_score,
)

# TODO: fix imports (make them relative)
from src.data.dataset import ECGDataset, HospitalDataset
from src.method.features import extract_sub_trajectories, extract_rri_data
from src.method.kernels import RBFKernel
from src.method.svm_classifier import SVCClassifier


def main(
    train_set: ECGDataset,
    val_set: ECGDataset,
    experiment_number: int,
    mixing_speed: int,
    offset: int,
    distance_fraction: float,
    cs: [float],
    scalar: float,
    n_jobs: int = 1,
    results_folder: str = "../../results",  # TODO: Flagged for refactoring.
    dname: str = "qrs_sub_trajecs",  # TODO: Flagged for refactoring.
    quantile: float = 0.5,
):  # pylint: disable=missing-function-docstring, too-many-arguments, too-many-locals, invalid-name

    mean_distance = train_set.distance_between_qrs_complexes(
        "mean", labels=train_set.RHYTHMS["AF"]
    )
    length_trajectory = int(distance_fraction * mean_distance)

    (train_trajectory_indices, _,) = extract_sub_trajectories(
        mixing_speed, offset, length_trajectory, train_set
    )
    val_trajectory_indices, _ = extract_sub_trajectories(
        mixing_speed, offset, length_trajectory, val_set
    )
    # create mask for elements that are dropped because not enough qrs
    # complexes are detected (see _extract_sub_trajectories for details)
    # reduce data further to consider only SR and AFIB
    train_mask = get_data_mask(
        train_set, train_trajectory_indices, n_instances_per_class=300
    )
    val_mask = get_data_mask(
        val_set, val_trajectory_indices, n_instances_per_class=200
    )
    train_spike_time_data = extract_rri_data(
        train_trajectory_indices, train_set.qrs_complexes[0]
    )
    val_spike_time_data = extract_rri_data(
        val_trajectory_indices, val_set.qrs_complexes[0]
    )

    # spike_time_bandwidth = 0.04
    # TODO: Flagged for refactoring. This is a heuristic and should go in the `heuristics` module.
    squared_distances = pairwise_distances(
        train_spike_time_data[train_mask],
    )
    band_width = np.sqrt(
        np.quantile(squared_distances[squared_distances > 0], q=quantile)
        / 2
        / scalar
    )

    spike_time_kernel = RBFKernel(band_width)

    score_functions = {
        "accuracy": accuracy_score,
        "f1-score": f1_score,
        "precision": precision_score,
        "recall": recall_score,
    }

    overall_scores = {}
    X_train = train_spike_time_data[train_mask]
    y_train = _get_binary_labels(train_mask, train_set)

    X_val = val_spike_time_data[val_mask]
    y_val = _get_binary_labels(val_mask, val_set)
    for c in cs:  # pylint: disable=invalid-name
        clf = SVCClassifier(kernel=spike_time_kernel, C=c, n_jobs=n_jobs)
        clf.fit(X_train, y_train)
        y_hat = clf.predict(X_val)

        scores = {
            score: function(y_val, y_hat)
            for score, function in score_functions.items()
        }

        overall_scores[c] = scores

    # TODO: dump results (configuration -> metrics)


def _get_binary_labels(mask, data_set):
    y_train = np.where(
        np.isin(
            data_set.labels[mask],
            [data_set.RHYTHMS["AFIB"], data_set.RHYTHMS["AF"]],
        ),
        0,
        1,
    )
    return y_train


# TODO: Flagged for refactoring. This should already be taken care of by the Dataset.
def get_data_mask(data_set, trajectory_indices, n_instances_per_class):
    mask_afib = (~np.isclose(trajectory_indices, 0).all(axis=(1, 2))) & (
        np.isin(
            data_set.labels, [data_set.RHYTHMS["AFIB"], data_set.RHYTHMS["AF"]]
        )
    )
    mask_healty = (~np.isclose(trajectory_indices, 0).all(axis=(1, 2))) & (
        ~np.isin(
            data_set.labels, [data_set.RHYTHMS["AFIB"], data_set.RHYTHMS["AF"]]
        )
    )
    rng = np.random.default_rng(seed=42)
    mask = np.zeros_like(mask_afib, dtype=np.bool)
    mask[
        rng.choice(
            np.where(mask_afib)[0], n_instances_per_class, replace=False
        )
    ] = True
    mask[
        rng.choice(
            np.where(mask_healty)[0],
            n_instances_per_class,
            replace=False,
        )
    ] = True
    return mask


if __name__ == "__main__":
    main(
        experiment_number=25,
        mixing_speed=0,
        offset=HospitalDataset.FREQUENCY * 5,
        distance_fraction=0.5,
        cs=np.logspace(-1, 1, 5),
        scalar=1,
    )
