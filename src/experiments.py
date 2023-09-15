from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from itertools import product
from typing import Generator, TypeVar, Generic, Callable, Iterable, Iterator, final, Final

import numpy as np

from .data.dataset import ECGDataset, SPHDataset
from .method.features import extract_normalized_rri, extract_smooth_pre_peak_trajectories
from .method.kernels import BaseKernel, RBFKernel, ProductKernel
from .method.svm_classifier import SVClassifier, SVMKMEClassifier
from .results import Outcome, Result, Description

KernelT = TypeVar("KernelT", bound=BaseKernel)
ClassifierT = TypeVar("ClassifierT", bound=SVClassifier)
ClassifierFactory = Callable[[KernelT, float, float], ClassifierT]
FeatureLike = np.ndarray | list[np.ndarray]


def make_binary_labels(labels: np.ndarray, values_1: set) -> np.ndarray:
    values_1 = list(values_1)
    return np.isin(labels, values_1).astype(int)


@dataclass(frozen=True)
class Parametrization:
    c: float
    c_class_weight_proportion: float


@dataclass(frozen=True)
class ParametrizationRRI(Parametrization):
    bandwidth: float


@dataclass(frozen=True)
class ParametrizationTrajectoryRRI(Parametrization):
    bandwidth_rri: float
    bandwidth_trajectory: float
    pre_peak_trajectory_time: float
    pre_peak_trajectory_encoding_dim: int


ParametrizationT = TypeVar("ParametrizationT", bound=Parametrization)


class Experiment(ABC, Generic[KernelT, ParametrizationT]):
    cs: Final[list[float]]
    c_class_weight_proportions: Final[list[float]]
    classifier_factory: Final[ClassifierFactory]
    result: Final[Result]

    def __init__(
        self,
        name: str,
        cs: Iterable[float],
        c_class_weight_proportions: Iterable[float],
        classifier_factory: ClassifierFactory
    ):
        cs = list(cs)
        c_class_weight_proportions = list(c_class_weight_proportions)

        assert all(c > 0 for c in cs)
        assert all(proportion > 0 for proportion in c_class_weight_proportions)

        self.cs = cs
        self.c_class_weight_proportions = c_class_weight_proportions
        self.classifier_factory = classifier_factory
        self.result = Result(name)

    @final
    def __call__(
        self,
        dataset_train: ECGDataset,
        dataset_validate: ECGDataset,
        af_labels_train: set,
        af_labels_validate: set,
        description: Description | None = None
    ) -> list[Outcome]:
        assert af_labels_train <= dataset_train.label_domain()
        assert af_labels_validate <= dataset_validate.label_domain()

        # make sure labels are binary (1 = AF, 0 = noAF)
        labels_train = make_binary_labels(dataset_train.labels, af_labels_train)
        labels_validate = make_binary_labels(dataset_validate.labels, af_labels_validate)

        outcomes = self.result.add({
            "dataset_train": dataset_train.description(),
            "dataset_validate": dataset_validate.description(),
            "af_labels_train": list(af_labels_train),
            "af_labels_validate": list(af_labels_validate)
        } | (description if description is not None else {}))

        for parametrization in self.parametrizations():
            features_train = self.features(dataset_train, parametrization)
            features_validate = self.features(dataset_validate, parametrization)
            kernel = self.kernel_factory(parametrization)

            n_positive = labels_train.sum()
            n_negative = labels_train.size - n_positive

            classifier = self.classifier_factory(
                kernel,
                parametrization.c,
                parametrization.c_class_weight_proportion * n_negative / n_positive
            )

            classifier.fit(features_train, labels_train)
            predictions_validate = classifier.predict(features_validate)
            description = asdict(parametrization)
            distances = classifier.decision_function(features_validate)

            outcome = Outcome.evaluate(description, dataset_validate, predictions_validate, labels_validate, distances)
            print(f"{outcome} \t {parametrization}")
            outcomes.append(outcome)

        return outcomes

    def _default_parametrizations(self) -> Iterator[Parametrization]:
        parameters = product(self.cs, self.c_class_weight_proportions)

        for c, c_class_weight_proportion in parameters:
            yield Parametrization(c, c_class_weight_proportion)

    @abstractmethod
    def parametrizations(self) -> Iterator[ParametrizationT]:
        raise NotImplementedError

    @abstractmethod
    def features(self, dataset: ECGDataset, parametrization: ParametrizationT) -> FeatureLike:
        raise NotImplementedError

    @abstractmethod
    def kernel_factory(self, parametrization: ParametrizationT) -> KernelT:
        raise NotImplementedError


class ExperimentRRI(Experiment[RBFKernel, ParametrizationRRI]):
    bandwidths: Final[list[float]]
    feature_cache: dict[ECGDataset, np.ndarray]

    def __init__(
        self,
        cs: Iterable[float],
        c_class_weight_proportions: Iterable[float],
        classifier_factory: ClassifierFactory,
        bandwidths: Iterable[float]
    ):
        bandwidths = list(bandwidths)
        assert all(bandwidth > 0 for bandwidth in bandwidths)

        super().__init__("svm rri", cs, c_class_weight_proportions, classifier_factory)
        self.bandwidths = bandwidths
        self.feature_cache = {}

    def parametrizations(self) -> Generator[ParametrizationRRI, Outcome, None]:
        for parametrization in super()._default_parametrizations():
            for bandwidth in self.bandwidths:
                yield ParametrizationRRI(
                    parametrization.c,
                    parametrization.c_class_weight_proportion,
                    bandwidth
                )

    def features(self, dataset: ECGDataset, parametrization: ParametrizationRRI) -> np.ndarray:
        if dataset not in self.feature_cache:
            normalized_rri = extract_normalized_rri(dataset.qrs_complexes)
            self.feature_cache[dataset] = normalized_rri[:, :, None]

        return self.feature_cache[dataset]

    def kernel_factory(self, parametrization: ParametrizationRRI) -> RBFKernel:
        return RBFKernel(parametrization.bandwidth)


class ExperimentTrajectoryRRI(Experiment[ProductKernel, ParametrizationTrajectoryRRI]):
    bandwidths_rri: Final[list[float]]
    bandwidths_trajectory: Final[list[float]]
    pre_peak_trajectory_times: Final[list[float]]
    pre_peak_trajectory_encoding_dims: Final[list[int]]

    def __init__(
        self,
        cs: Iterable[float],
        c_class_weight_proportions: Iterable[float],
        classifier_factory: ClassifierFactory,
        bandwidths_rri: Iterable[float],
        bandwidths_trajectory: Iterable[float],
        pre_peak_trajectory_times: Iterable[float],
        pre_peak_trajectory_encoding_dims: Iterable[int]
    ):
        bandwidths_rri = list(bandwidths_rri)
        bandwidths_trajectory = list(bandwidths_trajectory)
        pre_peak_trajectory_times = list(pre_peak_trajectory_times)
        pre_peak_trajectory_encoding_dims = list(pre_peak_trajectory_encoding_dims)

        assert all(bandwidth > 0 for bandwidth in bandwidths_rri)
        assert all(bandwidth > 0 for bandwidth in bandwidths_trajectory)
        assert all(trajectory_time > 0 for trajectory_time in pre_peak_trajectory_times)
        assert all(encoding_dim >= 1 for encoding_dim in pre_peak_trajectory_encoding_dims)

        super().__init__("svm trajectory rri", cs, c_class_weight_proportions, classifier_factory)
        self.bandwidths_rri = bandwidths_rri
        self.bandwidths_trajectory = bandwidths_trajectory
        self.pre_peak_trajectory_times = pre_peak_trajectory_times
        self.pre_peak_trajectory_encoding_dims = pre_peak_trajectory_encoding_dims

    def parametrizations(self) -> Iterator[ParametrizationTrajectoryRRI]:
        parameters = product(
            self.bandwidths_rri,
            self.bandwidths_trajectory,
            self.pre_peak_trajectory_times,
            self.pre_peak_trajectory_encoding_dims
        )

        for parametrization in super()._default_parametrizations():
            for bandwidth_rri, bandwidth_trajectory, trajectory_time, trajectory_encoding_dim in parameters:
                yield ParametrizationTrajectoryRRI(
                    parametrization.c,
                    parametrization.c_class_weight_proportion,
                    bandwidth_rri,
                    bandwidth_trajectory,
                    trajectory_time,
                    trajectory_encoding_dim
                )

    def features(self, dataset: ECGDataset, parametrization: ParametrizationTrajectoryRRI) -> list[np.ndarray]:
        normalized_rri = extract_normalized_rri(dataset.qrs_complexes)
        normalized_rri_trajectories = normalized_rri[:, :, None]

        normalized_pre_peak_trajectories = extract_smooth_pre_peak_trajectories(
            dataset.ecg_signals,
            dataset.qrs_complexes,
            parametrization.pre_peak_trajectory_time,
            dataset.FREQUENCY,
            parametrization.pre_peak_trajectory_encoding_dim
        )

        return [normalized_rri_trajectories, normalized_pre_peak_trajectories]

    def kernel_factory(self, parametrization: ParametrizationTrajectoryRRI) -> ProductKernel:
        kernel_rri = RBFKernel(parametrization.bandwidth_rri)
        kernel_trajectory = RBFKernel(parametrization.bandwidth_trajectory)
        return ProductKernel([kernel_rri, kernel_trajectory])


if __name__ == "__main__":
    ds_train = SPHDataset.load_train().filter(lambda entry: len(entry.qrs_complexes) > 7)
    ds_validate = SPHDataset.load_validate().filter(lambda entry: len(entry.qrs_complexes) > 7)

    af_classes_train = {ds_train.AFIB}
    af_classes_validate = {ds_validate.AFIB}

    ds_train = ds_train.balanced_binary_partition(af_classes_train, 1061)  # 1061
    ds_validate = ds_validate.balanced_binary_partition(af_classes_validate, 355)  # 355

    experiment = ExperimentRRI(
        cs=np.linspace(0.01, 20, 20),
        c_class_weight_proportions=np.logspace(-1, 1, 20, base=5),
        classifier_factory=SVMKMEClassifier,
        bandwidths=[0.05]
    )

    experiment(ds_train, ds_validate, af_classes_train, af_classes_validate)
    experiment.result.save()
