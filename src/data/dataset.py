from __future__ import annotations

import os.path
import pickle
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar, Generic, Callable, Iterable, final

import numpy as np

from src.data.qrs import PeakDetectionAlgorithm, XQRSPeakDetectionAlgorithm
from src.data.util import RANDOM_SEED, COATPath, SPHPath

TLabel = TypeVar("TLabel", str, int)


@dataclass(frozen=True)
class Identifier(ABC):
    pass


@dataclass(frozen=True)
class SPHIdentifier(Identifier):
    filename: str


@dataclass(frozen=True)
class COATIdentifier(Identifier):
    device_id: int
    patient_id: int


@dataclass(frozen=True)
class Entry(Generic[TLabel]):
    identifier: Identifier
    ecg_signal: np.ndarray
    qrs_complexes: np.ndarray
    label: TLabel


class ECGDataset(ABC, Generic[TLabel]):
    FREQUENCY: float

    n: int
    identifiers: list[Identifier]
    ecg_signals: list[np.ndarray]
    qrs_complexes: list[np.ndarray]
    labels: np.ndarray

    @classmethod
    def load_from_folder(cls, folder: Path, qrs_algorithm: PeakDetectionAlgorithm, lead: int = 0) -> ECGDataset:
        """ Creates a dataset instance from files in a given folder.

        The folder must contain an ECG file (.npy), a QRS complex file (.npy), a label file (.npy), and an identifiers
        file (.pickle). It is implicitly assumed that the order of the entries in these files match each other, i.e.
        that the i-th entry in one file corresponds to the i-th entry in another file.

        Args:
            folder: The folder from which to load
            qrs_algorithm: The peak extraction algorithm to use for R peak detection.
            lead: The lead to extract data for.

        Returns:
            The dataset in the given folder.
        """
        ecg_path = folder / f"ecg_lead_{lead}.npy"
        qrs_path = folder / f"qrs_{qrs_algorithm.name}_lead_{lead}.npy"
        label_path = folder / f"labels.npy"
        identifier_path = folder / "identifiers.pickle"

        for path in [ecg_path, qrs_path, label_path, identifier_path]:
            assert os.path.exists(path), f"{path} doesn't exist."

        with open(ecg_path, "rb") as ecg_file:
            # noinspection PyTypeChecker
            ecg_signals = np.load(ecg_file, allow_pickle=True)

        with open(qrs_path, "rb") as qrs_file:
            # noinspection PyTypeChecker
            qrs_complexes = np.load(qrs_file, allow_pickle=True)

        with open(label_path, "rb") as label_file:
            # noinspection PyTypeChecker
            labels = np.load(label_file)

        with open(identifier_path, "rb") as identifier_file:
            identifiers = pickle.load(identifier_file)

        return cls(
            identifiers,
            list(ecg_signals),
            list(qrs_complexes),
            labels
        )

    @final
    def __init__(
            self,
            identifiers: list[Identifier],
            ecg_signals: list[np.ndarray],
            qrs_complexes: list[np.ndarray],
            labels: np.ndarray
    ):
        assert all(ecg.ndim == 1 for ecg in ecg_signals)
        assert all(qrs.ndim == 1 for qrs in qrs_complexes)
        assert labels.ndim == 1

        self.n = len(identifiers)
        assert self.n == len(ecg_signals)
        assert self.n == len(qrs_complexes)
        assert self.n == labels.shape[0]

        self.identifiers = identifiers
        self.ecg_signals = [ecg.astype(np.float) for ecg in ecg_signals]
        self.qrs_complexes = [r_peaks.astype(np.int) for r_peaks in qrs_complexes]
        self.labels = labels

    def count_labels(self) -> dict[str, int]:
        return {
            str(key): int(count)
            for key, count in Counter(self.labels).items()
        }

    def label_domain(self) -> set[TLabel]:
        return set(self.labels)

    def subsample(self, chunks: dict[TLabel, int], seed: int = RANDOM_SEED) -> ECGDataset:
        """ Creates a random subset of the dataset containing the specified number of samples for each class.

        For each specified class, we uniformly choose the specified number of indices for that class without
        replacement. Finally, we subsample the dataset at the randomly drawn indices.
        Classes that are not specified will be dropped entirely.

        Args:
            chunks: The number of samples for each class (assumed to be 0 if a class is not present).
            seed: The random seed to set initially.

        Returns:
            A randomly sub-sampled dataset containing the specified number of samples per class.
        """
        np.random.seed(seed)
        indices = []

        # for each specified class, choose the specified number of indices for that class without replacement
        for label, size in chunks.items():
            label_indices, = np.where(self.labels == label)  # extract indices with that label
            chosen_indices = np.random.choice(label_indices, size, replace=False)  # randomly choose indices
            indices.append(chosen_indices)

        # subsample the dataset at the chosen indices
        indices = np.concatenate(indices)
        return self._subsample_from_indices(indices)

    def filter(self, condition: Callable[[Entry[TLabel]], bool]) -> ECGDataset:
        """ Creates a subset of the dataset containing exactly those entries that match a generic condition.

        Args:
            condition: The condition by which to filter the dataset.

        Returns:
            A subset of the dataset containing all those entries that match the condition.
        """
        return self._subsample_from_indices([
            index for index, entry in enumerate(self.__iter__())
            if condition(entry)
        ])

    def balanced_binary_partition(self, label_group: set[TLabel], size: int) -> ECGDataset:
        assert label_group < self.label_domain()
        np.random.seed(RANDOM_SEED)

        # obtain indices that partition the dataset in accordance to the label partition
        is_group_1 = np.isin(self.labels, list(label_group))
        is_group_2 = ~is_group_1
        indices_group_1, = np.where(is_group_1)
        indices_group_2, = np.where(is_group_2)
        assert indices_group_1.size >= size and indices_group_2.size >= size

        # uniformly sample the index partition without replacement
        group_1 = np.random.choice(indices_group_1, size, replace=False)
        group_2 = np.random.choice(indices_group_2, size, replace=False)
        partition = np.concatenate([group_1, group_2])

        # subsample the dataset from the randomly drawn indices
        return self._subsample_from_indices(partition)

    def _subsample_from_indices(self, indices: list[int] | np.ndarray) -> ECGDataset:
        assert all(0 <= index < self.n for index in indices)

        # filter identifiers, ECGs, QRS complexes, and labels at given indices
        identifiers = list(map(self.identifiers.__getitem__, indices))  # identifiers at indices
        ecg_signals = list(map(self.ecg_signals.__getitem__, indices))  # ECGs at indices
        qrs_complexes = list(map(self.qrs_complexes.__getitem__, indices))  # QRS complexes at indices
        labels = self.labels[indices]  # labels at indices

        # create new dataset using the consisting of the filtered entries
        return self._copy_factory(identifiers, ecg_signals, qrs_complexes, labels)

    def __or__(self, other: ECGDataset) -> ECGDataset:
        assert type(other) == type(self)
        identifiers_set = set(self.identifiers)

        identifiers = self.identifiers.copy()
        ecg_signals = self.ecg_signals.copy()
        qrs_complexes = self.qrs_complexes.copy()

        other_labels = []

        for entry in other:
            if entry.identifier not in identifiers_set:
                identifiers.append(entry.identifier)
                ecg_signals.append(entry.ecg_signal)
                qrs_complexes.append(entry.qrs_complexes)
                other_labels.append(entry.label)

        labels = np.concatenate([self.labels, other_labels])

        return self._copy_factory(identifiers, ecg_signals, qrs_complexes, labels)

    def __iter__(self) -> Iterable[Entry[TLabel]]:
        data = zip(self.identifiers, self.ecg_signals, self.qrs_complexes, self.labels)

        for identifier, ecg, qrs, label in data:
            yield Entry(identifier, ecg, qrs, label)

    def __repr__(self):
        return f"{type(self).__name__}({repr(self.count_labels())})"

    @abstractmethod
    def _copy_factory(
            self,
            identifiers: list[Identifier],
            ecg_signals: list[np.ndarray],
            qrs_complexes: list[np.ndarray],
            labels: np.ndarray
    ) -> ECGDataset:
        raise NotImplementedError


class COATDataset(ECGDataset):
    DURATION_IN_SEC = 60
    FREQUENCY = int(12_000 / DURATION_IN_SEC)

    # class labels
    noAF = 0
    AF = 1
    UNKNOWN = 2

    @staticmethod
    def load_train(qrs_algorithm: PeakDetectionAlgorithm = XQRSPeakDetectionAlgorithm()) -> COATDataset:
        return COATDataset.load_from_folder(COATPath.TRAIN_DATA, qrs_algorithm)

    @staticmethod
    def load_validate(qrs_algorithm: PeakDetectionAlgorithm = XQRSPeakDetectionAlgorithm()) -> COATDataset:
        return COATDataset.load_from_folder(COATPath.VALIDATE_DATA, qrs_algorithm)

    @staticmethod
    def load_test(qrs_algorithm: PeakDetectionAlgorithm = XQRSPeakDetectionAlgorithm()) -> COATDataset:
        return COATDataset.load_from_folder(COATPath.TEST_DATA, qrs_algorithm)

    def _copy_factory(
            self,
            identifiers: list[Identifier],
            ecg_signals: list[np.ndarray],
            qrs_complexes: list[np.ndarray],
            labels: np.ndarray
    ) -> COATDataset:
        return COATDataset(identifiers, ecg_signals, qrs_complexes, labels)


class SPHDataset(ECGDataset):
    DURATION_IN_SEC = 10
    FREQUENCY = 500

    # class labels
    AFIB = "AFIB"  # this is atrial fibrillation
    SA = "SA"
    SB = "SB"
    SR = "SR"
    SVT = "SVT"
    AT = "AT"
    AF = "AF"  # this is atrial flutter, NOT atrial fibrillation
    ST = "ST"
    AVRT = "AVRT"

    @staticmethod
    def _load_from_sub_folder(
            folder: Path,
            denoised: bool,
            qrs_algorithm: PeakDetectionAlgorithm
    ) -> SPHDataset:
        sub_folder = SPHPath.DENOISED_FOLDER if denoised else SPHPath.NOISY_FOLDER
        return SPHDataset.load_from_folder(folder / sub_folder, qrs_algorithm)

    @staticmethod
    def load_train(
            denoised: bool = True,
            qrs_algorithm: PeakDetectionAlgorithm = XQRSPeakDetectionAlgorithm()
    ) -> SPHDataset:
        return SPHDataset._load_from_sub_folder(SPHPath.TRAIN_DATA, denoised, qrs_algorithm)

    @staticmethod
    def load_validate(
            denoised: bool = True,
            qrs_algorithm: PeakDetectionAlgorithm = XQRSPeakDetectionAlgorithm()
    ) -> SPHDataset:
        return SPHDataset._load_from_sub_folder(SPHPath.VALIDATE_DATA, denoised, qrs_algorithm)

    @staticmethod
    def load_test(
            denoised: bool = True,
            qrs_algorithm: PeakDetectionAlgorithm = XQRSPeakDetectionAlgorithm()
    ) -> SPHDataset:
        return SPHDataset._load_from_sub_folder(SPHPath.TEST_DATA, denoised, qrs_algorithm)

    def _copy_factory(
            self,
            identifiers: list[Identifier],
            ecg_signals: list[np.ndarray],
            qrs_complexes: list[np.ndarray],
            labels: np.ndarray
    ) -> SPHDataset:
        return SPHDataset(identifiers, ecg_signals, qrs_complexes, labels)
