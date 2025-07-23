import random
import warnings
import zipfile
from abc import ABC, abstractmethod
from collections import Counter
from collections.abc import Sized
from functools import cached_property, cache
from pathlib import Path
from typing import Final, Self, Iterable, Literal, NamedTuple, Mapping

import jax
import jax.numpy as jnp
import neurokit2 as nk
import numpy as np
import pandas as pd
import tqdm
import tyro
from jax import Array
from scipy.io import loadmat
from wfdb.processing import XQRS

from src.util import download_file

type DatasetAlias = Literal["coat", "sph", "cinc"]
type Split = Literal["train", "validation", "test"]

type PeakExtractionAlias = Literal[
    "xqrs", "neurokit", "pantompkins1985", "christov2004", "elgendi2010", "hamilton2002", "rodrigues2021", "zong2003"
]

DIR_ROOT = Path(__file__).parent.parent

DIR_DATA = DIR_ROOT / "data" / "processed"
DIR_COAT = DIR_DATA / "coat"
DIR_SPH = DIR_DATA / "sph"
DIR_CINC = DIR_DATA / "cinc"

DIR_RAW_DATA = DIR_ROOT / "data" / "raw"
DIR_RAW_COAT = DIR_RAW_DATA / "coat"
DIR_RAW_SPH = DIR_RAW_DATA / "sph"
DIR_RAW_CINC = DIR_RAW_DATA / "cinc"

FILE_NAME_LABELS = "labels.npy"
FILE_NAME_ECGS = "ecgs.npy"
FILE_NAME_IDENTIFIERS = "identifiers.npy"

SAMPLING_FREQUENCY = {
    "coat": 200,
    "sph": 500,
    "cinc": 300
}


def file_name_peaks(peak_extraction: PeakExtractionAlias) -> str:
    return f"{peak_extraction}_peaks.npy"


class Dataset(Sized):
    frequency: float
    identifiers: Final[list[str]]
    ecgs: Final[Array]
    peak_indices: Final[Array]
    labels: Final[Array]
    label_num_mapping: Final[dict[int, str]]
    label_name_mapping: Final[dict[str, int]]

    @property
    def mask(self) -> Array:
        return self.peak_indices == -1

    @property
    def label_names(self) -> set[str]:
        return set(self.label_name_mapping)

    @property
    def label_nums(self) -> set[int]:
        return set(self.label_num_mapping)

    @cached_property
    def n_peaks(self) -> Array:
        return (~self.mask).sum(axis=-1)

    @cached_property
    def label_count(self) -> Counter[int]:
        return Counter(map(int, self.labels))

    @cached_property
    def label_name_count(self) -> Counter[str]:
        return Counter({
            self.label_num_mapping[label]: count
            for label, count in self.label_count.items()
        })

    @cached_property
    def peak_count(self) -> Counter[int]:
        return Counter(map(int, self.n_peaks))

    @cached_property
    def ecg_lengths(self) -> Array:
        return self.ecgs.shape[1] - self._ecg_trailing_zeros()

    @classmethod
    def from_disk(
            cls, base: Path, label_names: dict[int, str], peak_extraction: PeakExtractionAlias, frequency: float
    ) -> Self:
        identifiers = np.load(base / FILE_NAME_IDENTIFIERS, allow_pickle=True).tolist()

        ecgs = jnp.load(base / FILE_NAME_ECGS)
        labels = np.load(base / FILE_NAME_LABELS, allow_pickle=True)

        for label, name in label_names.items():
            labels[labels == name] = label

        labels = labels.astype(np.int32)
        labels = jnp.asarray(labels)

        peaks = jnp.load(base / file_name_peaks(peak_extraction))

        return cls(
            frequency=frequency,
            ecgs=ecgs,
            identifiers=identifiers,
            peak_indices=peaks,
            labels=labels,
            label_num_mapping=label_names
        )

    @classmethod
    @cache
    def load(cls, name: DatasetAlias, split: Split, peak_extraction: PeakExtractionAlias) -> Self:
        if name == "coat":
            return cls.coat(split, peak_extraction)
        elif name == "sph":
            return cls.sph(split, peak_extraction)
        elif name == "cinc":
            return cls.cinc(split, peak_extraction)
        else:
            raise ValueError(f"Unknown dataset {name}.")

    @classmethod
    def coat(cls, split: Split, peak_extraction: PeakExtractionAlias) -> Self:
        label_names = {
            0: "noAFIB",
            1: "AFIB",
            3: "unknown"
        }

        return cls.from_disk(
            base=DIR_COAT / split,
            label_names=label_names,
            peak_extraction=peak_extraction,
            frequency=SAMPLING_FREQUENCY["coat"]
        )

    @classmethod
    def sph(cls, split: Split, peak_extraction: PeakExtractionAlias) -> Self:
        label_names = {
            0: "SB",
            1: "SR",
            2: "AFIB",
            4: "ST",
            5: "AF",
            6: "SI",
            7: "SVT",
            8: "AT",
            9: "AVNRT",
            10: "AVRT",
            11: "SAAWR",
            12: "SA",
        }

        return cls.from_disk(
            base=DIR_SPH / split,
            label_names=label_names,
            peak_extraction=peak_extraction,
            frequency=SAMPLING_FREQUENCY["sph"]
        )

    @classmethod
    def cinc(cls, split: Split, peak_extraction: PeakExtractionAlias) -> Self:
        label_names = {
            0: "N",
            1: "A",
            2: "O",
            3: "~"
        }

        return cls.from_disk(
            base=DIR_CINC / split,
            label_names=label_names,
            peak_extraction=peak_extraction,
            frequency=SAMPLING_FREQUENCY["cinc"]
        )

    def __init__(
            self, identifiers: list[str], ecgs: Array, peak_indices: Array, labels: Array,
            label_num_mapping: dict[int, str], frequency: float
    ):
        assert len(label_num_mapping) == len(set(label_num_mapping.values()))

        self.frequency = frequency
        self.identifiers = identifiers
        self.ecgs = ecgs
        self.peak_indices = peak_indices
        self.labels = labels
        self.label_num_mapping = label_num_mapping
        self.label_name_mapping = {name: label for label, name in label_num_mapping.items()}

    def count_labels_names(self, labels: Iterable[str] | str) -> int:
        if isinstance(labels, str):
            labels = [labels]
        return sum(self.label_name_count[label] for label in labels)

    def _ecg_trailing_zeros(self):
        trailing_zeros = jnp.argmax(self.ecgs[:, ::-1] != 0, axis=-1)
        trailing_zeros = trailing_zeros.at[(self.ecgs == 0).all(axis=-1)].set(self.ecgs.shape[1])
        return trailing_zeros

    def identifier_indices(self, identifiers: Iterable[str]) -> Array:
        identifiers = set(identifiers)

        return jnp.array([
            i for i, identifier in enumerate(self.identifiers)
            if identifier in identifiers
        ])

    def binarize_labels(self, positive_classes: Iterable[str]) -> Array:
        positive_classes = set(positive_classes)

        assert all(name in self.label_name_mapping for name in positive_classes)

        positive_labels = jnp.array([self.label_name_mapping[name] for name in positive_classes])
        return jnp.isin(self.labels, positive_labels).astype(np.int32)

    def remove_labels(self, labels: Iterable[str] | str) -> Self:
        if isinstance(labels, str):
            labels = {labels}
        else:
            labels = set(labels)

        indices = jnp.array([
            i for i in range(len(self))
            if self.label_num_mapping[self.labels[i].item()] not in labels
        ])

        new_dataset = self[indices]

        return Dataset(
            frequency=self.frequency,
            identifiers=new_dataset.identifiers,
            ecgs=new_dataset.ecgs,
            peak_indices=new_dataset.peak_indices,
            labels=new_dataset.labels,
            label_num_mapping={
                label: name for label, name in self.label_num_mapping.items()
                if name not in labels
            }
        )

    def subsample_stratified_indices(self, n: int, key: Array) -> Array:
        assert 1 <= n <= len(self)

        target_sizes = {
            frozenset({label}): n * count // len(self)
            for label, count in self.label_count.items()
        }

        total = sum(target_sizes.values())

        for labels, target in target_sizes.items():
            if total < n and target < sum(self.label_count[label] for label in labels):
                target_sizes[labels] += 1
                total += 1

        return self.subsample_indices(target_sizes, key)

    def subsample_stratified(self, n: int, key: Array) -> Self:
        indices = self.subsample_stratified_indices(n, key)
        return self[indices]

    def subsample_stratified_constrained(self, labels: Iterable[str] | str, n: int, key: jax.Array) -> Self:
        labels = list(set(labels))

        total_labels = sum(self.label_name_count[label] for label in labels)
        assert n >= 1
        assert total_labels >= n

        fraction = n / total_labels

        target_sizes = {
            frozenset({label}): round(fraction * count)
            for label, count in self.label_count.items()
            if self.label_num_mapping[label] not in labels
        }

        target_sizes[frozenset(self.label_name_mapping[label] for label in labels)] = n

        return self.subsample(target_sizes, key)

    def subsample_binary_balanced_indices(self, n: int, positive_labels: Iterable[str], key: Array) -> Array:
        if isinstance(positive_labels, str):
            positive_labels = [positive_labels]
        else:
            positive_labels = list(positive_labels)

        negative_labels = set(self.label_names) - set(positive_labels)

        count_positive = self.count_labels_names(positive_labels)
        count_negative = self.count_labels_names(negative_labels)

        assert count_positive >= n // 2
        assert count_negative >= n // 2
        assert count_positive + count_negative >= n

        positive_labels = frozenset({self.label_name_mapping[name] for name in positive_labels})
        negative_labels = frozenset({self.label_name_mapping[name] for name in negative_labels})

        target_sizes = {
            positive_labels: n // 2,
            negative_labels: n // 2,
        }

        total = sum(target_sizes.values())

        if total < n and target_sizes[positive_labels] < count_positive:
            target_sizes[positive_labels] += 1
            total += 1

        if total < n and target_sizes[negative_labels] < count_negative:
            target_sizes[negative_labels] += 1
            total += 1

        return self.subsample_indices(target_sizes, key)

    def subsample_binary_balanced(self, n: int, positive_labels: Iterable[str], key: Array) -> Self:
        indices = self.subsample_binary_balanced_indices(n, positive_labels, key)
        return self[indices]

    def subsample_indices(self, target_sizes: dict[frozenset[int], int], key: Array) -> Array:
        indices = []

        for labels, size in target_sizes.items():
            key, key_sample = jax.random.split(key)

            labels = jnp.array([label for label in labels])
            label_indices = jnp.argwhere(jnp.isin(self.labels, labels)).flatten()
            sample_indices = jax.random.choice(key_sample, label_indices, (size,), replace=False)

            indices.append(sample_indices)

        return jnp.concatenate(indices)

    def subsample(self, target_sizes: dict[frozenset[int], int], key: Array) -> Self:
        indices = self.subsample_indices(target_sizes, key)
        return self[indices]

    def filter_min_peaks(self, min_peaks: int) -> Self:
        indices = jnp.argwhere(self.n_peaks >= min_peaks).flatten()
        return self[indices]

    def filter_ecg_length(self, length: float) -> Self:
        indices = jnp.argwhere(self.ecg_lengths >= length * self.frequency).flatten()
        return self[indices]

    def cut_peaks(self, n: int) -> Self:
        assert n >= 1
        assert jnp.all(self.n_peaks >= n)

        indices = jnp.arange(self.peak_indices.shape[1])
        unmasked_indices = jnp.where(~self.mask, indices[None], -1)
        last_unmasked_indices = jnp.sort(unmasked_indices, axis=-1)[:, -n:]
        last_unmasked_peak_indices = jnp.take_along_axis(self.peak_indices, last_unmasked_indices, axis=1)

        return Dataset(
            frequency=self.frequency,
            identifiers=self.identifiers,
            ecgs=self.ecgs,
            peak_indices=last_unmasked_peak_indices,
            labels=self.labels,
            label_num_mapping=self.label_num_mapping
        )

    def split_peaks(self, n_peaks: int, exact: bool = True):
        n_repetitions = self.peak_indices.shape[1] // n_peaks
        assert n_repetitions >= 1

        peak_indices = []
        ecgs = []
        identifiers = []
        labels = []

        peak_indices_np = np.array(self.peak_indices)
        ecgs_np = np.array(self.ecgs)

        for i in range(len(self)):
            identifier = self.identifiers[i]
            label = self.labels[i].item()

            for j in range(n_repetitions):
                peak_subset = peak_indices_np[i, j * n_peaks:(j + 1) * n_peaks]
                ecg_subset = ecgs_np[i, peak_subset[0]:peak_subset[-1]]

                if exact and np.any(peak_subset == -1):
                    continue
                elif np.all(peak_subset == -1):
                    continue

                peak_indices.append(peak_subset)
                ecgs.append(ecg_subset)
                identifiers.append(identifier)
                labels.append(label)

        ecgs = jnp.array(pad_2d(ecgs, fill_value=0.0, dtype=float))
        peak_indices = jnp.array(pad_2d(peak_indices, fill_value=-1, dtype=int))

        labels = jnp.array(labels)

        return Dataset(
            identifiers=identifiers,
            ecgs=ecgs,
            peak_indices=peak_indices,
            labels=labels,
            label_num_mapping=self.label_num_mapping,
            frequency=self.frequency,
        )

    def resample(self, frequency: float) -> Self:
        assert frequency > 0

        n_original = self.ecgs.shape[1]
        n_resampled = int(jnp.floor(n_original / self.frequency * frequency))

        times_original = jnp.arange(n_original) / self.frequency
        times_resampled = jnp.arange(n_resampled) / frequency

        ecgs = jnp.stack([
            jnp.interp(times_resampled, times_original, ecg)
            for ecg in self.ecgs
        ])

        return Dataset(
            identifiers=self.identifiers,
            ecgs=ecgs,
            peak_indices=self.peak_indices,
            labels=self.labels,
            label_num_mapping=self.label_num_mapping,
            frequency=frequency
        )

    def permute(self, key: Array) -> Self:
        indices = jnp.arange(len(self))
        indices = jax.random.permutation(key, indices)
        return self[indices]

    def remove_indices(self, indices: Array) -> Self:
        all_indices = jnp.arange(len(self))
        return self[jnp.setdiff1d(all_indices, indices)]

    def __len__(self) -> int:
        assert len(self.identifiers) == len(self.peak_indices)
        assert len(self.identifiers) == len(self.labels)
        return len(self.identifiers)

    def __or__(self, other: Self) -> Self:
        assert self.label_num_mapping == other.label_num_mapping

        def pad_2d(array_1: Array, array_2: Array) -> Array:
            array = jnp.full(
                shape=(array_1.shape[0] + array_2.shape[0], max(array_1.shape[1], array_2.shape[1])),
                fill_value=-1,
                dtype=array_1.dtype
            )

            array = array.at[:array_1.shape[0], :array_1.shape[1]].set(array_1)
            array = array.at[array_1.shape[0]:, :array_2.shape[1]].set(array_2)

            return array

        peak_indices = pad_2d(self.peak_indices, other.peak_indices)
        ecgs = pad_2d(self.ecgs, other.ecgs)

        return Dataset(
            frequency=self.frequency,
            identifiers=self.identifiers + other.identifiers,
            peak_indices=peak_indices,
            ecgs=ecgs,
            labels=jnp.concatenate([self.labels, other.labels]),
            label_num_mapping=self.label_num_mapping
        )

    def __getitem__(self, item: Array) -> Self:
        assert item.ndim == 1

        indices = jnp.asarray(item)

        return Dataset(
            frequency=self.frequency,
            identifiers=[self.identifiers[i] for i in indices],
            ecgs=self.ecgs[indices],
            peak_indices=self.peak_indices[indices],
            labels=self.labels[indices],
            label_num_mapping=self.label_num_mapping
        )


class Partition[T](NamedTuple):
    train: list[T]
    validation: list[T]
    test: list[T]


class IdentifierPartition(NamedTuple):
    train: list[str]
    validation: list[str]
    test: list[str]

    @classmethod
    def for_dataset(cls, name: DatasetAlias) -> Self:
        path_train = DIR_RAW_DATA / name / f"train.csv"
        path_validation = DIR_RAW_DATA / name / f"validation.csv"
        path_test = DIR_RAW_DATA / name / f"test.csv"

        if not path_train.exists():
            raise FileNotFoundError(f"File not found at {path_train}")

        if not path_validation.exists():
            raise FileNotFoundError(f"File not found at {path_validation}")

        if not path_test.exists():
            raise FileNotFoundError(f"File not found at {path_test}")

        with open(path_train, "r") as f:
            train = f.read().splitlines()

        with open(path_validation, "r") as f:
            validation = f.read().splitlines()

        with open(path_test, "r") as f:
            test = f.read().splitlines()

        return IdentifierPartition(train, validation, test)

    def split[T](self, collection: Mapping[str, T]) -> Partition[T]:
        train = [collection[identifier] for identifier in self.train]
        validation = [collection[identifier] for identifier in self.validation]
        test = [collection[identifier] for identifier in self.test]

        return Partition(
            train=train,
            validation=validation,
            test=test
        )


class PeakExtractor(ABC):
    @abstractmethod
    def __call__(self, ecg: np.ndarray, frequency: float) -> np.ndarray:
        raise NotImplementedError


class XQRSPeakExtractor(PeakExtractor):
    def __call__(self, ecg: np.ndarray, frequency: float) -> np.ndarray:
        xqrs = XQRS(sig=ecg, fs=frequency)
        xqrs.detect(verbose=False)
        return xqrs.qrs_inds


class NeuroKitPeakExtractor(PeakExtractor):
    method: Final[str]

    def __init__(self, method: str):
        super().__init__()
        self.method = method

    def __call__(self, ecg: np.ndarray, frequency: float) -> np.ndarray:
        ecg_clean = nk.ecg_clean(ecg, sampling_rate=frequency)
        _, peaks = nk.ecg_peaks(ecg_clean, sampling_rate=frequency, method=self.method, correct_artifacts=True)
        return peaks["ECG_R_Peaks"]


PEAK_EXTRACTORS: dict[PeakExtractionAlias, PeakExtractor] = {
    "xqrs": XQRSPeakExtractor(),
    "neurokit": NeuroKitPeakExtractor("neurokit"),
    "pantompkins1985": NeuroKitPeakExtractor("pantompkins1985"),
    "christov2004": NeuroKitPeakExtractor("christov2004"),
    "elgendi2010": NeuroKitPeakExtractor("elgendi2010"),
    "hamilton2002": NeuroKitPeakExtractor("hamilton2002"),
    "rodrigues2021": NeuroKitPeakExtractor("rodrigues2021"),
    "zong2003": NeuroKitPeakExtractor("zong2003")
}


def set_seed():
    random.seed(0)
    np.random.seed(0)


def pad_2d(arrays: list[np.ndarray], fill_value: np.ScalarType, dtype: np.typing.DTypeLike) -> np.ndarray:
    assert all(array.ndim == 1 for array in arrays)

    max_length = max(len(array) for array in arrays)
    padded_arrays = np.full((len(arrays), max_length), fill_value=fill_value, dtype=dtype)

    for i, array in enumerate(arrays):
        padded_arrays[i, :len(array)] = array

    return padded_arrays


def save_array(ecgs_array: np.ndarray, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "wb") as f:
        np.save(f, ecgs_array)


def save_ecgs(ecgs: list[np.ndarray], name: DatasetAlias, split: Split):
    ecgs_array = pad_2d(ecgs, fill_value=0, dtype=np.float32)
    save_array(ecgs_array, DIR_DATA / name / split / FILE_NAME_ECGS)


def save_peak_indices(peak_indices: list[np.ndarray], path: Path):
    peak_indices_array = pad_2d(peak_indices, fill_value=-1, dtype=np.int32)
    save_array(peak_indices_array, path)


def save_identifiers(identifiers: list[str], name: DatasetAlias, split: Split):
    identifiers = np.array(identifiers, dtype=object)
    save_array(identifiers, DIR_DATA / name / split / FILE_NAME_IDENTIFIERS)


def save_labels(labels: list[str], name: DatasetAlias, split: Split):
    labels = np.array(labels, dtype=object)
    save_array(labels, DIR_DATA / name / split / FILE_NAME_LABELS)


def extract_peak_indices(ecgs: list[np.ndarray], frequency: float, name: DatasetAlias, split: Split):
    for algorithm_name, peak_extractor in tqdm.tqdm(PEAK_EXTRACTORS.items(), desc=f"Extracting {split} peaks"):
        path = DIR_DATA / name / split / file_name_peaks(algorithm_name)

        if path.exists():
            warnings.warn(f"Skipping {algorithm_name} peak extraction. File already exists at {path}.")
            continue

        all_peak_indices = []

        for ecg in tqdm.tqdm(ecgs, desc=f"Algorithm: {algorithm_name}", leave=False):
            peak_indices = peak_extractor(ecg, frequency)
            all_peak_indices.append(peak_indices)

        save_peak_indices(all_peak_indices, path)


def preprocess_coat():
    set_seed()

    partition = IdentifierPartition.for_dataset("coat")

    path_labels = DIR_RAW_COAT / "COAT-EKGS_labels.csv"
    dir_ecgs = DIR_RAW_COAT / "COAT-EKGS"

    if not dir_ecgs.exists():
        raise FileNotFoundError(f"ECG directory not found at {dir_ecgs}")

    if not path_labels.exists():
        raise FileNotFoundError(f"Labels file not found at {path_labels}")

    save_identifiers(partition.train, name="coat", split="train")
    save_identifiers(partition.validation, name="coat", split="validation")
    save_identifiers(partition.test, name="coat", split="test")

    labels_df = pd.read_csv(path_labels)

    def label_name(label_num: int) -> str:
        if label_num == 0:
            return "noAFIB"
        elif label_num == 1:
            return "AFIB"
        elif label_num == 3:
            return "unknown"
        else:
            raise ValueError(f"Invalid label number: {label_num}")

    labels = partition.split({
        row["PatientId"]: label_name(row["ECG physician overread_0NoAF_1ScreenAF_3unknown"])
        for _, row in labels_df.iterrows()
    })

    save_labels(labels.train, name="coat", split="train")
    save_labels(labels.validation, name="coat", split="validation")
    save_labels(labels.test, name="coat", split="test")

    ecgs = partition.split({
        ecg_path.stem: np.loadtxt(ecg_path, delimiter=",", dtype=np.float32)
        for ecg_path in tqdm.tqdm(dir_ecgs.iterdir(), desc="Collecting ECGs")
    })

    save_ecgs(ecgs.train, name="coat", split="train")
    save_ecgs(ecgs.validation, name="coat", split="validation")
    save_ecgs(ecgs.test, name="coat", split="test")

    extract_peak_indices(ecgs.train, SAMPLING_FREQUENCY["coat"], name="coat", split="train")
    extract_peak_indices(ecgs.validation, SAMPLING_FREQUENCY["coat"], name="coat", split="validation")
    extract_peak_indices(ecgs.test, SAMPLING_FREQUENCY["coat"], name="coat", split="test")


def preprocess_sph():
    set_seed()

    partition = IdentifierPartition.for_dataset("sph")

    path_labels = DIR_RAW_SPH / "Diagnostics.xlsx"
    dir_ecgs = DIR_RAW_SPH / "ECGDataDenoised"
    dir_ecgs_zip = DIR_RAW_SPH / "ECGDataDenoised.zip"

    if not path_labels.exists():
        download_file(url="https://figshare.com/ndownloader/files/15653771", path=path_labels)
    if not dir_ecgs.exists() and not dir_ecgs_zip.exists():
        download_file(url="https://figshare.com/ndownloader/files/15652862", path=dir_ecgs_zip)

    if not dir_ecgs.exists() and dir_ecgs_zip.exists():
        print(f"unzipping {dir_ecgs_zip} to {dir_ecgs} ...")

        with zipfile.ZipFile(dir_ecgs.with_suffix(".zip"), "r") as zip_ref:
            zip_ref.extractall(dir_ecgs.parent)

    if not dir_ecgs.exists():
        raise FileNotFoundError(f"ECG directory not found at {dir_ecgs}")

    if not path_labels.exists():
        raise FileNotFoundError(f"Labels file not found at {path_labels}")

    save_identifiers(partition.train, name="sph", split="train")
    save_identifiers(partition.validation, name="sph", split="validation")
    save_identifiers(partition.test, name="sph", split="test")

    labels_df = pd.read_excel(path_labels)

    labels = partition.split({
        row["FileName"]: row["Rhythm"]
        for _, row in labels_df.iterrows()
    })

    save_labels(labels.train, name="sph", split="train")
    save_labels(labels.validation, name="sph", split="validation")
    save_labels(labels.test, name="sph", split="test")

    ecgs = partition.split({
        ecg_path.stem: np.loadtxt(ecg_path, delimiter=",", usecols=0, dtype=np.float32)
        for ecg_path in tqdm.tqdm(dir_ecgs.iterdir(), desc="Collecting ECGs")
    })

    save_ecgs(ecgs.train, name="sph", split="train")
    save_ecgs(ecgs.validation, name="sph", split="validation")
    save_ecgs(ecgs.test, name="sph", split="test")

    extract_peak_indices(ecgs.train, SAMPLING_FREQUENCY["sph"], name="sph", split="train")
    extract_peak_indices(ecgs.validation, SAMPLING_FREQUENCY["sph"], name="sph", split="validation")
    extract_peak_indices(ecgs.test, SAMPLING_FREQUENCY["sph"], name="sph", split="test")


def preprocess_cinc():
    set_seed()

    partition = IdentifierPartition.for_dataset("cinc")

    directory = DIR_RAW_CINC / "training2017"
    path_labels = directory / "REFERENCE.csv"
    directory_zip = DIR_RAW_CINC / "training2017.zip"

    if not directory.exists() and not directory_zip.exists():
        download_file(
            url="https://physionet.org/files/challenge-2017/1.0.0/training2017.zip?download",
            path=directory_zip
        )

    if not directory.exists() and directory_zip.exists():
        print(f"unzipping {directory_zip} to {directory} ...")

        with zipfile.ZipFile(directory_zip, "r") as zip_ref:
            zip_ref.extractall(directory.parent)

    if not directory.exists():
        raise FileNotFoundError(f"ECG directory not found at {directory}")

    if not path_labels.exists():
        raise FileNotFoundError(f"Labels file not found at {path_labels}")

    save_identifiers(partition.train, name="cinc", split="train")
    save_identifiers(partition.validation, name="cinc", split="validation")
    save_identifiers(partition.test, name="cinc", split="test")

    labels_df = pd.read_csv(path_labels, header=None, names=["identifier", "label"])

    labels = partition.split({
        row["identifier"]: row["label"]
        for _, row in labels_df.iterrows()
    })

    save_labels(labels.train, name="cinc", split="train")
    save_labels(labels.validation, name="cinc", split="validation")
    save_labels(labels.test, name="cinc", split="test")

    ecg_paths = (path for path in directory.iterdir() if path.name.endswith(".mat"))

    ecgs = partition.split({
        ecg_path.stem: loadmat(ecg_path.as_posix())["val"][0]
        for ecg_path in tqdm.tqdm(ecg_paths, desc="Collecting ECGs")
    })

    save_ecgs(ecgs.train, name="cinc", split="train")
    save_ecgs(ecgs.validation, name="cinc", split="validation")
    save_ecgs(ecgs.test, name="cinc", split="test")

    extract_peak_indices(ecgs.train, SAMPLING_FREQUENCY["cinc"], name="cinc", split="train")
    extract_peak_indices(ecgs.validation, SAMPLING_FREQUENCY["cinc"], name="cinc", split="validation")
    extract_peak_indices(ecgs.test, SAMPLING_FREQUENCY["cinc"], name="cinc", split="test")


def preprocess(dataset: DatasetAlias):
    if dataset == "coat":
        preprocess_coat()
    elif dataset == "sph":
        preprocess_sph()
    elif dataset == "cinc":
        preprocess_cinc()
    else:
        raise ValueError(f"Unknown dataset {dataset}")


if __name__ == "__main__":
    tyro.cli(preprocess, description="extract and preprocess database")
