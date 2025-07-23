from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generator, Self

import jax
from jax import Array

from src.data import Dataset, PeakExtractionAlias, DatasetAlias, Split
from src.rkhs import Kernel, pairwise_squared_mmd, gaussian_transformation
from src.util import generate_random_keys

NONE = object()


@dataclass(frozen=True)
class RandomizationConfig:
    seed: int

    def rng(self) -> Generator[Array, None, None]:
        return generate_random_keys(self.seed)


@dataclass(frozen=True)
class DatasetConfig:
    name: DatasetAlias
    peak_extraction: PeakExtractionAlias
    ignore_labels: list[str]
    positive_labels: list[str]
    min_peaks: int
    exact_peaks: int | None
    balance: list[str] | None
    subsample: int | None

    @classmethod
    def coat(
            cls,
            peak_extraction: PeakExtractionAlias = "xqrs",
            ignore_labels: list[str] = NONE,
            min_peaks: int = 50,
            exact_peaks: int | None = None,
            balance: list[str] | None = None,
            subsample: int | None = None,
    ) -> Self:
        if ignore_labels is NONE:
            ignore_labels = ["unknown"]

        return DatasetConfig(
            name="coat",
            peak_extraction=peak_extraction,
            ignore_labels=ignore_labels,
            positive_labels=["AFIB"],
            min_peaks=min_peaks,
            exact_peaks=exact_peaks,
            balance=balance,
            subsample=subsample
        )

    @classmethod
    def sph(
            cls,
            peak_extraction: PeakExtractionAlias = "xqrs",
            ignore_labels: list[str] = NONE,
            min_peaks: int = 8,
            exact_peaks: int | None = None,
            balance: list[str] | None = None,
            subsample: int | None = None,
    ) -> Self:
        if ignore_labels is NONE:
            ignore_labels = []

        return DatasetConfig(
            name="sph",
            peak_extraction=peak_extraction,
            ignore_labels=ignore_labels,
            positive_labels=["AFIB"],
            min_peaks=min_peaks,
            exact_peaks=exact_peaks,
            balance=balance,
            subsample=subsample
        )

    @classmethod
    def sph_r(
            cls,
            peak_extraction: PeakExtractionAlias = "xqrs",
            min_peaks: int = 8,
            exact_peaks: int | None = None,
            balance: list[str] | None = None,
            subsample: int | None = None,
    ) -> Self:
        return cls.sph(
            peak_extraction=peak_extraction,
            ignore_labels=["AF", "AT", "SA"],
            min_peaks=min_peaks,
            exact_peaks=exact_peaks,
            balance=balance,
            subsample=subsample
        )

    @classmethod
    def cinc(
            cls,
            peak_extraction: PeakExtractionAlias = "xqrs",
            ignore_labels: list[str] = NONE,
            min_peaks: int = 20,
            exact_peaks: int | None = None,
            balance: list[str] | None = None,
            subsample: int | None = None,
    ) -> Self:
        if ignore_labels is NONE:
            ignore_labels = ["~"]

        return DatasetConfig(
            name="cinc",
            peak_extraction=peak_extraction,
            ignore_labels=ignore_labels,
            positive_labels=["A"],
            min_peaks=min_peaks,
            exact_peaks=exact_peaks,
            balance=balance,
            subsample=subsample
        )

    @classmethod
    def cinc_r(
            cls,
            peak_extraction: PeakExtractionAlias = "xqrs",
            min_peaks: int = 20,
            exact_peaks: int | None = None,
            balance: list[str] | None = None,
            subsample: int | None = None,
    ) -> Self:
        return cls.cinc(
            peak_extraction=peak_extraction,
            ignore_labels=["~", "O"],
            min_peaks=min_peaks,
            exact_peaks=exact_peaks,
            balance=balance,
            subsample=subsample
        )

    def __post_init__(self):
        assert self.balance is None or len(self.balance) >= 0
        assert self.subsample is None or self.subsample >= 1
        assert self.min_peaks >= 0

    def preprocess(self, dataset: Dataset, key: Array) -> Dataset:
        dataset = dataset.remove_labels(self.ignore_labels)
        dataset = dataset.filter_min_peaks(self.min_peaks)

        if self.exact_peaks is not None:
            dataset = dataset.split_peaks(self.exact_peaks, exact=True)

        if self.balance is not None:
            key, key_subsample = jax.random.split(key)
            count_positive = dataset.count_labels_names(self.balance)
            n = 2 * min(count_positive, len(dataset) - count_positive)
            dataset = dataset.subsample_binary_balanced(n, self.balance, key)

        if self.subsample is not None:
            key, key_subsample = jax.random.split(key)
            n_subsample = min(self.subsample, dataset.count_labels_names(self.positive_labels))

            dataset = dataset.subsample_stratified_constrained(
                labels=self.positive_labels,
                n=n_subsample,
                key=key_subsample
            )

        return dataset

    def load(self, split: Split) -> Dataset:
        return Dataset.load(self.name, split=split, peak_extraction=self.peak_extraction)

    def train(self, key: Array) -> Dataset:
        dataset = self.load(split="train")
        return self.preprocess(dataset, key)

    def validation(self, key: Array) -> Dataset:
        dataset = self.load(split="validation")
        return self.preprocess(dataset, key)

    def test(self, key: Array) -> Dataset:
        dataset = self.load(split="test")
        return self.preprocess(dataset, key)

    def evaluation(self, test: bool, key: Array) -> Dataset:
        if test:
            return self.test(key)
        else:
            return self.validation(key)


@dataclass(frozen=True)
class KernelMatrixConfig(ABC):
    @abstractmethod
    def __call__(self, xs_1: Array, xs_2: Array, mask_1: Array, mask_2: Array) -> Array:
        raise NotImplementedError


@dataclass(frozen=True)
class DistributionalKernelMatrixConfig(KernelMatrixConfig):
    bandwidth_base: float
    bandwidth_mmd: float
    cached: bool = True

    def __call__(self, xs_1: Array, xs_2: Array, mask_1: Array, mask_2: Array) -> Array:
        kernel = Kernel.gaussian(self.bandwidth_base)
        squared_mmd = pairwise_squared_mmd(kernel, xs_1, xs_2, mask_1, mask_2, cached=self.cached)
        return gaussian_transformation(squared_mmd, self.bandwidth_mmd)
