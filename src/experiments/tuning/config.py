from abc import ABC, abstractmethod
from dataclasses import dataclass
from statistics import mean
from typing import Iterator, Any, NamedTuple, Literal

import jax
from frozendict import frozendict
from jax import numpy as jnp, Array

from src.config import RandomizationConfig, DatasetConfig
from src.data import Dataset
from src.features import features
from src.metrics import Run
from src.rkhs import Kernel, pairwise_squared_mmd, gaussian_transformation

type PerformanceCriterion = Literal["roc-auc", "f1"]


class Parametrization(NamedTuple):
    kernel_matrix: dict[str, Any]
    c: float
    rho: float


class CVIndexSplit(NamedTuple):
    train: Array
    validation: Array


@dataclass(frozen=True)
class KernelMatrixParameters(ABC):
    @abstractmethod
    def iter(self, xs: Array, mask: Array) -> Iterator[tuple[dict[str, Any], Array]]:
        raise NotImplementedError

    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError


@dataclass(frozen=True)
class DistributionalKernelMatrixParameters(KernelMatrixParameters):
    bandwidth_base: list[float]
    bandwidth_mmd: list[float]
    cached: bool

    def iter(self, xs: Array, mask: Array) -> Iterator[tuple[dict[str, Any], Array]]:
        for bandwidth_base in self.bandwidth_base:
            kernel = Kernel.gaussian(bandwidth_base)
            squared_mmd = pairwise_squared_mmd(kernel, xs, xs, mask, mask, cached=self.cached)

            for bandwidth_mmd in self.bandwidth_mmd:
                kernel_matrix = gaussian_transformation(squared_mmd, bandwidth_mmd)
                yield {"bandwidth_base": bandwidth_base, "bandwidth_mmd": bandwidth_mmd}, kernel_matrix,

    def __len__(self) -> int:
        return len(self.bandwidth_base) * len(self.bandwidth_mmd)


@dataclass(frozen=True)
class Config(RandomizationConfig):
    dataset: DatasetConfig
    kernel_matrix: KernelMatrixParameters
    c: list[float]
    rho: list[float]
    k_fold_cv: int
    criterion: PerformanceCriterion
    top_k_results: int | None

    @property
    def n_parametrizations(self) -> int:
        return len(self.kernel_matrix) * len(self.c) * len(self.rho)

    def iter_kernel_matrices(self, dataset: Dataset) -> Iterator[tuple[dict[str, Any], Array]]:
        xs, mask = features(dataset.peak_indices, dataset.mask)
        kernel_matrices = self.kernel_matrix.iter(xs, mask)

        for kernel_matrix_parameters, kernel_matrix in kernel_matrices:
            yield kernel_matrix_parameters, kernel_matrix

    def iter(self, dataset: Dataset) -> Iterator[tuple[Parametrization, Array]]:
        for kernel_matrix_parameters, kernel_matrix in self.iter_kernel_matrices(dataset):
            for c in self.c:
                for rho in self.rho:
                    parametrization = Parametrization(
                        kernel_matrix=frozendict(kernel_matrix_parameters),
                        c=c,
                        rho=rho,
                    )

                    yield parametrization, kernel_matrix

    def build_training_dataset(self) -> Dataset:
        key = next(self.rng())
        key_preprocess, key_permutation = jax.random.split(key)

        train = self.dataset.load(split="train")
        validation = self.dataset.load(split="validation")
        merged = self.dataset.preprocess(train | validation, key_preprocess)

        return merged.permute(key_permutation)

    def _cv_splits(self, dataset: Dataset, sizes: list[int], key: Array) -> list[CVIndexSplit]:
        assert sum(sizes) <= len(dataset)

        folds = []
        indices = jnp.arange(len(dataset))

        for size in sizes:
            key, key_subsample = jax.random.split(key)
            dataset_indices = dataset.subsample_stratified_indices(size, key_subsample)
            fold_indices = indices[dataset_indices]

            folds.append(fold_indices)
            indices = jnp.setdiff1d(indices, fold_indices)
            dataset = dataset.remove_indices(dataset_indices)

        splits = []

        for i in range(self.k_fold_cv):
            validation_indices = folds[i]
            training_indices = jnp.concatenate(folds[:i] + folds[i + 1:])
            split = CVIndexSplit(train=training_indices, validation=validation_indices)
            splits.append(split)

        return splits

    def cv_indices(self, dataset: Dataset, key: Array) -> list[CVIndexSplit]:
        q, r = divmod(len(dataset), self.k_fold_cv)
        sizes = [q + 1] * r + [q] * (self.k_fold_cv - r)
        return self._cv_splits(dataset, sizes, key)

    def performance(self, runs: list[Run]) -> float:
        if self.criterion == "roc-auc":
            criterion = lambda evaluation: evaluation.roc_auc
        elif self.criterion == "f1":
            criterion = lambda evaluation: evaluation.f1
        else:
            raise ValueError(f"Unknown performance criterion: {self.criterion}")

        return mean(criterion(run.evaluation_validation) for run in runs)


type Result = dict[Parametrization, list[Run]]
