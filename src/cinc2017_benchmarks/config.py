from dataclasses import dataclass

import jax.numpy as jnp

from src.config import DatasetConfig, RandomizationConfig
from src.data import Dataset
from src.metrics import Evaluation


@dataclass(frozen=True)
class Config(RandomizationConfig):
    dataset: DatasetConfig
    entry_id: str

    def build_dataset(self) -> Dataset:
        rng = self.rng()
        return self.dataset.test(next(rng)).permute(next(rng))


@dataclass(frozen=True)
class Result:
    predictions: dict[str, str]

    def evaluate(self, dataset: Dataset, positive_labels: list[str]) -> Evaluation:
        binary_predictions = jnp.array([
            int(self.predictions[identifier] == 'A')
            for identifier in dataset.identifiers
        ])

        binary_labels = dataset.binarize_labels(positive_labels)

        dummy_scores = jnp.ones_like(binary_predictions)

        return Evaluation.from_predictions(
            labels_binary=binary_labels,
            predictions=binary_predictions,
            scores=dummy_scores,
            dataset=dataset
        )
