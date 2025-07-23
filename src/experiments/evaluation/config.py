from dataclasses import dataclass
from typing import NamedTuple

from src.config import RandomizationConfig, DatasetConfig
from src.metrics import Run


@dataclass(frozen=True)
class Config(RandomizationConfig):
    tuning_run: str
    evaluation: dict[str, DatasetConfig]
    cache: bool


class RocCurve(NamedTuple):
    fpr: list[float]
    tpr: list[float]
    thresholds: list[float]


class SingleEvaluation(NamedTuple):
    run: Run
    roc_curve: RocCurve


type Result = dict[str, SingleEvaluation]
