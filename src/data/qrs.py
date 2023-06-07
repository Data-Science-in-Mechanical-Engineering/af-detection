from abc import ABC, abstractmethod

import numpy as np


class PeakExtractionAlgorithm(ABC):
    name: str

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def __call__(self, ecg: np.ndarray) -> np.ndarray:
        raise NotImplementedError
