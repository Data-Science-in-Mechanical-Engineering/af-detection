from __future__ import annotations

from abc import abstractmethod, ABC

import numpy as np
from sklearn.svm import SVC

from ..method.kernels import AbstractKernel


class BaseClassifier(ABC):
    kernel: AbstractKernel
    x: np.ndarray | None

    def __init__(self, kernel: AbstractKernel):
        """
        :param kernel: kernel method that is used to embed data into RKHS.
        Must already be completely initialised.
        :type kernel: AbstractKernel
        """
        self.kernel = kernel
        self.x = None

    @abstractmethod
    def fit(self, x: np.ndarray, y: np.ndarray):
        raise NotImplementedError

    @abstractmethod
    def predict(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class SVClassifier(BaseClassifier, ABC):
    classifier: SVC

    def __init__(self, kernel: AbstractKernel, c: float = 1.0):
        """ Constructs an SVM classifier.

        Args:
            kernel: The kernel method used to compute the similarity between two trajectories.
            c: Regularisation parameter for the SVM.
        """
        super().__init__(kernel)
        self.classifier = SVC(C=c, kernel="precomputed")

    @abstractmethod
    def compute_kernel_matrix(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def fit(self, x: np.ndarray, labels: np.ndarray):
        self.x = x
        kernel_matrix = self.compute_kernel_matrix(x, x)
        self.classifier.fit(kernel_matrix, labels)

    def predict(self, x: np.ndarray) -> np.ndarray:
        kernel_matrix = self.compute_kernel_matrix(x, self.x)
        return self.classifier.predict(kernel_matrix)

    def decision_function(self, x: np.ndarray):
        kernel_matrix = self.compute_kernel_matrix(x, self.x)
        return self.classifier.decision_function(kernel_matrix)


class SVMKMEClassifier(SVClassifier):
    def compute_kernel_matrix(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        pairwise_kernel = self.kernel.pairwise(x, y)
        assert pairwise_kernel.ndim == 4
        return pairwise_kernel.mean(axis=(-1, -2))


class SVMVarianceClassifier(SVClassifier):
    def compute_kernel_matrix(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return np.array([
            [self.kernel(xi.var().reshape(1), yj.var().reshape(1)) for yj in y]
            for xi in x
        ])


class SVMFeatureVectorClassifier(SVClassifier):
    def compute_kernel_matrix(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        assert x.shape[2] == x.shape[3] == y.shape[2] == y.shape[3] == 1

        n_features = min(x.shape[1], y.shape[1])
        x = x[:, :n_features, :, :]
        y = y[:, :n_features, :, :]

        return np.array([
            [self.kernel(xi.flatten(), yj.flatten()) for yj in y]
            for xi in x
        ])
