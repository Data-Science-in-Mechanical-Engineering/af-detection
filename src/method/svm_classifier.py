from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import final

import numpy as np
from sklearn.svm import SVC

from ..method.kernels import BaseKernel, ProductKernel, SimpleKernel


class BaseClassifier(ABC):
    kernel: BaseKernel
    x: np.ndarray | None

    def __init__(self, kernel: BaseKernel):
        self.kernel = kernel
        self.x = None

    def _check_validity(self, x: np.ndarray | list[np.ndarray]):
        if isinstance(self.kernel, SimpleKernel) and not isinstance(x, np.ndarray):
            raise TypeError("Kernel is of type SimpleKernel and can only deal with np.ndarrays as feature structure")
        elif isinstance(self.kernel, ProductKernel) and not isinstance(x, list):
            raise TypeError("Kernel is of type ProductKernel and can only deal with with a list as feature structure")

    @abstractmethod
    def fit(self, x: np.ndarray | list[np.ndarray], y: np.ndarray | list[np.ndarray]):
        raise NotImplementedError

    @abstractmethod
    def predict(self, x: np.ndarray | list[np.ndarray]) -> np.ndarray:
        raise NotImplementedError


class SVClassifier(BaseClassifier, ABC):
    classifier: SVC
    x: np.ndarray | None

    @final
    def __init__(self, kernel: BaseKernel, c: float = 1.0, class_weight_proportion: float = 1.0):
        """Constructs an SVM classifier.

        Args:
            kernel: The kernel method used to compute the similarity between two trajectories.
            c: TODO
            class_weight_proportion: TODO
        """
        super().__init__(kernel)
        self.classifier = SVC(C=c, class_weight={0: 1.0, 1: class_weight_proportion}, kernel="precomputed")
        self.x = None

    @abstractmethod
    def compute_kernel_matrix(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def fit(self, x: np.ndarray | list[np.ndarray], labels: np.ndarray):
        self._check_validity(x)
        self.x = x
        kernel_matrix = self.compute_kernel_matrix(x, x)
        self.classifier.fit(kernel_matrix, labels)

    def predict(self, x: np.ndarray | list[np.ndarray]) -> np.ndarray:
        self._check_validity(x)
        assert self.x is not None
        kernel_matrix = self.compute_kernel_matrix(x, self.x)
        return self.classifier.predict(kernel_matrix)

    def decision_function(self, x: np.ndarray):
        self._check_validity(x)
        kernel_matrix = self.compute_kernel_matrix(x, self.x)
        return self.classifier.decision_function(kernel_matrix)


class SVMKMEClassifier(SVClassifier):
    """SVM Classifier using kernel mean embedding as kernel matrix.

    We apply the kernel to each pair of trajectories of each pair of patients.
    We then take the mean over these kernel values to obtain the kernel mean embedding.
    """

    def compute_kernel_matrix(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return self.kernel.double_pairwise_kme(x, y)


class SVMMeanKernelClassifier(SVClassifier):
    """SVM Classifier using a kernel matrix of averaged pairwise trajectory distances.

    We apply the kernel to the mean distances between each pair of trajectories for each pair of patients to obtain the
    kernel matrix.
    """

    def compute_kernel_matrix(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return self.kernel.double_pairwise_mean(x, y)


class SVMVarianceClassifier(SVClassifier):
    """SVM Classifier using a kernel matrix of trajectory variances.

    We only consider the variances of trajectories as features.
    We then apply the kernel to the variances for each pair of patients to obtain the kernel matrix.
    """

    def compute_kernel_matrix(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        assert x.shape[2] == y.shape[2] == 1

        x_var = x.var(axis=1, keepdims=True)
        y_var = y.var(axis=1, keepdims=True)
        return self.kernel.double_pairwise(x_var, y_var).reshape(x.shape[0], y.shape[0])


class SVMFeatureVectorClassifier(SVClassifier):
    def compute_kernel_matrix(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        assert x.shape[2] == y.shape[2] == 1

        # make sure we have the same number of features between patients
        n_features = min(x.shape[1], y.shape[1])
        x = x[:, -n_features:, :]
        y = y[:, -n_features:, :]

        # make sure we have only one trajectory such that only the last dimension is compared per pair of patients
        x = x.reshape((x.shape[0], 1, n_features))
        y = y.reshape((y.shape[0], 1, n_features))

        # normalize vectors by the square root of their dimension
        # this is to make euclidean distances independent of the vector dimension (on expectation)
        x = x / math.sqrt(n_features)
        y = y / math.sqrt(n_features)

        return self.kernel.double_pairwise(x, y).reshape(x.shape[0], y.shape[0])
