import timeit
from abc import abstractmethod

import numpy as np
from sklearn.svm import SVC

from ..method.kernels import AbstractKernel


# NOTE: This is experimental
# TODO: This should be refactored into the RBF Kernel if we keep this
def vec_reduce_kernel(x: np.ndarray, y: np.ndarray, bandwidth: float) -> np.ndarray:
    if x.ndim == 4:
        x = x.reshape((*x.shape[:-2], -1))

    if y.ndim == 4:
        y = y.reshape((*y.shape[:-2], -1))

    assert x.ndim == 3
    assert y.ndim == 3
    assert x.shape[2] == y.shape[2]

    dist = np.subtract.outer(x.reshape(x.shape[0], -1), y.reshape(y.shape[0], -1)).transpose((0, 2, 1, 3)) ** 2
    exp = np.exp(-1 / 2 / (bandwidth ** 2) * dist)
    kme = exp.mean(axis=(2, 3))

    return kme


def _validate_shape(x: np.ndarray, y: np.ndarray):
    assert x.ndim == 4, (
        f"Input expected to be of shape (n_instances, n_sub_trajectories, "
        f"length_sub_trajectories, dim). However, input y has {x.ndim} != 4 "
        f"dimensions"
    )
    assert x.shape[0] == y.shape[0], (
        f"Number of instances and labels mismatch. Found {x.shape[0]} "
        f"training instances, but {y.shape[0]} labels."
    )


class BaseClassifier:
    def __init__(self, kernel: AbstractKernel):
        """
        :param kernel: kernel method that is used to embed data into RKHS.
        Must already be completely initialised.
        :type kernel: AbstractKernel
        """
        self._y = None
        self._x = None
        self._clf = None
        self.kernel = kernel

    @abstractmethod
    def fit(self, x: np.ndarray, y: np.ndarray):
        """
        Fit the classifier
        :param x: training data
        :type x: np.ndarray of shape (n_instances, n_trajectories, length_trajectory, dimension)
        :param y: labels
        :type y: np.ndarray of shape (n_instances,)
        :return: self
        :rtype: BaseClassifier
        """
        self._validate_and_store_data(x, y)

    def _validate_and_store_data(self, x, y):
        _validate_shape(x, y)
        self._x = x
        self._y = y
        return self

    @abstractmethod
    def predict(self, x: np.ndarray):
        """
        Predict the class labels for the given instances.
        :param x: Test samples
        :type x: np.ndarray of shape (n_queries, n_trajectories, length_trajectory, dimension)
        :return: Label for each sample
        :rtype: np.ndarray of shape (n_queries,)
        """
        raise NotImplementedError


class SVCClassifier(BaseClassifier):
    def __init__(self, kernel: AbstractKernel, C: float = 1.0):
        """
        :param kernel: kernel method that is used to compute the Maximum
        Mean Discrepancy (MMD). Must already be completely initialised.
        :type kernel: AbstractKernel
        :param C: regularisation parameter
        :type C: float
        """
        super().__init__(kernel)
        self._clf = SVC(C=C, kernel="precomputed")

    def fit(self, x: np.ndarray | list[np.ndarray], y: np.ndarray):
        """
        Fit the classifier
        :param x: training data
        :type x: np.ndarray of shape (n_instances, n_trajectories, length_trajectory, dimension)
        :param y: labels
        :type y: np.ndarray of shape (n_instances,)
        :return: self
        :rtype: SVCClassifier
        """
        self._x = x
        kernel_matrix = self.compute_kernel_matrix(x, x)
        self._clf.fit(kernel_matrix, y)
        return self

    def _reduce_kernel(self, x, y):
        kernel_matrix = self.kernel(x, y)
        reduced_kernel_value = np.mean(kernel_matrix)
        return reduced_kernel_value

    def compute_kernel_matrix(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        # return np.array([
        #     self._reduce_kernel(xi, yj)
        #     for xi in x
        #     for yj in y
        # ]).reshape(x.shape[0], y.shape[0])

        # TODO: This is experimental and should be refactored if we keep it
        return vec_reduce_kernel(x, y, self.kernel._bandwidth)

    def predict(self, x: np.ndarray | list[np.ndarray]):
        """
        Predict the class labels for the given instances.
        :param x: Test samples
        :type x: np.ndarray of shape (n_queries, n_trajectories, length_trajectory, dimension)
        :return: Label for each sample
        :rtype: np.ndarray of shape (n_queries,)
        """
        kernel_matrix = self.compute_kernel_matrix(x, self._x)
        return self._clf.predict(kernel_matrix)

    def decision_function(self, x: np.ndarray | list[np.ndarray]):
        """
        Predict the class labels for the given instances.
        :param x: Test samples
        :type x: np.ndarray of shape (n_queries, n_trajectories, length_trajectory, dimension)
        :return: Label for each sample
        :rtype: np.ndarray of shape (n_queries,)
        """
        kernel_matrix = self.compute_kernel_matrix(x, self._x)
        return self._clf.decision_function(kernel_matrix)
