from abc import ABC, abstractmethod

import numpy as np
from sklearn.metrics import pairwise_kernels


def _validate_pairwise_kernels_shape(x: np.ndarray, y: np.ndarray):
    assert x.ndim == y.ndim == 4, "Inputs must have shape (n_patients, m_trajectories, length, dim)."
    assert x.shape[2] == y.shape[2], "Trajectory length must match."
    assert x.shape[3] == y.shape[3], "Trajectory dimensionality must match."


class AbstractKernel(ABC):
    """
    Base class for kernel methods.
    Concrete kernels have to implement the _transform method.

    The kernel is expected to operate on three-dimensional data that can be
    described as (number_sub_trajectories, length_sub_trajectory,
    data_dimension).
    """

    def pairwise(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        _validate_pairwise_kernels_shape(x, y)

        return np.array([
            self._transform(xi, yj)
            for xi in x
            for yj in y
        ]).reshape((x.shape[0], y.shape[0], x.shape[1], y.shape[1]))

    @abstractmethod
    def _transform(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Abstract method to implement a pairwise kernel method.
        Kernel must operate on three-dimensional input data.
        :param x: feature array
        :type x: np.ndarray of shape (m_x, s, d), where
                 m_x is the number of sub trajectories,
                 s is the length of the sub trajectories, and
                 d is the dimension of the data
        :param y: second feature array
        :type y: np.ndarray of shape (m_x, s, d), where
                 m_y is the number of sub trajectories,
                 s is the length of the sub trajectories, and
                 d is the dimension of the data
        :return: pairwise kernel matrix between sub-trajectories in x and y
        :rtype: np.ndarray of shape (m_x, m_x) if y is None, else (m_x, m_y)
        """
        ...

    def __call__(self, x: np.ndarray, y: np.ndarray = None):
        if y is None:
            y = x
        # _validate_shape(x, y)
        return self._transform(x, y)


class RBFKernel(AbstractKernel):  # pylint: disable=too-few-public-methods
    """
    Implement the pairwise (generalised) RBF kernel.
    k(x, y) = exp(-1/(2*bandwidth**2) * ||x-y||^2).

    Here, ||x-y|| is either the euclidean distance if x and y are vectors,
    i.e. length_of_subtrajectories = 1. If length_of_subtrajectories > 1,
    # x and y can be interpreted as matrices and ||x-y|| is the Frobenius
    norm.
    Using the Frobenius norm is equivalent to think about the tensor kernel
    k^s((x_1, ..., x_s), (y_1, ..., y_s)) = k_i(x_1, y_1) * ... * k_s(x_s,
    y_s) = exp(1/(bandwidth**2) * sum(||x_i - y_i||_F)), where x_i, y_i in R^d
    if k_1 = ... = k_s, as
    """

    def __init__(self, bandwidth: float, n_jobs: int | None = None):
        """

        :param bandwidth: scaling factor that is used for every pairwise
        calculation.
        :type bandwidth: float
        :param n_jobs:  The number of jobs to use for the computation. This works by breaking
        down the pairwise matrix into n_jobs even slices and computing them in
        parallel. Currently, multithreading enforced.

        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors.
        :type n_jobs: int or None
        """
        self._bandwidth = bandwidth
        self._n_jobs = n_jobs

    def pairwise(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        if x.shape[0] * y.shape[0] < (x.shape[1] + x.shape[2] + x.shape[3]) * (y.shape[1] + y.shape[2] + y.shape[3]):
            return super(RBFKernel, self).pairwise(x, y)
        else:
            return self.vectorized_pairwise(x, y)

    def vectorized_pairwise(self, x: np, y: np.ndarray):
        _validate_pairwise_kernels_shape(x, y)
        x = x.reshape((*x.shape[:-2], -1))
        y = y.reshape((*y.shape[:-2], -1))

        pairwise_differences = x[:, None, :, None] - y[None, :, None, :]
        pairwise_squared_euclidean_distances = (pairwise_differences ** 2).sum(axis=-1)
        pairwise_rbf_kernel = np.exp(-1 / 2 / (self._bandwidth ** 2) * pairwise_squared_euclidean_distances)

        return pairwise_rbf_kernel

    def _transform(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        if x.ndim == 1 or y.ndim == 1:
            assert x.ndim == y.ndim == 1
            return np.exp(-1 / 2 / self._bandwidth * ((x - y) ** 2).sum())

        m_x, m_y = x.shape[0], y.shape[0]
        # pairwise_kernels of sklearn uses different form of scaling factor
        gamma = 1 / (2 * self._bandwidth ** 2)
        # reshape is possible because Frobenius norm of a (mxn) matrix is
        # equivalent to (m*n) vector.
        return pairwise_kernels(
            x.reshape(m_x, -1),
            y.reshape(m_y, -1),
            metric="rbf",
            gamma=gamma,
            n_jobs=self._n_jobs,
        )
