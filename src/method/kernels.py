from abc import ABC, abstractmethod

import numpy as np
from sklearn.metrics import pairwise_kernels


def _validate_shape(x, y):
    assert x.ndim == 3, (
        f"Input expected to be of shape ("
        f"n_sub_trajectories, length_sub_trajectories, "
        f"dim). However, input x has {x.ndim} != 3 dimensions"
    )
    assert y.ndim == 3, (
        f"Input expected to be of shape ("
        f"n_sub_trajectories, length_sub_trajectories, "
        f"dim). However, input y has {y.ndim} != 3 dimensions"
    )
    assert (
            x.shape[1] == y.shape[1]
    ), f"The length of the sub-trajectories mismatch: {x.shape[1]} != {y.shape[1]}"
    assert x.shape[2] == y.shape[2], (
        f"The dimension of a single measured data point mismatch: "
        f"{x.shape[2]} != {y.shape[2]}"
    )


class AbstractKernel(ABC):
    """
    Base class for kernel methods.
    Concrete kernels have to implement the _transform method.

    The kernel is expected to operate on three-dimensional data that can be
    described as (number_sub_trajectories, length_sub_trajectory,
    data_dimension).
    """

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

    def _transform(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
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
