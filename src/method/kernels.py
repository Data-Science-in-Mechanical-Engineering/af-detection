from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable

import numpy as np

DEFAULT_BATCH_SIZE = 200


def double_pairwise_squared_euclidean_distances(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """ Computes the squared euclidean distances between all pairs of instances and all pairs of trajectories.

    Args:
        x: Feature matrix of shape (n_instances_x, m_trajectories_x, dim_trajectory).
        y: Feature matrix of shape (n_instances_y, m_trajectories_y, dim_trajectory).

    Returns:
        The squared euclidean distances for all pairs of instances and all pairs of trajectories.
        The shape is (n_instances_x, n_instances_y, m_trajectories_x, m_trajectories_y).
    """
    assert x.ndim == y.ndim == 3, "Inputs must have shape (n_patients, m_trajectories, dim_trajectory)."
    assert x.shape[2] == y.shape[2], "Trajectory dimension must match."

    # for every pair of instances: for every pair of trajectories: compute their difference in the last dimension
    # shape: (n_instances_x, n_instances_y, m_trajectories_x, m_trajectories_y, dim_trajectory)
    double_pairwise_differences = x[:, None, :, None] - y[None, :, None, :]

    # for every pair of instances: for every pair of trajectories: compute their squared euclidean distance
    # shape: (n_instances_x, n_instances_y, m_trajectories_x, m_trajectories_y)
    return (double_pairwise_differences ** 2).sum(-1).reshape(x.shape[0], y.shape[0], x.shape[1], y.shape[1])


Kernel = Callable[[np.ndarray, np.ndarray], np.ndarray]


def compute_double_pairwise_kernel_matrix(
        x: np.ndarray,
        y: np.ndarray,
        kernel_aggregator: Kernel,
        batch_size: int
) -> np.ndarray:
    """ Computes a kernel matrix in a batched fashion.

    The computation of a kernel matrix may require much more memory in intermediate steps than the final kernel matrix
    itself. For this reason, some implementations may benefit from computing the kernel matrix in batches by sliding
    over the kernel matrix in strides and computing only that subsection of the kernel matrix at once.

    This function provides a generic interface that enables exactly that: We slide over the kernel matrix in strides and
    compute the kernel values for that slide. In the end, the full kernel matrix is returned.

    Args:
        x: Feature matrix of shape (n_instances_x, ...).
        y: Feature matrix of shape (n_instances_y, ...).
        kernel_aggregator: A function that takes batches of x and y (of shape (batch_x, ...), (batch_y, ...)) and return
                           the kernel values for the batch as an array of shape (batch_x, batch_y). Note that batch_x
                           and batch_y may not always be the batch size as the batch size may not perfectly divide
                           n_instances_x and n_instances_y.
        batch_size: The batch size used to slide over the kernel matrix in (quadratic) strides.

    Returns:
            The kernel matrix.
            The shape is (n_instances_x, n_instances_y).
    """

    n_instances_x, n_instances_y = x.shape[0], y.shape[0]

    # start from empty kernel matrix and fill in actual kernel values later
    kernel_matrix = np.empty((n_instances_x, n_instances_y), dtype=float)

    # slide over x instances in strides
    for i in range(0, n_instances_x, batch_size):
        x_batch = x[i:i + batch_size]

        # slide over y instances in strides
        for j in range(0, n_instances_y, batch_size):
            y_batch = y[j:j + batch_size]

            # compute the kernel matrix for the selected instance batch and store them in the full kernel matrix
            kernel_batch = kernel_aggregator(x_batch, y_batch)
            assert kernel_batch.shape == (x_batch.shape[0], y_batch.shape[0])
            kernel_matrix[i:i + batch_size, j:j + batch_size] = kernel_batch

    return kernel_matrix


class BaseKernel(ABC):
    @abstractmethod
    def double_pairwise(self, x: np.ndarray | list[np.ndarray], y: np.ndarray | list[np.ndarray]) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def double_pairwise_kme(
            self,
            x: np.ndarray | list[np.ndarray],
            y: np.ndarray | list[np.ndarray],
            batch_size: int = DEFAULT_BATCH_SIZE
    ) -> np.ndarray:
        raise NotImplementedError

    def double_pairwise_mean(
            self,
            x: np.ndarray | list[np.ndarray],
            y: np.ndarray | list[np.ndarray],
            batch_size: int = DEFAULT_BATCH_SIZE
    ):
        raise NotImplementedError

    @abstractmethod
    def transform(self, distances: np.ndarray | list[np.ndarray]) -> np.ndarray:
        # TODO: specify shape
        raise NotImplementedError

    # TODO: check type hint, from python 3.10 onwards possible to delete quotation marks
    #  check for any derived class as well
    @abstractmethod
    def __mul__(self, other) -> 'ProductKernel':
        raise NotImplementedError


class SimpleKernel(BaseKernel):
    def __mul__(self, other) -> ProductKernel:
        if isinstance(other, ProductKernel):
            return ProductKernel([self] + other.kernels)
        elif isinstance(other, SimpleKernel):
            return ProductKernel([self] + [other])
        else:
            raise ValueError(f"Multiplication of {self.__class__.__name__} with {type(other)} is not defined.")

    def double_pairwise(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Computes the kernel matrix for every pair of instances and trajectories.

        For every pair of instances we compute, for every pair of their trajectories, the squared euclidean distance
        between the trajectories. The kernel transformation is then applied to that distance.

        Args:
            x: Feature matrix of shape (n_instances_x, m_trajectories_x, dim_trajectory).
            y: Feature matrix of shape (n_instances_y, m_trajectories_y, dim_trajectory).

        Returns:
            The kernel matrix for every pair of instances and every pair of trajectories.
            The shape is (n_instances_x, n_instances_y, m_trajectories_x, m_trajectories_y).
        """

        # for every pair of instances: for every pair of trajectories: squared euclidean distance between trajectories
        # shape: (n_instances_x, n_instances_y, m_trajectories_x, m_trajectories_y)
        pairwise_distances = double_pairwise_squared_euclidean_distances(x, y)

        return self.transform(pairwise_distances)

    def double_pairwise_kme(self, x: np.ndarray, y: np.ndarray, batch_size: int = DEFAULT_BATCH_SIZE) -> np.ndarray:
        """ Computes the kernel matrix for every pair of instances as the KME of the trajectories.

        We compute the kernel values for every pair of trajectories of every pair of instances. The kernel value for two
        instances is then computed as the mean of the kernel values for all pairs of their trajectories.

        The kernel matrix is computed in a batched fashion to limit the amount of memory used in intermediate steps.

        Args:
            x: Feature matrix of shape (n_instances_x, m_trajectories_x, dim_trajectory).
            y: Feature matrix of shape (n_instances_y, m_trajectories_y, dim_trajectory).
            batch_size: Matrix batch stride to use for memory-efficient computation of the kernel matrix.

        Returns:
            The kernel matrix for every pair of instances.
            The shape is (n_instances_x, n_instances_y).
        """

        def kernel_aggregator(x_batch: np.ndarray, y_batch: np.ndarray) -> np.ndarray:
            kernel_batch = self.double_pairwise(x_batch, y_batch)
            return kernel_batch.mean(axis=(-1, -2)).reshape((x_batch.shape[0], y_batch.shape[0]))

        return compute_double_pairwise_kernel_matrix(x, y, kernel_aggregator, batch_size)

    def double_pairwise_mean(self, x: np.ndarray, y: np.ndarray, batch_size: int = DEFAULT_BATCH_SIZE):
        """ Computes the kernel matrix for every pair of instances over the average trajectory distances.

        For every pair of instances we compute, for every pair of their trajectories, the squared euclidean distance
        between the trajectories. We then average the distances per instance pair and apply the kernel
        transformation.

        The kernel matrix is computed in a batched fashion to limit the amount of memory used in intermediate steps.

        Args:
            x: Feature matrix of shape (n_instances_x, m_trajectories_x, dim_trajectory).
            y: Feature matrix of shape (n_instances_y, m_trajectories_y, dim_trajectory).
            batch_size: Matrix batch stride to use for memory-efficient computation of the kernel matrix.

        Returns:
            The kernel matrix for every pair of instances.
            The shape is (n_instances_x, n_instances_y).
        """

        def kernel_aggregator(x_batch: np.ndarray, y_batch: np.ndarray) -> np.ndarray:
            # for every instance pair: for every trajectory pair: squared euclidean distance between trajectories
            # shape: (n_instances_x_batch, n_instances_y_batch, m_trajectories_x_batch, m_trajectories_y_batch)
            pairwise_distances = double_pairwise_squared_euclidean_distances(x_batch, y_batch)

            # for every instance pair: mean squared euclidean distance between trajectories over their trajectory pairs
            # shape: (n_instances_x_batch, n_instances_y_batch)
            pairwise_distances_mean = pairwise_distances.mean(axis=(-1, -2))
            pairwise_distances_mean = pairwise_distances_mean.reshape((x_batch.shape[0], y_batch.shape[0]))

            return self.transform(pairwise_distances_mean)

        return compute_double_pairwise_kernel_matrix(x, y, kernel_aggregator, batch_size)


class RBFKernel(SimpleKernel):
    bandwidth: float

    def __init__(self, bandwidth: float):
        self.bandwidth = bandwidth

    def transform(self, distances: np.ndarray) -> np.ndarray:
        """ Implements a Gaussian radial basis function.

        Args:
            distances: The distances between points (can be any shape).

        Returns:
            The transformed distances as a measure of similarity.
        """
        return np.exp(-1 / 2 / (self.bandwidth ** 2) * distances)


class ProductKernel(BaseKernel):
    def __init__(self, kernels: list[BaseKernel]):
        assert len(kernels) >= 2, f"Length of kernels expected to be at least 2. However, it is {len(kernels)}"
        self.kernels = []
        for kernel in kernels:
            if isinstance(kernel, BaseKernel):
                self.kernels.append(kernel)
            elif isinstance(kernel, ProductKernel):
                self.kernels.extend(kernel.kernels)
            else:
                raise ValueError(f"Found Kernel in kernels that is of unexpected type {type(kernel)}")

    def __mul__(self, other) -> 'ProductKernel':
        if isinstance(other, ProductKernel):
            return ProductKernel(self.kernels + other.kernels)
        elif isinstance(other, BaseKernel):
            return ProductKernel(self.kernels + [other])
        else:
            raise ValueError(f"Multiplication of {self.__class__.__name__} with {type(other)} is not defined.")

    def double_pairwise(self, x: list[np.ndarray], y: list[np.ndarray]) -> np.ndarray:
        """
        Computes the kernel matrix for every pair of instances and trajectories.

        For every pair of instances we compute, for every pair of their trajectories, the squared euclidean distance
        between the trajectories. The kernel transformation is then applied to that distance per feature.
        Finally, we return the element-wise product of each simple kernel matrix.

        Args:
            x: List of features matrices [x_1, ..., x_n] where each x_i is a np.ndarray of shape (n_instances_x, m_x_trajectories, dim_trajcetory_i)
            y: List of features matrices [y_1, ..., y_n] where each y_i is a np.ndarray of shape (n_instances_y, m_y_trajectories, dim_trajcetory_i)

        Returns:
            The product kernel matrix for every pair of instances and every pair of trajectories.
            The shape is (n_instances_x, n_instances_y, m_trajectories_x, m_trajectories_y).
        """
        self._check_validity(x)
        self._check_validity(y)
        self._assert_equal_n_trajectories(x)
        self._assert_equal_n_trajectories(y)
        res = np.ones((len(x[0]), len(y[0]), x[0].shape[1], y[0].shape[1]))
        for i, kernel in enumerate(self.kernels):
            res *= kernel.double_pairwise(x[i], y[i])
        return res

    def double_pairwise_kme(
            self,
            x: list[np.ndarray],
            y: list[np.ndarray],
            batch_size: int = DEFAULT_BATCH_SIZE
    ) -> np.ndarray:
        """
        Computes the kernel matrix as the pairwise product of the kernel matrices of each simple kernel.
        Each single kernel matrix is the kernel matrix for every pair of instances as the KME of the trajectories.
        Each single kernel matrix is computed in a batched fashion to limit the amount of memory used in intermediate steps.

        Args:
            x: List of features matrices [x_1, ..., x_n] where each x_i is a np.ndarray of shape (n_instances_x, m_x_i_trajectories, dim_trajcetory_i)
            y: List of features matrices [y_1, ..., y_n] where each y_i is a np.ndarray of shape (n_instances_y, m_y_i_trajectories, dim_trajcetory_i)
            batch_size: Matrix batch stride to use for memory-efficient computation of the kernel matrix.

        Returns:
            The kernel matrix for every pair of instances.
            The shape is (n_instances_x, n_instances_y).
        """
        self._check_validity(x)
        self._check_validity(y)
        res = np.ones((len(x[0]), len(y[0])))
        for i, kernel in enumerate(self.kernels):
            res *= kernel.double_pairwise_kme(x[i], y[i], batch_size)
        return res

    def double_pairwise_mean(
            self,
            x: list[np.ndarray],
            y: list[np.ndarray],
            batch_size: int = DEFAULT_BATCH_SIZE
    ) -> np.ndarray:
        """
        Computes the kernel matrix as the pairwise product of the kernel matrices of each simple kernel.
        Each single kernel matrix is the kernel matrix for every pair of instances over the average trajectory distances.
        Each single kernel matrix is computed in a batched fashion to limit the amount of memory used in intermediate steps.

        Args:
            x: List of features matrices [x_1, ..., x_n] where each x_i is a np.ndarray of shape (n_instances_x, m_x_i_trajectories, dim_trajcetory_i)
            y: List of features matrices [y_1, ..., y_n] where each y_i is a np.ndarray of shape (n_instances_y, m_y_i_trajectories, dim_trajcetory_i)
            batch_size: Matrix batch stride to use for memory-efficient computation of the kernel matrix.

        Returns:
            The kernel matrix for every pair of instances.
            The shape is (n_instances_x, n_instances_y).
        """
        self._check_validity(x)
        self._check_validity(y)
        res = np.ones(len(x[0]), len(y[0]))
        for i, kernel in enumerate(self.kernels):
            res *= kernel.double_pairwise_mean(x[i], y[i], batch_size)
        return res

    def _check_validity(self, features: list[np.ndarray]):
        assert isinstance(features, list), "Parameter must be of type list"
        assert len(features) == len(self.kernels), (
            f"Number of number of kernels does not match number of features: {len(features)} != {len(self.kernels)}"
        )
        assert all((e.shape[0] == features[0].shape[0] for e in features)), (
            "Number of instances must be the same for all features."
        )

    @staticmethod
    def _assert_equal_n_trajectories(x):
        assert all((feature.shape[1] == x[0].shape[1] for feature in x)), (
            "Number of trajectories must be the same for all features."
        )

    def transform(self, distances: list[np.ndarray]) -> np.ndarray:
        # TODO: specify shape
        res = self.kernels[0].transform(distances[0])
        for i, kernel in enumerate(self.kernels[1:]):
            res *= kernel.transform(distances[i])
        return res
