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


class AbstractKernel(ABC):
    def single_pairwise(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """ Computes the kernel matrix for every pair of trajectories of two single instances.

        We add an instance dimension to both x and y and then apply the more general kernel matrix function for every
        pair of instances and every pair of their trajectories.

        Args:
            x: Feature matrix of shape (m_trajectories_x, dim_trajectory).
            y: Feature matrix of shape (m_trajectories_y, dim_trajectory).

        Returns:
            The kernel matrix for every pair of trajectories.
            The shape is (m_trajectories_x, m_trajectories_y).
        """
        assert x.ndim == y.ndim == 2, "Inputs must have shape (m_trajectories, dim_trajectory)."
        assert x.shape[1] == y.shape[1], "Trajectory dimension must match."

        # add instance dimensions
        x = x[None, :, :]  # shape: (1, m_trajectories_x, dim_trajectory)
        y = y[None, :, :]  # shape: (1, m_trajectories_y, dim_trajectory)

        # shape: (1, 1, m_trajectories_x, m_trajectories_y, dim_trajectory)
        pairwise_instance_kernel = self.double_pairwise(x, y)

        # shape: (m_trajectories_x, m_trajectories_y)
        return pairwise_instance_kernel[0, 0].reshape(x.shape[0], y.shape[0])

    def double_pairwise(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """ Computes the kernel matrix for every pair of instances and trajectories.

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

    @abstractmethod
    def transform(self, distances: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class RBFKernel(AbstractKernel):
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
