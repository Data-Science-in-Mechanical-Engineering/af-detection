from abc import ABC, abstractmethod

import numpy as np


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
            The shape is (n_instances_x, n_instances_y, m_trajectories_x, m_trajectories_y).
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

    def double_pairwise_mean(self, x: np.ndarray, y: np.ndarray):
        """ Computes the kernel matrix for every pair of instances over the average trajectory distances.

        For every pair of instances we compute, for every pair of their trajectories, the squared euclidean distance
        between the trajectories. We then average the distances per instance pair and apply the kernel
        transformation.

        Args:
            x: Feature matrix of shape (n_instances_x, m_trajectories_x, dim_trajectory).
            y: Feature matrix of shape (n_instances_y, m_trajectories_y, dim_trajectory).

        Returns:
            The kernel matrix for every pair of instances.
            The shape is (n_instances_x, n_instances_y).
        """
        # for every pair of instances: for every pair of trajectories: squared euclidean distance between trajectories
        # shape: (n_instances_x, n_instances_y, m_trajectories_x, m_trajectories_y)
        pairwise_distances = double_pairwise_squared_euclidean_distances(x, y)

        # for every pair of instances: mean squared euclidean distance between trajectories over their trajectory pairs
        # shape: (n_instances_x, n_instances_y)
        pairwise_distances_mean = pairwise_distances.mean(axis=(-1, -2))

        return self.transform(pairwise_distances_mean)

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
