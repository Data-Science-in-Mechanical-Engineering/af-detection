from abc import ABC, abstractmethod

import numpy as np


class AbstractKernel(ABC):
    def single_pairwise(self, x: np.ndarray, y: np.ndarray = None) -> np.ndarray:
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

    def double_pairwise(self, x: np.ndarray, y: np.ndarray = None) -> np.ndarray:
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
        assert x.ndim == y.ndim == 3, "Inputs must have shape (n_patients, m_trajectories, dim_trajectory)."
        assert x.shape[2] == y.shape[2], "Trajectory dimension must match."

        # for every pair of instances: for every pair of trajectories: compute their difference in the last dimension
        # shape: (n_instances_x, n_instances_y, m_trajectories_x, m_trajectories_y, dim_trajectory)
        double_pairwise_differences = x[:, None, :, None] - y[None, :, None, :]

        # for every pair of instances: for every pair of trajectories: compute their squared euclidean distance
        # shape: (n_instances_x, n_instances_y, m_trajectories_x, m_trajectories_y)
        double_pairwise_squared_euclidean_distances = (double_pairwise_differences ** 2).sum(-1).reshape(
            x.shape[0], y.shape[0], x.shape[1], y.shape[1]
        )

        return self.transform(double_pairwise_squared_euclidean_distances)

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
