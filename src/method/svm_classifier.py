# TODO: Consider removing the multithreading as it shouldn't allow for performance gains.

import numpy as np
from joblib import Parallel, delayed
from sklearn.svm import SVC

from src.method.kernels import AbstractKernel


class SVCClassifier:
    def __init__(
        self, kernel: AbstractKernel, C: float = 1.0, n_jobs: int = 1
    ):
        """
        :param kernel: kernel method that is used to compute the Maximum
        Mean Discrepancy (MMD). Must already be completely initialised.
        :type kernel: AbstractKernel
        :param C: regularisation parameter
        :type C: float
        :param n_jobs: number of workers (threads) that are used to compute
        the MMD in a pairwise manner
        :type n_jobs: int
        """
        self.kernel = kernel
        self.n_jobs = n_jobs
        self._clf = SVC(C=C, kernel="precomputed")

    def fit(self, x: np.ndarray, y: np.ndarray):
        """
        Fit the classifier
        :param x: training data
        :type x: np.ndarray of shape (n_instances, n_trajectories,
        lenght_trajectory, dimension)
        :param y: labels
        :type y: np.ndarray of shape (n_instances,)
        :return: self
        :rtype: SVCClassifier
        """
        super()._validate_and_store_data(x, y)
        kernel_matrix = np.array(
            Parallel(n_jobs=self._n_jobs, backend="threading")(
                delayed(self._reduce_kernel)(x[i], x[j])
                for i in range(x.shape[0])
                for j in range(x.shape[0])
            )
        ).reshape(x.shape[0], x.shape[0])

        self._clf.fit(kernel_matrix, y)
        return self

    def _reduce_kernel(self, x, y):
        kernel_matrix = self.kernel(x, y)
        reduced_kernel_value = np.mean(kernel_matrix)
        return reduced_kernel_value

    def predict(self, x: np.ndarray):
        """
        Predict the class labels for the given instances.
        :param x: Test samples
        :type x: np.ndarray of shape (n_queries, n_trajectories,
        lenght_trajectory, dimension)
        :return: Label for each sample
        :rtype: np.ndarray of shape (n_queries,)
        """
        kernel_matrix = np.array(
            Parallel(n_jobs=self._n_jobs, backend="threading")(
                delayed(self._reduce_kernel)(x[i], self._x[j])
                for i in range(x.shape[0])
                for j in range(self._x.shape[0])
            )
        ).reshape(x.shape[0], self._x.shape[0])
        return self._clf.predict(kernel_matrix)
