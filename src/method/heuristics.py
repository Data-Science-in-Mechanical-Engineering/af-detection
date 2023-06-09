# TODO: currently we have no heuristics that are actually used -> remove this module or come up with a good heuristic

import numpy as np
from sklearn.metrics import pairwise_distances


def kernel_bandwidth_heuristic(x: np.ndarray, quantile: float) -> float:
    squared_distances = pairwise_distances(x)
    quantile_distance = np.quantile(squared_distances[squared_distances > 0], q=quantile)
    bandwidth = np.sqrt(quantile_distance / 2)
    return float(bandwidth)
