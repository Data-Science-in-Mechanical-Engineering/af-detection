import numpy as np


def extract_rri(qrs_complexes: list[np.ndarray]) -> np.ndarray:
    assert all(rr_indices.ndim == 1 for rr_indices in qrs_complexes)

    complex_lengths = map(len, qrs_complexes)
    min_length = min(complex_lengths)

    equi_length_qrs_complexes = [rr_indices[-min_length:] for rr_indices in qrs_complexes]
    equi_length_qrs_complexes = np.array(equi_length_qrs_complexes)

    return np.diff(equi_length_qrs_complexes, axis=1)


def extract_normalized_rri(qrs_complexes: list[np.ndarray]) -> np.ndarray:
    rris = extract_rri(qrs_complexes)
    means = rris.mean(axis=1, keepdims=True)
    return rris / means
