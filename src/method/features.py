import numpy as np


def get_equi_length_qrs_complexes(qrs_complexes: list[np.ndarray]) -> np.ndarray:
    """ Parses a list of QRS complexes into a numpy array.

    In case there are QRS complexes with varying lengths (varying number of entries per instance), the number of entries
    per instance are truncated. We truncate from the left because early QRS measurements tend to be noisier than later
    ones.

    Args:
        qrs_complexes: A list of QRS complexes as numpy arrays of shape (m_peaks,).

    Returns:
        The (possibly truncated) QRS complexes as a numpy array of shape (n_instances, m_peaks).
    """
    assert all(rr_indices.ndim == 1 for rr_indices in qrs_complexes), "QRS complexes must be 1D per instance."

    min_length = min(map(len, qrs_complexes))  # minimal number of entries in QRS complexes

    # take fixed, minimal number of entries from the end of the lists
    equi_length_qrs_complexes = [rr_indices[-min_length:] for rr_indices in qrs_complexes]

    # return as numpy array of shape (n_instances, m_rris)
    return np.array(equi_length_qrs_complexes)


def extract_rri(qrs_complexes: list[np.ndarray]) -> np.ndarray:
    """ Extracts the interval length between R peaks from QRS complexes.

    The RRI is the difference between subsequent R peaks.
    Note that the scale of the outcome depends on the frequency of the ECG measurement.

    Args:
        qrs_complexes: A list of QRS complexes (R peak indices) as numpy arrays of shape (m_rris,).

    Returns:
        The RRIs as a numpy array of shape (n_instances, m_rris).
    """
    # get R peaks as numpy array - shape: (n_instances, m_peaks)
    equi_length_r_peaks = get_equi_length_qrs_complexes(qrs_complexes)

    # intervals between R peaks - shape: (n_instances, m_peaks - 1)
    return np.diff(equi_length_r_peaks, axis=1)


def extract_normalized_rri(qrs_complexes: list[np.ndarray]) -> np.ndarray:
    rris = extract_rri(qrs_complexes)
    means = rris.mean(axis=1, keepdims=True)
    return rris / means


def extract_pre_peak_trajectories(
        ecg_signals: list[np.ndarray],
        qrs_complexes: list[np.ndarray],
        length: int
) -> np.ndarray:
    assert len(ecg_signals) == len(qrs_complexes)
    n = len(ecg_signals)

    equi_length_qrs_complexes = get_equi_length_qrs_complexes(qrs_complexes)
    pre_peak_lengths = np.diff(equi_length_qrs_complexes, axis=1, prepend=0)
    sufficient_pre_peak_lengths = pre_peak_lengths > length

    assert len(qrs_complexes) == sufficient_pre_peak_lengths.shape[0]

    qrs_complexes = [
        r_peaks[sufficient_pre_peak_length]
        for r_peaks, sufficient_pre_peak_length in zip(equi_length_qrs_complexes, sufficient_pre_peak_lengths)
    ]

    equi_length_qrs_complexes = get_equi_length_qrs_complexes(qrs_complexes)
    assert equi_length_qrs_complexes.shape[0] == n
    m_peaks = equi_length_qrs_complexes.shape[1]

    trajectory_indices = np.tile(equi_length_qrs_complexes, length).reshape((n, m_peaks, length)) - np.arange(0, length)
    assert trajectory_indices.shape[0] == n

    trajectories = np.stack([
        ecg_signal[indices]
        for ecg_signal, indices in zip(ecg_signals, trajectory_indices)
    ])

    return trajectories


def extract_smooth_pre_peak_trajectories(
        ecg_signals: list[np.ndarray],
        qrs_complexes: list[np.ndarray],
        pre_peak_time: float,
        measurement_frequency: float,
        dim_encoding: int
) -> np.ndarray:
    trajectory_length = round(measurement_frequency * pre_peak_time)
    assert trajectory_length >= dim_encoding, f"Encoding dimensionality is too large for F={measurement_frequency}" \
                                              f"and T={pre_peak_time}"
    trajectory_length += dim_encoding - (trajectory_length % dim_encoding)

    assert trajectory_length % dim_encoding == 0
    interval_length = trajectory_length // dim_encoding

    trajectories = extract_pre_peak_trajectories(ecg_signals, qrs_complexes, trajectory_length)
    partitioned_trajectories = trajectories.reshape((*trajectories.shape[:-1], -1, interval_length))
    smooth_trajectories = partitioned_trajectories.mean(axis=-1)

    max_peak = np.abs(smooth_trajectories).max(axis=-1, keepdims=True)
    return smooth_trajectories / max_peak
