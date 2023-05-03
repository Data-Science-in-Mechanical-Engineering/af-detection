import numpy as np

from src.data.dataset import ECGDataset


def extract_sub_trajectories(
        mixing_speed: int,
        offset: int,
        length_trajectory: int,
        data_set: ECGDataset,
):
    """
    Get the sub-trajectories such that each sub-trajectory ends at an
    annotated qrs-complex (R). One sub-trajectory is formed by the
    length_trajectory samples in before the annotated qrs-complex.
    Some qrs-complexes are skipped such that the difference between two
    sub-trajectories is at least mixing_speed. Additionally,
    only qrs-complexes are considered after offset samples.
    For each subject, there might be a different number of resulting
    sub-trajectories, e.g. due to different heart beats. For all subjects
    the number of returned sub-trajectories is reduced to the smallest
    number of sub-trajectories per patient.
    :param mixing_speed: assumed mixing speed of the data, minimum
    difference between end of one sub-trajectory and start the next one.
    :type mixing_speed: int
    :param offset: sub trajectories are only considered after offset samples
    :type offset: int
    :param length_trajectory: length of the sub-trajectories (samples
     considered before annotated qrs-complex)
    :type length_trajectory: int
    :param data_set:
    :type data_set: HospitalDataset
    :return: indices, sub trajectories per patient
    :rtype: np.ndarray of shape(n_subjects, n_sub_trajectories, length_trajectory),
            np.ndarray of shape (n_subjects, n_sub-trajectories, length_trajectory, 1)
    """
    indices = []

    def sub_trajectory_indices(qrs_complex_end: int) -> range:
        return range(qrs_complex_end - length_trajectory, qrs_complex_end)

    for qrs_complexes in data_set.qrs_complexes[0]:
        # we start at the second complex to be able to calculate the difference
        # to the previous one
        i = 0
        while qrs_complexes[i] < length_trajectory + offset:
            i += 1

        instance_indices = [sub_trajectory_indices(qrs_complexes[i])]

        j = i + 1  # index of the next complex
        while j < len(qrs_complexes):
            while (
                    j < len(qrs_complexes)
                    and qrs_complexes[j] - qrs_complexes[i]
                    < length_trajectory + mixing_speed
            ):
                j += 1

            if j < len(qrs_complexes) - 1:
                instance_indices.append(
                    sub_trajectory_indices(qrs_complexes[j])
                )

            i = j

        indices.append(instance_indices)

    n_sub_trajectories = min(
        map(len, indices)
    )  # smallest number of sub trajectories among all instances

    # set n_sub_trajectories hard to 6 for SPHDataset
    # usually, n_sub_trajectories is 1 but here we want more data.
    THRESHOLD = 7  # TODO: Flagged for refactoring.
    n_sub_trajectories = max(THRESHOLD, n_sub_trajectories)
    # cut off "overflowing" sub trajectories
    # we do this to make sure all instances have the same number of sub trajectories
    indices = [
        instance_indices[-n_sub_trajectories:]
        if len(instance_indices) >= THRESHOLD
        else np.zeros((n_sub_trajectories, length_trajectory), dtype=np.int64)
        for instance_indices in indices
    ]

    # extract sub trajectories for every instance from the data set
    # shape: (n_instances, n_sub_trajectories, length_trajectory, 1)
    sub_trajectory_signal = np.array([
        data_set[i, instance_range]
        for i, instance_range in enumerate(indices)
    ])

    indices = np.array(
        indices
    )  # (n_instances, n_sub_trajectories, length_trajectory)

    return indices, sub_trajectory_signal


def extract_rri_data(
    sub_trajectory_indices: np.ndarray, qrs_complexes: list[np.ndarray]
) -> np.ndarray:
    rr_differences = extract_rri(
        sub_trajectory_indices, qrs_complexes
    )
    normalized_spike_time_differences = rr_differences / rr_differences.mean(
        axis=-1, keepdims=True
    )
    return normalized_spike_time_differences


def extract_rri(
    sub_trajectory_indices: np.ndarray, spike_indices: list[np.ndarray]
) -> np.ndarray:
    spike_time_differences = []
    sub_trajectory_starts = sub_trajectory_indices[:, :, 0]

    for instance_sub_trajectory_starts, instance_qrs_complexes in zip(
        sub_trajectory_starts, spike_indices
    ):
        spike_index = 0
        instance_spike_time_differences = []

        for trajectory_start in instance_sub_trajectory_starts:
            while instance_qrs_complexes[spike_index + 1] < trajectory_start:
                spike_index += 1

            current_spike_index = instance_qrs_complexes[
                spike_index
            ]  # spike in the current complex
            next_spike_index = instance_qrs_complexes[
                spike_index + 1
            ]  # first spike after current complex
            spike_time_difference = (
                current_spike_index - next_spike_index
            )  # difference in time steps between spikes

            instance_spike_time_differences.append(spike_time_difference)

        spike_time_differences.append(instance_spike_time_differences)

    return np.array(spike_time_differences)
