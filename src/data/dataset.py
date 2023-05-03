import csv
import os
import pickle
import zipfile
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from urllib import request

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from wfdb.processing import XQRS


@dataclass(frozen=True)
class Identifier:
    pass


@dataclass(frozen=True)
class HospitalDatasetIdentifier(Identifier):
    device_id: int
    patient_id: int


@dataclass(frozen=True)
class SPHIdentifier(Identifier):
    file_name: str


class ECGDataset(ABC):
    FREQUENCY = None
    DURATION_IN_SEC = None
    RHYTHMS = None

    # TODO: Consider removing this constructor as it seems to do nothing
    def __init__(self):
        self._qrs_complexes = None
        self._labels = None
        self._ecgs = None
        self._identifiers = None
        self._leads = [0]

    @property
    def qrs_complexes(self) -> list[np.ndarray]:
        """
        :return: extracted indices of the qrs complex for each stored ecg
        :rtype: List[np.ndarray]
        """
        return self._qrs_complexes

    @property
    def labels(self) -> np.ndarray:
        """
        :return:  labels for each instance
        :rtype: np.ndarray of shape (n_instances,)
        """
        return self._labels

    @property
    def ecgs(self) -> np.ndarray:
        """
        :return: all saved ecg recordings
        :rtype: np.ndarray of shape (n_instances, FREQUENCY *
        DURATION_IN_SEC, n_leads)
        """
        return self._ecgs

    @property
    def identifiers(self) -> dict[int, Identifier]:
        """
        :return: mapping from dataset_id to unique (file) identifier for the
        dataset
        :rtype: Dict[int, Identifier]
        """
        return self._identifiers

    @property
    def n_ecgs(self) -> int:
        """
        :return: number of stored ecgs
        :rtype: int
        """
        return self._ecgs.shape[0]

    def _get_qrs_complexes(self, ecgs: np.ndarray, offset: int) -> list[int]:
        """
        TODO: Flagged for refactoring.

        Returns the qrs_complex indices for the single(!) lead ecgs.
        :param ecgs: ecgs that qrs_complexes should be extracted from. Must
        be a single lead.
        :type ecgs: np.ndarray of shape (n_instances, length_trajectory)
        :param offset: number of seconds that are dropped at the beginning of
        the recording
        :type offset: int
        :return: List[int]
        :rtype: list of indices
        """
        qrs_complex_per_instance = []
        for ecg in ecgs:
            xqrs = XQRS(
                sig=ecg[offset * self.FREQUENCY:],
                fs=self.FREQUENCY,
            )
            xqrs.detect(verbose=False)
            qrs_indices = xqrs.qrs_inds + offset * self.FREQUENCY
            qrs_complex_per_instance.append(qrs_indices)
        return qrs_complex_per_instance

    def get_identifier(self, dataset_id: int) -> Identifier:
        """
        :param dataset_id: internal id within dataset class
        :type dataset_id: int
        :return: file identifier for the data file storing the patients ecg.
        :rtype: Identifier
        """
        return self._identifiers[dataset_id]

    def get_ids_with_label(self, label: int | list[int]) -> np.ndarray:
        """
        :param label: label
        :type label: int or List[int]
        :return: dataset indices with a label in the given labels.
        :rtype: np.ndarray
        """
        if isinstance(label, int):
            label = [label]
        # labels are one-dimensional, so extract the first indices of
        # returned tuple
        indices = np.where(np.isin(self._labels, label))[0]
        return indices

    def plot_ecg_for(
            self,
            dataset_id: Iterable[int] | int,
            duration: int = 60,
            offset: int = 0,
            lead: Iterable[int] | int = 0,
    ):
        """
        TODO: Flagged for refactoring.

        Plot the single-lead ecg for each patient within dataset_id
        from the offset of the recording onwards for duration seconds.
        :param dataset_id: identifier of the patients
        :type dataset_id: Union[Iterable[int], int]
        :param duration: length of plot in seconds, default 60, must not
        exceed DURATION_IN_SEC - offset
        :type duration: int
        :param offset: number of seconds that are dropped at the beginning of
        the recording, default 0
        :type offset: int
        :param lead: number of the lead(s) to plot (0-indexed), default 0
        :type lead: Union[Iterable[int], int]
        """
        assert (
                duration + offset <= self.DURATION_IN_SEC
        ), f"Duration (+offset) must be smaller than {self.DURATION_IN_SEC}"

        max_sample_index = self.FREQUENCY * (duration + offset)
        min_sample_index = self.FREQUENCY * offset
        seconds = np.linspace(
            offset, duration + offset, max_sample_index - min_sample_index
        )

        if isinstance(dataset_id, int):
            dataset_id = [dataset_id]
        leads = lead
        if isinstance(leads, int):
            leads = [lead]

        fig, ax = plt.subplots(
            len(leads),
            len(dataset_id),
            figsize=(len(dataset_id) * 5, len(leads) * 3),
        )
        # note the changed checkup when a single float was passed as parameter
        if len(dataset_id) * len(leads) == 1:
            ax = np.array(ax)
        ax = ax.reshape(len(leads), len(dataset_id))
        for j in range(len(leads)):
            for i, id_ in enumerate(dataset_id):
                ax[j, i].plot(
                    seconds, self[id_, min_sample_index:max_sample_index, j]
                )
                ax[j, i].grid()
        self._label_axis_for_ecg_plot(ax, fig, dataset_id, duration, offset)
        fig.tight_layout()
        plt.show()
        return fig, ax

    @abstractmethod
    def _label_axis_for_ecg_plot(
            self, axes, fig, dataset_id, duration, offset
    ):
        raise NotImplementedError

    def __getitem__(self, item):
        return self._ecgs[item]

    def get_abbrev_for_label_number(self, label: int) -> str:
        """
        :param label:  label number
        :type label: int
        :return: abbreviation for the heart condition encoded with label
        :rtype: str
        """
        keys = [k for k, v in self.RHYTHMS.items() if v == label]
        if not keys:
            raise ValueError(
                f"There is no abbreviation for label {label} "
                f"stored in this class"
            )
        return keys[0]

    @abstractmethod
    def distance_between_qrs_complexes(
            self,
            mode: float | str = "mean",
            labels: int | list[int] | None = None,
    ) -> int:
        """
        TODO: Flagged for refactoring.

        Calculate the (mean, quantile) distance between rri intervals.
        :param mode: Mode to aggregate the distance. Can either be "mean" or
         some value q between 0 and 1 for the q-th quantile.
        :type mode: int or str
        :param labels: consider only ecgs with these labels, if None all
        instances
        :type labels: Union[Optional[int], List[int]]
        :return: rri-interval
        :rtype: int
        """
        raise NotImplementedError


class HospitalDataset(ECGDataset):
    """
    This class is used to wrap the data for the ecg measurements.
    Each trajectory is assumed to be created by a 1-min single-lead ecg.
    The frequency of the ecg is expected to be 200Hz, i.e. 12,000 samples
    per minute.
    """

    DURATION_IN_SEC = 60
    FREQUENCY = int(1 / (DURATION_IN_SEC / 12000))
    SAMPLES_PER_MINUTE = FREQUENCY * 60
    RHYTHMS = {"noAF": 0, "AF": 1, "UNKNOWN": 3}

    noAF = 0
    AF = 1
    UNKNOWN = 3

    @classmethod
    def from_csv_files(cls, path: str):
        """
        Creates a new dataset from files given at the specified path.
        The files should be named in the following structure:
        <device-number>-PAT-<patient-id>.csv
        where device-number and id can be padded with leading zeros.
        The csv file stores the measurements comma seperated in a single row.
        All trajectories are assumed to be of equal length,
        i.e. trajectories longer than the shortest one are shortened
        :param path: path where the data files are stored, can be relative or absolute
        :type path: str
        """
        filenames = sorted(
            [file for file in os.listdir(path) if file.endswith(".csv")]
        )
        data = []
        sensor_mapping = {}
        for i, file in enumerate(filenames):
            device_id, _, patient_id = file.split(".")[0].split("-")
            device_id, patient_id = int(device_id), int(patient_id)
            with open(os.path.join(path, file), "r", encoding="UTF-8") as f:
                trajec = np.genfromtxt(f, delimiter=",")[..., np.newaxis]
            if len(trajec) < cls.SAMPLES_PER_MINUTE:
                print(
                    f"Recording for file "
                    f"{device_id:0>3}-PAT-{patient_id:0>4}.csv does only "
                    f"have {len(trajec)} samples instead of "
                    f"{cls.SAMPLES_PER_MINUTE}. Will not be recorded in "
                    f"this dataset."
                )
                continue
            data.append(trajec[: cls.SAMPLES_PER_MINUTE])
            sensor_mapping[i] = HospitalDatasetIdentifier(
                device_id, patient_id
            )
        data = np.stack(data)
        return cls(data, sensor_mapping)

    @classmethod
    def from_npy_file(cls, path: str):
        """
        Method to reload data set from disk that has been saved via the
        class save method
        :param path: location where dataset should be loaded from, should
        contain extension .npy
        :type path: str
        """
        with open(path, "rb") as f:
            data = np.load(f)
            sensor_mapping = np.load(f)
        sensor_mapping = {
            id_: HospitalDatasetIdentifier(device_id, patient_id)
            for [id_, device_id, patient_id] in sensor_mapping
        }
        return cls(data, sensor_mapping)

    def __init__(
            self,
            data: np.ndarray,
            sensor_mapping: dict[int, HospitalDatasetIdentifier],
            labels: np.ndarray | None = None,
            qrs_complexes: list[np.ndarray] | None = None,
    ):
        """

        :param data: ecg data
        :type data: np.ndarray of shape (n_trajectories, length_trajectory, n_dim)
        :param sensor_mapping: dictionary mapping the new dataset id to its sensor and patient id
        :type sensor_mapping: Dict[int, HospitalDatasetIdentifier]
        :param labels: label for data
        :type labels: np.ndarray of shape (n_trajectories,)
        :param qrs_complexes: annotations of qrs_complexes for the data
        :type qrs_complexes: list of np.ndarray
        """
        super().__init__()
        self._ecgs = data
        self._identifiers = sensor_mapping
        self._labels = labels
        self._qrs_complexes = qrs_complexes

    def add_labels_from_csv_file(self, path: str):
        """
        Method to load labels from a CSV version of the
        Excel sheet containing labels
        :param path: location where the labels should be loaded from, should
        contain extension .csv
        """
        device_patient_label = {}
        labels = np.array([-1] * self._ecgs.shape[0], dtype=int)
        header = True
        with open(path, "r", encoding="UTF-8") as f:
            reader = csv.reader(f, delimiter=",")
            for row in reader:
                if header:
                    header = False
                    continue
                device_id, patient_id = map(int, row[0].split("-")[::2])
                label = int(row[1])
                device_patient_label[(device_id, patient_id)] = label

        for id_, device_patient_id in self._identifiers.items():
            device_id = device_patient_id.device_id
            patient_id = device_patient_id.patient_id
            labels[id_] = device_patient_label[(device_id, patient_id)]
        if (labels == -1).any():
            raise ValueError("Some labels are left unassigned")
        self._labels = labels

    def add_labels_from_npy(self, path: str):
        """
        Method to reload labels from disk that has been saved via the
        class save_labels method
        :param path: location where labels should be loaded from, should
        contain extension .npy
        :type path: str
        """
        with open(path, "rb") as f:
            self._labels = np.load(f)

    def save_labels(self, path: str):
        """
        Save labels as binary file to disk.
        :param path: location where dataset should be saved to. Path should
        contain already the extension .npy
        :type path: str
        """
        with open(path, "wb") as f:
            np.save(f, self._labels)

    def save_qrs_complexes(self, path: str):
        """
        Save labels as binary file to disk.
        :param path: location where dataset should be saved to. Path should
        contain already the extension .npy
        :type path: str
        """
        with open(path, "wb") as f:
            np.savez(f, *self._qrs_complexes)

    def add_qrs_complexes_from_npy(self, path: str):
        """
        Method to reload qrs_complexes from disk that has been saved via the
        class save_qrs_complexes method
        :param path: location where qrs_complexes should be loaded from, should
        contain extension .npz
        :type path: str
        """
        with open(path, "rb") as f:
            qrs_complexes = np.load(f)
            self._qrs_complexes = [
                qrs_complexes[k] for k in qrs_complexes.files
            ]
        # current fix necessary because of generalisation and that data might
        # be stored without additional lead-list structure
        if isinstance(self._qrs_complexes[0], np.ndarray):
            self._qrs_complexes = [self._qrs_complexes]

    def save(self, path: str):
        """
        Save data set as binary file to disk.
        :param path: location where dataset should be saved to. Path should
        contain already the extension .npy
        :type path: str
        """
        sensor_mapping_list = [
            [k, v.device_id, v.patient_id]
            for k, v in self._identifiers.items()
        ]

        sensor_mapping_array = np.array(sensor_mapping_list)
        with open(path, "wb") as f:
            np.save(f, self._ecgs)
            np.save(f, sensor_mapping_array)

    def get_device_and_patient_id(self, dataset_id: int) -> Identifier:
        """
        Return the device and patient id for a given dataset id.
        This means that dataset_id contains the data that has been stored in
        <device-number>-PAT-<patient-id>.csv
        :param dataset_id: id for which device and patient id are requested
        :type dataset_id: int
        :return: (device_id, patient_id)
        :rtype: Identifier
        """
        return self._identifiers[dataset_id]

    def _label_axis_for_ecg_plot(
            self, axes, fig, dataset_id, duration, offset
    ):
        fig.supxlabel("Seconds")
        for i, ax in enumerate(axes.T):
            ax[0].set_ylabel("mm")
            ax[0].set_title(f"One lead ECG for patient {dataset_id[i]}")
        for ax in axes.flatten():
            ax.set_xlim(left=offset, right=duration + offset)

    def distance_between_qrs_complexes(
            self, mode: float | str = "mean", labels: int | None = None
    ) -> int:
        """
        Calculate the (mean, quantile) distance between rri intervals.
        :param mode: Mode to aggregate the distance. Can either be "mean" or
         some value q between 0 and 1 for the q-th quantile.
        :type mode: int or str
        :param labels: consider only ecgs from this label, if None consider
        the data with and without AF, i.e. unknown labels are ignored.
        :type labels: Optional[int]
        :return: rri-interval
        :rtype: int
        """
        if self.qrs_complexes is None:
            raise RuntimeError("Call calculate_qrs_complexes beforehand")
        if labels is None:
            labels = [HospitalDataset.AF, HospitalDataset.noAF]
        ids = self.get_ids_with_label(labels)

        rri_intervals = []
        for qrs_complex in [self.qrs_complexes[0][id_] for id_ in ids]:
            rri_interval = np.diff(qrs_complex)
            rri_intervals.extend(rri_interval.tolist())
        if mode == "mean":
            distance = int(np.mean(rri_intervals))
        else:
            distance = int(np.quantile(rri_intervals, q=mode))
        return distance

    @property
    def ids(self):  # pylint: disable=missing-function-docstring
        return sorted(list(self._identifiers.keys()))

    def calculate_qrs_complexes(self, offset: int):
        """
        Calculate the qrs complexes for all the data points from offset
        seconds onwoards.
        Set the attribute self.qrs_complexes
        :param offset: offset in seconds
        :type offset: int
        """
        self._qrs_complexes = self._get_qrs_complexes(self._ecgs, offset)


class SPHDataset(ECGDataset):  # pylint: disable=too-many-instance-attributes
    """
    The dataset from `"A 12-lead electrocardiogram database for arrhythmia
    research covering more than 10,000 patient"` (
    https://www.nature.com/articles/s41597-020-0386-x).

    The dataset consist of a 10sec 12-lead ECG of 10,646 patients.
    The sampling frequency is 500Hz.

    The sample ECGDataDenoised/MUSE_20180113_124215_52000.csv is less
    than 10sec long and is therefore discarded.
    The samples ECGDataDenoised/MUSE_20181222_*.csv are constant 0 and are
    therefore discarded.

    Thus, there are 10,605 instances, in total.

    The dataset is downloaded from
    https://figshare.com/collections/ChapmanECG/4560497/2.
    """

    DURATION_IN_SEC = 10
    FREQUENCY = 500

    RHYTHMS = {
        "SB": 0,
        "SR": 1,
        "AFIB": 2,
        "ST": 3,
        "AF": 4,
        "SA": 5,
        "SVT": 6,
        "AT": 7,
        "AVNRT": 8,
        "AVRT": 9,
        "SAAWR": 10,
    }

    # number of total patients/ecg files including those files that are
    # excluded
    _N_PATIENTS = 10646

    # list of files to be excluded from the downloaded data because there is
    # some error in the ecg recordings
    # TODO: update files for noisy data
    _EXCLUDE_FILES = [
        "ECGDataDenoised/MUSE_20180113_124215_52000.csv",
        "ECGDataDenoised/MUSE_20181222_204118_08000.csv",
        "ECGDataDenoised/MUSE_20181222_204121_42000.csv",
        "ECGDataDenoised/MUSE_20181222_204122_52000.csv",
        "ECGDataDenoised/MUSE_20181222_204123_64000.csv",
        "ECGDataDenoised/MUSE_20181222_204128_13000.csv",
        "ECGDataDenoised/MUSE_20181222_204131_50000.csv",
        "ECGDataDenoised/MUSE_20181222_204132_64000.csv",
        "ECGDataDenoised/MUSE_20181222_204140_77000.csv",
        "ECGDataDenoised/MUSE_20181222_204141_91000.csv",
        "ECGDataDenoised/MUSE_20181222_204143_03000.csv",
        "ECGDataDenoised/MUSE_20181222_204146_34000.csv",
        "ECGDataDenoised/MUSE_20181222_204154_20000.csv",
        "ECGDataDenoised/MUSE_20181222_204155_31000.csv",
        "ECGDataDenoised/MUSE_20181222_204156_45000.csv",
        "ECGDataDenoised/MUSE_20181222_204157_58000.csv",
        "ECGDataDenoised/MUSE_20181222_204158_72000.csv",
        "ECGDataDenoised/MUSE_20181222_204207_92000.csv",
        "ECGDataDenoised/MUSE_20181222_204212_44000.csv",
        "ECGDataDenoised/MUSE_20181222_204217_03000.csv",
        "ECGDataDenoised/MUSE_20181222_204218_14000.csv",
        "ECGDataDenoised/MUSE_20181222_204219_27000.csv",
        "ECGDataDenoised/MUSE_20181222_204222_63000.csv",
        "ECGDataDenoised/MUSE_20181222_204226_00000.csv",
        "ECGDataDenoised/MUSE_20181222_204227_13000.csv",
        "ECGDataDenoised/MUSE_20181222_204236_34000.csv",
        "ECGDataDenoised/MUSE_20181222_204237_47000.csv",
        "ECGDataDenoised/MUSE_20181222_204239_70000.csv",
        "ECGDataDenoised/MUSE_20181222_204240_84000.csv",
        "ECGDataDenoised/MUSE_20181222_204243_08000.csv",
        "ECGDataDenoised/MUSE_20181222_204245_36000.csv",
        "ECGDataDenoised/MUSE_20181222_204246_47000.csv",
        "ECGDataDenoised/MUSE_20181222_204248_77000.csv",
        "ECGDataDenoised/MUSE_20181222_204249_88000.csv",
        "ECGDataDenoised/MUSE_20181222_204302_49000.csv",
        "ECGDataDenoised/MUSE_20181222_204303_61000.csv",
        "ECGDataDenoised/MUSE_20181222_204306_99000.csv",
        "ECGDataDenoised/MUSE_20181222_204309_22000.csv",
        "ECGDataDenoised/MUSE_20181222_204310_31000.csv",
        "ECGDataDenoised/MUSE_20181222_204312_58000.csv",
        "ECGDataDenoised/MUSE_20181222_204314_78000.csv",
    ]

    def __init__(
            self,
            root: str,
            split: str,
            leads: list[int] | None = None,
            denoised_data: bool = True,
    ):
        """
        Download the data and save it to disk.
        Only the given leads are extracted and saved.
        :param root: root directory where the dataset should be saved.
        :type root: str
        :param split: either train, val or test
        :type split: str
        :param denoised_data: specify if the denoised data should
        be downloaded. If set to False, the raw noisy data is used.
        :type denoised_data: bool
        :param leads: leads to use, if None use all leads
        :type leads: Optional[List[int]]
        """
        super().__init__()
        assert split in ["train", "val", "test"], (
            f"split must be one of the following: train, val, test. However, "
            f"it is {split}"
        )

        self.root = Path(root)
        self.split = split
        self.denoised_data = denoised_data
        self._leads = sorted(leads)

        self._init_directory_structure()
        if not all(
                (self.numpy_dir / f"lead_{lead}.npy").exists()
                for lead in self._leads
        ):
            existing_leads = [
                lead_file.stem[-1]
                for lead_file in self.numpy_dir.iterdir()
                if lead_file.stem.startswith("lead")
            ]
            missing_leads = sorted(
                set(self._leads).difference(set(existing_leads))
            )
            if not self.raw_dir.exists():
                self.download()
            self._process(missing_leads)
        self.load()

    def _init_directory_structure(self):
        self.root.mkdir(parents=True, exist_ok=True)
        self.raw_dir.mkdir(exist_ok=True)
        self.numpy_dir.mkdir(exist_ok=True, parents=True)

    @property
    def raw_file_names(self):
        """
        File names of the dataset that must be downloaded.
        """
        raw_file_names = [
            "RhythmNames.xlsx",
            "ConditionNames.xlsx",
            "Diagnostics.xlsx",
            "AttributesDictionary.xlsx",
        ]
        if self.denoised_data:
            raw_file_names.append("ECGDataDenoised.zip")
        else:
            raw_file_names.append("ECGData.zip")
        return raw_file_names

    @property
    def raw_dir(self):
        """
        Path of directory where original files of dataset are stored
        """
        return self.root / "raw"

    @property
    def numpy_dir(self):
        """
        Path of directory where compressed numpy files are stored
        """
        sub_dir = "denoised" if self.denoised_data else "noisy"
        return self.root / sub_dir / self.split

    @property
    def file_name_to_download_url(self):
        """
        Mapping of file names to download to url where data can be downloaded.
        """
        return {
            "RhythmNames.xlsx": "https://figshare.com/ndownloader/files/15651296",
            "ConditionNames.xlsx": "https://figshare.com/ndownloader/files/15651293",
            "Diagnostics.xlsx": "https://figshare.com/ndownloader/files/15653771",
            "AttributesDictionary.xlsx": "https://figshare.com/ndownloader/files/15653762",
            "ECGDataDenoised.zip": "https://figshare.com/ndownloader/files/15652862",
            "ECGData.zip": "https://figshare.com/ndownloader/files/15651326",
        }

    def download(self):
        """
        Download the raw data files that are not yet downloaded
        """
        downloaded_files = [
            file.name for file in self.raw_dir.iterdir() if file.is_file()
        ]
        for file in self.raw_file_names:
            if file in downloaded_files:
                if not file.endswith(".zip"):
                    continue
                with zipfile.ZipFile(self.raw_dir / file) as zip_file:
                    # + 1 for information about the zip directory itself
                    if len(zip_file.infolist()) == self._N_PATIENTS + 1:
                        continue
            url = self.file_name_to_download_url[file]
            request.urlretrieve(url, self.raw_dir / file)

    def _process(self, leads):
        """
        Collect the data from all single files and calculate the qrs_complex.
        """
        meta_data = self._load_meta_data()
        file_names = self._get_file_names_from_meta_data(meta_data)
        labels = self._get_labels(meta_data)
        indices = self._get_stratified_indices(labels)
        self._labels = labels[indices]
        self._save_labels()
        self._identifiers = {
            id_: SPHIdentifier(file_name)
            for id_, file_name in enumerate(file_names[indices])
        }
        self._save_identifiers()

        self._extract_and_save_ecgs_and_qrs_complexes(
            file_names, indices, leads
        )

    def _extract_and_save_ecgs_and_qrs_complexes(
            self, file_names: np.ndarray, indices: np.ndarray, leads: list[int]
    ):
        zip_file = (
            "ECGDataDenoised.zip" if self.denoised_data else "ECGData.zip"
        )
        for lead in leads:
            ecg_recordings = []
            with zipfile.ZipFile(self.raw_dir / zip_file) as zf:
                # ignore first entry about zip directory itself
                self._check_equal_file_names(file_names, zf)
                for i in indices:
                    file = file_names[i]
                    with zf.open(file) as ecg_file:
                        ecg_record = pd.read_csv(
                            ecg_file, header=None
                        ).to_numpy()[:, lead]
                        ecg_recordings.append(ecg_record)
            ecg_recordings = np.stack(ecg_recordings)
            # set lead to zero as ecg_recordings only contain a single lead
            qrs_complexes = self._get_qrs_complexes(
                ecgs=ecg_recordings, offset=0
            )
            self._save_lead(lead, ecg_recordings, qrs_complexes)

    def _check_equal_file_names(self, file_names, zf):
        ecg_file_names = set(zf.namelist()[1:]).difference(
            set(self._EXCLUDE_FILES)
        )
        assert set(file_names) == set(
            ecg_file_names
        ), "Mismatch between label filenames and ecg filenames"

    def _save_identifiers(self):
        with open(self.numpy_dir / "identifiers.pickle", "wb") as f:
            pickle.dump(self._identifiers, f)

    def _save_lead(
            self, lead: int, data: np.ndarray, qrs_complexes: list[int]
    ):
        """
        Save the ecg recordings and the corresponding extracted
        qrs-complexes to disk.
        :param lead: number of the lead
        :type lead: int
        :param data: ecg recordings for the given lead
        :type data: np.ndarray of shape (n_instances, length_trajectory, 1)
        :param qrs_complexes: list of nd.arrays storing the indices
        :type qrs_complexes: List[np.ndarray]
        """
        with open(self.numpy_dir / f"lead_{lead}.npy", "wb") as f:
            np.save(f, data)
        with open(self.numpy_dir / f"qrs_complexes_{lead}.pickle", "wb") as f:
            pickle.dump(qrs_complexes, f)

    def _save_labels(self):
        """
        Save the ecgs, labels and the indices of the qrs_complexes to disk.
        """
        with open(self.numpy_dir / "labels.npy", "wb") as f:
            np.save(f, self._labels)

    # TODO: Flagged for refactoring.
    def _get_stratified_indices(self, labels):
        # creates overall an 60:20:20 split as 0.8*0.25=0.2 for
        # train/validation split
        """
        indices = np.arange(labels.shape[0])
        train_val_indices, test_indices = train_test_split(
            indices, random_state=42, stratify=labels, test_size=0.2
        )
        if self.split != "test":
            train_indices, val_indices = train_test_split(
                train_val_indices,
                random_state=10,
                stratify=labels[train_val_indices],
                test_size=0.25,
            )
        if self.split == "test":
            return test_indices
        if self.split == "val":
            return val_indices
        return train_indices
        """
        raise NotImplementedError

    def _get_labels(self, meta_data):
        exclude_file_names = list(
            map(lambda x: x.split("/")[1].split(".")[0], self._EXCLUDE_FILES)
        )
        labels_abbrev = meta_data["Rhythm"][
            ~meta_data["FileName"].isin(exclude_file_names)
        ]
        labels = np.array([self.RHYTHMS[label] for label in labels_abbrev])
        return labels

    def _get_file_names_from_meta_data(self, meta_data):
        # append name of zip directory and file extension to file name path
        zip_dir = "ECGDataDenoised/" if self.denoised_data else "ECGData/"
        file_names = (
            meta_data["FileName"]
            .apply(lambda x: zip_dir + x + ".csv")
            .tolist()
        )
        file_names = np.array(
            list(filter(lambda x: x not in self._EXCLUDE_FILES, file_names))
        )
        return file_names

    def _load_meta_data(self):
        meta_data_file_name = self.raw_dir / "Diagnostics.xlsx"
        with open(meta_data_file_name, "rb") as f:
            meta_data = pd.read_excel(f)
        return meta_data

    def load(self):
        """
        Load the compressed data.
        It is required that the data has once been saved to disk.
        """
        self._load_labels()
        self._load_identifiers()
        self._load_ecgs()
        self._load_qrs_complexes()

    def _load_ecgs(self):
        ecgs = []
        for lead in self._leads:
            with open(self.numpy_dir / f"lead_{lead}.npy", "rb") as f:
                ecg = np.load(f)
            ecgs.append(ecg)
        self._ecgs = np.stack(ecgs, axis=-1)

    def _load_qrs_complexes(self):
        qrs_complexes = []
        for lead in self._leads:
            with open(
                    self.numpy_dir / f"qrs_complexes_{lead}.pickle", "rb"
            ) as f:
                qrs_complex = pickle.load(f)
            qrs_complexes.append(qrs_complex)
        self._qrs_complexes = qrs_complexes

    def _load_labels(self):
        with open(self.numpy_dir / "labels.npy", "rb") as f:
            self._labels = np.load(f)

    def _load_identifiers(self):
        with open(self.numpy_dir / "identifiers.pickle", "rb") as f:
            self._identifiers = pickle.load(f)

    def _label_axis_for_ecg_plot(
            self, axes, fig, dataset_id, duration, offset
    ):
        fig.supxlabel("Seconds")
        fig.supylabel("mV")

        for i, ax in enumerate(axes.T):
            heart_condition = self.get_abbrev_for_label_number(
                self._labels[dataset_id[i]]
            )
            ax[0].set_title(f"Patient {dataset_id[i]} ({heart_condition})")
        for i, ax in enumerate(axes):
            ax[0].set_ylabel(f"Lead {self._leads[i]}")

        for ax in axes.flatten():
            ax.set_xlim(left=offset, right=duration + offset)

    def distance_between_qrs_complexes(
            self,
            mode: float | str = "mean",
            labels: int | list[int] | None = None,
    ) -> int:
        """
        Calculate the (mean, quantile) distance between rri intervals.
        :param mode: Mode to aggregate the distance. Can either be "mean" or
         some value q between 0 and 1 for the q-th quantile.
        :type mode: int or str
        :param labels: consider only ecgs with these labels, if None all
        instances
        :type labels: Union[Optional[int], List[int]]
        :return: rri-interval
        :rtype: int
        """
        if self._qrs_complexes is None:
            raise RuntimeError("Call calculate_qrs_complexes beforehand")

        if labels is None:
            labels = self.RHYTHMS.values()
        elif isinstance(labels, int):
            labels = [labels]
        ids = self.get_ids_with_label(labels).tolist()
        rri_intervals_lead = [[] for _ in range(len(self._qrs_complexes))]
        for i in range(len(self._leads)):
            qrs_complexes = np.array(self._qrs_complexes[i], dtype=object)[ids]
            for qrs_complex in qrs_complexes:
                rri_interval = np.diff(qrs_complex)
                rri_intervals_lead[i].extend(rri_interval.tolist())
        distances = []
        if mode == "mean":
            for rri_intervals in rri_intervals_lead:
                distances.append(int(np.mean(rri_intervals)))
        else:
            for rri_intervals in rri_intervals_lead:
                distances.append(int(np.quantile(rri_intervals, q=mode)))
        distances = np.array(distances)
        return distances
