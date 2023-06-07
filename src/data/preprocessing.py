import pickle
import random
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple, TypeVar, Generic, Final
from zipfile import ZipFile

import numpy as np
import tqdm

from .qrs import PeakDetectionAlgorithm
from ..data.dataset import Identifier, SPHIdentifier, COATIdentifier
from ..data.util import RANDOM_SEED

TLabel = TypeVar("TLabel")


@dataclass(frozen=True)
class Diagnostic(ABC, Generic[TLabel]):
    label: TLabel

    @abstractmethod
    def get_identifier(self) -> Identifier:
        raise NotImplementedError


@dataclass(frozen=True)
class SPHDiagnostic(Diagnostic[str]):
    filename: str

    def get_identifier(self) -> SPHIdentifier:
        return SPHIdentifier(self.filename)


@dataclass(frozen=True)
class COATDiagnostic(Diagnostic[int]):
    device_id: int
    patient_id: int

    @staticmethod
    def from_string_patient_id(rhythm: int, patient_id: str, delimiter="-PAT-"):
        split = patient_id.split(delimiter)
        clean_split = map(str.strip, split)
        device_id, patient_id = map(int, clean_split)
        return COATDiagnostic(rhythm, device_id, patient_id)

    def get_identifier(self) -> COATIdentifier:
        return COATIdentifier(self.device_id, self.patient_id)


TDiagnostic = TypeVar("TDiagnostic", bound=Diagnostic)
Diagnostics = list[TDiagnostic]
DiagnosticsGroup = dict[str, list[TDiagnostic]]
DiagnosticLeadInfo = dict[TDiagnostic, dict[int, np.ndarray]]


class Split(NamedTuple):
    train: Diagnostics
    validate: Diagnostics
    test: Diagnostics


def load_excluded(path: Path) -> set:
    with open(path) as excluded_file:
        lines = excluded_file.readlines()
        clean_lines = map(str.strip, lines)
        valid_lines = filter(len, clean_lines)

    return set(valid_lines)


def group_diagnostics_by_rhythm(diagnostics: Diagnostics) -> DiagnosticsGroup:
    """ Groups diagnostic entries by their rhythm annotations.

    Args:
        diagnostics: A list of diagnostic entries in an arbitrary order.

    Returns:
        A dictionary indexed by rhythm annotations mapping to the diagnostic entries with that label.
    """
    groups = defaultdict(list)

    for diagnostic in diagnostics:
        groups[diagnostic.label].append(diagnostic)

    return groups


def extract_qrs_complexes(
        ecg_leads: DiagnosticLeadInfo,
        qrs_algorithm: PeakDetectionAlgorithm,
        sampling_rate: int
) -> DiagnosticLeadInfo:
    """ Extracts QRS complexes from ECG signals.

    We represent the QRS complexes as the indices of RR peaks.

    Args:
        ecg_leads: A mapping from each diagnostic to a numpy ECG signal for each given lead.
        qrs_algorithm: The algorithm to use for R peak extraction.

    Returns:
        A mapping from each diagnostic to a numpy array of RR peaks for each given lead.
    """
    qrs_complexes = defaultdict(dict)

    progress = tqdm.tqdm(ecg_leads, unit="ECGs")
    progress.set_description("Extracting QRS complexes from ECG")

    for diagnostic in progress:
        for lead, ecg_signal in ecg_leads[diagnostic].items():
            qrs_complexes[diagnostic][lead] = qrs_algorithm(ecg_signal, sampling_rate=sampling_rate)

    return qrs_complexes


def split_train_validate_test(rhythm_groups: DiagnosticsGroup, p_train: float, p_validate: float) -> Split:
    """ Splits diagnostic entries into train / validation / test datasets.

    The relative frequency of rhythm annotations in the dataset as a whole are obtained.
    The percentage of test entries is inferred from the percentage of train and validate entries.

    Args:
        rhythm_groups: Diagnostic entries grouped by their rhythm annotation.
        p_train: Percentage of train entries.
        p_validate: Percentage of validate entries.

    Returns:
        A partition of the given diagnostic entries into train / validate / test subsets of given sizes.
    """
    assert p_train + p_validate <= 1, "It must be p_train + p_validate <= 1."
    assert p_train >= 0, "It must be p_train >= 0."
    assert p_validate >= 0, "It must be p_validate >= 0."

    random.seed(RANDOM_SEED)
    train, validate, test = [], [], []

    # we randomly partition into train / validate / test per rhythm annotations to maintain rhythm frequency
    for rhythm, diagnostics in rhythm_groups.items():
        # randomly shuffle a copy of the diagnostic entries
        diagnostics = diagnostics.copy()
        random.shuffle(diagnostics)

        train_offset = round(len(diagnostics) * p_train)  # index until end of train entries
        validate_offset = train_offset + round(len(diagnostics) * p_validate)  # index until end of validate entries

        # we will randomly shuffle the entries and partition into train / validate / test contiguously
        train.extend(diagnostics[:train_offset])
        validate.extend(diagnostics[train_offset:validate_offset])
        test.extend(diagnostics[validate_offset:])

    assert len(train) + len(validate) + len(test) == sum(map(len, rhythm_groups.values())), "Bug alert: Size mismatch."

    return Split(train, validate, test)


def save_diagnostics(path: Path, diagnostics: Diagnostics):
    """ Saves identifiers and labels to disk.

    Args:
        path: The path to the sub-folder of the SPH data folder to save the data to (train / validate / test).
        diagnostics: The diagnostics to save.
    """
    labels = [diagnostic.label for diagnostic in diagnostics]

    identifiers = [diagnostic.get_identifier() for diagnostic in diagnostics]

    with open(path / "identifiers.pickle", "wb+") as identifiers_file:
        pickle.dump(identifiers, identifiers_file)

    with open(path / "labels.npy", "wb+") as labels_file:
        # noinspection PyTypeChecker
        np.save(labels_file, labels)


def save_lead_signal(path: Path, signals: DiagnosticLeadInfo, filename_prefix: str):
    data = defaultdict(list)

    for lead_signals in signals.values():
        for lead, signal in lead_signals.items():
            data[lead].append(signal)

    for lead, signals in data.items():
        with open(path / f"{filename_prefix}_{lead}.npy", "wb+") as signal_file:
            # noinspection PyTypeChecker
            np.save(signal_file, np.array(signals, dtype=object))


def assert_diagnostic_ecg_qrs_format(
        diagnostics: Diagnostics,
        ecg_leads: DiagnosticLeadInfo,
        leads: set[int],
        qrs_leads: DiagnosticLeadInfo
):
    assert_message = "Bug alert: Format inconsistency between ECG signals and QRS complexes."
    assert len(ecg_leads) == len(qrs_leads) == len(diagnostics), assert_message
    assert all(diagnostic in diagnostics for diagnostic in diagnostics), assert_message
    assert all(lead in diagnostic_ecg for lead in leads for diagnostic_ecg in ecg_leads.values()), assert_message
    assert all(lead in diagnostic_qrs for lead in leads for diagnostic_qrs in ecg_leads.values()), assert_message
    assert all(
        all(len(ecg_leads[diagnostic][lead] == len(qrs_leads[diagnostic][lead])) for lead in leads)
        for diagnostic in diagnostics
    ), assert_message


class Preprocessing(ABC, Generic[TDiagnostic]):
    p_train: Final[float]
    p_validate: Final[float]
    leads: set[int]

    train_folder: Final[Path]
    validate_folder: Final[Path]
    test_folder: Final[Path]
    archive_folder: Final[Path]

    qrs_algorithm: Final[PeakDetectionAlgorithm]
    sampling_rate: int

    def __init__(
            self,
            p_train: float,
            p_validate: float,
            leads: set[int],
            train_folder: Path,
            validate_folder: Path,
            test_folder: Path,
            archive_path: Path,
            qrs_algorithm: PeakDetectionAlgorithm,
            sampling_rate: int
    ):
        assert len(leads) >= 1

        self.p_train = p_train
        self.p_validate = p_validate
        self.leads = leads
        self.train_folder = train_folder
        self.validate_folder = validate_folder
        self.test_folder = test_folder
        self.archive_folder = archive_path
        self.qrs_algorithm = qrs_algorithm
        self.sampling_rate = sampling_rate

    @abstractmethod
    def extract_diagnostics(self) -> Diagnostics[TDiagnostic]:
        raise NotImplementedError

    @abstractmethod
    def extract_ecg_signal(self, diagnostic: TDiagnostic, zip_file: ZipFile) -> dict[int, np.ndarray]:
        """ Extracts an ECG signals for every specified lead from a given archive and for a given diagnostic.

        Only the leads specified during construction should be considered.

        Args:
            diagnostic: The diagnostic (i.e. identifiers) for which to extract the ECG signals.
            zip_file: The archive in which to look for the raw ECG data for the diagnostic.

        Returns:
            A mapping from the specified leads to the extracted ECG signals for the given diagnostic.
        """
        raise NotImplementedError

    def make_split_folder(self, path: Path) -> Path:
        """ Determines the folder to which the training / validation / test data is stored.

        If the folder does not yet exist, it is created first.
        This method is necessary if there is more than one training / validation / test dataset (e.g. denoised, noisy)
        and we must determine a sub-folder of the given folder to store the dataset to.

        Args:
            path: The path where the data should be stored.

        Returns:
            An existing path to the folder where the dataset should actually be stored.
        """
        path.mkdir(parents=True, exist_ok=True)
        return path

    def extract_ecg_signals(self, diagnostics: Diagnostics[COATDiagnostic]) -> DiagnosticLeadInfo:
        """ Extracts the ECG signals corresponding to a list of diagnostics for each specified lead.

        We look for the ECGs in the archive specified during construction.

        Args:
            diagnostics: A list of diagnostics specifying which signals to construct.

        Returns:
            A mapping from diagnostics to a mapping from lead to extracted ECG signals for all diagnostics, leads.
        """
        assert self.archive_folder.name.endswith(".zip")

        with ZipFile(self.archive_folder) as zip_file:
            progress = tqdm.tqdm(diagnostics, unit="files")
            progress.set_description("Extracting ECG signals from disk")

            return {
                diagnostic: self.extract_ecg_signal(diagnostic, zip_file)
                for diagnostic in progress
            }

    def __call__(self):
        """ Preprocesses the dataset by saving ECGs and QRS complexes as numpy and diagnostics (i.e. labels) as pickle.

        1. We extract the diagnostic information (identifiers and rhythm labels) from disk.
        2. We split the diagnostics into training / validation / test datasets.
        3. For each sub-sample (i.e. training / validation / test), we extract ECG and QRS signals and store them.
        """
        diagnostics = self.extract_diagnostics()
        rhythm_groups = group_diagnostics_by_rhythm(diagnostics)
        split = split_train_validate_test(rhythm_groups, self.p_train, self.p_validate)

        paths = [self.train_folder, self.validate_folder, self.test_folder]
        data = [split.train, split.validate, split.test]

        for path, diagnostics in zip(paths, data):
            folder = self.make_split_folder(path)

            print(f"Start unpacking to {folder}")
            ecg_leads = self.extract_ecg_signals(diagnostics)
            qrs_leads = extract_qrs_complexes(ecg_leads, self.qrs_algorithm, self.sampling_rate)

            assert_diagnostic_ecg_qrs_format(diagnostics, ecg_leads, self.leads, qrs_leads)

            save_diagnostics(folder, diagnostics)
            save_lead_signal(folder, ecg_leads, "ecg_lead")
            save_lead_signal(folder, qrs_leads, f"qrs_{self.qrs_algorithm.name}_lead")
