import os.path
from pathlib import Path
from typing import Final
from urllib import request
from zipfile import ZipFile

import pandas as pd

from .dataset import COATDataset
from .qrs import PeakDetectionAlgorithm
from ..data.preprocessing import SPHDiagnostic, Diagnostics, DiagnosticLeadInfo, load_excluded, Preprocessing
from ..data.util import SPHPath

SPH_N_LEADS = 12


class RawSPHFile:
    RHYTHMS = "RhythmNames.xlsx"
    CONDITIONS = "ConditionNames.xlsx"
    DIAGNOSTICS = "Diagnostics.xlsx"
    ATTRIBUTES = "AttributesDictionary.xlsx"
    ECG_DENOISED_ARCHIVE = "ECGDataDenoised.zip"
    ECG_NOISY_ARCHIVE = "ECGData.zip"


DOWNLOAD_URLS = {
    RawSPHFile.RHYTHMS: "https://figshare.com/ndownloader/files/15651296",
    RawSPHFile.CONDITIONS: "https://figshare.com/ndownloader/files/15651293",
    RawSPHFile.DIAGNOSTICS: "https://figshare.com/ndownloader/files/15653771",
    RawSPHFile.ATTRIBUTES: "https://figshare.com/ndownloader/files/15653762",
    RawSPHFile.ECG_DENOISED_ARCHIVE: "https://figshare.com/ndownloader/files/15652862",
    RawSPHFile.ECG_NOISY_ARCHIVE: "https://figshare.com/ndownloader/files/15651326",
}

EXCLUDED_PATH = SPHPath.ROOT / "excluded-filenames.txt"
assert os.path.isfile(EXCLUDED_PATH), f"Explicitly exclude unsuited SPH files in {EXCLUDED_PATH}."


def _make_sph_raw_folder() -> Path:
    """ Obtains the folder in which all SPH data is obtained and makes sure this folder exists.

    Returns:
        The path to the (existing) SPH data folder.
    """
    folder = Path(SPHPath.RAW_DATA)
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def download_sph_raw():
    folder = _make_sph_raw_folder()
    downloaded_files = {file.name for file in folder.iterdir() if file.is_file()}

    print("Start downloading raw SPH data. Note that this may take some time ...")

    for file_name, download_url in DOWNLOAD_URLS.items():
        if file_name in downloaded_files:
            print(f"\t - Skipping {file_name} (already exists)")
            continue

        request.urlretrieve(download_url, folder / file_name)
        print(f"\t - Downloaded {file_name}")

    print("Finished downloading raw SPH data.")


class SPHPreprocessing(Preprocessing[SPHDiagnostic]):
    ecg_archive: Final[str]

    def __init__(
            self,
            p_train: float,
            p_validate: float,
            qrs_algorithm: PeakDetectionAlgorithm,
            leads: set[int] | None = None,
            denoised: bool = True,
    ):
        leads = {0} if leads is None else leads
        assert leads <= set(range(SPH_N_LEADS)), "Leads must be in {0, ..., 11}"

        ecg_archive = RawSPHFile.ECG_DENOISED_ARCHIVE if denoised else RawSPHFile.ECG_NOISY_ARCHIVE

        super(SPHPreprocessing, self).__init__(
            p_train,
            p_validate,
            leads,
            SPHPath.TRAIN_DATA,
            SPHPath.VALIDATE_DATA,
            SPHPath.TEST_DATA,
            SPHPath.RAW_DATA / ecg_archive,
            qrs_algorithm,
            COATDataset.FREQUENCY
        )

        self.ecg_archive = ecg_archive

    def make_split_folder(self, path: Path) -> Path:
        """ Determines the folder to which the training / validation / test data is stored.

        We modify the given path such that a dedicated sub-folder is chosen that differentiates between noisy and
        denoised training / validation / test data.
        If the folder does not yet exist, it is created first.

        :param path: The path where the data should be stored.
        :return: An existing path to the folder where the dataset should actually be stored.
        """
        path = path / self.ecg_archive.removesuffix(".zip")
        path.mkdir(parents=True, exist_ok=True)
        return path

    def extract_diagnostics(self) -> Diagnostics:
        """ Extracts filenames and the associated rhythm labels.

        Some entries are excluded (e.g. because there is an issue with the recording).
        Excluded entries are specified by their filename in the file specified in `SPH_EXCLUDED_PATH`.

        Returns:
            The diagnostic entries that are not explicitly excluded.
        """
        diagnostics_path = SPHPath.RAW_DATA / RawSPHFile.DIAGNOSTICS
        assert os.path.isfile(diagnostics_path), f"{diagnostics_path} wasn't found."

        # read in diagnostics file
        with open(diagnostics_path, "rb") as diagnostics_file:
            diagnostics = pd.read_excel(diagnostics_file)

        assert "FileName" in diagnostics, f"Diagnostics file should contain `FileName` column."
        assert "Rhythm" in diagnostics, f"Diagnostics file should contain `Rhythm` column."

        # read in excluded filenames
        excluded_filenames = load_excluded(EXCLUDED_PATH)

        # obtain diagnostic entries that are not excluded
        return [
            SPHDiagnostic(rhythm, filename)
            for filename, rhythm in zip(diagnostics["FileName"], diagnostics["Rhythm"])
            if filename not in excluded_filenames
        ]

    def extract_ecg_signal(self, diagnostic: SPHDiagnostic, zip_file: ZipFile) -> DiagnosticLeadInfo:
        # the CSV files are contained in a folder sharing the name of the archive itself
        archive_name = self.ecg_archive.removesuffix(".zip")
        filename = f"{archive_name}/{diagnostic.filename}.csv"

        with zip_file.open(filename) as ecg_file:
            file_ecg_signals = pd.read_csv(ecg_file).to_numpy()  # all ECG leads in numpy format
            return {lead: file_ecg_signals[:, lead] for lead in self.leads}  # ECG leads
