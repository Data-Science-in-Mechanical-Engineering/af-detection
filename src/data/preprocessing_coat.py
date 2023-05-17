import os
from zipfile import ZipFile

import numpy as np
import pandas as pd

from ..data.preprocessing import Diagnostics, COATDiagnostic, Preprocessing
from ..data.util import COATPath


class RawCOATFile:
    DIAGNOSTICS = "diagnostics.csv"
    ECG_ARCHIVE = "ecgs.zip"


LABEL_COLUMN = "ECG physician overread_0NoAF_1ScreenAF_3unknown"
ID_STRING_COLUMN = "PatientId"


class COATPreprocessing(Preprocessing[COATDiagnostic]):
    def __init__(self, p_train: float, p_validate: float):
        super(COATPreprocessing, self).__init__(
            p_train,
            p_validate,
            {0},
            COATPath.TRAIN_DATA,
            COATPath.VALIDATE_DATA,
            COATPath.TEST_DATA,
            COATPath.RAW_DATA / RawCOATFile.ECG_ARCHIVE
        )

    def extract_diagnostics(self) -> Diagnostics:
        diagnostics_path = COATPath.RAW_DATA / RawCOATFile.DIAGNOSTICS
        assert os.path.isfile(diagnostics_path), f"{diagnostics_path} wasn't found."
        assert diagnostics_path.name.endswith(".csv")

        # read in diagnostics file
        with open(diagnostics_path, "rb") as diagnostics_file:
            diagnostics = pd.read_csv(diagnostics_file)

        assert ID_STRING_COLUMN in diagnostics, f"Diagnostics file should contain `{ID_STRING_COLUMN}` column."
        assert LABEL_COLUMN in diagnostics, f"Diagnostics file should contain `{LABEL_COLUMN}` column."

        # obtain diagnostic entries that are not excluded
        return [
            COATDiagnostic.from_string_patient_id(rhythm, id_string)
            for id_string, rhythm in zip(diagnostics[ID_STRING_COLUMN], diagnostics[LABEL_COLUMN])
        ]

    def extract_ecg_signal(self, diagnostic: COATDiagnostic, zip_file: ZipFile) -> dict[int, np.ndarray]:
        # the CSV files are contained in a folder sharing the name of the archive itself
        filename = f"{diagnostic.device_id:03d}-PAT-{diagnostic.patient_id:04d}.csv"

        with zip_file.open(filename, "r") as ecg_file:
            # noinspection PyTypeChecker
            ecg = np.genfromtxt(ecg_file, delimiter=",")
            return {0: ecg}  # data has only one lead (indexed 0)


if __name__ == "__main__":
    processing = COATPreprocessing(0.6, 0.2)
    processing()
