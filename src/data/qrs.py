from abc import ABC, abstractmethod

import neurokit2 as nk
import numpy as np

from src.data.dataset import ECGDataset, SPHDataset


# TODO: integrate new peak extraction abstraction into preprocessing
class PeakExtractionAlgorithm(ABC):
    name: str

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def __call__(self, ecg: np.ndarray, sampling_rate: float) -> np.ndarray:
        raise NotImplementedError


def make_neurokit_peak_extraction_algorithm(name: str):
    class NeurokitAlgorithm(PeakExtractionAlgorithm):
        def __init__(self):
            super(NeurokitAlgorithm, self).__init__(name)

        def __call__(self, ecg: np.ndarray, sampling_rate: int) -> np.ndarray:
            _, info = nk.ecg_peaks(ecg, sampling_rate=sampling_rate)
            return info["ECG_R_Peaks"]

    return NeurokitAlgorithm


Neurokit = make_neurokit_peak_extraction_algorithm("neurokit")
PanTompkins1985 = make_neurokit_peak_extraction_algorithm("pantompkins1985")
Hamilton2002 = make_neurokit_peak_extraction_algorithm("hamilton2002")
Zong2003 = make_neurokit_peak_extraction_algorithm("zong2003")
Martinez2004 = make_neurokit_peak_extraction_algorithm("martinez2004")
Christov2004 = make_neurokit_peak_extraction_algorithm("christov2004")
Gamboa2008 = make_neurokit_peak_extraction_algorithm("gamboa2008")
Elgendi2010 = make_neurokit_peak_extraction_algorithm("elgendi2010")
Engzeemod2012 = make_neurokit_peak_extraction_algorithm("engzeemod2012")
Kalidas2017 = make_neurokit_peak_extraction_algorithm("kalidas2017")
Nabian2018 = make_neurokit_peak_extraction_algorithm("nabian2018")
Rodrigues2021 = make_neurokit_peak_extraction_algorithm("rodrigues2021")
Koka2022 = make_neurokit_peak_extraction_algorithm("koka2022")
Promac = make_neurokit_peak_extraction_algorithm("promac")


def extract_r_peaks(algorithm: PeakExtractionAlgorithm, dataset: ECGDataset) -> list[np.ndarray]:
    all_r_peaks = []

    for ecg in dataset.ecg_signals:
        r_peaks = algorithm(ecg, dataset.FREQUENCY)
        all_r_peaks.append(r_peaks)

    return all_r_peaks


if __name__ == "__main__":
    validate_data = SPHDataset.load_validate()
    alg = PanTompkins1985()

    x = extract_r_peaks(alg, validate_data)
    print(list(map(len, x)))
