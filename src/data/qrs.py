from abc import ABC, abstractmethod

import neurokit2 as nk
import numpy as np
from wfdb.processing import XQRS, correct_peaks


class PeakDetectionAlgorithm(ABC):
    name: str

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def __call__(self, ecg: np.ndarray, sampling_rate: float) -> np.ndarray:
        raise NotImplementedError


class XQRSPeakDetectionAlgorithm(PeakDetectionAlgorithm):
    def __init__(self, name=None):
        super(XQRSPeakDetectionAlgorithm, self).__init__("xqrs" if name is None else name)

    def __call__(self, ecg: np.ndarray, sampling_rate: float) -> np.ndarray:
        xqrs = XQRS(sig=ecg, fs=sampling_rate)
        xqrs.detect(verbose=False)
        return np.array(xqrs.qrs_inds)


class CorrectedXQRSPeakDetectionAlgorithm(XQRSPeakDetectionAlgorithm):
    def __init__(self, search_radius: int, smooth_window_size: int, peak_dir: str):
        super(CorrectedXQRSPeakDetectionAlgorithm, self).__init__("xqrs_corr")
        self.search_radius = search_radius
        self.smooth_window_size = smooth_window_size
        self.peak_dir = peak_dir

    def __call__(self, ecg: np.ndarray, sampling_rate: float):
        peaks = super().__call__(ecg, sampling_rate)
        corrected = correct_peaks(ecg, peaks, self.search_radius, self.smooth_window_size, self.peak_dir)
        return corrected


def make_neurokit_peak_extraction_algorithm(name: str):
    class NeurokitAlgorithm(PeakDetectionAlgorithm):
        def __init__(self):
            super(NeurokitAlgorithm, self).__init__(name)

        def __call__(self, ecg: np.ndarray, sampling_rate: int) -> np.ndarray:
            try:
                ecg = nk.ecg_clean(ecg, sampling_rate=sampling_rate)
                _, info = nk.ecg_peaks(ecg, sampling_rate=sampling_rate, method=self.name)
            except:
                return np.array([])

            return info["ECG_R_Peaks"]

    return NeurokitAlgorithm


Neurokit = make_neurokit_peak_extraction_algorithm("neurokit")
PanTompkins1985 = make_neurokit_peak_extraction_algorithm("pantompkins1985")
Hamilton2002 = make_neurokit_peak_extraction_algorithm("hamilton2002")
Zong2003 = make_neurokit_peak_extraction_algorithm("zong2003")
Christov2004 = make_neurokit_peak_extraction_algorithm("christov2004")
Gamboa2008 = make_neurokit_peak_extraction_algorithm("gamboa2008")
Elgendi2010 = make_neurokit_peak_extraction_algorithm("elgendi2010")
Kalidas2017 = make_neurokit_peak_extraction_algorithm("kalidas2017")
Nabian2018 = make_neurokit_peak_extraction_algorithm("nabian2018")
Rodrigues2021 = make_neurokit_peak_extraction_algorithm("rodrigues2021")
Promac = make_neurokit_peak_extraction_algorithm("promac")


ALL_WORKING_PEAK_DETECTION_ALGORITHMS = [
    XQRSPeakDetectionAlgorithm,
    CorrectedXQRSPeakDetectionAlgorithm,
    Hamilton2002,
    Zong2003,
    Christov2004,
    Elgendi2010,
    Rodrigues2021,
    Promac
]

ALL_PEAK_DETECTION_ALGORITHMS = ALL_WORKING_PEAK_DETECTION_ALGORITHMS + [
    Neurokit,
    PanTompkins1985,
    Gamboa2008,
    Kalidas2017,
    Nabian2018,
]
