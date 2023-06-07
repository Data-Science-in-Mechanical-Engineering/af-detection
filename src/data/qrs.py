from abc import ABC, abstractmethod

import neurokit2 as nk
import numpy as np
from wfdb.processing import XQRS


class PeakDetectionAlgorithm(ABC):
    name: str

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def __call__(self, ecg: np.ndarray, sampling_rate: float) -> np.ndarray:
        raise NotImplementedError


class XQRSPeakDetectionAlgorithm(PeakDetectionAlgorithm):
    def __init__(self):
        super(XQRSPeakDetectionAlgorithm, self).__init__("xqrs")

    def __call__(self, ecg: np.ndarray, sampling_rate: float) -> np.ndarray:
        xqrs = XQRS(sig=ecg, fs=sampling_rate)
        xqrs.detect(verbose=False)
        return np.array(xqrs.qrs_inds)


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

ALL_PEAK_DETECTION_ALGORITHMS = [
    XQRSPeakDetectionAlgorithm,
    Neurokit,
    PanTompkins1985,
    Hamilton2002,
    Zong2003,
    Christov2004,
    Gamboa2008,
    Elgendi2010,
    Kalidas2017,
    Nabian2018,
    Rodrigues2021,
    Promac
]
