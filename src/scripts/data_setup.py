from src.data.preprocessing_coat import COATPreprocessing
from src.data.preprocessing_sph import SPHPreprocessing, download_sph_raw
from src.data.qrs import XQRSPeakDetectionAlgorithm


def main():
    print("Processing DiagnoStick data...")
    coat_processor = COATPreprocessing(p_train=0.6, p_validate=0.2, qrs_algorithm=XQRSPeakDetectionAlgorithm())
    coat_processor()
    print("Done.\nProcessing SPH data...")

    download_sph_raw()
    sph_processor = SPHPreprocessing(p_train=0.6, p_validate=0.2, qrs_algorithm=XQRSPeakDetectionAlgorithm())
    sph_processor()
    print("Done.")

if __name__ == '__main__':
    main()