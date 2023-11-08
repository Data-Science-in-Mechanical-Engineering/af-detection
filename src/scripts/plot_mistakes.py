import numpy as np

from src.plots.ecg import plot_misclassifications
from src.results import Result, RESULTS_FOLDER
from src.data.dataset import SPHDataset, COATDataset, ECGDataset

def get_number_peaks(dataset: ECGDataset, q: float):
    return int(np.quantile((list(map(len, dataset.qrs_complexes))), q))


if __name__ == '__main__':
    COAT = COATDataset.load_train() | COATDataset.load_validate() | COATDataset.load_test()
    SPH = SPHDataset.load_train() | SPHDataset.load_validate() | SPHDataset.load_test()
    svm_rri_folder = RESULTS_FOLDER / 'svm rri'
    rname = 'test_imbalanced_cross'

    result = Result.from_json(svm_rri_folder / (rname + '.json'))
    plot_misclassifications(
        result, 
        datasets = {"COATDataset": COAT, "SPHDataset": SPH}, 
        n_peaks_by_dataset={"COATDataset": get_number_peaks(COAT, 0.01), "SPHDataset": get_number_peaks(SPH, 0.01)}, 
        root_folder=svm_rri_folder / rname, 
        setup_folder=lambda snapshot: f"peaks/all_peaks/{snapshot.setup['dataset_train']['name']}_{snapshot.setup['dataset_validate']['name']}"
    )
