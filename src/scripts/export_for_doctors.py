import csv
import numpy as np
from pathlib import Path
from typing import Callable

from src.results import Result, RESULTS_FOLDER, Snapshot
from src.data.dataset import SPHDataset, COATDataset, Identifier
from src.scripts.util import DatasetT

def get_number_peaks(dataset: DatasetT, q: float):
    return int(np.quantile((list(map(len, dataset.qrs_complexes))), q))

def get_r_peaks_times(dataset_validate: DatasetT, identifiers: list[Identifier]):
    entries = list(map(dataset_validate.get_by_identifier, identifiers))
    return list(map(lambda ecg_entry: ecg_entry.qrs_complexes, entries))

def export_mistakes_as_csv(mistakes_to_export, file_path):
    with file_path.open('w') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(("Identifier", "R-peaks position indices"))
        writer.writerows(mistakes_to_export)

def export_mistakes(
        result: Result,
        datasets_train: list[str],
        datasets_validate: dict[str, DatasetT],
        root_folder: Path,
        setup_folder: Callable[[Snapshot], Path]
    ):

    for snapshot in result:
        dataset_train_name = snapshot.setup["dataset_train"]["name"]
        dataset_validate_name = snapshot.setup["dataset_validate"]["name"]
        dataset_validate = datasets_validate.get(dataset_validate_name)
        if dataset_validate is None or dataset_train_name not in datasets_train:
            print(f"Skipping outcome of {dataset_train_name} / {dataset_validate_name}")
            continue

        snapshot_folder = root_folder / setup_folder(snapshot)
        false_positives_file = snapshot_folder / "false_positives.csv"
        false_negatives_file = snapshot_folder / "false_negatives.csv"

        snapshot_folder.mkdir(parents=True, exist_ok=True)

        for outcome in snapshot:
            false_positives_ids = outcome.false_positives
            false_positives_r_peaks_times = get_r_peaks_times(dataset_validate, false_positives_ids)
            false_negatives_ids = outcome.false_negatives
            false_negatives_r_peaks_times = get_r_peaks_times(dataset_validate, false_negatives_ids)

            false_positives_to_export = list(map(
                lambda idpeaks: [idpeaks[0],]+ list(idpeaks[1]),
                list(zip(false_positives_ids, false_positives_r_peaks_times))
            ))
            false_negatives_to_export = list(map(
                lambda idpeaks: [idpeaks[0],]+ list(idpeaks[1]), 
                list(zip(false_negatives_ids, false_negatives_r_peaks_times))
            ))

            export_mistakes_as_csv(false_positives_to_export, false_positives_file)
            export_mistakes_as_csv(false_negatives_to_export, false_negatives_file)

def export_training_indices(dataset: DatasetT, file_path: Path):
    af_labels = {dataset.AFIB}
    n_afib_train = dataset.count(*af_labels)
    balanced_dataset = dataset.balanced_binary_partition(af_labels, n_afib_train)
    with file_path.open('w') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerows(list(map(lambda id: [str(id)], balanced_dataset.identifiers)))


if __name__ == '__main__':
    COAT = COATDataset.load_train() | COATDataset.load_validate() | COATDataset.load_test()
    SPH = SPHDataset.load_train() | SPHDataset.load_validate() | SPHDataset.load_test()
    svm_rri_folder = RESULTS_FOLDER / 'svm rri'
    rname = 'test_imbalanced_cross'

    result = Result.from_json(svm_rri_folder / (rname + '.json'))
    export_mistakes(
        result, 
        datasets_train=["COATDataset"],
        # datasets = {"COATDataset": COAT, "SPHDataset": SPH}, 
        datasets_validate = {"COATDataset": COAT},
        root_folder=svm_rri_folder / rname, 
        setup_folder=lambda snapshot: f"export/mistakes/{snapshot.setup['dataset_train']['name']}_{snapshot.setup['dataset_validate']['name']}"
    )
    export_training_indices(
        COATDataset.load_train(),
        file_path=svm_rri_folder / rname / "export" / "training_set_identifiers.csv" 
    )
