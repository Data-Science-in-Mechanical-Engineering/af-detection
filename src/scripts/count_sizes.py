from src.data.dataset import COATDataset, SPHDataset, ECGDataset
from src.data.qrs import XQRSPeakDetectionAlgorithm
from src.scripts.util import SPHSetup, COATSetup
from typing import TypeVar

AFIBDataset = TypeVar("AFIBDataset", COATDataset, SPHDataset)

# def standard_preprocessing(ds: AFIBDataset, ):
#     ds.filter(lambda entry: len(entry.qrs_complexes) > STANDARD_SPH_MINIMUM_RRIS

def get_balanced(ds: AFIBDataset):
    n_afib = ds.count(ds.AFIB)
    return ds.balanced_binary_partition({ds.AFIB}, n_afib)

def get_len(ds: AFIBDataset):
    counts = ds.count_labels()
    return sum(counts.values())

def run():
    qrs_algorithm = XQRSPeakDetectionAlgorithm()
    dataset_groups = {'DiagnoStick': [
        COATSetup.from_standard_preprocessing(
            COATDataset.load_train(qrs_algorithm),COATDataset.load_train(qrs_algorithm)
        ).training,
        COATSetup.from_standard_preprocessing(
            COATDataset.load_validate(qrs_algorithm),COATDataset.load_validate(qrs_algorithm)
        ).training,
        COATSetup.from_standard_preprocessing(
            COATDataset.load_test(qrs_algorithm),COATDataset.load_test(qrs_algorithm)
        ).training,
    ],
        'SPH': [
        SPHSetup.from_standard_preprocessing(
            SPHDataset.load_train(qrs_algorithm),SPHDataset.load_train(qrs_algorithm)
        ).training,
        SPHSetup.from_standard_preprocessing(
            SPHDataset.load_validate(qrs_algorithm),SPHDataset.load_validate(qrs_algorithm)
        ).training,
        SPHSetup.from_standard_preprocessing(
            SPHDataset.load_test(qrs_algorithm),SPHDataset.load_test(qrs_algorithm)
        ).training,
    ]
    }
    
    # dataset_groups = {'DiagnoStick': [
    #     COATDataset.load_train(qrs_algorithm),
    #     COATDataset.load_validate(qrs_algorithm),
    #     COATDataset.load_test(qrs_algorithm)
    # ],
    #     'SPH': [
    #     SPHDataset.load_train(qrs_algorithm),
    #     SPHDataset.load_validate(qrs_algorithm),
    #     SPHDataset.load_test(qrs_algorithm)
    # ]
    # }

    labels = ['Training', 'Validation', 'Testing']
    for name, grp in dataset_groups.items():
        for ds_name, ds in zip(labels, grp):
            print(name, ':', ds_name, ':', get_len(ds))
            print(name, ':', ds_name, '(balanced) :', get_len(get_balanced(ds)))

if __name__ == '__main__':
    run()