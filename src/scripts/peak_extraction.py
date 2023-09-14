from argparse import Namespace, ArgumentParser

from .util import args_parse_c, args_parse_classifier, args_parse_bandwidth_rri, args_add_c, \
    args_add_classifier, args_add_bandwidth_rri, SPHSetup, finish_experiment
from ..data.dataset import SPHDataset
from ..data.qrs import ALL_WORKING_PEAK_DETECTION_ALGORITHMS
from ..experiments import ExperimentRRI


def main_peak_extraction_experiment_rri_sph(arguments: Namespace):
    cs = args_parse_c(arguments)
    classifier = args_parse_classifier(arguments)
    bandwidths_rri = args_parse_bandwidth_rri(arguments)

    experiment = ExperimentRRI(cs, [1], classifier, bandwidths_rri)

    for algorithm_cls in ALL_WORKING_PEAK_DETECTION_ALGORITHMS:
        algorithm = algorithm_cls()

        if arguments.test:
            dataset_validate = SPHDataset.load_test(qrs_algorithm=algorithm)
        else:
            dataset_validate = SPHDataset.load_test(qrs_algorithm=algorithm)

        dataset_train = SPHDataset.load_train(qrs_algorithm=algorithm)
        setup = SPHSetup.from_standard_preprocessing(dataset_train, dataset_validate)

        print(f"Algorithm: {algorithm.name}")

        experiment(
            setup.training,
            setup.validating,
            {setup.training.AFIB},
            {setup.validating.AFIB},
            {"peak_detection_algorithm": algorithm.name}
        )

        print("\n")

    finish_experiment(experiment.result)


if __name__ == "__main__":
    parser = ArgumentParser()

    args_add_c(parser)
    args_add_classifier(parser)
    args_add_bandwidth_rri(parser)

    parser.add_argument("--test", dest="test", default=False, action="store_true",
                        help="Whether or not to evaluate on the test dataset.")

    args = parser.parse_args()
    main_peak_extraction_experiment_rri_sph(args)
