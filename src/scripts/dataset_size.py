import math
from argparse import ArgumentParser, Namespace

from src.scripts.util import args_add_c, args_parse_c, args_parse_classifier, args_add_classifier, args_add_bandwidth_rri, args_parse_bandwidth_rri, args_add_rho, args_parse_rho, args_add_setup, args_parse_setup, Setup, finish_experiment
from src.experiments import Experiment, ExperimentRRI


def run_logsize_experiment(
        experiment: Experiment,
        setup: Setup,
        af_labels_train: set,
        af_labels_validate: set,
        repetitions: int,
        imbalanced_validating: bool
):
    assert repetitions >= 1
    assert af_labels_train <= setup.training.label_domain()
    assert af_labels_validate <= setup.validating.label_domain()

    if imbalanced_validating:
        dataset_validate = setup.validating
    else:
        dataset_validate = setup.validating.balanced_binary_partition(
            af_labels_validate,
            setup.validating.count(*af_labels_validate)
        )

    n_afib_instances_training = setup.training.count(setup.training.AFIB)
    log_n_afib_instances_training = int(math.log2(n_afib_instances_training))
    sizes = [2 ** k for k in range(log_n_afib_instances_training)] + [n_afib_instances_training]

    for size in sizes:
        for _ in range(repetitions):
            sub_sampled_dataset_train = setup.training.balanced_binary_partition(af_labels_train, size, None)

            experiment(
                sub_sampled_dataset_train,
                dataset_validate,
                af_labels_train,
                af_labels_validate
            )


def main_logsize_experiment_rri(arguments: Namespace):
    cs = args_parse_c(arguments)
    classifier = args_parse_classifier(arguments)
    bandwidths_rri = args_parse_bandwidth_rri(arguments)
    c_class_weight_proportions = args_parse_rho(arguments)
    setups = args_parse_setup(arguments)
    repetitions = arguments.repetitions

    experiment = ExperimentRRI(cs, c_class_weight_proportions, classifier, bandwidths_rri)

    for setup in setups:
        print("\n", f"Training: {setup.training} \t Validation: {setup.validating}", "\n")

        run_logsize_experiment(
            experiment,
            setup,
            {setup.training.AFIB},
            {setup.validating.AFIB},
            repetitions,
            arguments.imbalanced_validating
        )

    finish_experiment(experiment.result)


if __name__ == "__main__":
    parser = ArgumentParser()

    args_add_c(parser)
    args_add_classifier(parser)
    args_add_bandwidth_rri(parser)
    args_add_rho(parser)
    args_add_setup(parser, "cross")

    parser.add_argument("--repetitions", dest="repetitions", type=int, default=20,
                        help="Number of independent runs per fixed size.")
    parser.add_argument("--imbalanced_validating", dest="imbalanced_validating", 
                        default=False, action="store_true",
                        help="Whether or not to validate on the full dataset.")

    args = parser.parse_args()
    main_logsize_experiment_rri(args)
