from argparse import Namespace, ArgumentParser

from src.scripts.util import args_parse_c, args_parse_classifier, args_parse_bandwidth_rri, args_parse_setup, \
    args_add_c, args_add_classifier, args_add_bandwidth_rri, args_add_setup, finish_experiment
from src.experiments import ExperimentRRI


def main_performance_experiment_rri(arguments: Namespace):
    cs = args_parse_c(arguments)
    classifier = args_parse_classifier(arguments)
    bandwidths_rri = args_parse_bandwidth_rri(arguments)
    setups = args_parse_setup(arguments)

    experiment = ExperimentRRI(cs, [1], classifier, bandwidths_rri)

    for setup in setups:
        af_labels_train = {setup.training.AFIB}
        af_labels_validate = {setup.validating.AFIB}

        if arguments.imbalanced_training:
            ds_train = setup.training
        else:
            n_afib_train = setup.training.count(*af_labels_train)
            ds_train = setup.training.balanced_binary_partition(af_labels_train, n_afib_train)
        if arguments.imbalanced_validating:
            ds_validate = setup.validating
        else:
            n_afib_validate = setup.validating.count(*af_labels_validate)
            ds_validate = setup.validating.balanced_binary_partition(af_labels_validate, n_afib_validate)

        print("\n", f"Training: {ds_train}", "\n", f"Validation: {ds_validate}", "\n")

        experiment(
            ds_train,
            ds_validate,
            af_labels_train,
            af_labels_validate
        )

    finish_experiment(experiment.result)


if __name__ == "__main__":
    parser = ArgumentParser()

    args_add_c(parser)
    args_add_classifier(parser)
    args_add_bandwidth_rri(parser)
    args_add_setup(parser, "cross")

    parser.add_argument("--imbalanced_training", dest="imbalanced_training", 
                        default=False, action="store_true",
                        help="Whether or not to train on the full dataset.")
    parser.add_argument("--imbalanced_validating", dest="imbalanced_training", 
                        default=False, action="store_true",
                        help="Whether or not to validate on the full dataset.")

    args = parser.parse_args()
    main_performance_experiment_rri(args)
