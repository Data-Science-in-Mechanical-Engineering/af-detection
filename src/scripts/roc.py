from argparse import Namespace, ArgumentParser

from src.scripts.util import args_add_c, args_add_classifier, args_add_bandwidth_rri, args_add_setup, args_add_rho, args_parse_c, args_parse_rho, args_parse_classifier, \
    args_parse_bandwidth_rri, args_parse_setup, finish_experiment
from src.experiments import ExperimentRRI


def main_logsize_experiment_rri(arguments: Namespace):
    cs = args_parse_c(arguments, linear=True)
    c_class_weight_proportions = args_parse_rho(arguments)
    classifier = args_parse_classifier(arguments)
    bandwidths = args_parse_bandwidth_rri(arguments)
    setups = args_parse_setup(arguments)

    experiment = ExperimentRRI(cs, c_class_weight_proportions, classifier, bandwidths)

    for setup in setups:
        af_labels_training = {setup.training.AFIB}
        af_labels_validate = {setup.validating.AFIB}

        n_afib_train = setup.training.count(*af_labels_training)
        ds_train = setup.training.balanced_binary_partition(af_labels_training, n_afib_train)

        if arguments.imbalanced_validating:
            ds_validate = setup.validating
        else:
            n_afib_validate = setup.validating.count(*af_labels_validate)
            ds_validate = setup.validating.balanced_binary_partition(af_labels_validate, n_afib_validate)


        experiment(
            ds_train,
            ds_validate,
            af_labels_training,
            af_labels_validate
        )

    finish_experiment(experiment.result)


if __name__ == "__main__":
    parser = ArgumentParser()

    args_add_c(parser)
    args_add_classifier(parser)
    args_add_bandwidth_rri(parser)
    args_add_rho(parser)
    args_add_setup(parser, "in")
    parser.add_argument("--imbalanced_validating", dest="imbalanced_validating", 
                        default=False, action="store_true",
                        help="Whether or not to validate on the full dataset.")

    args = parser.parse_args()
    main_logsize_experiment_rri(args)
