from argparse import Namespace, ArgumentParser

from .util import args_add_c, args_add_classifier, args_add_bandwidth_rri, args_add_setup, args_add_rho, args_parse_c, args_parse_rho, args_parse_classifier, \
    args_parse_bandwidth_rri, args_parse_setup, finish_experiment
from ..experiments import ExperimentRRI


def main_logsize_experiment_rri(arguments: Namespace):
    cs = args_parse_c(arguments)
    c_class_weight_proportions = args_parse_rho(arguments)
    classifier = args_parse_classifier(arguments)
    bandwidths = args_parse_bandwidth_rri(arguments)
    setups = args_parse_setup(arguments)

    experiment = ExperimentRRI(cs, c_class_weight_proportions, classifier, bandwidths)

    for setup in setups:
        af_labels_training = {setup.training.AFIB}
        af_labels_validate = {setup.validating.AFIB}

        experiment(
            setup.training.balanced_binary_partition(af_labels_training, setup.training.count(*af_labels_validate)),
            setup.validating.balanced_binary_partition(af_labels_validate, setup.validating.count(*af_labels_validate)),
            {setup.training.AFIB},
            {setup.validating.AFIB}
        )

    finish_experiment(experiment.result)


if __name__ == "__main__":
    parser = ArgumentParser()

    args_add_c(parser)
    args_add_classifier(parser)
    args_add_bandwidth_rri(parser)
    args_add_rho(parser)
    args_add_setup(parser, "in")

    args = parser.parse_args()
    main_logsize_experiment_rri(args)
