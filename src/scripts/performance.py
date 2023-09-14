from argparse import Namespace, ArgumentParser

from src.experiments import ExperimentRRI
from src.scripts.util import args_parse_c, args_parse_classifier, args_parse_bandwidth_rri, args_parse_setup, \
    args_add_c, args_add_classifier, args_add_bandwidth_rri, args_add_setup, finish_experiment


def main_performance_experiment_rri(arguments: Namespace):
    cs = args_parse_c(arguments)
    classifier = args_parse_classifier(arguments)
    bandwidths_rri = args_parse_bandwidth_rri(arguments)
    setups = args_parse_setup(arguments)

    experiment = ExperimentRRI(cs, [1], classifier, bandwidths_rri)

    for setup in setups:
        print("\n", f"Training: {setup.training}", "\n", f"Validation: {setup.validating}", "\n")

        af_labels_train = {setup.training.AFIB}
        af_labels_validate = {setup.validating.AFIB}

        experiment(
            setup.training.balanced_binary_partition(af_labels_train, setup.training.count(*af_labels_train)),
            setup.validating.balanced_binary_partition(af_labels_validate, setup.validating.count(*af_labels_validate)),
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

    args = parser.parse_args()
    main_performance_experiment_rri(args)
