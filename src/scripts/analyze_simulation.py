import numpy as np
import matplotlib.pyplot as plt
import json
from collections import OrderedDict
import dataclasses
from argparse import ArgumentParser
from pathlib import Path

from src.results import RESULTS_FOLDER, Result, Snapshot, Outcome
from src.experiments import ParametrizationRRI



def parametrization_as_dict(parametrization: ParametrizationRRI) -> dict[str]:
    parameter_names = list(map(lambda f:f.name,dataclasses.fields(ParametrizationRRI)))
    return {pname: getattr(parametrization, pname) for pname in parameter_names}


def find_closest_parameter(snapshot, pname, ptarget):
    parameter_values = np.array(list(map(
        lambda outcome: outcome.parametrization[pname],
        snapshot
    )))
    closest_index = np.argmin(np.abs(parameter_values - ptarget))
    return parameter_values[closest_index]


def find_optimal_parameters(dataset_snapshot: Snapshot, parameter_names: list[str], metric: str, override_parametrization: ParametrizationRRI) -> ParametrizationRRI:
    for oname, override in parametrization_as_dict(override_parametrization).items():
        if override is not None:
            true_override = find_closest_parameter(dataset_snapshot, oname, float(override))
            dataset_snapshot = dataset_snapshot.filter(
                lambda outcome: outcome.parametrization[oname] == true_override
            )
    partition_by_parameters = dataset_snapshot.partition(
            lambda outcome: tuple(outcome.parametrization[pname] for pname in parameter_names)
        )
    for parameter_tuple, snapshot in partition_by_parameters.items():
        train_name = snapshot.setup['dataset_train']['name']
        validate_name = snapshot.setup['dataset_validate']['name']
        assert (n:=len(snapshot.outcomes)) == 1, f"Snapshot for configuration {train_name}, {validate_name}, {parameter_tuple} has more than one outcome (found {n})."

    optimal_parameters = ParametrizationRRI(**dict(zip(parameter_names, max(
        partition_by_parameters,
        key=lambda k: partition_by_parameters[k].outcomes[0].as_dict()['scores'][metric]
    ))))
    return optimal_parameters


def extract_parameter_sensitivity(
        dataset_snapshot: Snapshot, pname: str, optimal_parameters: ParametrizationRRI
    ) -> OrderedDict[ParametrizationRRI, Outcome]:
    filtering_parameters = parametrization_as_dict(optimal_parameters)
    filtering_parameters.pop(pname)
    parameter_partition = dataset_snapshot.filter(
        lambda outcome: all(
            outcome.parametrization[filter_pname] == parameter for filter_pname, parameter in filtering_parameters.items()
        )
    ).partition(lambda outcome: outcome.parametrization[pname])

    for parameter_tuple, snapshot in parameter_partition.items():
        assert (n:=len(snapshot.outcomes)) == 1, f"Snapshot for configuration {filtering_parameters}, {parameter_tuple} has more than one outcome (found {n})."
    parameter_outcomes = {
        parameter: snapshot.outcomes[0] 
        for parameter, snapshot in parameter_partition.items()
    }
    
    sorted_parameters = sorted(list(parameter_outcomes.keys()))
    sorted_outcomes = OrderedDict([
        (ParametrizationRRI(**{pname: parameter}, **filtering_parameters), parameter_outcomes[parameter])
        for parameter in sorted_parameters
    ])
    return sorted_outcomes


def plot_parameter_sensitivities(
        sensitivities: dict[str, OrderedDict[ParametrizationRRI, Outcome]], 
        excluded_parameters: list[str], save_dir: Path, dataset_key: tuple[str]):
    n_included_params = len(sensitivities) - len(excluded_parameters)
    assert n_included_params >= 0
    if n_included_params == 0:
        return None  # nothing to do

    fig, axes = plt.subplots(nrows=n_included_params, ncols=1)
    n_ax = 0
    for pname in sensitivities.keys():
        if pname in excluded_parameters:
            continue
        ax = axes[n_ax]
        sorted_outcomes = sensitivities[pname]
        parameter_values = [getattr(parametrization, pname) for parametrization, _ in sorted_outcomes.items()]
        scores_names = list(list(sorted_outcomes.values())[0].as_dict()['scores'].keys())
        scores = np.array([
            tuple(outcome.as_dict()['scores'][sname] for sname in scores_names) 
            for _, outcome in sorted_outcomes.items()
        ]).T
        for i, sname in enumerate(scores_names):
            ax.plot(parameter_values, scores[i, :], label=sname)
        ax.set_title(f"Influence of {pname}")
        ax.set_xlabel(pname)
        ax.legend()
        ax.set_xscale('log')
        n_ax += 1
    figname = f'{dataset_key[0].split("Dataset")[0]}_{dataset_key[1].split("Dataset")[0]}.pdf'
    fig.savefig(save_dir / figname, dpi=300)


def main(results: Result, metric: str, save_dir: Path, override_parametrization: ParametrizationRRI):
    parameter_names = list(map(lambda f:f.name,dataclasses.fields(ParametrizationRRI)))
    partitions_by_dataset = results.partition_by_datasets()
    optimal_parameters = {}
    for train_name, validate_name in partitions_by_dataset.keys():
        dataset_key = train_name, validate_name
        dataset_results = results.filter(
            lambda snapshot: (
                snapshot.setup["dataset_train"]["name"] == train_name
            ) and (
                snapshot.setup["dataset_validate"]["name"] == validate_name
            )
        )
        assert len(dataset_results.snapshots) == 1, 'Data set configuration run multiple times'
        dataset_snapshot = dataset_results.snapshots[0]
        optimal_parameters[dataset_key] = find_optimal_parameters(dataset_snapshot, parameter_names, metric, override_parametrization)
        sensitivities = {}
        excluded_parameters = []
        for pname in parameter_names:
            sensitivities[pname] = extract_parameter_sensitivity(
                dataset_snapshot, pname, optimal_parameters[dataset_key]
            )
            if len(sensitivities[pname]) <= 1:
                excluded_parameters.append(pname)
        plot_parameter_sensitivities(sensitivities, excluded_parameters, save_dir, dataset_key)

    serialized_optimal_parameters = {
        str(k): parametrization_as_dict(v) for k,v in optimal_parameters.items()
    }
    with (save_dir / 'optimal_parameters.json').open('w') as f:
        json.dump(serialized_optimal_parameters, f, indent=4)


def add_override_args(parser):
    for arg in ['c', 'bw', 'rho']:
        parser.add_argument(
            f'--{arg}', dest=arg, default=None,
            help=f'Specify to override the optimization for the parameter {arg}. Will pick the available closest to the specification.')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--f', dest='f', help='JSON file with results to analyze', type=str)
    parser.add_argument('--metric', dest='metric', help='Metric to use when computing the optimal parameters')
    add_override_args(parser)
    args = parser.parse_args()

    results_file = Path(RESULTS_FOLDER / args.f)
    save_dir = results_file.parent / results_file.stem
    save_dir.mkdir(exist_ok=True, parents=False)
    results = Result.from_json(results_file)
    override_parametrization = ParametrizationRRI(args.c, args.rho, args.bw)
    main(results, args.metric, save_dir, override_parametrization)
