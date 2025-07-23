import heapq
from statistics import mean, stdev
from typing import Literal

import pandas as pd
import tqdm
import tyro
from jax import Array
from typing_extensions import Callable

from src import expyro
from src.config import DatasetConfig
from src.data import DatasetAlias, PeakExtractionAlias, Dataset
from src.experiments.tuning.config import Config, Result
from src.experiments.tuning.config import DistributionalKernelMatrixParameters, Parametrization, CVIndexSplit
from src.expyro.experiment import Run as ExperimentRun
from src.metrics import Run, Evaluation
from src.util import DIR_RESULTS, move_experiment_run

type ExtendedDatasetAlias = DatasetAlias | Literal["sph-r", "cinc-r"]

K_FOLD_CV = 5


def _tuning_step(
        dataset: Dataset, kernel_matrix: Array, parametrization: Parametrization, splits: list[CVIndexSplit],
        positive_labels: list[str],
) -> list[Run]:
    runs = []

    for split in splits:
        dataset_train = dataset[split.train]
        dataset_evaluation = dataset[split.validation]

        kernel_matrix_train = kernel_matrix[split.train][:, split.train]
        kernel_matrix_evaluation = kernel_matrix[split.validation][:, split.train]

        run = Run.svm(
            kernel_matrix_train=kernel_matrix_train,
            kernel_matrix_evaluation=kernel_matrix_evaluation,
            dataset_train=dataset_train,
            dataset_evaluation=dataset_evaluation,
            positive_labels_train=positive_labels, positive_labels_evaluation=positive_labels,
            c=parametrization.c, rho=parametrization.rho
        )

        runs.append(run)

    return runs


@expyro.experiment(DIR_RESULTS, name="tuning")
def experiment(config: Config) -> Result:
    rng = config.rng()

    dataset = config.build_training_dataset()
    splits = config.cv_indices(dataset, next(rng))

    performance = {}
    best_runs = None

    with tqdm.tqdm(total=config.n_parametrizations, smoothing=0) as pbar:
        for parametrization, kernel_matrix in config.iter(dataset):
            runs = performance[parametrization] = _tuning_step(
                dataset=dataset,
                kernel_matrix=kernel_matrix,
                parametrization=parametrization,
                splits=splits,
                positive_labels=config.dataset.positive_labels
            )

            if best_runs is None or config.performance(runs) > config.performance(best_runs):
                best_runs = runs
                print("\n", parametrization)

                pbar.set_postfix_str(
                    f"F1: [{100 * min(run.evaluation_validation.f1 for run in best_runs):.2f}% "
                    f"{100 * max(run.evaluation_validation.f1 for run in best_runs):.2f}%] "
                    f"([{100 * min(run.evaluation_train.f1 for run in best_runs):.2f}% "
                    f"{100 * max(run.evaluation_train.f1 for run in best_runs):.2f}%]) "
                    f"AUC: [{100 * min(run.evaluation_validation.roc_auc for run in best_runs):.2f}% "
                    f"{100 * max(run.evaluation_validation.roc_auc for run in best_runs):.2f}%] "
                    f"([{100 * min(run.evaluation_train.roc_auc for run in best_runs):.2f}% "
                    f"{100 * max(run.evaluation_train.roc_auc for run in best_runs):.2f}%]) "
                    f"Acc: [{100 * min(run.evaluation_validation.accuracy for run in best_runs):.2f}% "
                    f"{100 * max(run.evaluation_validation.accuracy for run in best_runs):.2f}%] "
                    f"([{100 * min(run.evaluation_train.accuracy for run in best_runs):.2f}% "
                    f"{100 * max(run.evaluation_train.accuracy for run in best_runs):.2f}%]) "
                    f"Sen: [{100 * min(run.evaluation_validation.sensitivity for run in best_runs):.2f}% "
                    f"{100 * max(run.evaluation_validation.sensitivity for run in best_runs):.2f}%] "
                    f"([{100 * min(run.evaluation_train.sensitivity for run in best_runs):.2f}% "
                    f"{100 * max(run.evaluation_train.sensitivity for run in best_runs):.2f}%]) "
                    f"Spec: [{100 * min(run.evaluation_validation.specificity for run in best_runs):.2f}% "
                    f"{100 * max(run.evaluation_validation.specificity for run in best_runs):.2f}%] "
                    f"([{100 * min(run.evaluation_train.specificity for run in best_runs):.2f}% "
                    f"{100 * max(run.evaluation_train.specificity for run in best_runs):.2f}%]) "
                    f"Prec: [{100 * min(run.evaluation_validation.precision for run in best_runs):.2f}% "
                    f"{100 * max(run.evaluation_validation.precision for run in best_runs):.2f}%] "
                    f"([{100 * min(run.evaluation_train.precision for run in best_runs):.2f}% "
                    f"{100 * max(run.evaluation_train.precision for run in best_runs):.2f}%])"
                )

            pbar.update()

    if config.top_k_results is not None:
        best_parametrizations = heapq.nlargest(
            n=config.top_k_results,
            iterable=performance.keys(),
            key=lambda p: config.performance(performance[p])
        )

        performance = {
            parametrization: performance[parametrization]
            for parametrization in best_parametrizations
        }

    return performance


def main(
        dataset_name: ExtendedDatasetAlias,
        peak_extraction: PeakExtractionAlias,
        subsample: int | None,
        seed: int = 0,
        cache_only: bool = False,
):
    cs = [0.05, 0.1, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 5]

    kernel_matrices = DistributionalKernelMatrixParameters(
        bandwidth_base=[0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2],
        bandwidth_mmd=[0.1, 0.25, 0.5, 0.75, 1.0],
        cached=cache_only or subsample is None
    )

    datasets = {
        "coat": DatasetConfig.coat(peak_extraction=peak_extraction, subsample=subsample),
        "sph": DatasetConfig.sph(peak_extraction=peak_extraction, subsample=subsample),
        "cinc": DatasetConfig.cinc(peak_extraction=peak_extraction, subsample=subsample),
        "sph-r": DatasetConfig.sph_r(peak_extraction=peak_extraction, subsample=subsample),
        "cinc-r": DatasetConfig.cinc_r(peak_extraction=peak_extraction, subsample=subsample)
    }

    assert dataset_name in datasets, f"Unknown dataset name {dataset_name}"
    dataset = datasets[dataset_name]

    config = Config(
        seed=seed,
        dataset=dataset,
        kernel_matrix=kernel_matrices,
        c=cs,
        rho=[0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5],
        k_fold_cv=K_FOLD_CV,
        criterion="roc-auc",
        top_k_results=10
    )

    if cache_only:
        dataset = config.build_training_dataset()
        list(tqdm.tqdm(config.iter_kernel_matrices(dataset), total=len(config.kernel_matrix)))
        return

    if subsample is None:
        sub_dir = f"distributional/{dataset_name}/{peak_extraction}"
    else:
        sub_dir = f"distributional_sized/{dataset_name}/{peak_extraction}/n={subsample}"

    run_name = f"seed-{seed:03d}"

    target_dir = experiment.directory / experiment.name / sub_dir / run_name

    if target_dir.exists():
        print(f"Skipping {target_dir} because it already exists.")
    else:
        run = experiment(config)
        summarize_single_run(run.location.name)
        move_experiment_run(run, sub_dir=sub_dir, dir_name=run_name)


def format_multi_metric(runs: list[Run], metric: Callable[[Evaluation], float]) -> str:
    metrics = [metric(run_.evaluation_validation) for run_ in runs]
    return f"{100 * mean(metrics):.2f}% (±{100 * stdev(metrics):.2f}%)"


def summarize_single_run(identifier: str):
    run = experiment[identifier]
    config: Config = run.config
    result: Result = run.result

    parameters = list(sorted(
        result.keys(),
        key=lambda p: (
            config.performance(result[p]),
            mean(x.evaluation_validation.roc_auc for x in result[p]),
            mean(x.evaluation_validation.f1 for x in result[p]),
            mean(x.evaluation_validation.accuracy for x in result[p]),
            mean(x.evaluation_validation.sensitivity for x in result[p]),
        ),
        reverse=True
    ))

    data = []

    columns_kernel_matrix_parameters = [key for key in parameters[0].kernel_matrix.keys()]

    for parametrization in parameters:
        entries = [parametrization.kernel_matrix[key] for key in columns_kernel_matrix_parameters]

        entries.extend((
            parametrization.c,
            parametrization.rho,
            format_multi_metric(result[parametrization], lambda evaluation: evaluation.roc_auc),
            format_multi_metric(result[parametrization], lambda evaluation: evaluation.f1),
            format_multi_metric(result[parametrization], lambda evaluation: evaluation.accuracy),
            format_multi_metric(result[parametrization], lambda evaluation: evaluation.sensitivity),
            format_multi_metric(result[parametrization], lambda evaluation: evaluation.specificity),
            format_multi_metric(result[parametrization], lambda evaluation: evaluation.precision)
        ))

        data.append(entries)

    df = pd.DataFrame.from_records(
        data=data,
        columns=columns_kernel_matrix_parameters + [
            "C",
            "Rho",
            "AUC",
            "F1-score",
            "Accuracy",
            "Sensitivity",
            "Specificity",
            "Precision"
        ]
    )

    df.to_excel(run.location / f"summary.xlsx", index=False)


def get_best_parametrization(run: ExperimentRun[Config, Result]) -> Parametrization:
    best_parametrization = None
    best_score = -1

    for parametrization, cv_results in run.result.items():
        score = run.config.performance(cv_results)

        if best_parametrization is None or score > best_score:
            best_score = score
            best_parametrization = parametrization

    if best_parametrization is None:
        raise ValueError("No parametrization found")

    return best_parametrization


if __name__ == "__main__":
    tyro.cli(main)
