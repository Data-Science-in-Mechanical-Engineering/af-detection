from multiprocessing import Pool

import pandas as pd
import tqdm
import tyro
from typing import Literal, NamedTuple
from sklearn.metrics import roc_curve

from src import expyro
from src.config import DatasetConfig, DistributionalKernelMatrixConfig
from src.data import DatasetAlias
from src.experiments.evaluation.config import Config, Result, SingleEvaluation, RocCurve
from src.experiments.evaluation.plots import plot_confusion_matrices_evaluation
from src.experiments.tuning.__main__ import experiment as tuning, get_best_parametrization
from src.features import features
from src.metrics import Run
from src.util import DIR_RESULTS, set_plot_style, move_experiment_run


@expyro.plot(plot_confusion_matrices_evaluation, file_format="png")
@expyro.experiment(DIR_RESULTS, name="evaluation")
def experiment(config: Config) -> Result:
    rng = config.rng()

    tuning_run = tuning[config.tuning_run]
    parametrization = get_best_parametrization(tuning_run)

    kernel_matrix = DistributionalKernelMatrixConfig(
        bandwidth_base=parametrization.kernel_matrix["bandwidth_base"],
        bandwidth_mmd=parametrization.kernel_matrix["bandwidth_mmd"],
        cached=config.cache
    )

    dataset_train = tuning_run.config.build_training_dataset()
    xs_train, mask_train = features(dataset_train.peak_indices, dataset_train.mask)
    kernel_matrix_train = kernel_matrix(xs_train, xs_train, mask_train, mask_train)

    evaluations = {}

    for name_evaluation, evaluation_config in config.evaluation.items():
        dataset_evaluation = evaluation_config.test(next(rng))
        xs_evaluation, mask_evaluation = features(dataset_evaluation.peak_indices, dataset_evaluation.mask)
        kernel_matrix_evaluation = kernel_matrix(xs_evaluation, xs_train, mask_evaluation, mask_train)

        run, _, y_score = Run.svm(
            kernel_matrix_train=kernel_matrix_train,
            kernel_matrix_evaluation=kernel_matrix_evaluation,
            dataset_train=dataset_train,
            dataset_evaluation=dataset_evaluation,
            positive_labels_train=tuning_run.config.dataset.positive_labels,
            positive_labels_evaluation=evaluation_config.positive_labels,
            c=parametrization.c, rho=parametrization.rho,
            return_scores=True
        )

        labels_evaluation = dataset_evaluation.binarize_labels(evaluation_config.positive_labels)
        fpr, tpr, thresholds = roc_curve(labels_evaluation, y_score)

        evaluations[name_evaluation] = SingleEvaluation(
            run=run,
            roc_curve=RocCurve(fpr=fpr, tpr=tpr, thresholds=thresholds)
        )

    return evaluations


def main(tuning_sub_dir: str, tuning_seed: int, evaluation: DatasetAlias | Literal["all"], cache: bool = False):
    datasets = {
        "coat": DatasetConfig.coat(),
        "sph": DatasetConfig.sph(),
        "sph-r": DatasetConfig.sph_r(),
        "cinc": DatasetConfig.cinc(),
        "cinc-r": DatasetConfig.cinc_r()
    }

    if evaluation == "all":
        evaluation_datasets = datasets
    else:
        assert evaluation in datasets
        evaluation_datasets = {evaluation: datasets[evaluation]}

    run_name = f"seed-{tuning_seed:03d}"

    target_dir = experiment.directory / experiment.name / tuning_sub_dir / run_name

    if target_dir.exists():
        print(f"Skipping {target_dir} because it already exists.")
        return

    config = Config(
        seed=0,
        tuning_run=f"{tuning_sub_dir}/{run_name}",
        evaluation=evaluation_datasets,
        cache=cache,
    )

    run = experiment(config)
    summarize(run.location.name)
    move_experiment_run(run, sub_dir=tuning_sub_dir, dir_name=run_name)


def summarize(identifier: str):
    run = experiment[identifier]
    result: Result = run.result

    data = []

    for dataset_evaluation in result:
        evaluation = result[dataset_evaluation].run.evaluation_validation

        data.append((
            dataset_evaluation,
            f"{100 * evaluation.f1:.2f}%",
            f"{100 * evaluation.roc_auc:.2f}%",
            f"{100 * evaluation.accuracy:.2f}%",
            f"{100 * evaluation.sensitivity:.2f}%",
            f"{100 * evaluation.specificity:.2f}%",
            f"{100 * evaluation.precision:.2f}%",
        ))

    data = sorted(data, key=lambda x: x[1])

    df = pd.DataFrame.from_records(
        data=data,
        columns=["Validation dataset", "F1-score", "ROC AUC", "Accuracy", "Sensitivity", "Specificity", "Precision"]
    )

    df.to_excel(run.location / "summary.xlsx", index=False)


if __name__ == "__main__":
    set_plot_style()
    tyro.cli(main)
