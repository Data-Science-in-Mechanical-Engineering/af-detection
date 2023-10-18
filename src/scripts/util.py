from __future__ import annotations

from abc import abstractmethod
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from itertools import product
from typing import Generic, TypeVar, Iterator

import numpy as np

from ..data.dataset import SPHDataset, COATDataset
from ..method.svm_classifier import SVMKMEClassifier, SVMMeanKernelClassifier, SVMVarianceClassifier, \
    SVMFeatureVectorClassifier
from ..results import Result

STANDARD_SPH_MINIMUM_RRIS = 7
STANDARD_COAT_MINIMUM_RRIS = 50

DatasetTrainingT = TypeVar("DatasetTrainingT", COATDataset, SPHDataset)
DatasetValidatingT = TypeVar("DatasetValidatingT", COATDataset, SPHDataset)
DatasetT = DatasetTrainingT | DatasetValidatingT


@dataclass(frozen=True)
class Setup(Generic[DatasetTrainingT, DatasetValidatingT]):
    training: DatasetTrainingT
    validating: DatasetValidatingT

    @staticmethod
    @abstractmethod
    def standard_preprocessing(dataset: DatasetT) -> DatasetT:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def _load_standard_training() -> DatasetT:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def _load_standard_validate() -> DatasetT:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def _load_standard_test() -> DatasetT:
        raise NotImplementedError

    @classmethod
    def from_standard_preprocessing(cls, training: DatasetT, validating: DatasetT) -> Setup:
        return Setup(
            cls.standard_preprocessing(training),
            cls.standard_preprocessing(validating)
        )

    @classmethod
    def standard_validate(cls) -> Setup:
        return cls.from_standard_preprocessing(
            cls._load_standard_training(),
            cls._load_standard_validate()
        )

    @classmethod
    def standard_test(cls) -> Setup:
        return cls.from_standard_preprocessing(
            cls._load_standard_training(),
            cls._load_standard_test()
        )


class SPHSetup(Setup[SPHDataset, SPHDataset]):
    @staticmethod
    def _load_standard_training() -> DatasetT:
        return SPHDataset.load_train()

    @staticmethod
    def _load_standard_validate() -> DatasetT:
        return SPHDataset.load_validate()

    @staticmethod
    def _load_standard_test() -> DatasetT:
        return SPHDataset.load_test()

    @staticmethod
    def standard_preprocessing(dataset: SPHDataset) -> SPHDataset:
        return dataset.filter(lambda entry: len(entry.qrs_complexes) > STANDARD_SPH_MINIMUM_RRIS)


class COATSetup(Setup[COATDataset, COATDataset]):
    @staticmethod
    def _load_standard_training() -> DatasetT:
        return COATDataset.load_train()

    @staticmethod
    def _load_standard_validate() -> DatasetT:
        return COATDataset.load_validate()

    @staticmethod
    def _load_standard_test() -> DatasetT:
        return COATDataset.load_test()

    @staticmethod
    def standard_preprocessing(dataset: COATDataset) -> COATDataset:
        return dataset.filter(lambda entry: len(entry.qrs_complexes) > STANDARD_COAT_MINIMUM_RRIS) \
            .filter(lambda entry: entry.label in {COATDataset.AFIB, COATDataset.noAFIB})


@dataclass(frozen=True)
class MultiDatabaseSetup:
    sph: SPHSetup
    coat: COATSetup

    @classmethod
    def standard_validate(cls) -> MultiDatabaseSetup:
        return cls(
            SPHSetup.standard_validate(),
            COATSetup.standard_validate()
        )

    @classmethod
    def standard_test(cls) -> MultiDatabaseSetup:
        return cls(
            SPHSetup.standard_test(),
            COATSetup.standard_test()
        )

    @abstractmethod
    def __iter__(self) -> Iterator[Setup]:
        raise NotImplementedError


@dataclass(frozen=True)
class InDatabaseSetup(MultiDatabaseSetup):
    def __iter__(self) -> Iterator[Setup]:
        yield self.sph
        yield self.coat


@dataclass(frozen=True)
class CrossDatabaseSetup(MultiDatabaseSetup):
    def __iter__(self) -> Iterator[Setup]:
        datasets_training = [self.sph.training, self.coat.training]
        datasets_validating = [self.sph.validating, self.coat.validating]

        for dataset_training, dataset_validating in product(datasets_training, datasets_validating):
            yield Setup(dataset_training, dataset_validating)


def args_add_c(parser: ArgumentParser):
    group = parser.add_argument_group("SVM C Parameter")

    group.add_argument("--c_lower", dest="c_lower", type=float, default=1, help="Log start value for c parameter.")
    group.add_argument("--c_upper", dest="c_upper", type=float, default=1, help="Log end value for c parameter.")
    group.add_argument("--c_steps", dest="c_steps", type=int, default=1, help="Number of steps for c parameter.")


def args_add_classifier(parser: ArgumentParser):
    parser.add_argument(
        "--classifier",
        dest="classifier",
        type=str,
        default="KME",
        choices=["KME", "MEAN", "VAR", "FV"],
        help="Type of classifier."
    )


def args_add_bandwidth_rri(parser: ArgumentParser):
    group = parser.add_argument_group("Bandwidth RRI")

    group.add_argument("--bandwidth_rri_lower", dest="bandwidth_rri_lower", type=float, default=0.05,
                       help="Linear start value for RRI bandwidth parameter.")
    group.add_argument("--bandwidth_rri_upper", dest="bandwidth_rri_upper", type=float, default=0.05,
                       help="Linear end value for RRI bandwidth parameter.")
    group.add_argument("--bandwidth_rri_steps", dest="bandwidth_rri_steps", type=int, default=1,
                       help="Number of steps for RRI bandwidth parameter.")
    group.add_argument("--bandwidth_rri_logspace", dest="bandwidth_rri_logspace", default=False,
                       action="store_true", help="Whether or not use a logspace for grid search.")
    group.add_argument("--bandwidth_rri_base", dest="bandwidth_rri_base", type=float, default=10.0,
                       help="If using a logspace, base of the log. Ignored otherwise.")


def args_add_rho(parser: ArgumentParser):
    group = parser.add_argument_group("Bandwidth RRI")

    group.add_argument("--rho_lower", dest="rho_lower", type=float, default=-1,
                       help="Log start value for the class weight proportion parameter.")

    group.add_argument("--rho_upper", dest="rho_upper", type=float, default=1,
                       help="Log end value for the class weight proportion parameter.")

    group.add_argument("--rho_steps", dest="rho_steps", type=int, default=20,
                       help="Number of steps for the class weight proportion parameter.")

    group.add_argument("--rho_base", dest="rho_base", type=float, default=5,
                       help="Base for the log scale of the class weight proportion parameter.")


def args_add_setup(parser: ArgumentParser, default: str):
    group = parser.add_argument_group("Database Setup")

    group.add_argument("--setup", dest="setup", type=str, default=default,
                       choices=["in", "cross", "sph", "diagnostick"], help="Database(s) to use.")

    group.add_argument("--test", dest="test", default=False, action="store_true",
                       help="Whether or not to evaluate on the test dataset.")


def args_parse_c(arguments: Namespace):
    return np.logspace(arguments.c_lower, arguments.c_upper, arguments.c_steps)


def args_parse_classifier(arguments: Namespace):
    if arguments.classifier == "KME":
        return SVMKMEClassifier
    elif arguments.classifier == "MEAN":
        return SVMMeanKernelClassifier
    elif arguments.classifier == "VAR":
        return SVMVarianceClassifier
    elif arguments.classifier == "FV":
        return SVMFeatureVectorClassifier
    else:
        raise AttributeError(f"Invalid classifier '{arguments.classifier}'.")


def args_parse_bandwidth_rri(arguments: Namespace):
    if arguments.bandwidth_rri_logspace:
        space = np.logspace
        kwargs = {"base": arguments.bandwidth_rri_base}
    else:
        space = np.linspace
        kwargs = {}
    return space(
        arguments.bandwidth_rri_lower, arguments.bandwidth_rri_upper, arguments.bandwidth_rri_steps, **kwargs
    )


def args_parse_rho(arguments: Namespace):
    return np.logspace(arguments.rho_lower, arguments.rho_upper, arguments.rho_steps, base=arguments.rho_base)


def args_parse_setup(arguments: Namespace) -> Iterator[Setup]:
    setup_cls = None

    def get_setup_instance(cls):
        return cls.standard_test() if arguments.test else cls.standard_validate()

    if arguments.setup == "in":
        setup_cls = InDatabaseSetup
    elif arguments.setup == "cross":
        setup_cls = CrossDatabaseSetup

    if setup_cls is not None:
        yield from get_setup_instance(setup_cls)
        return

    if arguments.setup == "sph":
        setup_cls = SPHSetup
    elif arguments.setup == "diagnostick":
        setup_cls = COATSetup
    else:
        raise AttributeError(f"Invalid setup '{arguments.setup}'.")

    yield from [get_setup_instance(setup_cls)]


def finish_experiment(result: Result):
    results_path = result.save()
    print("\n", f"Finished. Results saved to {results_path}.")
