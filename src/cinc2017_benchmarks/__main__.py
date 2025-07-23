import csv
import getpass
import os
import shutil
import subprocess
import tarfile
import zipfile
from pathlib import Path
from statistics import mean
from typing import NamedTuple

import numpy as np
import pandas as pd
import tyro
import wfdb
from dotenv import load_dotenv
from wfdb.io.convert import wfdb_to_mat

from src import expyro
from src.cinc2017_benchmarks.config import Config, Result
from src.config import DatasetConfig
from src.data import DIR_ROOT, DatasetAlias
from src.expyro.experiment import Run
from src.util import download_file, move_experiment_run, DIR_RESULTS, generate_random_keys

load_dotenv()


def _is_empty(path: Path) -> bool:
    return not any(True for _ in path.iterdir())


DIR_CONTAINERS = DIR_ROOT / "cinc2017"
NAME_PREPARE_ENTRY_SH = "prepare-entry.sh"
PATH_PREPARE_ENTRY_SH = DIR_CONTAINERS / NAME_PREPARE_ENTRY_SH
NAME_CONTAINER_DEFINITION = "container.def"
NAME_CONTAINER_IMAGE = "container.sif"

DIR_DATA_PREPROCESSED = DIR_CONTAINERS / "data"


class EntryEvaluation(NamedTuple):
    entry_id: str
    dataset_name: str

    @property
    def dir(self) -> Path:
        return DIR_CONTAINERS / self.entry_id

    @property
    def dir_workspace(self) -> Path:
        return self.dir / self.dataset_name / "workspace"

    @property
    def dir_code(self) -> Path:
        return self.dir_workspace / "code"

    @property
    def dir_data(self) -> Path:
        return self.dir_code / "validation"

    @property
    def path_records(self) -> Path:
        return self.dir_data / "RECORDS"

    @property
    def path_container_definition(self) -> Path:
        return self.dir / NAME_CONTAINER_DEFINITION

    @property
    def path_container_image(self) -> Path:
        return self.dir / NAME_CONTAINER_IMAGE

    @property
    def path_prepare_entry_sh(self) -> Path:
        return self.dir_code / NAME_PREPARE_ENTRY_SH

    @property
    def path_identification_csv(self) -> Path:
        return DIR_DATA_PREPROCESSED / f"{self.dataset_name}.csv"

    @property
    def path_answers(self) -> Path:
        return self.dir_code / "answers.txt"

    @property
    def path_url(self) -> Path:
        return self.dir / "url.txt"

    def get_url(self) -> str:
        with open(self.path_url, 'r') as f:
            return f.read().strip()

    def _build_code(self):
        assert not self.dir_code.exists() or _is_empty(self.dir_code)

        self.dir_code.mkdir(exist_ok=True, parents=True)

        entry_url = self.get_url()
        archive_extension = entry_url.split(".")[-1]
        path_archive = self.dir_code / f"{self.entry_id}.{archive_extension}"

        download_file(entry_url, path_archive)

        if zipfile.is_zipfile(path_archive):
            with zipfile.ZipFile(path_archive, 'r') as zf:
                zf.extractall(self.dir_code)
        elif tarfile.is_tarfile(path_archive):
            with tarfile.open(path_archive, "r:*") as tf:
                tf.extractall(self.dir_code)
        else:
            raise ValueError(f"Unsupported archive format: {path_archive}")

        if self.dir_data.exists():
            shutil.rmtree(self.dir_data)

        if self.path_answers.exists():
            self.path_answers.unlink()

        path_archive.unlink()

    def _build_data(self):
        shutil.rmtree(self.dir_data, ignore_errors=True)
        dir_data = DIR_DATA_PREPROCESSED / self.dataset_name

        shutil.copytree(dir_data, self.dir_data)
        shutil.copy(dir_data / "RECORDS", self.path_records)

    def _write_records(self, names: list[str]):
        with open(self.path_records, 'w') as f:
            f.writelines([
                f"{name}\n" if i < len(names) - 1 else f"{name}"
                for i, name in enumerate(names)
            ])

    def build_directory_structure(self):
        assert self.dir.exists()
        assert self.path_url.exists()
        assert self.path_container_definition.exists()

        self.dir_workspace.mkdir(parents=True, exist_ok=True)

        if not self.dir_code.exists() or _is_empty(self.dir_code):
            self._build_code()

        assert self.dir_code.exists() and not _is_empty(self.dir_code)

        shutil.copy(PATH_PREPARE_ENTRY_SH, self.path_prepare_entry_sh)

        if not self.dir_data.exists() or _is_empty(self.dir_data) or not self.path_records.exists():
            self._build_data()

        assert self.dir_data.exists() and not _is_empty(self.dir_data)
        assert self.path_identification_csv.exists()
        assert self.path_records.exists()

        if self.path_answers.exists():
            shutil.copy(self.path_answers, self.path_answers.with_suffix(".backup"))

    def update_records(self):
        if not self.path_answers.exists():
            return

        answers = self.answers()
        identification = self.identification()
        records = self.records()

        unprocessed_records = {
            record for record in records
            if identification[record] not in answers
        }

        processed_records = {
            record for record in records
            if identification[record] in answers
        }

        for name in processed_records:
            (self.dir_data / f"{name}.hea").unlink(missing_ok=True)
            (self.dir_data / f"{name}.mat").unlink(missing_ok=True)

        self._write_records(list(sorted(unprocessed_records)))

    def build_container_image(self):
        if self.path_container_image.exists():
            return

        subprocess.run([
            "apptainer",
            "build",
            str(self.path_container_image),
            str(self.path_container_definition)
        ], cwd=self.dir)

    def identification(self) -> dict[str, str]:
        assert self.path_identification_csv.exists()
        identification = {}

        with open(self.path_identification_csv, 'r') as f:
            lines = f.readlines()
            lines = [line.strip() for line in lines]
            lines = [line for line in lines if len(line) > 0]

        for line in lines:
            name, identifier = line.split(",")
            identification[name] = identifier

        return identification

    def answers(self) -> dict[str, str]:
        assert self.path_answers.exists()
        identification = self.identification()
        answers = {}

        with open(self.path_answers, 'r') as f:
            lines = f.readlines()
            lines = [line.strip() for line in lines]
            lines = [line for line in lines if len(line) > 0]

        for line in lines:
            name, answer = line.split(",")
            answers[identification[name]] = answer

        return answers

    def records(self) -> list[str]:
        with open(self.path_records, 'r') as f:
            lines = f.readlines()

        lines = [line.strip() for line in lines]
        return [line for line in lines if len(lines) > 0]


def _get_path_matlab() -> Path:
    matlab_path = shutil.which("matlab")

    if matlab_path is None:
        raise RuntimeError("MATLAB is not in PATH")

    return Path(matlab_path)


def _build_fake_home(entry_id: str, dataset_name: str) -> Path:
    user = getpass.getuser()
    fake_home = Path(f"/tmp/fake-home") / entry_id / dataset_name / user
    fake_home.mkdir(exist_ok=True, parents=True)
    return fake_home


@expyro.experiment(DIR_RESULTS, name="cinc2017_benchmarks")
def experiment(config: Config) -> Result:
    evaluation = EntryEvaluation(config.entry_id, config.dataset.name)
    evaluation.build_directory_structure()
    evaluation.update_records()

    evaluation.build_container_image()

    path_fake_home = _build_fake_home(config.entry_id, config.dataset.name)
    assert path_fake_home.exists()

    path_matlab = _get_path_matlab()
    matlab_root = os.path.dirname(os.path.dirname(path_matlab))

    os.environ["MATLABROOT"] = matlab_root
    os.environ["MLM_LICENSE_SERVER"] = os.getenv("MLM_LICENSE_SERVER")
    os.environ["MATLAB_PREFDIR"] = "/tmp/my-matlab-prefs"

    if os.environ["MLM_LICENSE_SERVER"] is None:
        raise RuntimeError("MLM_LICENSE_SERVER is not set")

    os.makedirs(os.environ["MATLAB_PREFDIR"], exist_ok=True)

    user = getpass.getuser()

    subprocess.run([
        "apptainer",
        "run",
        "--no-home", "--contain", "--writable-tmpfs", "--cleanenv",
        "--pwd", "/workspace",
        f"--home", f"{path_fake_home}:/home/{user}",
        f"--bind", f"{path_fake_home}:/home/{user}",
        "--bind", f"{evaluation.dir_workspace}:/workspace",
        "--bind", "/cvmfs:/cvmfs:ro",
        "--env", f"PATH={path_matlab}/bin:{os.environ['PATH']}",
        "--env", f"MLM_LICENSE_FILE={os.environ['MLM_LICENSE_SERVER']}",
        "--env", f"MATLAB_PREFDIR={os.environ['MATLAB_PREFDIR']}",
        "--env", f"LD_LIBRARY_PATH=$MATLABROOT/bin",
        str(evaluation.path_container_image),
    ], cwd=evaluation.dir)

    return Result(
        predictions=evaluation.answers()
    )


def main(entry_id: str, dataset_name: DatasetAlias):
    if dataset_name == "coat":
        dataset = DatasetConfig.coat()
    elif dataset_name == "sph":
        dataset = DatasetConfig.sph()
    elif dataset_name == "cinc":
        dataset = DatasetConfig.cinc()
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    preprocess_data(dataset_name)

    config = Config(
        seed=0,
        dataset=dataset,
        entry_id=entry_id
    )

    run = experiment(config)
    summarize(run)
    move_experiment_run(run, sub_dir=dataset_name, dir_name=f"{entry_id}")


def preprocess_data(dataset_name: DatasetAlias):
    rng = generate_random_keys(seed=0)

    if dataset_name == "coat":
        dataset = DatasetConfig.coat()
    elif dataset_name == "sph":
        dataset = DatasetConfig.sph()
    elif dataset_name == "cinc":
        dataset = DatasetConfig.cinc()
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    dir_data = DIR_DATA_PREPROCESSED / dataset.name
    path_records = dir_data / "RECORDS"
    dir_data.mkdir(exist_ok=True, parents=True)

    if dir_data.exists():
        return

    dataset = dataset.test(next(rng)).permute(next(rng))

    identification = {}

    for i, (identifier, ecg) in enumerate(zip(dataset.identifiers, dataset.ecgs, strict=True)):
        name = f"A{i + 1:05d}"
        identification[name] = identifier

        wfdb.wrsamp(
            record_name=name,
            fs=dataset.frequency,
            units=["mV"],
            sig_name=["ECG"],
            p_signal=np.array(ecg).reshape(-1, 1),
            fmt=["16"],
            write_dir=dir_data
        )

        wfdb_to_mat((dir_data / name).as_posix())

        shutil.move(f"./{name}m.mat", dir_data / f"{name}.mat")
        shutil.move(f"./{name}m.hea", dir_data / f"{name}.hea")

        (dir_data / f"{name}.dat").unlink()

    names = [name for name in identification.keys()]

    with open(path_records, 'w') as f:
        f.writelines([
            f"{name}\n" if i < len(names) - 1 else f"{name}"
            for i, name in enumerate(names)
        ])

    with open(DIR_DATA_PREPROCESSED / f"{dataset_name}.csv", 'w') as f:
        csv.writer(f).writerows(identification.items())


def summarize(run: Run[Config, Result]):
    dataset = run.config.build_dataset()
    evaluation = run.result.evaluate(dataset, run.config.dataset.positive_labels)

    frac_noisy = mean(
        int(prediction == '~')
        for prediction in run.result.predictions.values()
    )

    df = pd.DataFrame.from_records(
        data=[(
            f"{evaluation.f1 * 100:.2f}%",
            f"{evaluation.accuracy * 100:.2f}%",
            f"{evaluation.sensitivity * 100:.2f}%",
            f"{evaluation.specificity * 100:.2f}%",
            f"{evaluation.precision * 100:.2f}%",
            f"{frac_noisy * 100:.2f}%",
        )],
        columns=[
            "F1-score",
            "Accuracy",
            "Sensitivity",
            "Specificity",
            "Precision",
            "Classified as noise"
        ]
    )

    df.to_excel(run.location / f"summary.xlsx", index=False)


if __name__ == "__main__":
    tyro.cli(main)
