from pathlib import Path
from typing import Generator, Literal, Hashable, Iterable, Sized

import jax
import matplotlib
import requests
import seaborn as sns
import tqdm
from jax import Array

from src.expyro.experiment import Run

type RNG = Generator[Array, None, None]

DIR_RESULTS = Path(__file__).parent.parent / "results"
DIR_RESULTS.mkdir(exist_ok=True, parents=True)

DIR_FIGURES = Path(__file__).parent.parent / "figures"
DIR_FIGURES.mkdir(exist_ok=True, parents=True)


def generate_random_keys(seed: int) -> RNG:
    key = jax.random.PRNGKey(seed)

    while True:
        key, subkey = jax.random.split(key)
        yield subkey


FIGURE_FULL_WIDTH = 6.85  # in
TEXT_FONT_SIZE = 10
COLOR_GRID = "#cfcfcf"


def set_plot_style(grid: Literal["white", "dark", "whitegrid", "darkgrid", "ticks"] = "whitegrid"):
    sns.set_style(grid)
    matplotlib.rcParams.update({"font.size": TEXT_FONT_SIZE})
    matplotlib.rcParams["mathtext.fontset"] = "stix"
    matplotlib.rcParams["font.family"] = "STIXGeneral"
    matplotlib.rcParams["grid.color"] = COLOR_GRID
    matplotlib.rcParams["axes.edgecolor"] = COLOR_GRID
    matplotlib.rcParams["grid.linewidth"] = 0.5
    matplotlib.rcParams["xtick.color"] = COLOR_GRID
    matplotlib.rcParams["ytick.color"] = COLOR_GRID
    matplotlib.rcParams["xtick.labelcolor"] = "black"
    matplotlib.rcParams["ytick.labelcolor"] = "black"
    matplotlib.rcParams["lines.dash_capstyle"] = "round"
    matplotlib.rcParams["lines.solid_capstyle"] = "round"


LABEL_NAMES_COAT = {
    "AFIB": "AF",
    "noAFIB": "not AF",
    "unknown": "unknown"
}

LABEL_NAMES_SPH = {
    "AFIB": "AF",
    "AF": "AFlut",
    "AT": "AT",
    "SA": "SA",
    "SB": "SB",
    "SR": "SR",
    "ST": "ST",
    "SVT": "SVT",
    "AVNRT": "AVNRT",
    "SAAWR": "SAAWR",
    "AVRT": "AVRT"
}

LABEL_NAMES_CINC = {
    "A": "AF",
    "O": "other",
    "~": "noisy",
    "N": "normal"
}


def get_label_names(dataset_name: str) -> dict[str, str]:
    if dataset_name == "coat":
        return LABEL_NAMES_COAT
    elif dataset_name == "sph" or dataset_name == "sph-r":
        return LABEL_NAMES_SPH
    elif dataset_name == "cinc" or dataset_name == "cinc-r":
        return LABEL_NAMES_CINC
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")


def get_dataset_label(dataset_name: str) -> str:
    if dataset_name == "coat":
        return "DiagnoStick"
    elif dataset_name == "sph":
        return "SPH"
    elif dataset_name == "sph-r":
        return "SPH-"
    elif dataset_name == "cinc":
        return "CinC"
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")


def move_experiment_run(run: Run, sub_dir: str | None, dir_name: str | None = None):
    if sub_dir is None:
        new_parent_dir = run.location.parent
    else:
        new_parent_dir = run.location.parent / sub_dir

    if dir_name is None:
        dir_name = run.location.name

    new_parent_dir.mkdir(parents=True, exist_ok=True)

    i = 1
    unique_dir_name = dir_name
    while (new_parent_dir / unique_dir_name).exists():
        unique_dir_name = f"{dir_name} ({i})"
        i += 1

    run.location.rename(new_parent_dir / unique_dir_name)


type Color = str | tuple[float, float, float]

LINE_STYLES = ["-", "--", "-.", ":"]


def make_color_palette[T: Hashable](palette: str, values: Sized and Iterable[T]) -> dict[T, Color]:
    palette = sns.color_palette(palette, len(values))

    return {
        value: palette[i]
        for i, value in enumerate(values)
    }


def make_line_styles[T: Hashable](values: Iterable[T]) -> dict[T, str]:
    return {
        value: LINE_STYLES[i % len(LINE_STYLES)]
        for i, value in enumerate(values)
    }


def download_file(url: str, path: Path):
    if path.exists():
        raise FileExistsError(f"File already exists at {path}")

    print(f"Downloading {url} to {path} ...")

    response = requests.get(url, stream=True)
    response.raise_for_status()
    total = int(response.headers.get("content-length", 0))

    with (
        open(path, "wb") as f,
        tqdm.tqdm(total=total, unit="iB", unit_scale=True, desc=f"Downloading {path.name}") as pbar
    ):
        for chunk in response.iter_content(chunk_size=1024):
            if not chunk:
                continue

            f.write(chunk)
            pbar.update(len(chunk))
