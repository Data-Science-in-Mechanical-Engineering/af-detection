from pathlib import Path
from typing import Callable

import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm

from .util import Style
from ..results import Snapshot, Result, RESULTS_FOLDER, Outcome

Metric = Callable[[Outcome], float]


def plot_roc(
    snapshot: Snapshot,
    metric_x: Metric,
    metric_y: Metric,
    label_metric_x: str,
    label_metric_y: str,
    line_cs: set[float],
    highlighted_cs: set[float],
    path: Path | None = None
):
    outcomes_by_c = snapshot.filter(lambda outcome: outcome.parametrization["c"] in line_cs) \
        .partition(lambda outcome: outcome.parametrization["c"])

    highlighted_outcomes_by_c = {
        c: outcomes
        for c, outcomes in outcomes_by_c.items()
        if c in highlighted_cs
    }

    highlighted_data = {"x": [], "y": [], "rho": []}
    min_rho, max_rho = 0, 0

    for c, outcomes in highlighted_outcomes_by_c.items():
        outcomes = list(outcomes_by_c[c])
        highlighted_data["x"].extend(map(metric_x, outcomes))
        highlighted_data["y"].extend(map(metric_y, outcomes))

        rhos = [outcome.parametrization["c_class_weight_proportion"] for outcome in outcomes]
        min_rho = min(min_rho, min(rhos))
        max_rho = max(max_rho, max(rhos))

        highlighted_data["rho"].extend(rhos)

    for outcomes in outcomes_by_c.values():
        xs = map(metric_x, outcomes)
        ys = map(metric_y, outcomes)
        sns.lineplot(x=xs, y=ys, color="#c7ddf2", errorbar=None)

    point_colors = Style.VIBRANT_COLOR_PALETTE

    ax = sns.scatterplot(
        data=highlighted_data,
        x="x",
        y="y",
        hue="rho",
        legend=None,
        zorder=100,
        hue_norm=LogNorm(),
        palette=point_colors
    )

    plt.xlabel(label_metric_x, size=Style.LABEL_FONT_SIZE)
    plt.ylabel(label_metric_y, size=Style.LABEL_FONT_SIZE)
    plt.tick_params(labelsize=Style.LABEL_FONT_SIZE)

    sm = plt.cm.ScalarMappable(cmap=point_colors, norm=plt.Normalize(min_rho, max_rho))
    ax.figure.colorbar(sm, ax=ax, label="Rho")
    ax.figure.axes[-1].tick_params(labelsize=Style.LABEL_FONT_SIZE)
    ax.figure.axes[-1].yaxis.label.set_size(Style.LABEL_FONT_SIZE)

    if path is None:
        plt.show()
    else:
        plt.savefig(path, dpi=300)

    plt.clf()


if __name__ == "__main__":
    RESULT = Result.from_json(RESULTS_FOLDER / "svm rri/validate-roc-cross-database.json") \
        .partition_by_datasets()

    SPH_RESULT = RESULT["SPHDataset", "SPHDataset"]
    COAT_RESULT = RESULT["COATDataset", "COATDataset"]

    LINE_CS = {1.0621052631578947, 10.531052631578946, 20}
    CS = {10.531052631578946}

    plot_roc(
        SPH_RESULT.snapshots[0],
        lambda outcome: 1 - outcome.specificity,
        lambda outcome: outcome.recall,
        "False Positive Rate (1 - Specificity)",
        "True Positive Rate (Sensitivity)",
        LINE_CS,
        CS,
        RESULTS_FOLDER / "roc-sph.pdf"
    )