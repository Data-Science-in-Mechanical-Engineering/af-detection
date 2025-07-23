# Robust Screening of Atrial Fibrillation with Distribution Classification

The structure of this project is the following:

    .
    ├── cache                       # `joblib` cache of kernel matrices
    ├── cinc2017                    # configurations for the baselines from the CinC 2017 challenge
    ├── data                        # where datasets will be stored (not part of the repo)
    ├── figures                     # where figures will be stored (not part of the repo)
    ├── results                     # where results will be stored (not part of the repo)
    ├── src                         # Source files
    │   ├── cinc2017_benchmarks     # Scripts for running and evaluating the CinC 2017 challemge baselines
    │   ├── experiments             # Main scripts for configuring and running our experiments
    │   ├── expyro                  # Utility package for experiment management
    │   ├── figures                 # Scripts for generating figures
    |   config.py                   # Configurations shared by multiple experiments
    |   data.py                     # Preprocessing and loading of datasets
    |   features.py                 # Computation of normalized RRis
    |   metrics.py                  # Evaluation of classifiers
    |   rkhs.py                     # Computation of distributional kernel matrix
    |   util.py                     # Miscellaneous
    ├── requirements.txt            # .txt-file with package specifications
    └── README.md

## Installation

Create an environment called ``af-detection`` with Python 3.12.6, pull the content from this repo into the
environment, and install all needed packages with:

```bash
cd af-detection
source <path/to/venv>/bin/activate
pip install -r requirements.txt
```

## Reproducing numerical results

All of our numerical experiments can be reproduced from the command line. By running the following commands, you can
reproduce our results using the configurations used in the paper.

The outcome of every run is saved to `./results`.

**Note:** Our implementation refers to the dataset `MyDiagnostick` as `coat`.

### Hyperparameter tuning

We select hyperparameters by cross-validation, optimizing for the mean AUROC on the held-out fold. Run the following
command to reproduce our hyperparameter search for the desired dataset and peak extraction method.

```bash
python -m src.experiments.tuning \ 
  --dataset-name=<coat | sph | cinc> \ 
  --peak-extraction=<xqrs | neurokit | pantompkins1985 | christov2004 | elgendi2010 | hamilton2002 | rodrigues2021 | zong2003> \
  --subsample=None
```

### Performance evaluation

To evaluate our distributional classifier, run the following command for the desired dataset. We automatically use the
best-performing parametrization from the hyperparameter search. This means that you have to run the hyperparameter
search before you can evaluate a model.

```bash
python -m src.experiments.tuning \ 
  --tuning-sub-dir="distributional/<DATASET NAME>/<PEAK EXTRACTION NAME>" \
  --tuning-seed=0 \
  --evaluation="all"
```

### Data efficiency

We evaluate data efficiency be re-running many times our entire training pipeline, including hyperparameter selection,
for randomly selected data subsets of different sizes. We then evaluate on the entire test set. For this experiment, we
always use the `xqrs` peak extraction method.

To tune parameters for many different random dataset selections, first run

```bash
for i in $(seq 0 99);
do
  python -m src.experiments.tuning \ 
      --dataset-name=<coat | sph | cinc> \ 
      --peak-extraction=xqrs \
      --subsample=<DATASET SIZE> \
      --seed=$i
done
```

for dataset sizes `10`, `25`, `100`, `200`. Note that the dataset size here refers to the number of AF examples in the
dataset, with the full dataset being sampled proportionally in a stratified fashion.

Then, evaluate each run separately.

```bash
for i in $(seq 0 99);
do
  python -m src.experiments.tuning \ 
  --tuning-sub-dir="distributional_sized/<DATASET NAME>/xqrs/n={DATASET SIZE}" \
  --tuning-seed=$i \
  --evaluation=<DATASET NAME>
done
```

### Baseline evaluation

We try to replicate the
[original environment of the CinC 2017 challenge](https://archive.physionet.org/challenge/sandbox/) as closely as
possible, using an [`apptainer`](https://github.com/apptainer/apptainer) container. Make sure you have `apptainer`
installed.

Then, run the following command on the desired challenge submission

```bash
python -m src.cinc2017_benchmarks \
  --entry-id=<shreyasi-datta | shenda-hong | morteza-zabihi | ruhi-mahajan> \
  --dataset-name="coat"
```

Note that each submission may take several hours to finish. You must have MATLAB installed on your system. We used
MATLAB 2019b. Our implementation expects that there is a `.env` file in the project's root directory, with a variable
`MLM_LICENSE_SERVER` pointing to a valid MATLAB license server.

## Reproducing figures

After having run the experiments, you can reproduce the corresponding figures, by running the following commands:

```bash
python -m src.figures.confusion
python -m src.figures.dataset_size
python -m src.figures.roc
```