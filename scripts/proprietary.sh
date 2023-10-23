#!/bin/bash

export PYTHONPATH=$PYTHONPATH:.

python src/scripts/proprietary.py -f ./data/coat/raw/stick_diagnostics.xlsx --imbalanced
# python src/scripts/proprietary.py -f ./data/coat/raw/stick_diagnostics.xlsx