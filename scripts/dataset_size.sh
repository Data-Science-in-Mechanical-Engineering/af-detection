#!/bin/bash

export PYTHONPATH=$PYTHONPATH:.

python src/scripts/dataset_size.py \
    --c_lower 1 \
    --c_upper 1 \
    --c_steps 1 \
    --classifier MEAN \
    --bandwidth_rri_lower 0.1 \
    --bandwidth_rri_upper 0.1\
    --bandwidth_rri_steps 1 \
    --rho_lower 1\
    --rho_upper 1\
    --rho_steps 1\
    --rho_base 0.33\
    --setup cross \
    --repetitions 50 \
    --imbalanced_validating 