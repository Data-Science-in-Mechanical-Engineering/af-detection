#!/bin/bash

export PYTHONPATH=$PYTHONPATH:.

python src/scripts/roc.py \
    --c_lower 0.01 \
    --c_upper 20 \
    --c_steps 20 \
    --rho_lower -1\
    --rho_upper 1\
    --rho_steps 20\
    --rho_base 5\
    --bandwidth_rri_lower -1 \
    --bandwidth_rri_upper -1 \
    --bandwidth_rri_steps 1 \
    --bandwidth_rri_logspace \
    --bandwidth_rri_base 10.0 \
    --setup cross \
    --imbalanced_validating \
    --test