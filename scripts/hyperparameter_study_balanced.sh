#!/bin/bash

export PYTHONPATH=$PYTHONPATH:.

python src/scripts/performance.py \
    --c_lower 1 \
    --c_upper 1 \
    --c_steps 1 \
    --rho_lower -1\
    --rho_upper 1\
    --rho_steps 20\
    --classifier KME \
    --bandwidth_rri_lower -2 \
    --bandwidth_rri_upper 0 \
    --bandwidth_rri_steps 20 \
    --bandwidth_rri_logspace \
    --bandwidth_rri_base 10.0 \
    --setup cross \
    --imbalanced_validating