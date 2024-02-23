#!/bin/bash

export PYTHONPATH=$PYTHONPATH:.

python src/scripts/performance.py \
    # --qrs_name corrected_xqrs \
    # --search_radius 36 \
    # --smooth_window_size 50 \
    --c_lower 1 \
    --c_upper 1 \
    --c_steps 1 \
    --rho_lower 1\
    --rho_upper 1\
    --rho_steps 1\
    --rho_base 0.33\
    --classifier KME \
    --bandwidth_rri_lower -1 \
    --bandwidth_rri_upper -1 \
    --bandwidth_rri_steps 1 \
    --bandwidth_rri_logspace \
    --bandwidth_rri_base 10.0 \
    --setup cross \
    --imbalanced_validating\
    --test