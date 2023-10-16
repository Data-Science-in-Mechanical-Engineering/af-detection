#!/bin/bash

export PYTHONPATH=$PYTHONPATH:.

python src/scripts/performance.py \
    --c_lower -1 \
    --c_upper 1 \
    --c_steps 11 \
    --classifier KME \
    --bandwidth_rri_lower -1 \
    --bandwidth_rri_upper 3 \
    --bandwidth_rri_steps 11 \
    --bandwidth_rri_logspace \
    --bandwidth_rri_base 10.0 \
    --setup cross \
    --imbalanced_validating