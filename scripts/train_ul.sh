#!/bin/bash

export PYTHONPATH="${PYTHONPATH}:../"
echo $PYTHONPATH

CUDA_VISIBLE_DEVICES=0 python lm_main.py \
    --dataset paraNMT \
    --vocab-size 100000 \
    --loss-type UL \
    --root ./data ;