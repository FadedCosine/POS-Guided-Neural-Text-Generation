#!/bin/bash

export PYTHONPATH="${PYTHONPATH}:../"
echo $PYTHONPATH

CUDA_VISIBLE_DEVICES=7 python lm_main.py \
    --dataset paraNMT \
    --vocab-size 100000 \
    --loss-type plain \
    --lower \
    --root ./data ;
