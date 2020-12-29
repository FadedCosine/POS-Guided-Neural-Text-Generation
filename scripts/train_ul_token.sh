#!/bin/bash

export PYTHONPATH="${PYTHONPATH}:../"
echo $PYTHONPATH

CUDA_VISIBLE_DEVICES=1 python lm_main.py \
    --dataset wiki103\
    --loss-type unlikelihood-token \
    --root ./data ;

