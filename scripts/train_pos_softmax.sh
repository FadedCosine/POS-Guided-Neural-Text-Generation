#!/bin/bash

export PYTHONPATH="${PYTHONPATH}:../"
echo $PYTHONPATH

CUDA_VISIBLE_DEVICES=0 python lm_main.py \
    --dataset paraNMT \
    --vocab-size 100000 \
    --loss-type F2v1 \
    --model-checkpoint data/checkpoint/paraNMT/_POS_layer_6_lr_0.0001_cutoffs_17_core_epoch_4 \
    --root ./data ;
