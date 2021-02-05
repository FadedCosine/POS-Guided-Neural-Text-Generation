#!/bin/bash

export PYTHONPATH="${PYTHONPATH}:../"
echo $PYTHONPATH

CUDA_VISIBLE_DEVICES=2 python lm_main.py \
    --dataset paraNMT \
    --vocab-size 100000 \
    --loss-type plain \
    --model-checkpoint data/checkpoint/paraNMT/_MLE_layer_6_lr_0.0001_cutoffs_17_core_epoch_3 \
    --root ./data ;
