#!/bin/bash

export PYTHONPATH="${PYTHONPATH}:../"
echo $PYTHONPATH

CUDA_VISIBLE_DEVICES=0 python lm_main.py \
    --dataset wiki103\
    --loss-type unlikelihood-token \
    --encoder-class FP \
    --root ./data/emnlp \
    --vocab-size 30000 \
#    --model-checkpoint data/bugs/_unlikelihood-token_layer_12_lr_0.0002_cutoffs_6_epoch_9 ;
