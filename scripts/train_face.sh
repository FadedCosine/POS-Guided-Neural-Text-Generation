#!/bin/bash

export PYTHONPATH="${PYTHONPATH}:../"
echo $PYTHONPATH

CUDA_VISIBLE_DEVICES=2 python lm_main.py \
    --dataset paraNMT \
    --vocab-size 100000 \
    --loss-type face \
    --root ./data \
    --finetune \
    --model-checkpoint data/checkpoint/paraNMT/_plain_layer_6_lr_0.0001_cutoffs_17_core_epoch_3 ; # require initial checkpoint from mle loss
