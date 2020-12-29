#!/bin/bash

export PYTHONPATH="${PYTHONPATH}:../"
echo $PYTHONPATH

CUDA_VISIBLE_DEVICES=0 python lm_main.py \
    --dataset wiki103 \
    --loss-type face \
    --root ./data \
    --finetune \
    --model-checkpoint data/checkpoint/wiki103/_plain_layer_12_lr_0.0001_cutoffs_17_core_epoch_9 ; # require initial checkpoint from mle loss
