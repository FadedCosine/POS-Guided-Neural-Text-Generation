#!/bin/bash

export PYTHONPATH="${PYTHONPATH}:../"
echo $PYTHONPATH

CUDA_VISIBLE_DEVICES=2 python main.py \
    --dataset paraNMT \
    --vocab-size 100000 \
    --loss-type FACE \
    --root ./data \
    --finetune \
    --model-checkpoint data/checkpoint/paraNMT/_MLE_layer_6_lr_0.0001_cutoffs_17_core_epoch_3 ; # require initial checkpoint from mle loss
