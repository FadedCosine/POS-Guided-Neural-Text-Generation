#!/bin/bash

export PYTHONPATH="${PYTHONPATH}:../"
echo $PYTHONPATH

CUDA_VISIBLE_DEVICES=0 python lm_main.py \
    --dataset paraNMT \
    --vocab-size 100000 \
    --loss-type experimental3 \
    --model-checkpoint /data/zhaozx/PosSampling/POS-Train-seq2seq/data/checkpoint/paraNMT/_experimental3_layer_6_lr_0.0001_cutoffs_17_core_epoch_4 \
    --root ./data ;
