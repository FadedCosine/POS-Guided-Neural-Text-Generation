#!/bin/bash

export PYTHONPATH="${PYTHONPATH}:./"
echo $PYTHONPATH

Mode=POS
Data=wikitext-103
CUDA_VISIBLE_DEVICES=0 python lm_ppl_eval.py \
    --saved-path data/checkpoint/$Data/_$Mode''_layer_12_lr_0.0001_cutoffs_17_core_epoch_9 \
    --dataset $Data \
    --loss-type $Mode \
    --root ./data \
    --vocab-size 200000;
