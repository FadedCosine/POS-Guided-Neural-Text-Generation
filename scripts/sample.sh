#!/bin/bash

export PYTHONPATH="${PYTHONPATH}:../"
echo $PYTHONPATH

Mode=plain
Data=paraNMT
for P in 0.9 
do
    CUDA_VISIBLE_DEVICES=2 python lm_sample.py \
        --saved-path data/checkpoint/$Data/_$Mode''_layer_6_lr_0.0001_cutoffs_14_core_epoch_3 \
        --dataset $Data \
        --loss-type $Mode \
        --top-p $P \
        --sampling-mode 0 \
        --root ./data \
        --nprefix 50 \
        --ngenerate 100 \
        --vocab-size 100000;
done