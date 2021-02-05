#!/bin/bash

export PYTHONPATH="${PYTHONPATH}:../"
echo $PYTHONPATH

Mode=UL
Data=paraNMT
for P in 0.8 0.9 0.7 0.6 0.5 0.4 0.3
do
    CUDA_VISIBLE_DEVICES=2 python sample.py \
        --saved-path data/checkpoint/$Data/_$Mode''_layer_6_lr_0.0001_cutoffs_17_core_epoch_4 \
        --dataset $Data \
        --loss-type $Mode \
        --top-p $P \
        --sampling-mode 2 \
        --root ./data \
        --nprefix 50 \
        --ngenerate 100 \
        --vocab-size 100000;
done