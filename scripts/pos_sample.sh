#!/bin/bash

export PYTHONPATH="${PYTHONPATH}:../"
echo $PYTHONPATH

Mode=POS
Data=wikitext-103

for M in 3
do
  CUDA_VISIBLE_DEVICES=0 python lm_sample.py \
      --saved-path data/checkpoint/$Data/_$Mode''_layer_6_lr_0.0001_cutoffs_17_core_epoch_8 \
      --dataset $Data \
      --loss-type $Mode \
      --top-p 0.3 \
      --pos-top-p 0.9 \
      --sampling-mode $M \
      --root ./data \
      --nprefix 50 \
      --ngenerate 100 \
      --vocab-size 100000;
done