#!/bin/bash

export PYTHONPATH="${PYTHONPATH}:../"
echo $PYTHONPATH

Mode=POS
Data=wikitext-103

for P in 0.3 0.4 0.5 0.6 0.7 0.8 0.9
do
  CUDA_VISIBLE_DEVICES=0 python sample.py \
      --saved-path data/checkpoint/$Data/_$Mode''_layer_6_lr_0.0001_cutoffs_17_core_epoch_8 \
      --dataset $Data \
      --loss-type $Mode \
      --top-p 0.3 \
      --pos-top-p 0.9 \
      --sampling-mode 3 \
      --root ./data \
      --nprefix 50 \
      --ngenerate 100 \
      --vocab-size 100000;
done