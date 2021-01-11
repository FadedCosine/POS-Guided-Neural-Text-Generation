#!/bin/bash

export PYTHONPATH="${PYTHONPATH}:../"
echo $PYTHONPATH

Mode=experimental3
Data=paraNMT
for P in 0.4
do
  CUDA_VISIBLE_DEVICES=0 python lm_sample.py \
      --saved-path data/checkpoint/$Data/_$Mode''_layer_6_lr_0.0001_cutoffs_17_core_epoch_3 \
      --dataset $Data \
      --loss-type $Mode \
      --top-p $P \
      --pos-top-p 0.9 \
      --sampling-mode 3 \
      --root ./data \
      --nprefix 50 \
      --ngenerate 100 \
      --vocab-size 100000;
done
