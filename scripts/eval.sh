#!/bin/bash

export PYTHONPATH="${PYTHONPATH}:../"
echo $PYTHONPATH

Mode=plain
Data=wiki103

for P in 0.3 0.4 0.5 0.6 0.7 0.8 0.9
do
    CUDA_VISIBLE_DEVICES=4 python eval_seq2seq_metric.py \
        --dataset $Data \
        --folderpath data/sampled/$Data/prefix-50_nsample-100 \
        --top-p $P \
        --top-k 0
done