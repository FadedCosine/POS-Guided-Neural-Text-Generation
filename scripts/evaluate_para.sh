#!/bin/bash

export PYTHONPATH="${PYTHONPATH}:../"
echo $PYTHONPATH

Mode=MLE
Data=paraNMT

for P in 0.8
do
    CUDA_VISIBLE_DEVICES=4 python eval_seq2seq_metric.py \
        --dataset $Data \
        --folderpath data/sampled/$Data/prefix-50_nsample-100 \
        --top-p $P \
        --top-k 0
done