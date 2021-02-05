#!/bin/bash

export PYTHONPATH="${PYTHONPATH}:../"
echo $PYTHONPATH

Mode=MLE
Data=wikitext-103

for P in 0.5
do
    CUDA_VISIBLE_DEVICES=4 python eval_LM_metric.py \
        --folderpath data/sampled/$Data/prefix-50_nsample-100 \
        --top-p $P \
        --top-k 0
done