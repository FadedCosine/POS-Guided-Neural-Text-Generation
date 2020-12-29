#!/bin/bash

export PYTHONPATH="${PYTHONPATH}:../"
echo $PYTHONPATH

for path in data/sampled/wiki103/overlapping/*; do
  CUDA_VISIBLE_DEVICES=0 python eval_corpus_wise_metric.py \
    --folderpath $path
done
#CUDA_VISIBLE_DEVICES=0 python eval_corpus_wise_metric.py \
#  --folderpath 'data/sampled/wiki103/topk-1_temp-1.0'
