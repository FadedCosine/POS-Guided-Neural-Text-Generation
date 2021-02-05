#!/bin/bash

export PYTHONPATH="${PYTHONPATH}:../"
echo $PYTHONPATH

for path in data/sampled/wikitext-103/overlapping/*; do
  CUDA_VISIBLE_DEVICES=0 python eval_corpus_wise_metric.py \
    --folderpath $path
done
