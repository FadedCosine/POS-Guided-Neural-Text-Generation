#!/bin/bash

export PYTHONPATH="./"
echo $PYTHONPATH

DATASET=wiki103
python basic_util/data_loader.py \
    --root ../data \
    --dataset $DATASET \
    --encoder-class spbpe \
    --vocab-size 30000;