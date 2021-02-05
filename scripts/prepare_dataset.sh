#!/bin/bash

export PYTHONPATH="./"
echo $PYTHONPATH

DATASET=paraNMT
python data_processing/data_loader_with_POS.py  \
    --root ./data \
    --dataset $DATASET \
    --encoder-class spbpe \
    --vocab-size 100000;