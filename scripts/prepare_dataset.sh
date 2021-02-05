#!/bin/bash

export PYTHONPATH="./"
echo $PYTHONPATH

DATASET=wikitext-103
python data_processing/data_loader_with_POS.py  \
    --dataset $DATASET \
    --vocab-size 200000;