#!/bin/bash

export PYTHONPATH="${PYTHONPATH}:../"
echo $PYTHONPATH

CUDA_VISIBLE_DEVICES=2 python main.py \
    --dataset wikitext-103 \
    --rnn-type LSTM \
    --vocab-size 200000 \
    --loss-type MLE \
    --root ./data ;
