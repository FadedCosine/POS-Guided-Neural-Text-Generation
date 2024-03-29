#!/bin/bash

export PYTHONPATH="${PYTHONPATH}:../"
echo $PYTHONPATH

CUDA_VISIBLE_DEVICES=2 python main.py \
    --dataset wikitext-103 \
    --vocab-size 100000 \
    --loss-type MoS \
    --root ./data ;
