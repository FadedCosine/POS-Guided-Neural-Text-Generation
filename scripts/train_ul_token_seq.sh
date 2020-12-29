#!/bin/bash

export PYTHONPATH="${PYTHONPATH}:../"
echo $PYTHONPATH

CUDA_VISIBLE_DEVICES=0 python lm_main.py \
    --dataset wiki103\
    --loss-type unlikelihood-token-seq \
    --root ./data \
    --finetune \
    --model-checkpoint data/wiki103/... ; # require initial checkpoint from ul-token loss
