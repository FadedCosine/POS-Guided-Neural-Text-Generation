#!/bin/bash

export PYTHONPATH="${PYTHONPATH}:../"
echo $PYTHONPATH

python data_processing/filter_para_data.py \
    --data-path ./data/paraNMT \
    --filter-len 10 \
    --corenlp-path # to be specified.
