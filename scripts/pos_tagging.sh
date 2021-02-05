#!/bin/bash

export PYTHONPATH="${PYTHONPATH}:../"
echo $PYTHONPATH

python data_processing/pos_tagging.py \
    --input-data-file ./data/wikitext-103/wiki.test.tokens \
    --output-data-file ./data/wikitext-103/pos_tagged_test.txt ;

python data_processing/pos_tagging.py \
    --input-data-file ./data/wikitext-103/wiki.valid.tokens \
    --output-data-file ./data/wikitext-103/pos_tagged_valid.txt ;

python data_processing/pos_tagging.py \
    --input-data-file ./data/wikitext-103/wiki.train.tokens \
    --output-data-file ./data/wikitext-103/pos_tagged_train.txt ;

python data_processing/pos_tagging.py \
    --input-data-file ./data/paraNMT/test.txt \
    --output-data-file ./data/wikitext-103/pos_tagged_test.txt ;

python data_processing/pos_tagging.py \
    --input-data-file ./data/paraNMT/valid.txt \
    --output-data-file ./data/wikitext-103/pos_tagged_valid.txt ;

python data_processing/pos_tagging.py \
    --input-data-file ./data/paraNMT/train.txt \
    --output-data-file ./data/wikitext-103/pos_tagged_train.txt ;
