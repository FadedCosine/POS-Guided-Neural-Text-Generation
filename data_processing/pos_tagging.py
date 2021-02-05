import sys
import logging
import random
import numpy as np
import os
# from nltk.corpus import wordnet as wn
import argparse
import torch
import pickle
from transformers import GPT2Tokenizer, TransfoXLTokenizer
from shutil import copyfile
import collections
from multiprocessing.pool import Pool
import multiprocessing

from functools import partial

try:
    multiprocessing.set_start_method('spawn')
except RuntimeError:
    pass

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)
parser = argparse.ArgumentParser()

parser.add_argument("--input-data-file", type=str, default="./data/wikitext-103/wiki.test.tokens")
parser.add_argument("--output-data-file", type=str, default="./data/wikitext-103/pos_tagged_test.txt")

args = parser.parse_args()

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

subword_models = ["gpt2-medium", "gpt2"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
logger.info("device: {}, n_gpu {}".format(device, n_gpu))
punct_masks = ["#", "$", "''", "(", ")", ",", ".", ":", "``"]
upper_pos_dir = {
    "NOUN": [ "FW", "NN", "NNS", "NNP", "NNPS"],
    "PRON": [ "PRP", "PRP$", "WP", "WP$", "EX"],
    "ADJ": [ "JJ", "JJR", "JJS"],
    "ADV": [ "RB", "RBR", "RBS", "RP", "WRB", "UH"],
    "VERB": [ "MD", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ"],
    "NUM": ["CD"],
    "ART": ["DT", "PDT", "WDT"],
    "PREP": ["IN", "POS", "TO"],
    "CONJ": ["CC"],
    "SYM": ["SYM", "LS", "#", "$", "''", "(", ")", ",", ".", ":", "``"]
}
punct_marks = ["-", "--", "---", ",", ".", "?", ":", "'", '"', "!", "`", "$", "#", "...", "(", ")", "[", "]", "{", "}"]
down_pos_to_up = {down_pos:up_pos for up_pos, down_pos_list in upper_pos_dir.items() for down_pos in down_pos_list}

pos2word = {}

def is_caption(line_split):
    return line_split[0] == '=' and  line_split[-1] == '='
    

def pos_tag_by_core(read_file_name, write_dirty_name):
    from nltk.parse import CoreNLPParser
    pos_tagger = CoreNLPParser(url="http://localhost:9876", tagtype='pos')
    read_file = open(read_file_name, "r", encoding="utf-8")
    write_file = open(write_dirty_name, "w", encoding="utf-8")
    for idx, line in enumerate(read_file):
        line_split = line.strip().split()
        if len(line_split) != 0 and not is_caption(line_split):
            pos_result = pos_tagger.tag(line_split)
            for word_pos in pos_result:
                write_file.write(word_pos[0])
                write_file.write(" ")
                write_file.write(word_pos[1])
                write_file.write(" ")
            write_file.write("\n")
        if idx % 1000 == 0:
            logging.info("Finish tag {} lines.".format(idx))
    read_file.close()
    write_file.close()

def main():
  
    logging.info("tagging data")
    pos_tag_by_core(args.input_data_file, args.output_data_file)

    
if __name__ == '__main__':
    main()
