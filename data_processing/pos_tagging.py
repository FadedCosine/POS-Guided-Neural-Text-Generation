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
parser.add_argument('--model-class', choices=['gpt2', 'gpt2-medium', 'tran_xl'], default='tran_xl')
parser.add_argument("--input_data_file", type=str, default="./data/wikitext-103/wiki.test.tokens")
parser.add_argument("--output_data_file", type=str, default="./data/wikitext-103/pos_tagged_test.txt")
parser.add_argument("--tagger_dir", type=str, default="../en-pos-ontonotes-v0.5.pt")
args = parser.parse_args()

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

model_path = {"gpt2":'/data/zhaozx/PosSampling/gpt2_model',
        "tran_xl":'/data/zhaozx/PosSampling/transformer-xl',
        }
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



def pos_tag_by_flair(read_file_name, write_dirty_name):
    from flair.models import SequenceTagger
    from flair.data import Sentence
    tagger = SequenceTagger.load(args.tagger_dir)
    read_file = open(read_file_name, "r", encoding="utf-8")
    write_file = open(write_dirty_name, "w", encoding="utf-8")
    for idx, line in enumerate(read_file):
        line_split = line.strip().split()
        if len(line_split) != 0 and not is_caption(line_split):
            sentence = Sentence(line)
            tagger.predict(sentence)
            token_pos_list = sentence.to_tagged_string().split(' ')
            # print(token_pos_list)
            assert len(token_pos_list) % 2 == 0
            for token_idx in range(0, len(token_pos_list), 2):
                token = token_pos_list[token_idx]
                pos = token_pos_list[token_idx + 1]
                if pos2word.get(pos) == None:
                    pos2word[pos] = set()
                pos2word[pos].add(token)

        # write_file.write(" ".join([item.split(">")[0] for item in sentence.to_tagged_string().split("<")[1:]]) + '\n')
        write_file.write(" ".join(token_pos_list) + '\n')
        if idx % 1000 == 0:
            logging.info("Finish tag {} lines.".format(idx))
  
    read_file.close()
    write_file.close()


def main():
  
    # MODEL_CLASSES = {
    # "gpt2": GPT2Tokenizer,
    # "tran_xl": TransfoXLTokenizer,
    # }
    
    # tokenizer_class = MODEL_CLASSES[args.model_class]
    # tokenizer = tokenizer_class.from_pretrained(model_path[args.model_class])
    # pos_tag_by_flair(args.input_data_file, args.output_data_file, tokenizer, args.output_data_dir + args.model_class + "_valid_flair.pos",)
    
    logging.info("tagging data")
    pos_tag_by_core(args.input_data_file, args.output_data_file)

    
if __name__ == '__main__':
    main()
