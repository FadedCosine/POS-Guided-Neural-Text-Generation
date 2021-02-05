import os
import string
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.metrics.pairwise import cosine_similarity
from nltk.translate.bleu_score import corpus_bleu
from stanfordcorenlp import StanfordCoreNLP
import argparse
from tqdm import tqdm
import collections
import random
import logging
logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                    datefmt='%d-%m-%Y:%H:%M:%S')
logging.getLogger().setLevel(logging.DEBUG)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description='sow training')
parser.add_argument('--data-path', default='./data/paraNMT',
                    help='data path')
parser.add_argument('--filter-len', default=10, type=int,
                    help='least valid sentence token number')
parser.add_argument('--corenlp-path', help='the path to the stanford corenlp')

class wordEmbedding(object):
    def __init__(self, filename):
        f = open(filename, 'r')
        self.vocab2id = {}
        self.id2vocab = {}
        self.vectors = []

        id = 0
        for line in f.readlines():
            word = line.strip().split()[0]
            vector = np.array([float(val)  for val in line.split()[1:]])
            self.id2vocab[id] = word
            self.vocab2id[word] = id
            self.vectors.append(vector)
            id += 1

        self.id2vocab[len(self.id2vocab)] = "UNK"
        self.vocab2id["UNK"] = len(self.vocab2id)

        self.vectors.append(np.zeros(50))
        self.vectors = np.array(self.vectors)

    def get_index(self, word):
        return self.vocab2id[word] if word in self.vocab2id else self.vocab2id["UNK"]

    def get_index_list(self, tokens):
        return [self.get_index(t) for t in tokens]


def similarity_matrix(tokens1, tokens2, wordEmbed):
    tokens_idx1 = wordEmbed.get_index_list(tokens1)
    tokens_idx2 = wordEmbed.get_index_list(tokens2)
    embed1 = np.take(wordEmbed.vectors, tokens_idx1, axis=0)
    embed2 = np.take(wordEmbed.vectors, tokens_idx2, axis=0)
    mat = cosine_similarity(embed1, embed2)
    return mat

def len_distribution(filename):
    len_counter = collections.Counter()
    len_list = []
    with open(filename, 'r') as f:
        for line in f:
            line_split = line.split('\t')
            avg_len = (len(line_split[0].split(' ')) + len(line_split[1].split(' '))) // 2
            # len_counter[avg_len] += 1
            len_list.append(avg_len)
    print("平均长度: ", np.mean(len_list) ) #59
    print("中位数: ", sorted(len_list)[int(len(len_list)/2)]) #35
    

def create_paranmt(read_filename, write_path, corenlp_path, high=1, low=0, min_seq_len=10, sampler_num=None):
    nlp = StanfordCoreNLP(corenlp_path, lang='en')
    data = []
    with open(read_filename, 'r') as f:
        cnt = 0
        # print(len(list(f.readlines())))
        for line in tqdm(f.readlines(), mininterval=10,):
            line = line.strip('\n').split('\t')
            # print(line)
            if float(line[2]) >= low and float(line[2]) <= high:
                # if len(line[0].strip('\n').split()) <= 30 and len(line[1].strip('\n').split()) <= 30:
                src = line[0].split()
                tgt = line[1].split()
                if len(src) < min_seq_len or len(tgt) <min_seq_len:
                    continue
                cnt += 1
                if corpus_bleu([[src]], [tgt], weights=(0.25, 0.25, 0.25, 0.25)) * 100 < 10:
                    data.append(line[:2])
                    # if len(data) % 1000 == 0:
                        # print(data[-1])
                    # print(len(data), cnt)
    logger.info("total data len is {}.".format(len(data)))
    random.shuffle(data)
    if sampler_num is not None and len(data) > sampler_num:
        data = random.sample(data, sampler_num)
    
    with open(os.path.join(write_path, "test.txt"), "w") as write_file:
        for pair in tqdm(data[-3000:], mininterval=10,):
            write_file.write(" ".join(nlp.word_tokenize(pair[0])))
            write_file.write("\n")
            write_file.write(" ".join(nlp.word_tokenize(pair[1])))
            write_file.write("\n")

    with open(os.path.join(write_path, "valid.txt"), "w") as write_file:
        for pair in tqdm(data[-9000:-3000], mininterval=10,):
            write_file.write(" ".join(nlp.word_tokenize(pair[0])))
            write_file.write("\n")
            write_file.write(" ".join(nlp.word_tokenize(pair[1])))
            write_file.write("\n")

    with open(os.path.join(write_path, "train.txt"), "w") as write_file:
        for pair in tqdm(data[:-9000], mininterval=10,):
            write_file.write(" ".join(nlp.word_tokenize(pair[0])))
            write_file.write("\n")
            write_file.write(" ".join(nlp.word_tokenize(pair[1])))
            write_file.write("\n")
           
if __name__ == '__main__':
    args = parser.parse_args()
    create_paranmt(os.path.join(args.data_path, "para-nmt-50m.txt"), args.data_path, 0.8, 0.7, 10)
