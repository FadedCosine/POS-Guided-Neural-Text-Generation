import os
import string
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.metrics.pairwise import cosine_similarity
import argparse
import collections
import logging
logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                    datefmt='%d-%m-%Y:%H:%M:%S')
logging.getLogger().setLevel(logging.DEBUG)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description='sow training')
parser.add_argument('--input_file', default='./data/para-nmt-50m/para-nmt-50m.txt',
                    help='input file')
parser.add_argument('--output_file', default='./data/para-nmt-50m/filtered_para.txt',
                    help='output_file file')
parser.add_argument('--wordEmbed_file', default='./data/word_embedding/glove.6B.50d.txt',
                    help='glove embeddings file')
parser.add_argument('--filter_len', default=10, type=int,
                    help='least valid sentence token number')

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
    with open(input_file, 'r') as f:
        for line in f:
            line_split = line.split('\t')
            avg_len = (len(line_split[0].split(' ')) + len(line_split[1].split(' '))) // 2
            # len_counter[avg_len] += 1
            len_list.append(avg_len)
    print("平均长度: ", np.mean(len_list) ) #59
    print("中位数: ", sorted(len_list)[int(len(len_list)/2)]) #35
    


if __name__ == '__main__':
    args = parser.parse_args()
    input_file = args.input_file

    # len_distribution(input_file)
    output_file = open(args.output_file, 'w')
    wordEmbed_file = args.wordEmbed_file

    stopWords = set(stopwords.words('english'))
    punctuation = list(string.punctuation)

    wordEmbed = wordEmbedding(wordEmbed_file)
    logger.info("finished reading word embeddings")

    total = 0
    line_idx = 0
    with open(input_file, 'r') as f:
        for line in f:
            line_idx += 1
            if line_idx % 100000 == 0:
                logger.info("finish {}/51400000 lines".format(line_idx))
            line_split = line.split('\t')

            sent1 = line_split[0].lower()
            sent2 = line_split[1].lower()

            tokens1 = word_tokenize(sent1)
            tokens2 = word_tokenize(sent2)
        
            tokens1 = [w for w in tokens1 if w not in stopWords and w not in punctuation]
            tokens2 = [w for w in tokens2 if w not in stopWords and w not in punctuation]

            if len(tokens1) < args.filter_len or len(tokens2) < args.filter_len:
                continue

            mat = similarity_matrix(tokens1, tokens2, wordEmbed)
            max_indices = np.argmax(mat, axis=1)

            diff = 0
            count = 0
            for row_idx in range(mat.shape[0]):
                diff += np.abs(row_idx - max_indices[row_idx])
                count += 1

            diff = float(diff) / (float(count) * float(count))
            ratio = float(mat.shape[0]) / float(mat.shape[1])

            if 0.75 < ratio < 1.5:
                if diff > 0.35:
                    output_file.write(line_split[0] + '\n')
                    output_file.write(line_split[1] + '\n')
                    total += 1

    logger.info(total)
    output_file.close()