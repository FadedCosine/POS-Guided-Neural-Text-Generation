import collections
import pickle
import os
import logging
from os.path import join, exists
from os import makedirs


logger = logging.getLogger(__name__)

def load_vocab(vocab_file):
    vocab_list = []
    f =  open(vocab_file, 'r')
    for line in f:
        line = line.strip()
        if line != "UNK":
            vocab_list.append(line)
    return vocab_list

class TokenTokenizer(object):
    def __init__(self, vocab_file, vocab_size, add_special_token=False):
        """
        默认unk为0， padding为vocab_size+1, bos为vocab_size+2， eos为vocab_size+3
        """
        self.token_vocab = load_vocab(vocab_file)[:vocab_size]
        self.unk_id = 0
        self.padding_id = vocab_size + 1
        
        self.word2id = {word: index + 1 for index, word in enumerate(self.token_vocab)}
        self.word2id['<unk>'] = self.unk_id
        self.word2id['<padding>'] = self.padding_id
        
        if add_special_token:
            self.bos_id = vocab_size + 2
            self.eos_id = vocab_size + 3
            self.word2id['<bos>'] = self.bos_id
            self.word2id['<eos>'] = self.eos_id
        
        
        self.vocab_size = len(self.word2id)
        self.id2word = {i: w for w, i in self.word2id.items()}

    def convert_words_to_ids(self, words):
        """Converts a sequence into ids using the vocab."""
        ids = []
        for word in words:
            ids.append(self.word2id.get(word, self.unk_id))
        return ids
    def convert_word_to_id(self, word):
        """Converts a word into id using the vocab."""
        return self.word2id.get(word, self.unk_id)

    def convert_ids_to_words(self, ids):
        """Converts a sequence of ids into words using the vocab."""
        words = []
        for i in ids:
            if i != self.padding_id:
                words.append(self.id2word.get(i, "<unk>"))
        return words

    def convert_id_to_word(self, id):
        """Converts a id into word using the vocab."""
        return self.id2word.get(id, "<unk>")

class POSTokenizer(object):
    def __init__(self, vocab_file, add_special_token=False): # 所有的pos全部都用上，不会有unk
        """
        思考pos2word，只有有意义的pos算在pos_tokenizer的vocab_size当中，
        eos也在pos2word当中，eos的token list当中只有一个eos，所以为了方便，在之前的pos id之后紧跟着eos id
        
        """
        self.tag_vocab = load_vocab(vocab_file)
        
        self.pos2id = {word: index for index, word in enumerate(self.tag_vocab)}
        
        if add_special_token:
            self.bos_id = len(self.tag_vocab) + 1
            self.eos_id = len(self.tag_vocab)
            self.padding_id = len(self.tag_vocab) + 2
            self.pos2id['<bos>'] = self.bos_id
            self.pos2id['<eos>'] = self.eos_id
            self.meaningful_vocab_size = len(self.tag_vocab) + 1 # 加上了eos的pos，eos:[eos]，而不加bos，因为不可能生成bos的pos
        else:
            self.padding_id = len(self.tag_vocab)
            self.meaningful_vocab_size = len(self.tag_vocab)
            
        self.pos2id['<padding>'] = self.padding_id
        
        self.id2pos = {i: w for w, i in self.pos2id.items()}
        
        

    def convert_tags_to_ids(self, tags):
        """Converts a sequence of tags into ids using the vocab."""
        ids = []
        for tag in tags:
            # if self.pos2id.get(tag, "<unk>") == "<unk>":
            #     tag = 'NN'
            if i != self.padding_id:
                ids.append(self.pos2id[tag])

        return ids
    def convert_tag_to_id(self, tag):
        """Converts a sequence of tags into ids using the vocab."""
        # 把vocab中的tag全部用上时这没有<unk>
        # if self.pos2id.get(tag, "<unk>") == "<unk>":
        #     tag = 'NN'
        return self.pos2id[tag]
        
    def convert_ids_to_tags(self, ids):
        """Converts a sequence of ids into tags using the vocab."""
        tags = []
        for i in ids:
            tags.append(self.id2pos[i])
        return tags
