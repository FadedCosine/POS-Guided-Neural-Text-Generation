from contextlib import contextmanager
from multiprocessing import Process, Pool
from indexer.encoder import *
from basic_util.files import *
import pickle

@contextmanager
def poolcontext(*args,**kwargs):
    pool=Pool(*args,**kwargs)
    yield pool
    pool.terminate()

class LMIndexer:
    def __init__(self, root, prefix, encoder_class=tokenizers.SentencePieceBPETokenizer, vocab_size=30000):
        self.encoder = self.load_encoder(encoder_class, root, prefix)
        self.map_dic, self.inv_dic = self.load_mapper(root, prefix)
        self.vocab_size = vocab_size
        self.root = root
        self.prefix = prefix

    def decode(self, indexed, map=True):
        if map and self.map_dic is not None:
            indexed = self.reverse_map(indexed)
        return self.encoder.decode(indexed)

    def encode(self, text, map=True):
        encoded = self.encoder.encode(text).ids
        if map and self.map_dic is not None:
            encoded = self.convert_map(encoded)
        return encoded

    def convert_map(self, encoded):
        def convert_token(token):
            if key_type(token) in dic:
                return dic[key_type(token)]
            else:
                return len(dic) - 1
        assert self.map_dic is not None
        dic = self.map_dic
        key_type = type(list(self.map_dic)[0])
        converted = [convert_token(i) for i in encoded]
        return converted

    def reverse_map(self, encoded):
        return [int(self.inv_dic[i]) for i in encoded]

    def learn_encoder(self, path):
        self.encoder.train(path, vocab_size=self.vocab_size)
        print("self.root is ", self.root)
        print("self.prefix is ", self.prefix)
        # self.encoder.save(self.root, self.prefix)

    def learn_mapper(self, encoded):
        prob, dic = self.count(encoded,self.vocab_size)
        inv_dict = dict(zip(dic.values(), dic.keys()))
        base_name = os.path.join(self.root, self.prefix)
        dic_name = base_name + '-dic.pkl'
        prob_name = base_name + '-probs.pkl'
        pickle.dump(prob, open(prob_name, 'wb'))
        pickle.dump(dic, open(dic_name, 'wb'))
        self.map_dic, self.inv_dic = dic, inv_dict

    def load_mapper(self, root, prefix):
        base_name = os.path.join(root, prefix)
        dic_name = base_name + '-dic.pkl'
        if os.path.exists(dic_name):
            dic = load_pkl(dic_name)
            inv_dic = dict(zip(dic.values(), dic.keys()))
        else:
            dic=None
            inv_dic=None
        return dic, inv_dic

    def load_encoder(self, encoder_class, directory_path, encoder_filename):
        base_name = os.path.join(directory_path, encoder_filename)
        if encoder_class == tokenizers.BertWordPieceTokenizer:
            vocab_name = base_name + '-vocab.txt'
            if os.path.exists(vocab_name):
                print('trained encoder loaded')
                self.istrained = True
                return encoder_class(vocab_name)
            else:
                self.istrained = False
                print('encoder needs to be trained')
                return encoder_class()
        else:
            vocab_name = base_name + '-vocab.json'
            merge_name = base_name + '-merges.txt'
            if os.path.exists(vocab_name) and os.path.exists(merge_name):
                print('trained encoder loaded')
                self.istrained = True
                if encoder_class == tokenizers.SentencePieceBPETokenizer:
                    return encoder_class(vocab_name, merge_name)
                else:
                    return encoder_class(vocab_name, merge_name, lowercase=True)
            else:
                self.istrained = False
                print('encoder needs to be trained')
                if encoder_class == tokenizers.SentencePieceBPETokenizer:
                    return encoder_class()
                else:
                    return encoder_class(lowercase=True)

    def count(self, tl, vocab_size=30000):
        import collections
        cnter = collections.Counter()
        cnter.update(tl)
        for i in range(vocab_size):
            if i not in cnter:
                cnter[i] = 1

        tot = 0
        cum_prob = [0]
        for i in cnter.most_common():
            tot += i[1]
        for i in cnter.most_common():
            cum_prob.append(cum_prob[-1] + i[1] / tot)
        cum_prob.pop(0)
        new_dict = dict([(int(old[0]), int(new)) for (new, old) in enumerate(cnter.most_common())])
        return cum_prob, new_dict

