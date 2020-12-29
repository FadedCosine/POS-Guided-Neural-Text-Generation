"""Byte pair encoding utilities"""

import regex as re
from functools import lru_cache
import tokenizers
from basic_util.files import *
from indexer.morph_tagger import *


cachedir = '~/kogpt2/'
vocab_info = {
    'url':
        'https://kobert.blob.core.windows.net/models/kogpt2/tokenizer/kogpt2_news_wiki_ko_cased_818bfa919d.spiece',
    'fname': 'kogpt2_news_wiki_ko_cased_818bfa919d.spiece',
    'chksum': '818bfa919d'
}


def clean_line(line):
    line = re.sub('\n+','\t',line).strip('\r\n\t ')
    return line.lower()


def convert_idx_list(line, dic):
    converted = []
    for token in line:
        converted.append(dic[str(token)])
    return converted


def rollback_idx(indices,inv_dic):
    new = []
    for ind in indices:
        if ind in inv_dic:
            new.append(int(inv_dic[ind]))
    return new


@lru_cache()
def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a corresponding list of unicode strings.
    The reversible bpe codes work on unicode strings.
    This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
    When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
    This is a signficant percentage of your normal, say, 32K bpe vocab.
    To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
    And avoids mapping to whitespace/control characters the bpe code barfs on.
    """
    bs = list(range(ord("!"), ord("~")+1))+list(range(ord("¡"), ord("¬")+1))+list(range(ord("®"), ord("ÿ")+1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8+n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


def get_pairs(word):
    """Return set of symbol pairs in a word.
    Word is represented as tuple of symbols (symbols being variable-length strings).
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


class Encoder:
    def __init__(self, encoder_path, mmap_dict=None, errors='replace'):
        with open(os.path.join(encoder_path, 'encoder.json'), 'r') as f:
            encoder = json.load(f)
        with open(os.path.join(encoder_path, 'vocab.bpe'), 'r', encoding="utf-8") as f:
            bpe_data = f.read()
        bpe_merges = [tuple(merge_str.split()) for merge_str in bpe_data.split('\n')[1:-1]]
        self.encoder = encoder
        self.decoder = {v:k for k,v in self.encoder.items()}
        self.errors = errors # how to handle errors in decoding
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v:k for k, v in self.byte_encoder.items()}
        self.bpe_ranks = dict(zip(bpe_merges, range(len(bpe_merges))))
        self.cache = {}
        self.mmap_dict = mmap_dict
        self.inv_mmap = dict(zip(mmap_dict.values(), mmap_dict.keys()))
        self.mmap = True if mmap_dict is not None else False

        # Should haved added re.IGNORECASE so BPE merges can happen for capitalized versions of contractions
        self.pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

    def bpe(self, token):
        if token in self.cache:
            return self.cache[token]
        word = tuple(token)
        pairs = get_pairs(word)

        if not pairs:
            return token

        while True:
            bigram = min(pairs, key = lambda pair: self.bpe_ranks.get(pair, float('inf')))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except:
                    new_word.extend(word[i:])
                    break

                if word[i] == first and i < len(word)-1 and word[i+1] == second:
                    new_word.append(first+second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        word = ' '.join(word)
        self.cache[token] = word
        return word

    def encode(self, text):
        # text = clean_line(text)
        bpe_tokens = []
        for token in re.findall(self.pat, text):
            token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
            bpe_tokens.extend(self.encoder[bpe_token] for bpe_token in self.bpe(token).split(' '))
        if self.mmap:
            convert_idx_list(bpe_tokens, self.mmap_dict)
        return bpe_tokens

    def decode(self, tokens):
        if self.mmap:
            tokens = rollback_idx(tokens, self.inv_mmap)
        text = ''.join([self.decoder[token] for token in tokens])
        text = bytearray([self.byte_decoder[c] for c in text]).decode('utf-8', errors=self.errors)
        return text


class SPEncoder:
    def __init__(self, ):
        self.tokenizer = self.load_tokenizer()
        self.vocab = self.load_vocab()
        self.padding_idx = self.vocab[self.vocab.padding_token]
        self.eos_idx = self.vocab[self.vocab.eos_token]
        self.bos_idx = self.vocab[self.vocab.bos_token]
        self.sep_idx = self.vocab[self.vocab.sep_token]
        self.vocab_size = len(self.vocab)

    @staticmethod
    def load_vocab():
        import gluonnlp as nlp
        from kogpt2.utils import download as _download
        vocab_file = _download(vocab_info['url'],
                               vocab_info['fname'],
                               vocab_info['chksum'],
                               cachedir=cachedir)
        vocab_b_obj = nlp.vocab.BERTVocab.from_sentencepiece(vocab_file,
                                                             mask_token=None,
                                                             sep_token='<usr>',
                                                             cls_token=None,
                                                             unknown_token='<unk>',
                                                             padding_token='<pad>',
                                                             bos_token='<s>',
                                                             eos_token='</s>')
        return vocab_b_obj

    @staticmethod
    def load_tokenizer():
        from gluonnlp.data import SentencepieceTokenizer
        from kogpt2.utils import get_tokenizer
        tok_path = get_tokenizer()
        tok = SentencepieceTokenizer(tok_path)
        return tok

    def encode(self, text, bos=True, eos=True):
        toked = self.tokenizer(text)
        res = self.vocab[toked]
        if bos:
            res = [self.vocab[self.vocab.bos_token]] + res
        if eos:
            res += [self.vocab[self.vocab.eos_token]]
        return res

    def decode(self, indice):
        temp = ''.join(self.vocab.to_tokens(indice))
        return temp.replace('▁', ' ')


# class HFEncoder:
#     def __init__(self, dir_path, prefix, encoder_class=tokenizers.SentencePieceBPETokenizer,
#                  tokenizer_class=Mecab_Tokenizer, imap_dict_name=None, split_jamo=False):
#         if split_jamo:
#             assert encoder_class ==tokenizers.BertWordPieceTokenizer
#             space_symbol='쀍'
#         else:
#             space_symbol = '‐'
#         self.encoder = self.load_encoder(encoder_class,dir_path,prefix)
#         self.tokenizer = tokenizer_class(space_symbol=space_symbol, jamo=split_jamo)
#         if imap_dict_name:
#             mmap_dict_path = os.path.join(dir_path, imap_dict_name)
#             mmap_dict = load_pkl(mmap_dict_path)
#             self.mmap_dict = mmap_dict
#             self.inv_mmap = dict(zip(mmap_dict.values(), mmap_dict.keys()))
#             self.mmap = True
#         else:
#             self.mmap = False
#         self.split_jamo = split_jamo

#     def load_encoder(self, encoder_class, directory_path, encoder_filename):
#         base_name = os.path.join(directory_path, encoder_filename)
#         if encoder_class == tokenizers.BertWordPieceTokenizer:
#             vocab_name = base_name + '-vocab.txt'
#             return encoder_class(vocab_name)
#         else:
#             vocab_name = base_name + '-vocab.json'
#             merge_name = base_name + '-merges.txt'
#             return encoder_class(vocab_name, merge_name)

#     def encode(self,text):
#         if self.split_jamo:
#             text = self.tokenizer.text_to_morphs(text,True)
#         encoded = self.encoder.encode(text).ids
#         if self.mmap:
#             encoded = convert_idx_list(encoded,self.mmap_dict)
#         return encoded

#     def decode(self,indexed):
#         if self.mmap:
#             indexed = rollback_idx(indexed, self.inv_mmap)
#         decoded = self.encoder.decode(indexed)
#         if self.split_jamo:
#             decoded = self.tokenizer.morphs_to_text(decoded)
#         return decoded

