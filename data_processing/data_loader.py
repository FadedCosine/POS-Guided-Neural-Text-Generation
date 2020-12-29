from torchtext.datasets import WikiText103, WikiText2, PennTreebank
import tokenizers
import os
from torchtext import data
from indexer.indexing import LMIndexer
import json
import argparse

tokenizer_savenames = {tokenizers.SentencePieceBPETokenizer:'SPBPE', tokenizers.CharBPETokenizer:'CBPE'}
tokenizer_map = {'spbpe':tokenizers.SentencePieceBPETokenizer, 'charbpe':tokenizers.CharBPETokenizer}
acc_to_full = {'wiki103':'wikitext-103', 'wiki2': 'wikitext-2', 'ptb': 'penn-treebank'}
dataset_map = {'wikitext-103':WikiText103, 'wikitext-2': WikiText2, 'penn-treebank':PennTreebank}


def get_trainpath(dataset_name):
    if 'wiki' in dataset_name:
        temp = os.path.join(acc_to_full[dataset_name], acc_to_full[dataset_name])
        path = os.path.join(temp, 'wiki.train.tokens')
        # path = os.path.join(acc_to_full[dataset_name], 'wiki.train.tokens')
    elif 'ptb' in dataset_name:
        path = os.path.join(acc_to_full[dataset_name], 'ptb.train.txt')
    return path


def prepair_dataset(root, dataset_name='wiki103', tokenizer_class=tokenizers.SentencePieceBPETokenizer, vocab_size=30000):
    assert dataset_name in ['wiki103', 'wiki2', 'ptb']
    data_class = dataset_map[acc_to_full[dataset_name]]
    data_class.download(root)
    train_path = os.path.join(root,get_trainpath(dataset_name))

    enc = LMIndexer(os.path.join(root,acc_to_full[dataset_name]),
                    '{}_{}'.format(tokenizer_savenames[tokenizer_class], vocab_size),
                    tokenizer_class, vocab_size)
    print(train_path)
    enc.learn_encoder(train_path)
    print("finish learn_encoder")
    TEXT = data.Field(tokenize=enc.encode, use_vocab=False)

    #learn index_mapper
    train, _, _ = data_class.splits(text_field=TEXT, root=root, newline_eos=False)
    enc.learn_mapper(train[0].text)
    print("finish learn index_mapper")
    #mapped encoded
    train, valid, test = data_class.splits(text_field=TEXT, root=root, newline_eos=False)
    print("finish mapped encoded")
    encoded_savepath = os.path.join(os.path.join(root,acc_to_full[dataset_name]),'encoded_{}_{}'.format(tokenizer_savenames[tokenizer_class], vocab_size))
    if not os.path.exists(encoded_savepath):
        os.makedirs(encoded_savepath)

    json.dump(train[0].text, open(os.path.join(encoded_savepath,'train'),'w'))
    json.dump(valid[0].text, open(os.path.join(encoded_savepath,'valid'),'w'))
    json.dump(test[0].text, open(os.path.join(encoded_savepath,'test'),'w'))



def get_parser():
    parser=argparse.ArgumentParser()
    parser.add_argument("--root", type=str,
                        help='parent directory path')
    parser.add_argument("--dataset", type=str, default=r"wiki103",
                        help='directory where input data is stored')
    parser.add_argument("--encoder-class", type=str, default=r"spbpe",
                        help='encoder will be stored with this name')
    parser.add_argument("--vocab-size", type=int, default=10000)
    return parser.parse_args()


if __name__ == "__main__":
    args = get_parser()
    tokenizer_class = tokenizer_map[args.encoder_class]
    print(args.vocab_size)
    prepair_dataset(args.root, args.dataset, tokenizer_class, args.vocab_size)
