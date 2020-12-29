import os
import pandas as pd
import collections
import re
import pickle
from basic_util.files import *
import argparse


def count(fl, targets=['input_context'], checks=['input_keyword'], vocab_size=10000):
    cnter = collections.Counter()
    s = set()
    for filename in fl:
        cur_df = pd.read_pickle(filename)
        for target in targets:
            texts = cur_df[target].tolist()
            for i in texts:
                cnter.update(i[1:])
                s.add(i[0])
    #check
    for filename in fl:
        cur_df = pd.read_pickle(filename)
        for check in checks:
            texts = cur_df[check].tolist()
            for i in texts:
                s.update(i)
    for i in s:
        if i not in cnter:
            cnter[i] = 1
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


def convert_idx(filename, dic, targets:list):
    key_type = type(list(dic)[0])
    cur_df = pd.read_pickle(filename)
    for target in targets:
        new = []
        for line in cur_df[target].tolist():
            converted = []
            for token in line:
                if key_type(token) in dic:
                    converted.append(dic[key_type(token)])
                else:
                    converted.append(len(dic)-1)
            new.append(converted)
        cur_df[target] = new
    return cur_df


class IMap:
    def __init__(self, dir_path, prefix):
        self.probs_path, self.dic_path, self.file_path = self.get_path(dir_path, prefix)
        self.vocab_size = self.get_vocab_size(prefix)
        self.dic = self.load_dic(self.dic_path)

    @staticmethod
    def load_dic(path):
        if os.path.exists(path):
            return load_pkl(path)
        else:
            return None

    @staticmethod
    def get_vocab_size(prefix):
        target = prefix.split('_')[-1]
        try:
            return int(target)
        except ValueError:
            print('invalid prefix format vocab size set to 20000')
            return 20000

    @staticmethod
    def get_path(dir_path, prefix):
        probs_path = os.path.join(dir_path, '{}_probs.json'.format(prefix))
        dic_path = os.path.join(dir_path, '{}_dic.json'.format(prefix))
        file_path = os.path.join(dir_path,'{}_indexed'.format(prefix))
        return probs_path, dic_path, file_path

    def learn_dic(self, count_name, check_names):
        if not self.dic:
            print('start imap')
            probs, dic = count(get_files(self.file_path),count_name,check_names,self.vocab_size)
            self.dic = dic
            pickle.dump(probs, open(self.probs_path, 'wb'))
            pickle.dump(dic, open(self.dic_path, 'wb'))
        else:
            print('imap exists')

    def convert_and_save(self, targets:list):
        fl = get_files(self.file_path)
        print(fl)
        if not self.dic:
            raise ValueError('dictionary is empty')
        for filename in fl:
            cur_df = convert_idx(filename,self.dic, targets)
            new_filename = re.sub(r'indexed/','indexed_new/',filename)
            if not os.path.exists(os.path.dirname(new_filename)):
                os.makedirs(os.path.dirname(new_filename))
            cur_df.to_pickle(new_filename)


def get_parser():
    parser=argparse.ArgumentParser()
    parser.add_argument("--base-name", type=str,
                        help='parent directory path')
    parser.add_argument("--dir-path", type=str,
                        help='directory where input data is stored')
    parser.add_argument("--count-names", type=str, nargs='*')
    parser.add_argument("--check-names", type=str, nargs='*')
    parser.add_argument("--convert-names", type=str, nargs='*')
    return parser


if __name__ =='__main__':
    parser = get_parser()
    args = parser.parse_args()
    imap = IMap(args.dir_path, args.base_name)
    imap.learn_dic(args.count_names, args.check_names)
    imap.convert_and_save(args.convert_names)