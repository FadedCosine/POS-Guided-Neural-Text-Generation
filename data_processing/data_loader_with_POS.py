"""

使用flair标注的dirty file，以及token和POS词表（按照词频排列，高频词在前），构建encode之后的文件，以及pos2word

"""
import collections
import os
import argparse
import indexer.tokenizer as tokenizer
import logging
import json
import pickle
import numpy as np

logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                    datefmt='%d-%m-%Y:%H:%M:%S')
logging.getLogger().setLevel(logging.DEBUG)
logger = logging.getLogger(__name__)

flair_dirty_files = {
    "wikitext-103":
        {
            "test": "./data/wikitext-103/tran_xl_test_flair.pos_dirty",
            "valid": "./data/wikitext-103/tran_xl_valid_flair.pos_dirty",
            "train": "./data/wikitext-103/tran_xl_train_flair.pos_dirty",
        },
    
}
core_dirty_files = {
    "wikitext-103":
    {
        "test": "./data/wikitext-103/pos_tagged_test.txt",
        "valid": "./data/wikitext-103/pos_tagged_valid.txt",
        "train": "./data/wikitext-103/pos_tagged_train.txt",
    },
    "paraNMT":
    {
        "test": "./data/paraNMT/pos_tagged_test.txt",
        "valid": "./data/paraNMT/pos_tagged_valid.txt",
        "train": "./data/paraNMT/pos_tagged_train.txt",
    }
}

punct_marks = ["-", "--", "---", ",", ".", "?", ":", "'", '"', "!", "`", "$", "#", "...", "(", ")", "[", "]", "{", "}"]

def tagging_list_is_caption(tagging_list):
    return tagging_list[0] == '=' and  tagging_list[-2] == '='

def build_vocab_with_flair(write_path):
    """使用flair标注的dirty构建pos和token的词表

    Args:
        write_path ([string]): [写vocab的路径]
    """
    token_counter = collections.Counter()
    pos_counter = collections.Counter()
    for split, fname in flair_dirty_files.items():
        logging.info("Reading {} file.".format(split))
        dirty_file = open(fname, 'r')
        for idx, line in enumerate(dirty_file):
            token_pos_list = line.strip().split(" ")
            if len(token_pos_list) != 1 and not tagging_list_is_caption(token_pos_list):
                for token_idx in range(0, len(token_pos_list), 2):
                    token = token_pos_list[token_idx]
                    pos = token_pos_list[token_idx + 1].split(">")[0].split("<")[-1]
                    pos_counter[pos] += 1
                    token_counter[token] += 1
            if idx % 1000 == 0:
                logging.info("Finish reading {} lines.".format(idx))
        dirty_file.close()

    pos_vocab = pos_counter.most_common()
    with open(os.path.join(write_path,"flair_pos.vocab"), "w+") as f:
        for pos in pos_vocab:
            f.write("{}\n".format(pos[0]))
    token_vocab = token_counter.most_common()     
    with open(os.path.join(write_path,"flair_token.vocab"), "w+") as f:
        for token_item in token_vocab:
            token = token_item[0]
            if token != "UNK":
                f.write("{}\n".format(token[0]))


def build_vocab_with_core(args, write_path):
    """使用standford core nlp标注的dirty构建pos和token的词表

    Args:
        write_path ([string]): [写vocab的路径]
    """
    token_counter = collections.Counter()
    pos_counter = collections.Counter()
    for split, fname in core_dirty_files[args.dataset].items():
        logging.info("Reading {} file.".format(split))
        dirty_file = open(fname, 'r')
        for idx, line in enumerate(dirty_file):
            token_pos_list = line.strip().split(" ")
            # core nlp 标注的文本已经删除了标题和空行
            for token_idx in range(0, len(token_pos_list), 2):
                if args.lower:
                    token = token_pos_list[token_idx].lower()
                else:
                    token = token_pos_list[token_idx]
                pos = token_pos_list[token_idx + 1]
                pos_counter[pos] += 1
                token_counter[token] += 1
            if idx % 1000 == 0:
                logging.info("Finish reading {} lines.".format(idx))
        dirty_file.close()

    pos_vocab = pos_counter.most_common()
    with open(os.path.join(write_path,"core_pos.vocab"), "w+") as f:
        for pos in pos_vocab:
            f.write("{}\n".format(pos[0]))
    token_vocab = token_counter.most_common()     
    with open(os.path.join(write_path,"core_token.vocab"), "w+") as f:
        for token_item in token_vocab:
            token = token_item[0]
            if token != "<unk>": #词汇表中不包括 <unk>
                if token == "-LRB-":
                    token = "("
                elif token == "-RRB-":
                    token = ")"
                f.write("{}\n".format(token))

def prepare_dataset_with_flair(args, data_path, tokenizer, pos_tokenizer):
    """使用flair标注的dirty，和由词表构建的tokenizer，来把token和pos转换成token_id和pos_id，并构建pos2word

    Args:
        write_path ([string]): [写vocab的路径]
    """
    pos2word = [set() for i in range(len(pos_tokenizer.tag_vocab))]
    # tokenizer.vocab_size+2 包括了 unk 和 pad
    token_in_pos_id = np.zeros((len(pos_tokenizer.tag_vocab), tokenizer.vocab_size+2), dtype=np.int32)
 
    # token_counter 用于计算词频，在F2-softmax当中有用
    token_counter = collections.Counter()
    for split, fname in flair_dirty_files.items():
        logging.info("Prepare {} file.".format(split))
        dirty_file = open(fname, 'r')
        all_token_list = []
        all_pos_list = []
        for idx, line in enumerate(dirty_file):
            token_pos_list = line.strip().split(" ")
            if len(token_pos_list) != 1 and not tagging_list_is_caption(token_pos_list):
                for token_idx in range(0, len(token_pos_list), 2):
                    token = token_pos_list[token_idx]
                    pos = token_pos_list[token_idx + 1].split(">")[0].split("<")[-1]
                    token_id = tokenizer.convert_word_to_id(token)
                    token_counter[token_id] += 1
                    pos_id = pos_tokenizer.convert_tag_to_id(pos)
                    all_token_list.append(token_id)
                    all_pos_list.append(pos_id)
                    # 不能把unk排除在pos2word之外，因为y当中必然也存在unk
                    # if token_id != tokenizer.unk_id:
            
                    pos2word[pos_id].add(token_id)
            if idx % 1000 == 0:
                logging.info("Finish preparing {} lines.".format(idx))
        pickle.dump(all_token_list, open(os.path.join(data_path,'flair_{split}_{size}.token'.format(split=split, size=args.vocab_size)), 'wb'))
        pickle.dump(all_pos_list, open(os.path.join(data_path,'flair_{split}_{size}.pos'.format(split=split, size=args.vocab_size)), 'wb'))
    # 对于从表中存在，但是文件中却没有出现的token_id置为1
    # for id in range(tokenizer.vocab_size):
    #     if id not in token_counter:
    #         token_counter[id] = 1
    tot = 0
    cum_prob = [0]
    for i in token_counter.most_common():
        tot += i[1]
    # cum_prob中是累计的词频
    for i in token_counter.most_common():
        cum_prob.append(cum_prob[-1] + i[1] / tot)
    cum_prob.pop(0) # 移除第一个元素
    # new_dict 得到{token_id: 该token的词频从高到低的排名(从0开始)}
    token2order_dict = dict([(int(token_count[0]), int(idx)) for (idx, token_count) in enumerate(token_counter.most_common())])

    pickle.dump(cum_prob, open(os.path.join(data_path, 'flair_{size}_probs.pkl'.format(size=args.vocab_size)), 'wb'))
    pickle.dump(token2order_dict, open(os.path.join(data_path, 'flair_{size}_token2order.pkl'.format(size=args.vocab_size)), 'wb'))
    
    pos2word = np.array([np.array(list(i))  for i in pos2word])
    for pos_id, pos_i_vocab in enumerate(pos2word):
        for token_in_pos_i_id, token_id in enumerate(pos_i_vocab):
            token_in_pos_id[pos_id][token_id] = token_in_pos_i_id

    with open(os.path.join(data_path,'flair_{size}_pos2word.pkl'.format(size=args.vocab_size)), "wb") as writer:
        pickle.dump(pos2word, writer)
    with open(os.path.join(data_path,'flair_{size}_token_in_pos_id.pkl'.format(size=args.vocab_size)), "wb") as writer:
        pickle.dump(token_in_pos_id, writer)

def prepare_dataset_with_core(args, data_path, tokenizer, pos_tokenizer, dataset="wiki103", filter_bad_items=True, unk_rate=0.3, min_encode_len=10):
    """使用standford core nlp标注的dirty，和由词表构建的tokenizer，来把token和pos转换成token_id和pos_id，并构建pos2word

    Args:
        write_path ([string]): [写vocab的路径]
        filter_bad_items ([bool]): [是否过滤unk太多的数据]
    """
    # meaningful_vocab_size包括了原本的pos和eos
    pos2word = [set() for i in range(pos_tokenizer.meaningful_vocab_size)]
    # POS为 EOS的token list当中只有token为EOS这一个token
    pos2word[pos_tokenizer.eos_id].add(tokenizer.eos_id)
    # tokenizer.vocab_size 包括了 unk 和 pad，如果filter_bad_items为True，则还包括了bos和eos
    token_in_pos_id = np.zeros((pos_tokenizer.meaningful_vocab_size, tokenizer.vocab_size), dtype=np.int32)

    # token_counter 用于计算词频，在F2-softmax当中有用
    token_counter = collections.Counter()
    encode_decode_file = open("encode_decode_file_para.txt", "w")
    for split, fname in core_dirty_files[args.dataset].items():
        logging.info("Prepare {} file.".format(split))
        dirty_file = open(fname, 'r')
        all_token_list = []
        all_pos_list = []
        for idx, line in enumerate(dirty_file):
            token_pos_list = line.strip().split(" ")
            token_list = []
            pos_list = []

            for token_idx in range(0, len(token_pos_list), 2):
                if args.lower:
                    token = token_pos_list[token_idx].lower()
                else:
                    token = token_pos_list[token_idx]
                pos = token_pos_list[token_idx + 1]
                token_id = tokenizer.convert_word_to_id(token)
                token_counter[token_id] += 1
                pos_id = pos_tokenizer.convert_tag_to_id(pos)
                if dataset == "wiki103":
                    all_token_list.append(token_id)
                    all_pos_list.append(pos_id)
                elif dataset == "paraNMT":
                    token_list.append(token_id)
                    pos_list.append(pos_id)
                # 不能把unk排除在pos2word之外，因为y当中必然也存在unk
                # if token_id != tokenizer.unk_id:
                pos2word[pos_id].add(token_id)
            if dataset == "paraNMT":
                token_list = [tokenizer.bos_id] + token_list + [tokenizer.eos_id]
                pos_list = [pos_tokenizer.bos_id] + pos_list + [pos_tokenizer.eos_id]
                all_token_list.append(token_list)
                all_pos_list.append(pos_list)
            if idx % 1000 == 0:
                logging.info("Finish preparing {} lines.".format(idx))
        if dataset == "paraNMT" and filter_bad_items:
            # source sentence 或 target sentence中的unk数量大于一定的数值则过滤
            logger.info("Original all_token_list len is {}".format(len(all_token_list)))
            logger.info("Original all_pos_list len is {}".format(len(all_pos_list)))
            filtered_token_list = []
            filtered_pos_list =[]
            for idx in range(0, len(all_token_list), 2):
                if len(all_token_list[idx]) < min_encode_len or len(all_token_list[idx + 1]) < min_encode_len:
                    continue
                token_list_x = np.array(all_token_list[idx])
                token_list_y = np.array(all_token_list[idx + 1])
                x_unk_rate = (token_list_x == token_tokenizer.unk_id).sum() / len(token_list_x)
                y_unk_rate = (token_list_y == token_tokenizer.unk_id).sum() / len(token_list_y)
                if x_unk_rate > unk_rate or y_unk_rate > unk_rate :
                    continue
                else:
                    filtered_token_list.append(all_token_list[idx])
                    filtered_token_list.append(all_token_list[idx + 1])
                    filtered_pos_list.append(all_pos_list[idx])
                    filtered_pos_list.append(all_pos_list[idx + 1])
               
            all_token_list = filtered_token_list
            all_pos_list = filtered_pos_list
            logger.info("Filtered all_token_list len is {}".format(len(all_token_list)))
            logger.info("Filtered all_pos_list len is {}".format(len(all_pos_list)))
            for token_list in all_token_list:
                encode_decode_file.write(" ".join(token_tokenizer.convert_ids_to_words(token_list)) + '\n')
        pickle.dump(all_token_list, open(os.path.join(data_path,'core_{split}_{size}.token'.format(split=split, size=args.vocab_size)), 'wb'))
        pickle.dump(all_pos_list, open(os.path.join(data_path,'core_{split}_{size}.pos'.format(split=split, size=args.vocab_size)), 'wb'))
  
    tot = 0
    cum_prob = [0]
    for i in token_counter.most_common():
        tot += i[1]
    # cum_prob中是累计的词频
    for i in token_counter.most_common():
        cum_prob.append(cum_prob[-1] + i[1] / tot)
    cum_prob.pop(0) # 移除第一个元素
    # new_dict 得到{token_id: 该token的词频从高到低的排名(从0开始)}
    token2order_dict = dict([(int(token_count[0]), int(idx)) for (idx, token_count) in enumerate(token_counter.most_common())])

    pickle.dump(cum_prob, open(os.path.join(data_path, 'core_{size}_probs.pkl'.format(size=args.vocab_size)), 'wb'))
    pickle.dump(token2order_dict, open(os.path.join(data_path, 'core_{size}_token2order.pkl'.format(size=args.vocab_size)), 'wb'))
    
    pos2word = np.array([np.array(list(i))  for i in pos2word])
    for pos_id, pos_i_vocab in enumerate(pos2word):
        for token_in_pos_i_id, token_id in enumerate(pos_i_vocab):
            token_in_pos_id[pos_id][token_id] = token_in_pos_i_id

    with open(os.path.join(data_path,'core_{size}_pos2word.pkl'.format(size=args.vocab_size)), "wb") as writer:
        pickle.dump(pos2word, writer)
    with open(os.path.join(data_path,'core_{size}_token_in_pos_id.pkl'.format(size=args.vocab_size)), "wb") as writer:
        pickle.dump(token_in_pos_id, writer)

def get_parser():
    parser=argparse.ArgumentParser()
    parser.add_argument("--tag_tool", type=str, default="core",
                        help='parent directory path')
    parser.add_argument("--lower", action="store_true", help="If lower the text")
    parser.add_argument("--vocab_size", type=int, default=270000)
    parser.add_argument("--dataset", type=str, default="wikitext-103")
    return parser.parse_args()


if __name__ == '__main__':
    args = get_parser()
    data_path = "./data/{}/".format(args.dataset)
    VOCAB_PATH = {
        "flair" : (data_path + "flair_token.vocab", data_path + "flair_pos.vocab"),
        "core" : (data_path + "core_token.vocab", data_path + "core_pos.vocab")
    }
    BUILD_VOCAB_FUNCTION = {
        "flair" : build_vocab_with_flair,
        "core" : build_vocab_with_core
    }
    PREPARE_DATASET_FUNCTION = {
        "flair" : prepare_dataset_with_flair,
        "core" : prepare_dataset_with_core
    }
    token_vocab_path, pos_vocab_path = VOCAB_PATH[args.tag_tool]
    build_vocab = BUILD_VOCAB_FUNCTION[args.tag_tool]
    prepare_dataset = PREPARE_DATASET_FUNCTION[args.tag_tool]
    if not os.path.exists(token_vocab_path) or not os.path.exists(pos_vocab_path):
        logger.info("build vocab")
        build_vocab(args, data_path)
 
    logger.info("require vocab size is {}".format(args.vocab_size))
    if args.dataset == "wiki103":
        add_special_token= False
    elif args.dataset == "paraNMT":
        add_special_token = True
    token_tokenizer = tokenizer.TokenTokenizer(token_vocab_path, args.vocab_size, add_special_token=add_special_token)
    pos_tokenizer = tokenizer.POSTokenizer(pos_vocab_path,add_special_token=add_special_token)
    prepare_dataset(args, data_path, token_tokenizer, pos_tokenizer, dataset=args.dataset, filter_bad_items=True, unk_rate=0.1)
    
    
    

    



    
