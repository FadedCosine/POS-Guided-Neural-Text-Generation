import pandas as pd
import os
import indexer.tokenizer as tokenizer
import argparse
import json

def get_parser():
    parser=argparse.ArgumentParser()
    parser.add_argument("--tag_tool", type=str, default="core",
                        help='parent directory path')
    parser.add_argument("--vocab_size", type=int, default=100000)
    parser.add_argument("--dataset", type=str, default="paraNMT")
    parser.add_argument(
            "--topw_dir",
            type=str, default="topp-0.0-topk-3-temp-1"
        )
    parser.add_argument(
            "--sample_filename",
            type=str, default="experimental3_mode-3-pos-topp-0.0-topk-20"
        )
    return parser.parse_args()

if __name__ == '__main__':
    args = get_parser()
    data_path = "./data/{}/".format(args.dataset)
    VOCAB_PATH = {
        "flair" : (data_path + "flair_token.vocab", data_path + "flair_pos.vocab"),
        "core" : (data_path + "core_token.vocab", data_path + "core_pos.vocab")
    }
    token_vocab_path, pos_vocab_path = VOCAB_PATH[args.tag_tool]
    sampled_file_path = "./data/sampled/{}/prefix-50_nsample-100/{}".format(args.dataset, args.topw_dir)

    enc_file_name = args.sample_filename
    df = pd.read_pickle(os.path.join(sampled_file_path, enc_file_name))
    if args.dataset == "wiki103":
        add_special_token= False
    elif args.dataset == "paraNMT":
        add_special_token = True
    token_tokenizer = tokenizer.TokenTokenizer(token_vocab_path, args.vocab_size, add_special_token=add_special_token)
    pos_tokenizer = tokenizer.POSTokenizer(pos_vocab_path, add_special_token=add_special_token)
    outfile = open(os.path.join(sampled_file_path, enc_file_name + ".json"), "w")
    for prefix, decoded_predict, decoded_true in zip(df["prefix"], df["decoded_predict"], df["decoded_true"]):
        o = {}
        o["prefix"] = " ".join(token_tokenizer.convert_ids_to_words(prefix))
        print("decoded_predict is ", decoded_predict)
        o["decoded_predict"] = " ".join(token_tokenizer.convert_ids_to_words(decoded_predict))
        o["decoded_true"] = " ".join(token_tokenizer.convert_ids_to_words(decoded_true))
        print(json.dumps(o), file=outfile, flush=True)
 

