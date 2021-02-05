import pandas as pd
import os
import numpy as np
import indexer.tokenizer as tokenizer
import argparse
import json
import random
import logging
logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                    datefmt='%d-%m-%Y:%H:%M:%S')
logging.getLogger().setLevel(logging.DEBUG)
logger = logging.getLogger(__name__)

def get_files(path):
    filename = []
        # Directory
    for (dirpath, _, fnames) in os.walk(path):
        for fname in fnames:
            print(fname)
            if 'iternums' not in path and fname.endswith(".json"):
                filename.append(fname)
    return filename

def get_parser():
    parser=argparse.ArgumentParser()
    parser.add_argument("--tag_tool", type=str, default="core",
                        help='parent directory path')
    parser.add_argument('--top-p',type=float)
    parser.add_argument('--top-k',type=int)
    parser.add_argument('--folderpath',type=str)
    parser.add_argument('--sample-num',type=int, default=100, help="Sample for human evaluation")
    return parser.parse_args()

def main():
    args = get_parser()

    random.seed(316)
    folderpath = os.path.join(args.folderpath, "topp-{p}-topk-{k}-temp-1".format(p=args.top_p, k=args.top_k))

    filenames = sorted(get_files(folderpath))
    print("filenames is ", filenames)
    
    filtered_indices = []
    sample_indices = []
    for filename in filenames:
        logger.info("read : {}".format(filename))
 
        src_outfile = open(os.path.join("human_evaluation_samples", filename.replace(".json", ".src")), 'w')
        tgt_outfile = open(os.path.join("human_evaluation_samples", filename.replace(".json", ".tgt")), 'w')
        hyp_outfile = open(os.path.join("human_evaluation_samples", filename.replace(".json", ".hyp")), 'w')
        
        src_list = []
        hyp_list = []
        tgt_list = []
        
        if len(filtered_indices) == 0:
            all_idx = 0
            f_src = open(os.path.join(folderpath, filename.replace(".json", "_src.txt")), "r")
            f_tgt = open(os.path.join(folderpath, filename.replace(".json", "_tgt.txt")), "r")
            f_hyp = open(os.path.join(folderpath, filename.replace(".json", "_hyp.txt")), "r")
            for line in f_src:
                prefix = line.split(" ")
                decoded_predict = f_hyp.readline().split(" ")
                decoded_true = f_tgt.readline().split(" ")
                x_unk_rate = (np.array(prefix) == "<unk>").sum() / len(prefix)
                y_unk_rate = (np.array(decoded_predict) == "<unk>").sum() / len(prefix)
                if y_unk_rate > 0 or x_unk_rate > 0:
                    all_idx += 1
                    continue
                else:
                    src_list.append(prefix)
                    hyp_list.append(decoded_predict)
                    tgt_list.append(decoded_true)
                    filtered_indices.append(all_idx)
                    all_idx += 1
            f_src.close()
            f_tgt.close()
            f_hyp.close()
        else:
            all_idx = 0
            filtered_idx = 0
            f_src = open(os.path.join(folderpath, filename.replace(".json", "_src.txt")), "r")
            f_tgt = open(os.path.join(folderpath, filename.replace(".json", "_tgt.txt")), "r")
            f_hyp = open(os.path.join(folderpath, filename.replace(".json", "_hyp.txt")), "r")

            for line in f_src:
                prefix = line.split(" ")
                decoded_predict = f_hyp.readline().split(" ")
                decoded_true = f_tgt.readline().split(" ")
                if filtered_idx >= len(filtered_indices):
                    break
                if filtered_indices[filtered_idx] == all_idx:
                    src_list.append(prefix)
                    hyp_list.append(decoded_predict)
                    tgt_list.append(decoded_true)
                    filtered_idx += 1
                all_idx += 1
            f_src.close()
            f_tgt.close()
            f_hyp.close()
        if len(sample_indices) == 0:
            sample_indices = sorted(random.sample(range(len(src_list)), args.sample_num))

        idx = 0
        sample_idx = 0
        for prefix, decoded_predict, decoded_true in zip(src_list, hyp_list, tgt_list):
            if sample_idx >= args.sample_num:
                break
            if sample_indices[sample_idx] == idx:
                sample_idx += 1
                src_outfile.write("{}: ".format(sample_idx) + " ".join(prefix))
                src_outfile.write('\n\n')
                hyp_outfile.write("{}: ".format(sample_idx) + " ".join(decoded_predict))
                hyp_outfile.write('\n\n')
                tgt_outfile.write("{}: ".format(sample_idx) + " ".join(decoded_true))
                tgt_outfile.write('\n\n')
            idx += 1
        src_outfile.close()
        tgt_outfile.close()
        hyp_outfile.close()
    

if __name__ =='__main__':
    main()


