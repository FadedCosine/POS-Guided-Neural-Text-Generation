import nltk
from util.evaluate_utils import *
import pandas as pd
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluating Sampling Text')
    parser.add_argument(
            "--topw_dir",
            type=str, default="topp-0.0-topk-3-temp-1"
        )
    parser.add_argument(
            "--sample_filename",
            type=str, default="experimental3_mode-3-pos-topp-0.0-topk-20"
        )
   
    args = parser.parse_args()
    sampling_file_path = "./data/sampled/wiki103/prefix-50_nsample-100/{}".format(args.topw_dir)
    print("read : {}".format(os.path.join(sampling_file_path, args.sample_filename)))
    df = pd.read_pickle(os.path.join(sampling_file_path, args.sample_filename))
    
    n_samples = 500
    all_predict_sentences = []
    all_gt_sentence = []
    for prefix, decoded_predict, decoded_true in zip(df["prefix"], df["decoded_predict"], df["decoded_true"]):
        all_predict_sentences.append(decoded_predict)
        all_gt_sentence.append(decoded_true)
    for n in range(2,5):
        CR, NRR, Div = ngram_crnrr(all_predict_sentences, all_predict_sentences, gram_n=n)
        print('CRR in {}grams is {}'.format(n, CR))
        print('NRR in {}grams is {}'.format(n, NRR))
    
    print("pred repetition is {}".format(count_repetition(all_predict_sentences)))
    print("gt repetition is {}".format(count_repetition(all_gt_sentence)))
    self_blue = count_self_bleu_score(all_predict_sentences, n_samples)
    count_zipf = count_zipf(all_predict_sentences)
    avg_pred_stats, avg_gt_stats, avg_pred_jaccard = distinct_ngram(all_predict_sentences, all_gt_sentence)
    
    idx = 1
    for pred_distinct_n, gt_distinct_n, pred_jaccard_n in zip(avg_pred_stats, avg_gt_stats, avg_pred_jaccard):
        print('pred_distinct_{}grams is {}'.format(idx, round(pred_distinct_n,3)))
        print('gt_distinct_{}grams is {}'.format(idx, round(gt_distinct_n,3)))
        print('pred_jaccard_{}grams is {}'.format(idx, round(pred_jaccard_n,3)))
        idx += 1
    
    REPEAT_CONTEXT_LENGTHS = [16, 32, 64, 128]
    for cl in REPEAT_CONTEXT_LENGTHS:
        avg_pred_repeat, avg_pred_wrong_repeat, avg_gt_repeat = repeat_at_1(all_predict_sentences, all_gt_sentence, cl)
        print('repeat_at_1/{}: {}'.format(cl, round(avg_pred_repeat,3)))
        print('wrong_repeat_at_1/{}: {}'.format(cl, round(avg_pred_wrong_repeat,3)))
        print('human_repeat_at_1/{}: {}'.format(cl, round(avg_gt_repeat,3)))
    pred_id_set = set()
    gt_id_set = set()
    for sentence in all_predict_sentences:
        for id in sentence:
            pred_id_set.add(id)
    for sentence in all_gt_sentence:
        for id in sentence:
            gt_id_set.add(id)
    print("uniq pred : {}".format(len(pred_id_set)))
    print("uniq gt : {}".format(len(gt_id_set)))
        
       
    