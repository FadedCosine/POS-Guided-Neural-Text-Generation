import nltk
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
import random
import spacy
import math
import os
import json
import numpy as np
from functools import partial
from multiprocessing.pool import Pool
from collections import Counter
from scipy import stats
import operator
from nltk import ngrams
from collections import defaultdict, Counter
import torch

import logging
logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                    datefmt='%d-%m-%Y:%H:%M:%S')
logging.getLogger().setLevel(logging.DEBUG)
logger = logging.getLogger(__name__)

def bleu_i(weights, all_sentences, smoothing_function, i):
    # noinspection PyTypeChecker
    return sentence_bleu(
        references=all_sentences[:i] + all_sentences[i + 1:],
        hypothesis=all_sentences[i],
        weights=weights,
        smoothing_function=smoothing_function)

def count_self_bleu_score(all_sentences, n_sample, max_n = 5):
    if n_sample > len(all_sentences):
        n_sample = len(all_sentences) / 3
    random.seed(0)
    nlp = spacy.load('en_core_web_sm', disable=['parser', 'tagger', 'ner'])
    nlp.add_pipe(nlp.create_pipe('sentencizer'))

    smoothing_function = SmoothingFunction().method1
    bleu_scores = []
    with Pool(processes=os.cpu_count()) as pool:
        for n_gram in range(1, max_n+1):
            if n_gram == 1:
                weights = (1.0, 0, 0, 0)
            elif n_gram == 2:
                weights = (0.5, 0.5, 0, 0)
            elif n_gram == 3:
                weights = (1.0 / 3, 1.0 / 3, 1.0 / 3, 0)
            elif n_gram == 4:
                weights = (0.25, 0.25, 0.25, 0.25)
            elif n_gram == 5:
                weights = (0.2, 0.2, 0.2, 0.2, 0.2)
            else:
                raise ValueError
            bleu_scores.append(
                list(
                    pool.imap_unordered(
                        partial(bleu_i, weights, all_sentences, smoothing_function),
                        random.sample(range(len(all_sentences)), n_sample))
                ))
            logger.info(f"\nself bleu-{n_gram} = {sum(bleu_scores[n_gram - 1]) / n_sample}")
     
    return [sum(bleu_scores[idx]) / n_sample for idx in range(max_n)]

def count_zipf(all_sentences):
    cnt = Counter()
    logging.info("zipf value\tregression r value \tregression p value")
    N = len(all_sentences)
    for sentence in all_sentences:
        cnt.update(sentence)
    xs = np.arange(1, min(len(cnt), N)+1)
    ys = np.array(sorted(cnt.values(), key=operator.neg)[:N])
    a, b, r, p, std = stats.linregress(np.log(xs), np.log(ys))
    logging.info(f"{-a}\t{-r}\t{p}")
    return -a

def count_repetition(all_sentences, max_n=100):
   
    n_repeated_examples = 0
    for gen in all_sentences:
        rev_gen = list(reversed(gen))
        last_n_repeats = [0] * max_n

        for n in range(1, max_n + 1):
            n_repeat = 1
            while len(rev_gen[n*n_repeat:n*(n_repeat+1)]) == n and \
                    rev_gen[n*n_repeat:n*(n_repeat+1)] == rev_gen[:n]:
                n_repeat += 1
            last_n_repeats[n - 1] = n_repeat
        max_repeated_n = max(range(max_n), key=lambda x: last_n_repeats[x])

        if last_n_repeats[max_repeated_n] > 1 and (max_repeated_n+1 >= 3 or last_n_repeats[max_repeated_n] > 50):
            n_repeated_examples += 1
    logger.info("number of repeating examples: {}".format(n_repeated_examples))
    return n_repeated_examples


def distinct_ngram_(pred_token_list, tgt_token_list, lb=1, gb=5):
    # logger.info("token_list is {}".format(token_list))
    pred_stats, gt_stats, ja_stats = [], [], []
    for n in range(lb, gb):
        pred_ngs = [ng for ng in ngrams(pred_token_list, n)]
        pred_set = set(pred_ngs)
        tgt_ngs = [ng for ng in ngrams(tgt_token_list, n)]
        tgt_set = set(tgt_ngs)
        if len(pred_ngs) == 0:
            print("len of pred_ngs is 0")
            pred_stats.append(0)
        else:
            pred_stats.append(len(pred_set)/len(pred_ngs))
        if len(tgt_ngs) == 0:
            print("len of tgt_ngs is 0")
            gt_stats.append(0)
        else:
            gt_stats.append(len(tgt_set)/len(tgt_ngs))
        union_set = pred_set | tgt_set
        if len(union_set) == 0:
            print("len of union set is 0")
            ja_stats.append(0)
        else:
            ja_stats.append(len(pred_set & tgt_set) / len(union_set))

    return pred_stats, gt_stats, ja_stats

def distinct_ngram(all_sentences, all_gt_sentences):
    all_pred_stats, all_gt_stats = [], []
    all_pred_jaccard = []
    for pred_sentence, gt_sentence in zip(all_sentences, all_gt_sentences):
        pred_stats, gt_stats, pred_jaccard = distinct_ngram_(pred_sentence, gt_sentence)
        all_pred_stats.append(pred_stats)
        all_gt_stats.append(gt_stats)
        all_pred_jaccard.append(pred_jaccard)
    all_pred_stats = np.array(all_pred_stats)
    all_gt_stats = np.array(all_gt_stats)
    all_pred_jaccard = np.array(all_pred_jaccard)
    _, n = all_pred_stats.shape
    avg_pred_stats = [np.mean(all_pred_stats[:, i]) for i in range(n)]
    avg_gt_stats = [np.mean(all_gt_stats[:, i]) for i in range(n)]
    avg_pred_jaccard = [np.mean(all_pred_jaccard[:, i]) for i in range(n)]
    return avg_pred_stats, avg_gt_stats, avg_pred_jaccard

def get_dic(lines, gram_n, pad=None):
    """
    得到lineszhong维度为gram_n的n gram的标准化分布
    """
    ngram_dic = {}
    for blocks in lines:
        blocks_ngrams = ngrams(blocks, gram_n)
     
        for gram in blocks_ngrams:
            # print("gram is ", gram)
            if gram in ngram_dic:
                ngram_dic[gram] += 1
            else:
                ngram_dic[gram] = 1
    total = 0.0
    for k,v in ngram_dic.items():
        total += v
    for k,v in ngram_dic.items():
        ngram_dic[k] = ngram_dic[k] / total
    return ngram_dic

def ngram_crnrr(predictions, targets, number=None, gram_n=3, n_part=1):
    ref_grams = get_dic(predictions, gram_n)
    if number is None:
        hypo_sents_all = predictions
    else:
        hypo_sents_all = predictions[:number]
    assert len(hypo_sents_all) % n_part == 0
    cr_all = []
    nrr_all = []
    base_all = []
    for k in range(n_part):
        part_size = len(hypo_sents_all) / n_part
  
        hypo_sents = hypo_sents_all[int(k*part_size) : int((k+1)*part_size)]
        hypo_grams = get_dic(hypo_sents, gram_n)
        cr = 0.0
        nrr = 0.0
        for k,v in hypo_grams.items():
            if k in ref_grams:
                cr += (v * ref_grams[k])
            nrr -= (v ** 2)
        base = 0.0
        for k,v in ref_grams.items():
            base += (v ** 2)
        cr_all.append(cr)
        nrr_all.append(nrr)
        base_all.append(base)

    CR = np.mean(cr_all)
    NRR = np.mean(nrr_all)
    Div = np.mean(base_all) - 2*CR - NRR

    return CR, NRR, Div

def repeat_at_1_(predictions, targets, context_length):
  
    # predictions : (batch_size, seq_len)
    # targets :  (batch_size, seq_len)
    predictions = torch.Tensor(predictions).unsqueeze(0)
    targets = torch.Tensor(targets).unsqueeze(0)
    batch_size = predictions.size(0)
    T = targets.size(1)
    # print("predictions : ", predictions.size())
    if predictions.size(1) != T:
        return 0, 0, 0
    targets = targets.unsqueeze(1)
    
    # targets :  (batch_size, 1, seq_len)
    # T x T where prev_targets[t, :] = [y_1,...,y_t-1, -1, -1,..., -1]
    prev_targets = targets.expand(batch_size, T, T).tril().masked_fill_(torch.ones_like(targets.expand(batch_size, T, T)).byte().triu().bool(), -1)

    # each row t is [-1, ..., -1, y_{t-k-1}, ..., y_{t-1}, -1, ..., -1] where k is context length
    prev_targets = prev_targets.masked_fill_(torch.ones_like(targets.expand(batch_size, T, T)).byte().tril(-(context_length+1)).bool(), -1)
    repeat_at_1 = (predictions.unsqueeze(-1) == prev_targets)
    has_repeat_at_1 = repeat_at_1.sum(1).gt(0)
    total_repeat_at_1 = has_repeat_at_1.sum().float() / predictions.numel()
    is_incorrect = (predictions != targets.contiguous().view(batch_size, -1)).view(batch_size, -1, 1)
    total_wrong_repeat_at_1 = ((repeat_at_1 * is_incorrect).sum(1).gt(0)).sum().float() / predictions.numel()

    total_human_repeat_at_1 = (prev_targets == targets.view(batch_size, T, 1)).sum(1).gt(0).sum().float() / targets.numel()
        # 分别表示在context len的上文中，所预测的单词重复的数量、重复的单词中错误预测的数量和ground truth单词中重复的数量
    return total_repeat_at_1.item(), total_wrong_repeat_at_1.item(), total_human_repeat_at_1.item()


def repeat_at_1(all_sentences, all_gt_sentence, context_length):
    all_pred_repeat, all_pred_wrong_repeat, all_gt_repeat = [], [], []
    for pred_sentence, gt_sentence in zip(all_sentences, all_gt_sentence):
        total_repeat_at_1, total_wrong_repeat_at_1, total_human_repeat_at_1 = repeat_at_1_(pred_sentence, gt_sentence, context_length)
        if total_repeat_at_1 != 0:
            all_pred_repeat.append(total_repeat_at_1)
            all_pred_wrong_repeat.append(total_wrong_repeat_at_1)
            all_gt_repeat.append(total_human_repeat_at_1)

    avg_pred_repeat = np.mean(all_pred_repeat)
    avg_pred_wrong_repeat = np.mean(all_pred_wrong_repeat)
    avg_gt_repeat = np.mean(all_gt_repeat)
    return avg_pred_repeat, avg_pred_wrong_repeat, avg_gt_repeat
