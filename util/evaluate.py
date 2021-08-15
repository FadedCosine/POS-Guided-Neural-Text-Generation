import re
from itertools import chain
import collections
import math
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction, ngrams, brevity_penalty
from collections import Counter
from fractions import Fraction
from .wer import *
import numpy as np
from rouge import Rouge
import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def words_dist(cum_probs, texts):
    def compute_boundaries(word, indices):
        for idx, target in enumerate(indices):
            if word < target:
                return idx
        return len(indices)
    def get_indices(cum_prob):
        x = [0.4, 0.7, 0.9]
        cur = 0
        res = []
        for i in x:
            while cum_prob[cur] < i:
                cur += 1
            res.append(cur)
        return res

    indices = get_indices(cum_probs)
    res = [0, 0, 0, 0]
    for text in texts:
        for word in text:
            res[compute_boundaries(word, indices)] += 1
    return np.array(res) / sum(res)

def repetition(hyps):
    max_n = 100
    n_repeated_examples = 0
    for obj in hyps:
        gen = obj
        rev_gen = list(reversed(gen))
        last_n_repeats = [0] * max_n

        for n in range(1, max_n + 1):
            n_repeat = 1
            while len(rev_gen[n * n_repeat:n * (n_repeat + 1)]) == n and \
                    rev_gen[n * n_repeat:n * (n_repeat + 1)] == rev_gen[:n]:
                n_repeat += 1
            last_n_repeats[n - 1] = n_repeat
        max_repeated_n = max(range(max_n), key=lambda x: last_n_repeats[x])

        if last_n_repeats[max_repeated_n] > 1 and (max_repeated_n + 1 >= 3 or last_n_repeats[max_repeated_n] > 50):
            n_repeated_examples += 1
    return n_repeated_examples / len(hyps)


def pad_sequence(sequence, n, pad_left=False, pad_right=False,
                 left_pad_symbol=None, right_pad_symbol=None):
    """
    Returns a padded sequence of items before ngram extraction.
        list(pad_sequence([1,2,3,4,5], 2, pad_left=True, pad_right=True, left_pad_symbol='<s>', right_pad_symbol='</s>'))
        ['<s>', 1, 2, 3, 4, 5, '</s>']
        list(pad_sequence([1,2,3,4,5], 2, pad_left=True, left_pad_symbol='<s>'))
        ['<s>', 1, 2, 3, 4, 5]
        list(pad_sequence([1,2,3,4,5], 2, pad_right=True, right_pad_symbol='</s>'))
        [1, 2, 3, 4, 5, '</s>']
    :param sequence: the source humelo to be padded
    :type sequence: sequence or iter
    :param n: the degree of the ngrams
    :type n: int
    :param pad_left: whether the ngrams should be left-padded
    :type pad_left: bool
    :param pad_right: whether the ngrams should be right-padded
    :type pad_right: bool
    :param left_pad_symbol: the symbol to use for left padding (default is None)
    :type left_pad_symbol: any
    :param right_pad_symbol: the symbol to use for right padding (default is None)
    :type right_pad_symbol: any
    :rtype: sequence or iter
    """
    sequence = iter(sequence)
    if pad_left:
        sequence = chain((left_pad_symbol,) * (n - 1), sequence)
    if pad_right:
        sequence = chain(sequence, (right_pad_symbol,) * (n - 1))
    return sequence


def ngrams(sequence, n, pad_left=False, pad_right=False,
           left_pad_symbol=None, right_pad_symbol=None):
    """
    Return the ngrams generated from a sequence of items, as an iterator.
    For example:
        from nltk.basic_util import ngrams
            list(ngrams([1,2,3,4,5], 3))
        [(1, 2, 3), (2, 3, 4), (3, 4, 5)]
    Wrap with list for a list version of this function.  Set pad_left
    or pad_right to true in order to get additional ngrams:
            list(ngrams([1,2,3,4,5], 2, pad_right=True))
        [(1, 2), (2, 3), (3, 4), (4, 5), (5, None)]
            list(ngrams([1,2,3,4,5], 2, pad_right=True, right_pad_symbol='</s>'))
        [(1, 2), (2, 3), (3, 4), (4, 5), (5, '</s>')]
            list(ngrams([1,2,3,4,5], 2, pad_left=True, left_pad_symbol='<s>'))
        [('<s>', 1), (1, 2), (2, 3), (3, 4), (4, 5)]
            list(ngrams([1,2,3,4,5], 2, pad_left=True, pad_right=True, left_pad_symbol='<s>', right_pad_symbol='</s>'))
        [('<s>', 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, '</s>')]
    :param sequence: the source humelo to be converted into ngrams
    :type sequence: sequence or iter
    :param n: the degree of the ngrams
    :type n: int
    :param pad_left: whether the ngrams should be left-padded
    :type pad_left: bool
    :param pad_right: whether the ngrams should be right-padded
    :type pad_right: bool
    :param left_pad_symbol: the symbol to use for left padding (default is None)
    :type left_pad_symbol: any
    :param right_pad_symbol: the symbol to use for right padding (default is None)
    :type right_pad_symbol: any
    :rtype: sequence or iter
    """

    sequence = pad_sequence(sequence, n, pad_left, pad_right,
                            left_pad_symbol, right_pad_symbol)

    history = []
    while n > 1:
        history.append(next(sequence))
        n -= 1
    for item in sequence:
        history.append(item)
        yield tuple(history)
        del history[0]


def rogue(hyps, refs, avg=True):
    hyps = [' '.join(map(str, i)) for i in hyps]
    refs = [' '.join(map(str, i)) for i in refs]
    rouge = Rouge()
    scores = rouge.get_scores(hyps, refs, avg=avg)
    return [scores['rouge-1']['f'], scores['rouge-2']['f'], scores['rouge-l']['f']]

def self_wer(reference, hypothesis):
    score = 0.0
    for refer, hypo in zip(reference, hypothesis):
        score += wer(refer, hypo)   
    return score / len(reference)
    
def distinct_n_sentence_level(sentence, n):
    """
    Compute distinct-N for a single sentence.
    :param sentence: a list of words.
    :param n: int, ngram.
    :return: float, the metric value.
    """
    if len(sentence) == 0:
        return 0.0  # Prevent a zero division
    distinct_ngrams = set(ngrams(sentence, n))
    return len(distinct_ngrams) / len(sentence)


def bleu_upto(reference, hypothesis, n_gram):
    res = []
    for i in range(1,n_gram+1):
        res.append(calc_bleu_ngram(reference, hypothesis, i))
    return res

def calc_bleu_ngram(reference, hypothesis, n_gram):
    score = 0.0
    ratio = 1 / n_gram

    cc = SmoothingFunction()

    for refer, hypo in zip(reference, hypothesis):
        # refer.index()
        score += sentence_bleu([refer], hypo, (ratio,) * n_gram, cc.method1)

    return score / len(reference)


def distinct_upto(sentences, n):
    sentences = [i for i in sentences if len(i) > 5]
    res = []
    for i in range(1,n+1):
        res.append(distinct_n_corpus_level(sentences, i))
    return res


def distinct_n_corpus_level(sentences, n):
    """
    Compute average distinct-N of a list of sentences (the corpus).
    :param sentences: a list of sentence.
    :param n: int, ngram.
    :return: float, the average value.
    """
    

    return sum(distinct_n_sentence_level(sentence, n) for sentence in sentences) / len(sentences)


def bleu_single(reference,hypothesis,n_gram):
    ratio=1/n_gram
    cc = SmoothingFunction()
    return sentence_bleu([reference],hypothesis,(ratio,)*n_gram,cc.method1)


def bleu_multiples(references,hypothesis,n_gram):
    ratio=1/n_gram
    score = 0
    cnt = 0
    for i in hypothesis:
        score += sentence_bleu(references,i,(ratio,)*n_gram)
        cnt += 1
    return score / cnt


def count(x, n_gram):
    cnter = collections.Counter()
    for line in x:
        ngram_res = []
        temp = [-1] * (n_gram - 1) + line + [-1] * (n_gram - 1)
        for i in range(len(temp) + n_gram - 1):
            ngram_res.append(str(temp[i:i + n_gram]))
        cnter.update(ngram_res)
    return cnter

from collections import defaultdict,Counter
from nltk import ngrams

def ngram_metrics(token_list, pad=30001):
    if pad in token_list:
        token_list = token_list[:token_list.index(pad)]  # remove possible padding
    stats = defaultdict(float)
    for n in range(1, 5):
        ngs = [ng for ng in ngrams(token_list, n)]
        counter = Counter([ng for ng in ngrams(token_list, n)])
        stats['pct_repeat_%dgrams' % n] = 1.0 - len(counter) / len(ngs)
    return stats


def seq_rep_n(corpus):

    score=0.0
    total_n=len(corpus)

    for token_list in corpus:
        score+=ngram_metrics(token_list)["pct_repeat_4grams"]

    return score/total_n

def compute_probs(cnter,token_lists):
    tot = 0
    probs = []
    for i in cnter:
        tot+= cnter[i]
    for i in token_lists:
        if i in cnter:
            probs.append(cnter[i] / tot)
        else:
            probs.append(1e-10)
    return np.array(probs)


def kld(references, hypotheses, n_gram):

    r_cnter = count(references,n_gram)
    h_cnter = count(hypotheses,n_gram)

    s = set(r_cnter.keys())
    s.update(h_cnter.keys())
    s = list(s)
    r_probs = compute_probs(r_cnter, s)
    h_probs = compute_probs(h_cnter, s)
    kld = np.sum(r_probs * np.log(r_probs/h_probs))
    return kld


def entropy(x):
    cnter = collections.Counter()
    for line in x:
        cnter.update(line)
    tot = 0
    prob=[]
    for i in cnter:
        tot += cnter[i]
    for i in cnter:
        prob.append(cnter[i]/tot)
    prob = np.array(prob)
    ent = np.sum(prob * np.log(prob))
    return -ent


def ms_jaccard(ref,hyp,n_gram):
    res = []
    for i in range(1,1+n_gram):
        rc = count(ref,i)
        hc = count(hyp,i)
        n_gram_set = set(rc.keys())
        n_gram_set.update(hc.keys())
        rprob= compute_probs(rc,n_gram_set)
        hprob= compute_probs(hc,n_gram_set)
        numerator = np.sum(np.minimum(rprob,hprob))
        denominator = np.sum(np.maximum(rprob,hprob))
        res.append(numerator / denominator)
    score = []
    for i in range(1,1+n_gram):
        score.append(geo_mean(res[:i]))
    return score


class Refcnts:
    def __init__(self,references,n):
        self.ref_mcnts = {i: ref_cnts1(references, i) for i in range(1, n + 1)}
        self.ref_lens = [len(i) for i in references]
        self.n = n

    def bleu(self, hypothesis):
        bleu_scores = {i: [] for i in range(1, self.n + 1)}
        for hyp in hypothesis:
            # print(p_denominators,p_numerators)
            p_numerators = Counter()  # Key = ngram order, and value = no. of ngram matches.
            p_denominators = Counter()  # Key = ngram order, and value = no. of ngram in ref.
            for i in range(1, self.n + 1):
                p_i = modified_precision(self.ref_mcnts[i], hyp, i)
                # print(p_i)
                p_numerators[i] = p_i.numerator
                p_denominators[i] = p_i.denominator
            hyp_len = len(hyp)
            ref_len = closest_ref_length(iter(self.ref_lens), hyp_len)
            bp = brevity_penalty(ref_len, hyp_len)
            for i in range(1, self.n + 1):
                if p_numerators[i] == 0: p_numerators[i] = 1e-100
                s = (1 / i * math.log(p_numerators[j] / p_denominators[j]) for j in range(1, i + 1))
                s = bp * math.exp(math.fsum(s))
                bleu_scores[i].append(s)

        return [np.mean(bleu_scores[i]) for i in range(1, self.n + 1)]

    def ms_jaccard(self, ref, hyp, n_gram):
        rc = count(ref, n_gram)
        hc = count(hyp, n_gram)
        n_gram_set = set(rc.keys())
        n_gram_set.update(hc.keys())
        rprob = compute_probs(rc, n_gram_set)
        hprob = compute_probs(hc, n_gram_set)
        numerator = np.sum(np.minimum(rprob, hprob))
        denominator = np.sum(np.maximum(rprob, hprob))
        return numerator / denominator


def build_refcnts(references,n):
    ref_mcnts = {i:ref_cnts1(references, i) for i in range(1,n+1)}
    ref_lens = [len(i) for i in references]
    return ref_mcnts, ref_lens


def bleu(ref_mcnts, ref_lens, hypothesis, n):
    # print(ref_mcnts)
    # numerator, denominator = 0, 0
    bleu_scores = {i:[] for i in range(1,n+1)}
    for hyp in hypothesis:
        # print(p_denominators,p_numerators)
        p_numerators = Counter()  # Key = ngram order, and value = no. of ngram matches.
        p_denominators = Counter()  # Key = ngram order, and value = no. of ngram in ref.
        for i in range(1,n+1):
            p_i = modified_precision(ref_mcnts[i], hyp, i)
            # print(p_i)
            p_numerators[i] = p_i.numerator
            p_denominators[i] = p_i.denominator
        hyp_len = len(hyp)
        ref_len = closest_ref_length(iter(ref_lens), hyp_len)
        bp = brevity_penalty(ref_len, hyp_len)
        for i in range(1,n+1):
            if p_numerators[i] == 0: p_numerators[i] = 1e-100
            s = (1/i * math.log(p_numerators[j] / p_denominators[j]) for j in range(1,i+1))
            s = bp * math.exp(math.fsum(s))
            bleu_scores[i].append(s)

    return [np.mean(bleu_scores[i]) for i in range(1,n+1)]


def selfbleu(x, n):
    x_mcnts = {i: ref_cnts2(x, i) for i in range(1, n + 1)}
    x_lens = [len(i) for i in x]
    bleu_scores = {i:[] for i in range(1,n+1)}
    for idx, hyp in enumerate(x):
        p_numerators = Counter()  # Key = ngram order, and value = no. of ngram matches.
        p_denominators = Counter()  # Key = ngram order, and value = no. of ngram in ref.
        for i in range(1, n + 1):
            p_i = modified_precision(x_mcnts[i], hyp, i,True)
            p_numerators[i] = p_i.numerator
            p_denominators[i] = p_i.denominator
        hyp_lengths = len(hyp)
        ref_lengths = closest_ref_length(iter(x_lens[:idx] + x_lens[idx+1:]), hyp_lengths)
        bp = brevity_penalty(ref_lengths, hyp_lengths)
        for i in range(1,n+1):
            if p_numerators[i] == 0: p_numerators[i] = 1e-100
            s = (1 / i * math.log(p_numerators[j] / p_denominators[j]) for j in range(1, i + 1))
            s = bp * math.exp(math.fsum(s))
            bleu_scores[i].append(s)
    return [np.mean(bleu_scores[i]) for i in range(1,n+1)]

#
# def selfbleu(x,n):
#     logits = []
#     bleu_scores = []
#     for i in range(1,n+1):
#         logit = selfbleu_logit(x,i)
#         logits.append(logit)
#         bleu_score = geo_mean(logits)
#         bleu_scores.append(bleu_score)
#     return bleu_scores

def geo_mean(iterable):
    a = np.array(iterable)
    return a.prod()**(1.0/len(a))


def ref_cnts1(references,n):
    ref_mcnts = {}
    for reference in references:
        reference_counts = (
            Counter(ngrams(reference, n)) if len(reference) >= n else Counter()
        )
        for i in reference_counts:
            if i not in ref_mcnts: ref_mcnts[i] = reference_counts[i]
            elif ref_mcnts[i] < reference_counts[i]: ref_mcnts[i] = reference_counts[i]
    return ref_mcnts


def ref_cnts2(references,n):
    ref_mcnts = {}
    for reference in references:
        reference_counts = (
            Counter(ngrams(reference, n)) if len(reference) >= n else Counter()
        )
        for i in reference_counts:
            if i not in ref_mcnts: ref_mcnts[i] = [reference_counts[i],0]
            elif ref_mcnts[i][-1] < reference_counts[i]:
                if ref_mcnts[i][0] < reference_counts[i]:
                    ref_mcnts[i] = [reference_counts[i],ref_mcnts[i][0]]
                else:
                    ref_mcnts[i][-1] = reference_counts[i]
    return ref_mcnts


def modified_precision(ref_mcnts, hypothesis,n, isself=False):
    counts = Counter(ngrams(hypothesis, n)) if len(hypothesis) >= n else Counter()
    hyp_mcnts = {}
    for ngram in counts:
        if ngram in ref_mcnts: hyp_mcnts[ngram] = ref_mcnts[ngram]
        else : hyp_mcnts[ngram] = 0
    if isself:
        clipped_counts = {
            ngram: min(count, ref_mcnts[ngram][1]) if count == ref_mcnts[ngram][0] else min(count, ref_mcnts[ngram][0])
            for ngram, count in counts.items()
        }
    else:
        clipped_counts = {
            ngram: min(count, ref_mcnts.get(ngram,0)) for ngram, count in counts.items()
        }

    numerator = sum(clipped_counts.values())
    # Ensures that denominator is minimum 1 to avoid ZeroDivisionError.
    # Usually this happens when the ngram order is > len(reference).
    denominator = max(1, sum(counts.values()))

    return Fraction(numerator, denominator, _normalize=False)


def closest_ref_length(ref_lens, hyp_len):
    """
    This function finds the reference that is the closest length to the
    hypothesis. The closest reference length is referred to as *r* variable
    from the brevity penalty formula in Papineni et. al. (2002)

    :param references: A list of reference translations.
    :type references: list(list(str))
    :param hyp_len: The length of the hypothesis.
    :type hyp_len: int
    :return: The length of the reference that's closest to the hypothesis.
    :rtype: int
    """
    closest_ref_len = min(
        ref_lens, key=lambda ref_len: (abs(ref_len - hyp_len), ref_len)
    )
    return closest_ref_len

def count_avg_pos(sentences, pos="JJ"):
    from nltk.parse import CoreNLPParser
    try:
        pos_tagger = CoreNLPParser(url="http://localhost:9876", tagtype='pos')
    except:
        logging.info("load pos_tagger on http://localhost:9876 failed!")
        return -1
    target_pos_num = 0
    for sentence in sentences:
        pos_result = pos_tagger.tag(sentence)
        for word_pos in pos_result:
            if word_pos[1] == pos:
                target_pos_num += 1
    return target_pos_num / len(sentences)

