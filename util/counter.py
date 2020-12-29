import collections
import pandas as pd
import numpy as np
import re
import os


def count(fl,target='input_context',checks='input_keyword', vocab_size=10000):
    cnter = collections.Counter()
    s = set()
    for filename in fl:
        cur_df = pd.read_pickle(filename)
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


def convert_and_save(fl,dic,targets:list):
    for filename in fl:
        cur_df = convert_idx(filename,dic,targets)
        new_filename = re.sub(r'indexed/','indexed_new/',filename)
        if not os.path.exists(os.path.dirname(new_filename)):
            os.makedirs(os.path.dirname(new_filename))
        cur_df.to_pickle(new_filename)


def convert_idx(filename, dic, targets:list):
    key_type = type(list(dic)[0])
    cur_df = pd.read_pickle(filename)
    for target in targets:
        new = []
        for line in cur_df[target].tolist():
            converted = []
            for token in line:
                converted.append(dic[key_type(token)])
            new.append(converted)
        cur_df[target] = new
    return cur_df

def old_compute_cutoffs(probs,n_cutoffs):
    cutoffs = []
    cut_prob = 1/n_cutoffs
    cnt = 0
    target_probs = cut_prob
    for idx,prob in enumerate(probs):
        if prob>target_probs:
            cutoffs.append(idx + 1)
            target_probs += cut_prob
            cnt +=1
            if cnt >= n_cutoffs -1:
                break
    return cutoffs


def uniform_cutoffs(probs,n_cutoffs):
    per_cluster_n = len(probs) // n_cutoffs
    return [per_cluster_n * i for i in range(1,n_cutoffs)]


def compute_cutoffs(probs,n_cutoffs):
    def rebalance_cutprob():
        remaining_prob = 1 - prior_cluster_prob
        n = n_cutoffs - cnt
        return remaining_prob / n
    cutoffs = []
    probs = probs
    cut_prob = 1/n_cutoffs
    cnt = 0
    prior_cluster_prob = 0.0
    prior_idx = 0
    for idx, prob in enumerate(probs):
        cluster_cumprob = prob - prior_cluster_prob
        if cluster_cumprob > cut_prob:
            if idx != prior_idx:
                cutoffs.append(idx)
                prior_cluster_prob = probs[idx-1]
                prior_idx = idx
            else:
                cutoffs.append(idx+1)
                prior_cluster_prob = probs[idx]
                prior_idx = idx + 1
            cnt += 1
            cut_prob = rebalance_cutprob()
            if cnt >= n_cutoffs -1:
                break
    return cutoffs


def cumulative_to_indivisual(cum_prob):
    cum_prob.insert(0, 0)
    new = []
    for i in range(1,len(cum_prob)):
        new.append(cum_prob[i] - cum_prob[i - 1])
    cum_prob.pop(0)
    return new


def normalized_entropy(x):
    if len(x) ==1:
        return 1.0
    x = np.array(x)
    x = x / np.sum(x)
    entropy = -np.sum(x*np.log2(x))
    z = np.log2(len(x))
    return entropy / z


def cluster_probs(probs,cutoffs):
    p = [probs[cutoffs[0]-1]]
    for l,r in zip(cutoffs[:-1], cutoffs[1:]):
        p.append(probs[r-1]-probs[l-1])
    p.append(1.0-probs[cutoffs[-1]])
    return p


def ideal_cutoffs(probs,lower=2,upper=None):
    ind_probs = cumulative_to_indivisual(probs)
    ideal = None
    max_mean = 0
    if not upper:
        upper = int(1 / probs[0])
    for target in range(lower,upper+1):
        mean = []
        cutoffs = compute_cutoffs(probs,target)
        added_cutoffs = [0] + cutoffs + [len(probs)]
        for i in range(target):
            cluster = ind_probs[added_cutoffs[i]:added_cutoffs[i + 1]]
            mean.append(normalized_entropy(cluster))
        cluster_prob = cluster_probs(probs,cutoffs)
        head = normalized_entropy(cluster_prob)
        tail = np.sum(np.array(mean)) / np.array(mean).nonzero()[0].size
        mean = head * tail
        # print(head, tail, mean)
        if mean > max_mean:
            max_mean = mean
            ideal = cutoffs
    return ideal



