import torch.nn.functional as F
import torch
from torch.nn.modules.loss import _Loss
from collections import Counter
import torch.nn as nn
import numpy as np
from .loss_utils import *

class FactorizedLoss(_Loss):
    def __init__(self, padding_idx:int):
        super(FactorizedLoss, self).__init__()
        self.padding_idx = padding_idx

    def forward(self, y_hat, y):
        nll = y_hat
        ny = y.numel()
        padding_mask = y == self.padding_idx
        padding_indices = padding_mask.nonzero().squeeze(1)
        padding_size = padding_indices.size(0)
        nll[padding_indices] = 0
        return torch.mean(nll) * (ny / (ny-padding_size))


class PlainLoss(_Loss):
    def __init__(self, padding_idx:int):
        super(PlainLoss, self).__init__()
        self.padding_idx = padding_idx
        self.criteria = torch.nn.CrossEntropyLoss(ignore_index=padding_idx)

    def forward(self, y_hat, y):
        return self.criteria(y_hat,y)


class CandidateLoss(_Loss):
    """Applies a (1-p(x_nt)) loss to each negative target ('candidate') x_nt."""

    def __init__(self, rank_alpha, candidate_type='prev_context', padding_idx:int=0):
        super(CandidateLoss, self).__init__()
        self.rank_alpha = rank_alpha
        self.candidate_type = candidate_type
        self.padding_idx=padding_idx

    def forward(self, net_output,target:torch.LongTensor, reduce=True, compute_custom_metrics=True):
        # net_output = model(**sample['net_input'])
        # target = model.get_targets(sample, net_output)
        target = target.view(-1)

        #
        # # -- mle loss
        # lprobs = model.get_normalized_probs(net_output, log_probs=True)
        lprobs=torch.log_softmax(net_output,dim=-1)

        # print(lprobs.shape)
        # print(target.shape)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        true_token_lprobs = F.nll_loss(
            lprobs,
            target,
            ignore_index=self.padding_idx,
            reduction='none',
        )
        mle_loss = true_token_lprobs.sum()
        # -- unlikelihood loss
        # Maximize (1 - p(x_nt)) for negative target tokens x_nt (equivalently minimize -log(1-p(x_nt)))
        # - form negative targets
        with torch.no_grad():
            # E.g. DABCC | D | EFFGD => {A,B,C} are negative targets.
            if self.candidate_type == 'prev_context':
                # Make 'the triangle'.
                ctx_cands = target.unsqueeze(0).expand(target.size(0), target.size(0))
                ctx_cands_ = torch.ones_like(ctx_cands).triu() * self.padding_idx
                ctx_cands = ctx_cands.tril(-1) + ctx_cands_
                # Don't include the target for that timestep as a negative target.
                ctx_cands = ctx_cands.masked_fill(ctx_cands == target.unsqueeze(1), self.padding_idx)
                negative_targets = torch.zeros_like(lprobs).scatter_(1, ctx_cands, 1)
                negative_targets[(target == self.padding_idx).nonzero().squeeze()] = 0

            else:
                raise NotImplementedError('candidate type %s' % self.candidate_type)

        # - compute loss
        one_minus_probs = torch.clamp((1.0 - lprobs.exp()), min=1e-5)
        custom_loss = -torch.log(one_minus_probs) * negative_targets
        custom_loss = custom_loss.sum()
        loss = mle_loss + self.rank_alpha * custom_loss
        n_tokens=torch.sum((target != self.padding_idx).to(torch.long))
        return loss/n_tokens




class SequencePenaltyCriterion(_Loss):
    def __init__(self, sequence_ngram_n,sequence_prefix_length,sequence_completion_length,sequence_candidate_type,sequence_tune_rate=0.5,mask_p=0.5):
        super(SequencePenaltyCriterion, self).__init__()
        self.sequence_ngram_n = sequence_ngram_n
        self.sequence_prefix_length = sequence_prefix_length
        self.sequence_completion_length = sequence_completion_length
        self.sequence_candidate_type = sequence_candidate_type
        self.mask_p = mask_p
        self.sequence_tune_rate=sequence_tune_rate

    def forward(self, model, inp_ids, lens, label, reduce=True, generator=None):

        seq_len = inp_ids.size(1)
        # make total number of tokens equal to the sequence length (for memory purposes)
        n_batches = seq_len // (self.sequence_prefix_length + self.sequence_completion_length)
        batch = batch_input_sequence_by_prefix_length(inp_ids,
                                                      prefix_length=self.sequence_prefix_length)
        batch = batch[:n_batches]
        lens=torch.LongTensor([batch.size(-1)]*n_batches).to("cuda")
        inp_batch=(batch,lens)
        pred_toks, lprobs = generate_batch(model,inp_batch ,completion_length=self.sequence_completion_length)

        if self.sequence_candidate_type == 'repeat':
            mask = ngram_repeat_mask(pred_toks, self.sequence_ngram_n).type_as(lprobs)
        elif self.sequence_candidate_type == 'random':
            mask = torch.bernoulli(torch.zeros_like(pred_toks, dtype=torch.float).fill_(self.mask_p))
        lprobs=torch.log_softmax(lprobs,-1)
        pred_lprobs = lprobs.view(-1, lprobs.size(2)).gather(1, pred_toks.view(-1, 1))
        one_minus_probs = torch.clamp((1.0 - pred_lprobs.exp()), min=1e-20).view(pred_toks.size(0), pred_toks.size(1))
        loss = -torch.log(one_minus_probs)*mask
        loss = loss.sum()

        ntokens = pred_toks.numel()  # number of output tokens (tokens in completions)

        return loss/ntokens


class FACELoss(_Loss):
    def __init__(self, padding_idx: int, vocab_size, ignore_freq_index: list, ft="out", wt="pre"):
        """
        out mode must be in finetune procedure. you should transfer pretrained weight

        """
        super(FACELoss, self).__init__()

        self.ft = ft
        self.wt = wt
        self.cp = "none"
        self.ignore_freq_index = ignore_freq_index
        self.padding_idx = padding_idx
        self.word_freq = np.zeros(vocab_size)
        self.ignore_freq = ignore_freq_index
        self.lm_criterion = nn.CrossEntropyLoss(ignore_index=padding_idx, size_average=False)
        self.beta = 1e-4

    def forward(self, logits, y_true):
        """
        :param logits: (bsz, lens ,vocab_size)
        :param y_true:  (bsz ,lens)
        :param y_preds:  (bsz ,lens)
        :return:
        """

        logits = logits.view(-1, logits.size(-1)).contiguous()
        y_preds = torch.argmax(logits, -1)
        n_targets = y_true.numel()

        preds_clean = self.clean_preds(y_preds)

        if self.ft == "gt":
            self.update_frequency(self.clean_preds(y_true))

        elif self.ft == "out":
            self.update_frequency(preds_clean)

        if self.wt == "pre":
            self.lm_criterion.weight = self.loss_weight()
            lm_loss = self.lm_criterion(logits, y_true.view(-1))  # standard loss
        elif self.wt == "post":
            self.lm_criterion.reduction = 'none'
            lm_loss = self.lm_criterion(logits, y_true.view(-1).view(-1))

            device = lm_loss.device
            freq_pred = self.word_freq[y_preds.view(-1).cpu().numpy()]
            freq_pred = torch.FloatTensor(freq_pred).to(device)

            freq_GT = self.word_freq[y_true.view(-1).cpu().numpy()]
            freq_GT = torch.FloatTensor(freq_GT).to(device)
            total_freq = self.word_freq.sum()

            weight = 1 + F.relu(freq_pred - freq_GT) / total_freq
            lm_loss = torch.matmul(lm_loss, weight)
        else:
            lm_loss = self.criterion(logits, y_true)
        return lm_loss / n_targets

    def update_frequency(self, preds):
        curr = Counter(preds)
        for k, v in curr.items():
            if k == self.padding_idx:  # do not suppress END token
                continue
            self.word_freq[k] += v

    def clean_preds(self, preds):
        preds = preds.tolist()
        freq_labels = [l for l in preds if l not in self.ignore_freq_index]

        return freq_labels


    def loss_weight(self):
        RF = self.word_freq / self.word_freq.sum()  # relative frequency
        a = -1 / RF.max()

        weight = a * RF + 1
        weight = weight / weight.sum() * len(weight)  # normalization

        return torch.FloatTensor(weight).cuda()
        # return torch.FloatTensor(weight)




