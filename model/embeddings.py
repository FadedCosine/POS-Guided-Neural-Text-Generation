import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .ops import *
from typing import Dict, Optional, Tuple, Any, List

class Word_Embedding(nn.Module):
    def __init__(self,morph_size:int, pos_size:int, morph_lstm_hidden_size:int, morph_embedding_size:int,
                 pos_embedding_size, cutoffs:list, div_val:int, dropout:float=0.0, adaptive_embedding=False):
        super(Word_Embedding, self).__init__()

        self.morph_size = morph_size
        self.pos_size = pos_size
        self.morph_embedding_size = morph_embedding_size
        self.tag_embedding_size = pos_embedding_size
        self.morph_hidden_size = morph_lstm_hidden_size
        if not adaptive_embedding:
            self.morph = nn.Embedding(morph_size+1,morph_embedding_size,morph_size)
        else:
            self.morph = Adaptive_Embedding(morph_size+1,morph_embedding_size,morph_embedding_size,cutoffs,div_val)

        self.pos = nn.Embedding(pos_size+1,pos_embedding_size,pos_size)
        self.padding = nn.ConstantPad1d(3,0.0)
        self.Ks = [2,3,4]

        self.embedding_size = morph_lstm_hidden_size
        # self.embedding_size = len(self.Ks) * filter_size + self.morph_embedding_size + self.tag_embedding_size
        self.morph_rnn = nn.LSTM(morph_embedding_size+pos_embedding_size, morph_lstm_hidden_size, 1,
                                 batch_first=True, bidirectional=False)
        self.dropout = nn.Dropout(dropout)

    def morph_embedding(self,morphs,tags, morphs_lens):
        """

        :param morphs: torch.Tensor size of [batch, sentence_length, word_length]
        :param tags: torch.Tensor size of [batch, sentence_length, word_length]
        :param morphs_lens: torch.Tensor size of [batch, sentence_length]
        :return: torch.Tensor size of [batch,sentence_lengths,embedding_size]
        """
        b,s,w = morphs.size()
        # print(morphs.size(),tags.size(),morphs_lens.size())
        m = self.morph(morphs)
        t = self.pos(tags)
        c = torch.cat([m,t],-1) #[batch,sentence_lengths,word_lengths,embedding]
        # c = c.sum(-2)
        c = c.view(b*s,w,-1)
        morphs_lens = morphs_lens.view(b*s)
        lens_mask = mask_lengths(morphs_lens)
        zero_up_morphs_lens = torch.max(morphs_lens,torch.ones_like(morphs_lens,dtype=morphs_lens.dtype))
        rnned = run_rnn(c,zero_up_morphs_lens,self.morph_rnn)
        rnned = rnned * lens_mask.to(rnned.dtype).unsqueeze(-1)

        pooled = last_pool(rnned,zero_up_morphs_lens)
        outs = pooled.view(b,s,-1)

        return outs, rnned.view(b,s,w,-1)
        # return c

    def forward(self,morphs, pos, morphs_lens):
        mres = self.morph_embedding(morphs, pos, morphs_lens)
        # cres = self.character_embedding(characters,characters_lens)
        # res = torch.cat([mres, cres],-1)
        return mres


class Adaptive_Embedding(nn.Module):
    def __init__(self, vocab_size:int, base_embedding_dim:int, projection_dim:int, cutoffs:List, div_val=1):
        super(Adaptive_Embedding, self).__init__()
        self.n_embeddings = len(cutoffs) + 1
        self.projection_dim = projection_dim
        self.scale = projection_dim**0.5
        self.cutoffs = [0] + cutoffs + [vocab_size]
        self.embedding_dims = [base_embedding_dim // (div_val**i) for i in range(self.n_embeddings)]
        # print(self.embedding_dims)
        self.embeddings = nn.ModuleList([nn.Embedding(self.cutoffs[i+1]-self.cutoffs[i],
                                                      self.embedding_dims[i])
                                         if i != self.n_embeddings - 1
                                         else nn.Embedding(self.cutoffs[i+1]-self.cutoffs[i] + 1,
                                                           self.embedding_dims[i],self.cutoffs[i+1]-self.cutoffs[i])# for UNK
                                         for i in range(self.n_embeddings)])
        self.proj = nn.ModuleList([nn.Linear(i,projection_dim) for i in self.embedding_dims])

    def forward(self,x):
        flat_x = x.view(-1)
        total_embedding = torch.zeros(flat_x.size(0),self.projection_dim)
        for i in range(self.n_embeddings):
            l,r = self.cutoffs[i], self.cutoffs[i+1]
            if i == self.n_embeddings - 1:
                r +=1
            mask = (flat_x >=l) & (flat_x<r)
            indices = mask.nonzero().squeeze()
            if indices.numel() == 0:
                continue
            x_i = flat_x[indices] - l
            target_embedding = self.embeddings[i](x_i)
            projected_embedding = self.proj[i](target_embedding)
            total_embedding[indices] = projected_embedding
        total_embedding = total_embedding.view(*x.size(), self.projection_dim)
        total_embedding.mul_(self.scale)
        return total_embedding


class Position_Embedding(nn.Module):
    def __init__(self, embedding_dim:int):
        super(Position_Embedding, self).__init__()

        self.embedding_dim = embedding_dim

        inv_freq = 1 / (10000 ** (torch.arange(0.0, embedding_dim, 2.0) / embedding_dim))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, pos_seq):
        # pos_seq.size() = (query_lengths)
        if len(pos_seq.size()) ==1:
            sinusoid = torch.ger(pos_seq, self.inv_freq)
        elif len(pos_seq.size()) ==2:
            sinusoid = torch.einsum('ab,c->abc',pos_seq,self.inv_freq)
        pos_emb = torch.cat([sinusoid.sin(), sinusoid.cos()], dim=-1)
        return pos_emb
