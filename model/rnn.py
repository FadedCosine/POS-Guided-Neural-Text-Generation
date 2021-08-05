import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .softmax import *

class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type: str, vocab_size:int, embedding_dim: int, hidden_dim: int, nlayers: int, cutoffs:list, padding_index: int, experimental_loss=0, dropout=0.5, tie_weights=False, pos2word=None, token_in_pos_id=None, expert_dim=512, n_experts=15, ):
        super(RNNModel, self).__init__()
        self.model_type = 'RNN'
        self.vocab_size = vocab_size
        self.drop = nn.Dropout(dropout)
        self.word_embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_index) # 
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(embedding_dim, hidden_dim, nlayers, batch_first=True,dropout=dropout)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
            except KeyError:
                raise ValueError( """An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            self.rnn = nn.RNN(embedding_dim, hidden_dim, nlayers, batch_first=True, nonlinearity=nonlinearity, dropout=dropout)

        self.experimental_loss = experimental_loss
        if experimental_loss == 1:
            self.final = Factorized_SoftmaxV2(vocab_size, hidden_dim, cutoffs, padding_index)
        elif experimental_loss == 2:
            self.final = Factorized_Softmax(vocab_size, hidden_dim, cutoffs, padding_index)
        elif experimental_loss == 3:
            if pos2word is None or token_in_pos_id is None:
                raise ValueError('pos2word or token_in_pos_id must be specified!')
            self.final = POS_Guided_Softmax(vocab_size, hidden_dim, pos2word, token_in_pos_id, padding_index)
        elif experimental_loss == 4:
            self.final = MixofSoftmax(vocab_size, hidden_dim, expert_dim, n_experts, padding_index)
        else:
            self.final = nn.Linear(hidden_dim, vocab_size, bias=False)

        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        # if tie_weights:
        #     if nhid != ninp:
        #         raise ValueError('When using the tied flag, nhid must be equal to emsize')
        #     self.decoder.weight = self.word_embedding.weight

        self.init_weights()

        self.rnn_type = rnn_type
        self.hidden_dim = hidden_dim
        self.nlayers = nlayers

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.word_embedding.weight, -initrange, initrange)

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        if self.rnn_type == 'LSTM':
            
            return (weight.new_zeros(self.nlayers, bsz, self.hidden_dim),
                    weight.new_zeros(self.nlayers, bsz, self.hidden_dim))
        else:
            return weight.new_zeros(self.nlayers, bsz, self.hidden_dim)

    def get_hidden(self, dec_input, hidden):
        bs, qs = dec_input.size()
        emb = self.drop(self.word_embedding(dec_input))
        _, hidden = self.rnn(emb, hidden)
        return hidden

    def forward(self, dec_input, dec_output, hidden, dec_output_POS=None):
        bs, qs = dec_input.size()
        emb = self.drop(self.word_embedding(dec_input))
   
        out, hidden = self.rnn(emb, hidden)
        out = self.drop(out)
       
        out = out[:,:-1]
        out = out.contiguous().view(bs*(qs-1),-1)
 

        if self.experimental_loss in [1, 2]:
            dec_output = dec_output.contiguous().view(-1)
            final = self.final(out,dec_output)
        elif self.experimental_loss == 3:
            dec_output = dec_output.contiguous().view(-1)
            dec_output_POS = dec_output_POS.contiguous().view(-1)
            final = self.final(out, dec_output, dec_output_POS)
        else:
            final = self.final(out)
        return final, hidden

    def sampling(self, dec_input, hidden, sampling_mode, top_w):
        bs, qs = dec_input.size()
        emb = self.drop(self.word_embedding(dec_input))
   
        out, hidden = self.rnn(emb, hidden)
        out = self.drop(out)
        
        # out = out[:,-1]
        out = out.contiguous().view(bs * qs,-1)
        if sampling_mode == 1:
            ishard = True 
            out = self.final.hard_cluster_logit(out, top_w, ishard)
        elif sampling_mode == 2:
            ishard = False 
            out = self.final.hard_cluster_logit(out, top_w, ishard)
        elif sampling_mode == 3:
            out = self.final.pos_sampling(out, top_w)
        elif sampling_mode == 0 and self.experimental_loss != 0:
            out = self.final.soft_cluster_logit(out)
        elif sampling_mode == 0 and self.experimental_loss == 0:
            out = self.final(out)
        out = out.contiguous().view(bs, qs, -1)
        return out, hidden
       