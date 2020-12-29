import torch
import torch.nn as nn
from .embeddings import *
from .softmax import *
from .initializer import *
from .layers import *


class Att_Base(nn.Module):
    def __init__(self, hidden_dim:int, n_head:int, head_dim:int,
                 dropout_rate:float, dropatt_rate:float=0.0, pre_lnorm=False):
        super(Att_Base, self).__init__()
        self.n_head = n_head
        self.hidden_dim = hidden_dim
        self.head_dim = head_dim
        self.dropout_rate = dropout_rate

        self.kv_net = nn.Linear(hidden_dim, 2 * n_head * head_dim, bias=False)
        self.q_net = nn.Linear(hidden_dim, n_head * head_dim, bias=False)

        self.dropout = nn.Dropout(dropout_rate)
        self.dropatt = nn.Dropout(dropatt_rate)
        self.o_net = nn.Linear(n_head * head_dim, hidden_dim, bias=False)
        self.layer_norm = nn.LayerNorm(hidden_dim)

        self.scale = 1 / (head_dim ** 0.5)
        self.pre_lnorm = pre_lnorm


class Multihead_Att(Att_Base):
    def __init__(self, hidden_dim:int, n_head:int, head_dim:int,
                 dropout_rate:float, dropatt_rate:float=0.0, pre_lnorm=False):
        super(Multihead_Att, self).__init__(hidden_dim, n_head, head_dim, dropout_rate, dropatt_rate, pre_lnorm)

    def attend(self, query, key, value, mask):
        bs, qs, hs = query.size()
        ks = key.size(1)
        ms = ks-qs
        # print("key size is ", key.size())
        # print(query.size(),key.size(),value.size(),rel.size())
        #reshaping
        k = key.view(bs,ks,self.n_head,self.head_dim)
        v = value.view(bs,ks,self.n_head,self.head_dim)
        q = query.view(bs,qs,self.n_head,self.head_dim)

        att_score = torch.einsum('bqnd,bknd->bqkn',q,k)
        att_score.mul_(self.scale)

        #attend
        if mask is None:
            print('mask is none')
            mask = torch.ones((qs,ks)).byte()
            mask = mask.triu(1+ms) ==0
        # print(mask.size())
        encoder_mask = mask.bool()
        att_score.masked_fill_(encoder_mask.unsqueeze(-1), -float('inf'))
        # print(att_score)
        att_prob = torch.softmax(att_score,2)
        att_prob = self.dropatt(att_prob)

        attended = torch.einsum('bqkn,bknd->bqnd',att_prob,v)
        out = self.o_net(attended.contiguous().view(bs,qs,-1))
        out = self.dropout(out)
        return out

    def forward(self, x, mem, mask, ed_att=False):
        """
        :param x: input, input.size() = [batch_size, input_len, hidden_dim]
        :param mem:  memory, input.size() = [batch_size, memory_len, hidden_dim]
        :param decoder_mask: position_embedding, pos_ebd.size() = [input_len + memory_len, hidden_dim]
        :param ed_att:
        :param encoder_mask:
        :return:
        """
        if mem is None:
            mem = torch.Tensor().to(x.device).to(x.dtype)
        if ed_att:
            c = mem
        else:
            c = torch.cat([mem,x],1)
            #! 这是因为mem是之前decoder已经计算过的，而不是context，在seq2seq的问题中直接用可能会存在问题
        if self.pre_lnorm:
            c = self.layer_norm(c)
            x = self.layer_norm(x)

        #projection
        kv = self.kv_net(c)
      
        key, value = kv.chunk(2,-1)
        
        query = self.q_net(x)
   
        out = self.attend(query,key,value,mask)

        # if ed_att:
        #     return out
        out = x + out
        if not self.pre_lnorm:
            out = self.layer_norm(out)
        return out


class Rel_Multihead_Att(Att_Base):
    def __init__(self, hidden_dim:int, n_head:int, head_dim:int,
                 dropout_rate:float, dropatt_rate:float=0.0, pre_lnorm=False):
        super(Rel_Multihead_Att, self).__init__(hidden_dim, n_head, head_dim,
                 dropout_rate, dropatt_rate, pre_lnorm)

        self.r_net = nn.Linear(self.hidden_dim, self.n_head * self.head_dim, bias=False)


    def _left_shift(self, x:torch.Tensor)->torch.Tensor:
        """
        :param x: x.size() = [batch_size, q_len, k_len, n_head]
        x[0,:,:,0] =
        [[[9,8,7,6,5,4,3,2,1,0],
          [9,8,7,6,5,4,3,2,1,0],
          [9,8,7,6,5,4,3,2,1,0]]]]

        :param zero_triu:
        :return: left_shifted tensor of x by the index along query axis
        x[0,:,:,0] =
        [[[7,6,5,4,3,2,1,0,0,0], -> left shifted by 2
          [8,7,6,5,4,3,2,1,0,0], -> left shifted by 1
          [9,8,7,6,5,4,3,2,1,0]]]] ->shifted 0

        """
        bs,qs,ks,hs = x.size()
        zero_pad = torch.zeros((bs, qs, 1,hs),
                               device=x.device, dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x], dim=2)  #[b,q,k+1,n]

        x_padded = x_padded.view(bs, ks+1, qs, hs)

        x = x_padded[:,1:].view_as(x)

        ones = torch.ones((qs, ks),device=x.device, dtype=x.dtype)
        x = x * torch.tril(ones, ks-bs)[None,:, :, None]

        return x


    def attend(self, query, key, value, rel, rr_bias, rw_bias, mask):
        bs, qs, hs = query.size()
        ks = key.size(1)
        ms = ks-qs

        # print(query.size(),key.size(),value.size(),rel.size())
        #reshaping
        k = key.view(bs, ks, self.n_head, self.head_dim)
        v = value.view(bs, ks, self.n_head, self.head_dim)
        q = query.view(bs, qs, self.n_head, self.head_dim)
        r = rel.view(qs, self.n_head, self.head_dim)

        rwq = q + rw_bias[None, None]
        AC = torch.einsum('bqnd,bknd->bqkn', rwq, k)

        rrq = q + rr_bias[None, None]
        BD = torch.einsum('bqnd,knd->bqkn', rrq, r)
        BD = self._left_shift(BD)
        #attend
        if mask is None:
            print('mask is none')
            mask = torch.ones((qs,ks)).byte()
            mask = mask.triu(1+ms) ==0
        # print(mask.size())
        mask = mask.bool()

        att_score = AC + BD
        att_score.mul_(self.scale)
        att_score.masked_fill_(mask.unsqueeze(-1), -float('inf'))
        # print(att_score)
        att_prob = torch.softmax(att_score,2)
        att_prob = self.dropatt(att_prob)

        attended = torch.einsum('bqkn,bknd->bqnd',att_prob,v)
        out = self.o_net(attended.contiguous().view(bs,qs,-1))
        out = self.dropout(out)
        return out

    def forward(self,x, mem, mask, pos_emb, rr_bias, rw_bias):
        """
        :param x: input, input.size() = [batch_size, input_len, hidden_dim]
        :param mem:  memory, input.size() = [batch_size, memory_len, hidden_dim]
        :param pos_ebd: position_embedding, pos_ebd.size() = [input_len + memory_len, hidden_dim]
        :param mask: size = [batch_size, query_len, memory_len]
        :param rr_bias : attention bias
        :param rw_bias : attention bias
        :return:
        """
        if mem is None:
            mem = torch.Tensor().to(x.device).to(x.dtype)
        c = torch.cat([mem,x],1)

        if self.pre_lnorm:
            c = self.layer_norm(c)
            x = self.layer_norm(x)

        #projection
        kv = self.kv_net(c)
        key, value = kv.chunk(2,-1)
        query = self.q_net(x)
        rel = self.r_net(pos_emb)

        out = self.attend(query, key, value, rel, rr_bias, rw_bias, mask)
        out = x + out
        if not self.pre_lnorm:
            out = self.layer_norm(out)
        return out


class Transformer_Block(nn.Module):
    def __init__(self,hidden_dim:int, projection_dim:int, n_heads:int, head_dim:int,
                 dropout_rate:float,dropatt_rate:float,pre_lnorm:bool=False,rel_att=True):
        super(Transformer_Block, self).__init__()
        self.multihead_att = Rel_Multihead_Att(hidden_dim,n_heads,head_dim,dropout_rate,dropatt_rate,pre_lnorm) \
            if rel_att else Multihead_Att(hidden_dim,n_heads,head_dim,dropout_rate,dropatt_rate,pre_lnorm)
        self.feedforward = Residual_FF(hidden_dim,projection_dim,dropout_rate,pre_lnorm)

    def forward(self, x, mem, mask, *args):
        out = self.multihead_att(x, mem, mask,*args)
        out = self.feedforward(out)
        return out

class Transformer_Base(nn.Module):
    def __init__(self, vocab_size:int, seq_len:int, hidden_dim: int, projection_dim: int,
                 n_heads: int, head_dim: int, n_layers:int,
                 dropout_rate: float, dropatt_rate: float, padding_index : int,
                 pre_lnorm: bool = False, same_lengths:bool = False, rel_att=True,
                 transformer_type=Transformer_Block):
        super(Transformer_Base, self).__init__()
        # self.word_embedding = Adaptive_Embedding(vocab_size,word_embedding_dim,hidden_dim,cutoffs,div_val)
        self.n_layers = n_layers
        self.same_lengths = same_lengths
        self.seq_len = seq_len
        self.rel_att = rel_att
        self.word_embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=padding_index) # sparse=True
        self.posisition_embedding = nn.Embedding(seq_len, hidden_dim)
        # self.posisition_embedding = Position_Embedding(hidden_dim)

        if rel_att:
            self.rw_bias = nn.Parameter(torch.Tensor(n_heads, head_dim))
            self.rr_bias = nn.Parameter(torch.Tensor(n_heads, head_dim))

        self.dropout = nn.Dropout(dropout_rate)
        # if not self.embedding_equal_hidden:
        #     self.embedding_proj = nn.Linear(word_embedding_dim,hidden_dim,bias=False)
        self.main_nets = nn.ModuleList([transformer_type(hidden_dim,projection_dim,n_heads,head_dim,
                                                         dropout_rate,dropatt_rate,pre_lnorm,rel_att)
                                        for i in range(n_layers)])

    def get_emb(self, x, mem=None):
        bs, qs = x.size()
        ms = mem[0].size(1) if mem is not None else 0
        ks = qs + ms
        emb = self.word_embedding(x)
        # if not self.embedding_equal_hidden:
        #     emb = self.embedding_proj(emb)
        # emb = self.dropout(emb)

        # pos_indicator = torch.arange(ks-1,-1,-1.0).to(x.device)
        pos_indicator = torch.arange(ms, ks, 1).clamp_max_(self.seq_len).to(emb.device)
        pos_ebd = self.posisition_embedding(pos_indicator)

        # relative_embedding
        if self.rel_att:
            emb = self.dropout(emb)
            pos_ebd = self.dropout(pos_ebd)

        else:
            emb = pos_ebd + emb
            emb = self.dropout(emb)

        return emb, pos_ebd

    def get_mask(self, x, mem, inp_masks, is_decoder):
        bs, qs = x.size()
        ms = mem[0].size(1) if mem is not None else 0
        ks = qs + ms
        ones = torch.ones((qs, ks)).byte().to(x.device)
        if is_decoder:
            dec_mask = ones.triu(1 + ms)
        else:
            dec_mask = torch.zeros_like(ones)
        if self.same_lengths:
            dec_mask = dec_mask + ones.tril(-qs)
        if ms:
            inp_masks = torch.cat([torch.zeros(bs,ms,dtype=inp_masks.dtype,device=x.device),inp_masks],1)
     
        mask = (inp_masks.unsqueeze(1) + dec_mask.unsqueeze(0)) > 0
        return mask


class Transformer_Decoder(Transformer_Base):
    def __init__(self, vocab_size:int, seq_len:int, hidden_dim: int, projection_dim: int,
                 n_heads: int, head_dim: int, n_layers:int, cutoffs:list,
                 dropout_rate: float, dropatt_rate: float, padding_index : int,
                 pre_lnorm: bool = False, same_lengths:bool = False, rel_att=True, experimental_loss=False,
                 pos2word=None, token_in_pos_id=None):
        super(Transformer_Decoder, self).__init__(vocab_size,seq_len,hidden_dim,projection_dim,n_heads,head_dim,
                                                n_layers,dropout_rate,dropatt_rate,padding_index,pre_lnorm,
                                                same_lengths,rel_att,Transformer_Block)
        self.experimental_loss = experimental_loss
        if experimental_loss == 1:
            self.final = Factorized_SoftmaxV2(vocab_size, hidden_dim, cutoffs, padding_index)
        elif experimental_loss == 2:
            self.final = Factorized_Softmax(vocab_size, hidden_dim, cutoffs, padding_index)
        elif experimental_loss == 3:
            if pos2word is None or token_in_pos_id is None:
                raise ValueError('pos2word or token_in_pos_id must be specified!')
            self.final = POS_Guided_Softmax(vocab_size, hidden_dim, pos2word, token_in_pos_id, padding_index)
        else:
            self.final = nn.Linear(hidden_dim, vocab_size, bias=False)
        # self.final = Adaptive_Softmax(vocab_size,hidden_dim,cutoffs,div_val)


    def compute_hidden(self, x, mem, inp_lens, is_decoder=True):
        """
        :param x: input, input.size() = [batch_size, seq_len]
        :param mem: list of memories [mem1,mem2, ...memn], n equal to the number of layers
          memory[0].size() = [batch_size, memory_len, hidden_size]
        :return:
        """
        inp_masks = mask_lengths(inp_lens,reverse=True).byte()
        emb, pos_ebd = self.get_emb(x, mem)
        mask = self.get_mask(x, mem, inp_masks, is_decoder)
        out = emb
        new_mem = []
        for i in range(self.n_layers):
            new_mem.append(out)
            mem_i = mem[i] if mem is not None else None
            main_inp = (out, mem_i, mask, pos_ebd, self.rr_bias, self.rw_bias) if self.rel_att else \
                (out, mem_i, mask)
            out = self.main_nets[i](*main_inp)
        out = self.dropout(out)
        return out, new_mem

    def sampling(self, dec_input, dec_input_len, memory, sampling_mode, top_w):
        """
            sampling when the model is trained with experimental loss
        """
        bs, qs = dec_input.size()
        out, mem = self.compute_hidden(dec_input, memory, dec_input_len)
        # out： [batch , (seq_len - 1), hidden_dim]
        out = out[:, :-1]
        # out： [batch * (seq_len - 1), hidden_dim]
        out = out.contiguous().view(bs * (qs - 1), -1)
        if sampling_mode == 1:
            ishard = True 
            out = self.final.hard_cluster_logit(out, top_w, ishard)
        elif sampling_mode == 2:
            ishard = False 
            out = self.final.hard_cluster_logit(out, top_w, ishard)
        elif sampling_mode == 3:
            out = self.final.pos_sampling(out, top_w)
        else:
            out = self.final.soft_cluster_logit(out)
        return out, mem

    def forward(self, dec_input, dec_input_len, dec_output, dec_output_len, dec_output_POS=None, memory=None):
        bs, qs = dec_input.size()
        out, mem = self.compute_hidden(dec_input, memory, dec_input_len)
        #取out[:,:-1]是因为batch_generator输入的x是完全的完全的句子，而y是x[:, 1:]
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
        return final, mem


if __name__ == '__main__':
    vocab_size = 300000
    cutoffs = [20000,80000,200000]
    div_val = 4
    hidden_dim = 500
    word_embedding_dim = 500
    projection_dim = 1000
    n_heads = 10
    head_dim = 50
    n_layers = 10
    dropout_rate = 0.2
    dropatt_rate = 0.1

    m = Transformer_Decoder(vocab_size,word_embedding_dim,hidden_dim,projection_dim,n_heads,head_dim,n_layers,cutoffs,div_val,
                          dropout_rate,dropatt_rate)
    print(m.main_nets[0].multihead_att.vec_u)
    i = Initializer('normal',0.01,0.1)
    i.initialize(m)

    print(m.main_nets[0].multihead_att.vec_u)

