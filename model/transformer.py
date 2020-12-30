import torch
import torch.nn as nn
from .transformer_gpt2 import Transformer_Base
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

        self.k_net = nn.Linear(hidden_dim, n_head * head_dim, bias=False)
        self.v_net = nn.Linear(hidden_dim, n_head * head_dim, bias=False)
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
        # print("att_score size is : ", att_score.size())
        # print("encoder_mask size is : ", encoder_mask.size())
        att_score.masked_fill_(encoder_mask.unsqueeze(-1), -float('inf'))
        # print(att_score)
        att_prob = torch.softmax(att_score,2)
        att_prob = self.dropatt(att_prob)


        attended = torch.einsum('bqkn,bknd->bqnd',att_prob,v)
        out = self.o_net(attended.contiguous().view(bs,qs,-1))
        out = self.dropout(out)
        return out

    def forward(self, x, mask, context=None):
        """
        :param x: input, input.size() = [batch_size, input_len, hidden_dim]
        :param decoder_mask: position_embedding, pos_ebd.size() = [input_len + memory_len, hidden_dim]
        :param context:  context from encoder, context.size() = [batch_size, input_x_len, hidden_dim]
        :return:
        """
        if context is None:
            context = x 
     
        if self.pre_lnorm:
            context = self.layer_norm(context)
            x = self.layer_norm(x)

        #projection
        key = self.k_net(context)
        value = self.v_net(context)
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

    def forward(self,x, mask, pos_emb, rr_bias, rw_bias, context=None):
        """
        :param x: input, input.size() = [batch_size, input_len, hidden_dim]
        :param pos_ebd: position_embedding, pos_ebd.size() = [input_len + memory_len, hidden_dim]
        :param mask: size = [batch_size, query_len, memory_len]
        :param rr_bias : attention bias
        :param rw_bias : attention bias
        :param context:  context from encoder, context.size() = [batch_size, input_x_len, hidden_dim]
        :return:
        """
        if context is None:
            context = x 
     
        if self.pre_lnorm:
            context = self.layer_norm(context)
            x = self.layer_norm(x)

        #projection
        key = self.k_net(context)
        value = self.v_net(context)
        query = self.q_net(x)
        rel = self.r_net(pos_emb)

        out = self.attend(query, key, value, rel, rr_bias, rw_bias, mask)
        out = x + out
        if not self.pre_lnorm:
            out = self.layer_norm(out)
        return out


class Transformer_Encoder_Block(nn.Module):
    def __init__(self,hidden_dim:int, projection_dim:int, n_heads:int, head_dim:int,
                 dropout_rate:float,dropatt_rate:float,pre_lnorm:bool=False,rel_att=True):
        super(Transformer_Encoder_Block, self).__init__()
        self.multihead_att = Rel_Multihead_Att(hidden_dim,n_heads,head_dim,dropout_rate,dropatt_rate,pre_lnorm) \
            if rel_att else Multihead_Att(hidden_dim,n_heads,head_dim,dropout_rate,dropatt_rate,pre_lnorm)
        self.feedforward = Residual_FF(hidden_dim,projection_dim,dropout_rate,pre_lnorm)

    def forward(self, x, padding_mask, *args):
        out = self.multihead_att(x, padding_mask,*args)
        out = self.feedforward(out)
        return out

class Transformer_Decoder_Block(nn.Module):
    def __init__(self,hidden_dim:int, projection_dim:int, n_heads:int, head_dim:int,
                 dropout_rate:float,dropatt_rate:float,pre_lnorm:bool=False,rel_att=True):
        super(Transformer_Decoder_Block, self).__init__()
        self.masked_multihead_att = Rel_Multihead_Att(hidden_dim,n_heads,head_dim,dropout_rate,dropatt_rate,pre_lnorm) \
            if rel_att else Multihead_Att(hidden_dim,n_heads,head_dim,dropout_rate,dropatt_rate,pre_lnorm)
        self.multihead_att = Rel_Multihead_Att(hidden_dim,n_heads,head_dim,dropout_rate,dropatt_rate,pre_lnorm) \
            if rel_att else Multihead_Att(hidden_dim,n_heads,head_dim,dropout_rate,dropatt_rate,pre_lnorm)
        self.feedforward = Residual_FF(hidden_dim,projection_dim,dropout_rate,pre_lnorm)

    def forward(self, x, context_mask, padding_mask, context, *args):
        # 注意，masked_multihead_att和multihead_att这两个attn的mask是不一样的
        # dec_mask包括了tgt_input的padding mask和context mask
        out = self.masked_multihead_att(x, context_mask, *args)
        out = self.multihead_att(x, padding_mask, context=context, *args)
        out = self.feedforward(out)
        return out
    
        

class Transformer(nn.Module):
    def __init__(self, vocab_size:int, seq_len:int, hidden_dim: int, projection_dim: int,
                 n_heads: int, head_dim: int, n_layers:int, cutoffs:list,
                 dropout_rate: float, dropatt_rate: float, padding_index : int,
                 pre_lnorm: bool = False, same_lengths:bool = False, rel_att=True, experimental_loss=False,
                 pos2word=None, token_in_pos_id=None):
        super(Transformer, self).__init__()
        self.rel_att = rel_att # encoder与decoder是否使用相对位置编码应该是一致的 
        self.encoder = Transformer_Base(vocab_size, seq_len, hidden_dim, projection_dim, n_heads, 
                head_dim, n_layers, dropout_rate, dropatt_rate, padding_index, pre_lnorm, same_lengths,
                rel_att, Transformer_Encoder_Block)
        self.decoder = Transformer_Base(vocab_size, seq_len, hidden_dim, projection_dim, n_heads, 
                head_dim, n_layers, dropout_rate, dropatt_rate, padding_index, pre_lnorm, same_lengths,
                rel_att, Transformer_Decoder_Block)
        self.seq_len = seq_len
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
    def compute_enc_context(self, enc_input, enc_input_lens):
        """
        :param x: input, input.size() = [batch_size, seq_len]
        :return:
        """
        inp_masks = mask_lengths(enc_input_lens, max_len=self.seq_len, reverse=True).byte()
        emb, pos_ebd = self.encoder.get_emb(enc_input)
        # 其实通过padding_id也可以得到paddding_mask
        # enc_padding_mask = mask_padding(enc_input, self.padding_index)
        enc_padding_mask = self.encoder.get_mask(enc_input, None, inp_masks, False)
        enc_out = emb
        enc_mem = []
        for i in range(self.encoder.n_layers):
            # print("Encoder layer : ", i)
            enc_mem.append(enc_out)
            mem_i = None
            main_inp = (enc_out, enc_padding_mask, pos_ebd, self.encoder.rr_bias, self.encoder.rw_bias) if self.rel_att else \
                (enc_out, enc_padding_mask)
            enc_out = self.encoder.main_nets[i](*main_inp)
        enc_out = self.encoder.dropout(enc_out)
        return enc_out, enc_mem

    def forward(self, enc_input, enc_input_len, target, target_len, target_POS=None, memory=None):
        context, enc_mem = self.compute_enc_context(enc_input, enc_input_len)
        # 因为是seq2seq的任务，而已经不是纯文本生成的问题了，x就是x，y就是y，所以不用out[:,:-1]
        y_input = target[..., :-1]
        y_output = target[..., 1:]
        bs, qs = y_output.size()
        
        y_input_len = target_len - 1
        dec_masks = mask_lengths(y_input_len, max_len=self.seq_len-1, reverse=True).byte()
        dec_emb, dec_pos_ebd = self.decoder.get_emb(y_input, None)
        
        dec_context_mask = self.decoder.get_mask(y_input, None, dec_masks, True)
        bs, dec_seq_len = dec_masks.size()
        """ 
        必须要搞清楚enc_dec_padding_mask的作用， Q * K之后矩阵大小为 [ batch_size, q_seq_len, k_seq_len]
        其中[1]维度上的q_seq_len，一定是Decoder输入的文本的长度，因为我们对Decoder输入的文本的每一个位置都要做预测，
        所以mask不可能是mask在这个维度上，相反，是对k_seq_len维度上，因为k_seq_len在Encoder中是来自于Encoder输入本身，
        在Decoder中是来自于Encoder提取出的context，enc_dec_padding_mask的作用在这个维度上要mask掉padding的出的score
        """
        enc_padding_masks = mask_lengths(enc_input_len, max_len=self.seq_len, reverse=True).byte()
        enc_dec_padding_mask = enc_padding_masks.unsqueeze(1).expand(bs, dec_seq_len, context.size(1)).bool()

        out = dec_emb
        
        for i in range(self.decoder.n_layers):
            # print("Decoder layer : ", i)
            main_inp = (out, dec_context_mask, enc_dec_padding_mask, context, dec_pos_ebd, self.decoder.rr_bias, self.decoder.rw_bias) if self.rel_att else \
                (out, dec_context_mask, enc_dec_padding_mask, context)
            out = self.decoder.main_nets[i](*main_inp)
        out = self.decoder.dropout(out)
        # out: [batch, seq_len, hidden_dim]
        
        out = out.contiguous().view(bs * qs, -1)
       
        if self.experimental_loss in [1, 2]:
            y_output = y_output.contiguous().view(-1)
            final = self.final(out,y_output)
        elif self.experimental_loss == 3:
            y_output = y_output.contiguous().view(-1)
            y_pos_tgt = target_POS[..., 1:].contiguous().view(-1)
            final = self.final(out, y_output, y_pos_tgt)
        else:
            final = self.final(out)
     
        return final, enc_mem
       
    def decode_step(self, enc_input_len, dec_input, dec_input_len, context, sampling_mode, top_w):
        """ 在decoding阶段，外部已经通过compute_enc_context计算出encoder的context，每次输入当前时刻已经生成的所有y，预测下一个时刻的y_next，
           注意batch当中有可能有已经生成完的

        Args:
            enc_input_len ([torch.LongTensor]): encoder input length, to calculate the enc_dec_padding_mask, in decoding step 
            dec_input ([torch.LongTensor]): decoder input
            dec_input_len ([torch.LongTensor]): decoder input length
            context : the encoder input's representations from encoder
            sampling_mode ([int]): sampling mode
                0: Linear layer to output logits
                1: Hard Cluster Logits, using topk to sample Cluster. Final logits can keep multi cluster's tokens
                2: Hard Cluster Logits, using topp sampling to sample Cluster. Final logits can keep multi cluster's tokens
                3: Only for POS sampling, using topp sampling to sample POS. Final logits can only keep one POS's tokens
            top_w ([int or float]): 
                top k 's k value, if type is int,
                top p 's p value, if type is float,
                sampling_mode in [1, 2, 3], will ensure the output Logits keep enough tokens' logits for top k or top p sampling.

        Returns:
            out: Finial token logits, before softmax.
        """
        bs, qs = dec_input.size()
        dec_masks = mask_lengths(dec_input_len, reverse=True).byte()
       
        dec_emb, dec_pos_ebd = self.decoder.get_emb(dec_input, None)
        dec_context_mask = self.decoder.get_mask(dec_input, None, dec_masks, True)
        bs, dec_seq_len = dec_masks.size()
        enc_padding_masks = mask_lengths(enc_input_len, max_len=self.seq_len, reverse=True).byte()
        enc_dec_padding_mask = enc_padding_masks.unsqueeze(1).expand(bs, dec_seq_len, context.size(1)).bool()
        
        out = dec_emb
        for i in range(self.decoder.n_layers):
            
            main_inp = (out, dec_context_mask, enc_dec_padding_mask, context, dec_pos_ebd, self.decoder.rr_bias, self.decoder.rw_bias) if self.rel_att else \
                (out, dec_context_mask, enc_dec_padding_mask, context)
            out = self.decoder.main_nets[i](*main_inp)
        out = self.decoder.dropout(out)
        # out: [batch, seq_len, hidden_dim]
        out = out.contiguous().view(bs * qs, -1)
     
        if sampling_mode == 0:
            out = self.final(out)
        elif sampling_mode == 1:
            ishard = True 
            out = self.final.hard_cluster_logit(out, top_w, ishard)
        elif sampling_mode == 2:
            ishard = False 
            out = self.final.hard_cluster_logit(out, top_w, ishard)
        elif sampling_mode == 3:
            out = self.final.pos_sampling(out, top_w)
        else:
            out = self.final.soft_cluster_logit(out)
        return out

    def beam_search(self, enc_input, enc_input_len, dec_input, beam_size=None,
				 max_sequence_length=None, length_normalization_factor=0.0, top_p=0, top_k = 0,
				 get_attention=False):
        context, enc_mem = self.compute_enc_context(enc_input, enc_input_len)
        generator = SequenceGenerator(
			decode_step=self.decode_step,
			beam_size=beam_size,
			max_sequence_length=max_sequence_length,
			get_attention=get_attention,
			length_normalization_factor=length_normalization_factor)
        return generator.beam_search(dec_input, top_k=top_k, top_p=top_p)

    