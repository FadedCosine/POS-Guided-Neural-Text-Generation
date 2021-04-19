from model.ops import mask_lengths
import re
import torch
import numpy as np
import torch.nn.functional as F
from copy import deepcopy
import logging
logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                    datefmt='%d-%m-%Y:%H:%M:%S')
logging.getLogger().setLevel(logging.DEBUG)
logger = logging.getLogger(__name__)

def top_k_logits(logits, k):
    if k == 0:
        # no truncation
        return logits
    else:
        values, _ = torch.topk(logits, k=k)
        min_values = values[:, -1, None]
        return torch.where(
            logits < min_values,
            torch.ones_like(logits, dtype=logits.dtype) * float("-inf"),
            logits,
        )


def top_p_logits(logits, p):
    """
    Nucleus sampling
    注意此时logits还不是概率
    """
    batch = logits.size(0)
    sorted_logits, _ = torch.sort(logits, descending=True, dim=-1)
    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
    a = torch.arange(0,batch).to(logits.device)
    b = torch.max(torch.sum(cumulative_probs <= p, dim=-1) - 1, torch.Tensor([0]).long().to(logits.device))
    min_values = sorted_logits[a,b].to(logits.device)
    return torch.where(
        logits < min_values[:,None],
        torch.ones_like(logits) * float("-inf"),
        logits,
    )

def top_k_top_p_filtering(logits, top_w, filter_value=-float('Inf')):   
    with torch.no_grad():
        if isinstance(top_w, int): 
            top_k = min(top_w, logits.size(-1))  # Safety check
            if top_k > 0:
                #logits : [batch_size, seq_len, vocab_size] or [batch_size, vocab_size]
                # Remove all tokens with a probability less than the last token of the top-k
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = filter_value
        else:
            top_p = top_w
            if top_p > 0.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift the indices to the right to keep also the first token above the threshold
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                placehold = torch.zeros_like(sorted_indices_to_remove)
                batch_indices = placehold.scatter_(dim=-1, index=sorted_indices, src=sorted_indices_to_remove).bool()
                logits[batch_indices] = filter_value

            # indices_to_remove = sorted_indices[sorted_indices_to_remove]
            # logits[indices_to_remove] = filter_value
    return logits

def pos_filtering(pos_prev, tok_logits, pos2word_dir=None):
    """
    使用预测出的pos_prev [batch_size * seq_len, 1]，过滤出输入的tok_logits对应pos的token的概率
    """
    #pos_prev :  [batch_size * seq_len, 1]
    #tok_logits: [batch_size * seq_len, vocab_size]
    device = tok_logits.device
    with torch.no_grad():
        if pos2word_dir is not None:
            # [batch_size * seq_len, word_list]
            corres_tokens = pos2word_dir[pos_prev.cpu()]
  
            for idx, tok_lists in enumerate(corres_tokens):
                # tok_lists : [word_list] 
                filter_mask = torch.ones(tok_logits[idx].size()).bool().to(device)
                filter_mask[tok_lists] = 0
                filter_mask = filter_mask.bool()
                tok_logits[idx][filter_mask] = float("-inf")
    return tok_logits

def gathered_input(indexed):
    device = indexed.device
    # print(indexed.size())
    bs, l = indexed.size()
    lens = torch.LongTensor([l + 1] * bs).to(device)
    indexed = torch.cat([indexed, torch.LongTensor([0] * bs)[:, None].to(device)], 1)
    return bs, l, (indexed,lens)


def get_mem(model,inp):
    istuple = True if isinstance(inp, tuple) else False
    with torch.no_grad():
        if istuple:
            title, context, title_len, context_len = inp
            context = context[:,:-1]
            context_len = torch.clamp_min(context_len - 1,0)
            _, mem = model.compute_hidden((title,context,title_len,context_len,None))
        else:
            bs, l = inp.size()
            lens = torch.LongTensor([l - 1] * bs).to(inp.device)
            _, mem = model.compute_hidden(inp[:,:-1],None,lens)
    return mem, inp

@torch.no_grad()
def LM_sampling(model, lengths, inp, top_w, temparature, experimental_loss, sampling_mode=0, pos_top_w=10):
    model.eval()
    top_whatever = top_k_logits if isinstance(top_w, int) else top_p_logits
    probs = None
    res = inp
    # res = torch.LongTensor([]).to(inp.device)
    cnt = 0
    is_rnn_model = hasattr(model, 'model_type') and model.model_type == 'RNN'
    if is_rnn_model:
        bs, _ = inp.size()
        hidden = model.init_hidden(bs)
        # hidden = model.get_hidden(inp[:, :-1], hidden)
        # output, hidden = model(inp, hidden)
    else:
        mem, inp = get_mem(model, inp)
        mem=[m.to(torch.float) for m in mem]
    for _ in range(lengths):
        cnt+=1
   
        with torch.no_grad():
            if is_rnn_model:
                
                if experimental_loss == 1 or experimental_loss == 2 :
                    logits, hidden = model.sampling(inp, hidden, sampling_mode, top_w)
                elif experimental_loss == 3:
                    logits, hidden = model.sampling(inp, hidden, sampling_mode, pos_top_w)
                else:
                    logits, hidden = model.sampling(inp, hidden, sampling_mode, None)
               
            else:
                # l是加了一个0之前的batch中每个文本的长度
                bs, l, inp = gathered_input(inp[:,-1:])
                dec_input, input_len = inp
                if experimental_loss == 1 or experimental_loss == 2 :
                    logits, new_mem = model.sampling(dec_input, input_len, mem, sampling_mode, top_w)
                elif experimental_loss == 3:
                    logits, new_mem = model.sampling(dec_input, input_len, mem, sampling_mode, pos_top_w)
                else:
                    logits, new_mem = model(dec_input, input_len, None, None, memory=mem)
                # sampling输出的logits还不是最终的概率
                # new_mem=[m.to(torch.float) for m in new_mem]
                mem = [torch.cat([mem[i], new_mem[i][:,:-1]],1) for i in range(len(mem))]
            logits = logits[:,-1,:] / temparature
            logits = top_whatever(logits, top_w)
            saved_logits = logits
            
            sampled = torch.multinomial(torch.softmax(logits,-1),1)
            # logger.info("logits is {}".format(logits))
       
            res = torch.cat([res,sampled],1)
            temp_probs = torch.softmax(saved_logits, -1)
   
            probs = torch.cat([probs,temp_probs[torch.arange(len(sampled)),sampled.squeeze(1)][:,None]],1) \
                if probs is not None else temp_probs[torch.arange(len(sampled)),sampled.squeeze(1)][:,None]
            inp = sampled
  
    return res.tolist(), probs.tolist()

@torch.no_grad()
def seq2seq_sampling(model, max_decoding_len, tokenizer, inp, top_w, temparature, experimental_loss, sampling_mode=0, pos_top_w=10, sampling_num=1):
    top_whatever = top_k_logits if isinstance(top_w, int) else top_p_logits
    x, x_lens, x_pos, y, y_len, y_pos = inp
    context, enc_mem = model.compute_enc_context(x, x_lens)
    batch_size, seq_len = x.size()
    sample_results = [[[] for j in range(sampling_num)] for i in range(batch_size)]
    for sample_idx in range(sampling_num):
        # dec_result = [[] for i in range(batch_size)]
        existence = [True] * batch_size
        num_left = batch_size
        next_y = torch.ones(batch_size, 1).fill_(tokenizer.bos_id).type_as(x).to(x.device)
        for step in range(max_decoding_len):
            if num_left == 0:
                break
            with torch.no_grad():
                bs, l = next_y.size()
                lens = torch.LongTensor([l] * bs).to(x.device)
                if experimental_loss == 1 or experimental_loss == 2 :
                    logits = model.decode_step(x_lens, next_y, lens, context, sampling_mode, top_w)
                elif experimental_loss == 3:
                    logits = model.decode_step(x_lens, next_y, lens, context, sampling_mode, pos_top_w)
                else:
                    logits = model.decode_step(x_lens, next_y, lens, context, 0, None) # 除了f2和pos，其他的模型的最后一次皆为一层linear，必有sampling_mode为0
                # sampling输出的logits还不是最终的概率，而是一个logits，需要经过softmax
                
                logits = top_whatever(logits, top_w)
                logits = logits.view(bs,l,-1)
                logits = logits[:,-1,:] / temparature
                
                saved_logits = logits
                sampled = torch.multinomial(torch.softmax(logits,-1),1)
                
                next_y = torch.cat([next_y, sampled],dim=-1)
                
                
                for batch_idx in range(bs):
                    if existence[batch_idx] == False:
                        continue
                    cur_token_id = next_y[batch_idx, -1].item()
                    if cur_token_id == tokenizer.eos_id:
                        existence[batch_idx] = False
                        num_left -= 1
                    else:
                        # dec_result[batch_idx].append(cur_token_id) # check if token id is str?
                        sample_results[batch_idx][sample_idx].append(cur_token_id)
        torch.cuda.empty_cache()
    return sample_results