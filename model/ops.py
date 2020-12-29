import torch
import torch.nn as nn
import torch.optim as optim
import math

def mask_lengths(lengths:torch.LongTensor, max_len:torch.long=None,reverse=False)->torch.Tensor:
    """
    :param lengths: [batch_size] indicates lengths of sequence
    :return: [batch_size, max_len] ones for within the lengths zeros for exceeding lengths

    [4,2] -> [[1,1,1,1]
              ,[1,1,0,0]]
    """
    device = lengths.device
    if not max_len:
        max_len = torch.max(lengths).item()
    idxes = torch.arange(0,max_len,out=torch.LongTensor(max_len)).unsqueeze(0).to(device)
    masks = (idxes<lengths.unsqueeze(1)).byte()
    if reverse:
        masks = masks ==0
    return masks

def mask_padding(src, padding_id):
    device = src.device
    batch_size, seq_len = src.size()
    src_mask = (src == padding_id).unsqueeze(1).expand(-1, seq_len, -1)
    return src_mask

def detach_memory(mem:list,positions:torch.Tensor):
    new_mem = []
    for i in range(len(mem)):
        new_mem.append(torch.zeros_like(mem[i]))
        new_mem[i][:,positions] = mem[i].detach()
    return new_mem


def perm(x,p,ml,tl):
    """
    :param x: input, size = [batch, textlen, morphlen]
    :param ml: morph_lengths, size = [batch, textlen]
    :param l: text_lengths, size = [batch]
    :return:
    """
    perm = torch.randperm(x.size(1),device=x.device)
    mask = mask_lengths(tl,reverse=True)
    x = x[:,perm]
    p = p[:,perm]
    mask = mask[:,perm]
    ml = ml[:,perm]
    return x,p,ml,mask,perm


def rel_pos(mem,perm):
    if mem is not None:
        ms = mem[0].size(1)
    else:
        ms = 0
    m = torch.arange(-ms-2,0,device=perm.device,dtype=torch.half)
    cated = torch.cat([m,perm.to(m.dtype)])
    pos = cated[:, None] - cated[None]
    return pos[-len(perm):]

def dec_masking(mem,x,mask,same_length=False):
    if mem is not None:
        ms = mem[0].size(1)
    else:
        ms = 0
    qs = x.size(1)
    dec_mask = torch.ones(qs,ms+2+qs,dtype=mask.dtype,device=mask.device)
    upper = dec_mask.triu(3+ms)[None]
    if same_length:
        dec_mask = upper + dec_mask.tril(-qs)
    else:
        dec_mask = upper
    mask = torch.cat([torch.zeros(ms+2,dtype=mask.dtype,device=mask.device).expand(mask.size(0),-1),mask],1)
    dec_mask = mask[:,None] + dec_mask
    return dec_mask>0


def last_pool(x,seq_lengths):
    device = x.device
    row_indices = torch.arange(0, x.size(0)).long().to(device)
    col_indices = seq_lengths - 1

    last_tensor = x[row_indices, col_indices, :]
    return last_tensor

def reorder_sequence(x,index):
    x2 = torch.empty_like(x)
    x2[index,:,:] = x
    return x2

def run_rnn(x,lengths,rnn):
    sorted_lengths, sort_index = lengths.sort(0, True)
    x_sorted = x.index_select(0, sort_index)
    packed_input = nn.utils.rnn.pack_padded_sequence(x_sorted, sorted_lengths, batch_first=True)
    packed_output, _ = rnn(packed_input, None)
    out_rnn, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
    out_rnn = reorder_sequence(out_rnn, sort_index)
    return out_rnn

def halve_lr(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr'] *= 0.7

def get_optim(model,lr,optimizer='adam'):
    if optimizer =='adam':
        sparse_p = []
        dense_p = []
        for p in model.named_parameters():
            if 'word_embedding' in p[0]:
                sparse_p.append(p[1])
            else:
                dense_p.append(p[1])
        sparse_optimizer = optim.SparseAdam(sparse_p,lr)
        optimizer = optim.Adam(dense_p,lr)
        return sparse_optimizer, optimizer


def gelu(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
