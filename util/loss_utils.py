import torch

def generate_batch(model,inp,completion_length):
     # mem, inp = get_mem(model, inp)
    model.eval()
    res = inp[0]
    bsz,prefix_length=inp[0].size()
    lprobs=None
    temp_inp = torch.LongTensor([0] * bsz).to("cuda")

    for _ in range(completion_length):
        inp_id,lens=inp
        inp_id=torch.cat([inp_id, temp_inp[:, None]], -1)
        lens+=1
        inp=(inp_id,lens)

        if model.experimental_loss:
            logits, new_mem = model.sampling(inp + (None, False, 1))
        else:
            logits, new_mem = model(inp + (None, None))
        # mem = [torch.cat([mem[i], new_mem[i][:, :-1]], 1) for i in range(len(mem))]
        logits=logits.view(bsz,-1,logits.size(-1))[:,-1]

        if lprobs is not None:
            lprobs=torch.cat([lprobs,logits[:,None]],1)
        else:
            lprobs = logits[:,None]
        sampled=torch.argmax(logits,-1,keepdim=True)
        res = torch.cat([res, sampled], 1)

        # if model.experimental_loss:
        #     title, cind, tls, cls = inp
        #     cind = sampled
        #     inp = (title, cind, tls, cls)
        # else:
        inp,lens=inp
        inp=torch.cat([inp[:,:-1],sampled],-1)
        inp=(inp,lens)

#        print(res[:,prefix_length:].shape)
#        print(lprobs.shape)

    return res[:,prefix_length:].contiguous(),lprobs.contiguous()


def batch_input_sequence_by_prefix_length(input_sequence, prefix_length):
    seq_len = input_sequence.size(1)
    new_seq_len = (seq_len//prefix_length)*prefix_length
    input_sequence = input_sequence[:, :new_seq_len]
    batch = input_sequence.reshape(-1, prefix_length).contiguous()
    return batch


def ngram_repeat_mask(xs, n):
    mask = torch.zeros_like(xs)
    for i, x in enumerate(xs):
        seen = set()
        xl = x.tolist()
        for j in range(len(x)-n):
            ng = tuple(xl[j:j+n])
            if ng in seen:
                mask[i, j:j+n] = 1
            seen.add(ng)
    return mask
