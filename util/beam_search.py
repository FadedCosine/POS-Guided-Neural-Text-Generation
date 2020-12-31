from .sampling import top_k_logits, top_p_logits
import torch
from torch.nn.functional import log_softmax
import heapq

class Sequence(object):
    """Represents a complete or partial sequence."""

    def __init__(self, output, logprob, score, length_normalization_factor=0, length_normalization_const=5.):
        """Initializes the Sequence.
        Args:
          output: List of word ids in the sequence.
          logprob: Log-probability of the sequence.
          score: Score of the sequence.
        """
        self.output = output
        self.logprob = logprob
        if length_normalization_factor > 0:
            L = length_normalization_const
            length_penalty = (L + len(output)) / (L + 1)
            self.score = score / (length_penalty ** length_normalization_factor)
        else:
            self.score = score

    def __cmp__(self, other):
        """Compares Sequences by score."""
        assert isinstance(other, Sequence)
        if self.score == other.score:
            return 0
        elif self.score < other.score:
            return -1
        else:
            return 1

    # For Python 3 compatibility (__cmp__ is deprecated).
    def __lt__(self, other):
        assert isinstance(other, Sequence)
        return self.score < other.score

    # Also for Python 3 compatibility.
    def __eq__(self, other):
        assert isinstance(other, Sequence)
        return self.score == other.score

class TopN(object):
    """Maintains the top n elements of an incrementally provided set."""

    def __init__(self, n):
        self._n = n
        self._data = []

    def size(self):
        assert self._data is not None
        return len(self._data)

    def push(self, x):
        """Pushes a new element."""
        assert self._data is not None
        if len(self._data) < self._n:
            heapq.heappush(self._data, x)
        else:
            heapq.heappushpop(self._data, x)

    def extract(self, sort=False):
        """Extracts all elements from the TopN. This is a destructive operation.
        The only method that can be called immediately after extract() is reset().
        Args:
          sort: Whether to return the elements in descending sorted order.
        Returns:
          A list of data; the top n elements provided to the set.
        """
        assert self._data is not None
        data = self._data
        self._data = None
        if sort:
            data.sort(reverse=True)
        return data

    def reset(self):
        """Returns the TopN to an empty state."""
        self._data = []

@torch.no_grad()
def decode_step(model, x_lens, dec_input, context, top_w, temperature, experimental_loss, beam_size=5, sampling_mode=0, pos_top_w=10):
    """
    docstring
    """
    top_whatever = top_k_logits if isinstance(top_w, int) else top_p_logits
    bs, l = dec_input.size()
    lens = torch.LongTensor([l] * bs).to(dec_input.device)
    if experimental_loss == 1 or experimental_loss == 2 :
        logits = model.decode_step(x_lens, dec_input, lens, context, sampling_mode, top_w)
    elif experimental_loss == 3:
        logits = model.decode_step(x_lens, dec_input, lens, context, sampling_mode, pos_top_w)
    else:
        logits = model.decode_step(x_lens, dec_input, lens, context, 0, None)
    logits = logits.view(bs,l,-1)
    logits = logits[:,-1,:]
    if top_w > 0:
        logits = top_whatever(logits, top_w) / temperature
        # print("logits size is ", logits.size())
        logprobs = log_softmax(logits, dim=-1)
        probs = logprobs.exp().to("cpu")
        # print("probs size is ", probs.size())
        try:
            words = torch.multinomial(probs, num_samples=beam_size)
        except RuntimeError:
            words = torch.multinomial(probs, num_samples=1)
        logprobs_ = []
        for w,l in zip(words, logprobs):
            logprobs_.append(l[w])
    else:
        logprobs = log_softmax(logits, dim=1)
        logprobs_, words = logprobs.topk(beam_size, 1)
        # print("words is ", words)
    return words, logprobs_


@torch.no_grad()
def seq2seq_beam_search(model, max_decoding_len, tokenizer, inp, top_w, temperature, experimental_loss, beam_size=5, sampling_mode=0, pos_top_w=10, sample_num=5, length_normalization_factor=0, length_normalization_const=5.):
    """ Beam Search for seq2seq task

    Args:
        model ([type]): [description]
        max_decoding_len ([type]): [description]
        tokenizer ([type]): [description]
        inp ([type]): [description]
        top_w ([type]): [description]
        temperature ([type]): [description]
        experimental_loss ([type]): [description]
        beam_size (int, optional): [description]. Defaults to 5.
        sampling_mode (int, optional): [description]. Defaults to 0.
        pos_top_w (int, optional): [description]. Defaults to 10.
        length_normalization_factor (float, optional): [description]. Defaults to 0.2.
        length_normalization_const ([type], optional): [description]. Defaults to 5..

    Returns:
        seqs: [[beam_size * Sequence] * batch_size]
    """

    x, x_lens, x_pos, y, y_len, y_pos = inp
    context, enc_mem = model.compute_enc_context(x, x_lens)
    batch_size, seq_len = x.size()
    dec_result = [[] for i in range(batch_size)]
    existence = [True] * batch_size
    num_left = batch_size
    initial_input = torch.ones(batch_size, 1).fill_(tokenizer.bos_id).type_as(x).to(x.device)
    partial_sequences = [TopN(beam_size) for _ in range(batch_size)]
    complete_sequences = [TopN(beam_size) for _ in range(batch_size)]

    words, logprobs = decode_step(model, x_lens, initial_input, context, top_w, temperature, experimental_loss, beam_size=beam_size, sampling_mode=sampling_mode, pos_top_w=pos_top_w)
    words = words
    logprobs = logprobs
    for b in range(batch_size):
        # Create first beam_size candidate hypotheses for each entry in
        # batch
        for k in range(len(words[b])):
            seq = Sequence(
                output=[words[b][k].item()],
                logprob=logprobs[b][k].item(),
                score=logprobs[b][k].item(),
                length_normalization_factor=length_normalization_factor,
                length_normalization_const=length_normalization_const)
            partial_sequences[b].push(seq)
            
    # Run beam search.
    for _ in range(max_decoding_len - 1):
        partial_sequences_list = [p.extract() for p in partial_sequences]
        for p in partial_sequences:
            p.reset()
        # Keep a flattened list of parial hypotheses, to easily feed
        # through a model as whole batch
        # flattened_partial : [batch_size * beam_size] Sequence

        flattened_partial = [
            s for sub_partial in partial_sequences_list for s in sub_partial]
        # input_feed : [batch_size * beam_size] input
        input_feed = [c.output for c in flattened_partial]
        if len(input_feed) == 0:
            # We have run out of partial candidates; happens when
            # beam_size=1
            break

        # Feed current hypotheses through the model, and recieve new outputs and states
        # logprobs are needed to rank hypotheses
        input_feed = torch.Tensor(input_feed).type_as(x).to(x.device)
        x_lens_beamed = [] # 在decode_step时需要x_len来做enc_dec_padding_mask，在beam search下，decode的input的[0]维度会因为beam size的搜索而扩张，因此也需要扩张x_len的[0]维度
        context_beamed = [] # context同理
        for batch_idx in range(len(partial_sequences_list)):
            for beam_idx in range(len(partial_sequences_list[batch_idx])):
                x_lens_beamed.append(x_lens[batch_idx].item())
                context_beamed.append(context[batch_idx].tolist())
        x_lens_beamed = torch.Tensor(x_lens_beamed).type_as(x_lens).to(x.device)

        context_beamed = torch.Tensor(context_beamed).type_as(context).to(x.device)
        # print("input_feed size is : ", input_feed.size())
        # print("x_lens_beamed size is : ", x_lens_beamed.size())
        # print("context_beamed size is : ", context_beamed.size())
        words, logprobs = decode_step(model, x_lens_beamed, input_feed, context_beamed, top_w, temperature, experimental_loss, beam_size=beam_size, sampling_mode=sampling_mode, pos_top_w=pos_top_w)
        words = words
        logprobs = logprobs
        idx = 0
        for b in range(batch_size):
            # For every entry in batch, find and trim to the most likely
            # beam_size hypotheses
            for partial in partial_sequences_list[b]:
                for k in range(len(words[idx])):
                    w = words[idx][k].item()
                    output = partial.output + [w]
                    logprob = partial.logprob + logprobs[idx][k].item()
                    score = logprob

                    if w == tokenizer.eos_id:
                        beam = Sequence(output, logprob, score,length_normalization_factor=length_normalization_factor,
                            length_normalization_const=length_normalization_const)
                        complete_sequences[b].push(beam)
                        if length_normalization_factor == 0: # 意味着不进行长度惩罚,此时生成eos之后直接结束当前item文本的beam search
                            partial_sequences[b].reset()
                            break
                            
                    else:
                        beam = Sequence(output, logprob, score,length_normalization_factor=length_normalization_factor,
                            length_normalization_const=length_normalization_const)
                        partial_sequences[b].push(beam)
                idx += 1
    # If we have no complete sequences then fall back to the partial sequences.
    # But never output a mixture of complete and partial sequences because a
    # partial sequence could have a higher score than all the complete
    # sequences.
    for b in range(batch_size):
        if not complete_sequences[b].size():
            complete_sequences[b] = partial_sequences[b]
    seqs = [complete.extract(sort=True)[0] for complete in complete_sequences] # 返回score最大的
    return seqs

