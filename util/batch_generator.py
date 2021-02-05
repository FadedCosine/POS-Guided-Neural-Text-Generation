import pandas as pd
import random
import numpy as np
import torch
from torch.utils.data import IterableDataset, Dataset, DataLoader
from keras.preprocessing.sequence import pad_sequences
import math

class Base_Batchfier(IterableDataset):
    def __init__(self, batch_size: int = 32, seq_len=512, minlen=50, maxlen: int = 4096,
                 criteria: str = 'lens',
                 padding_index=70000, epoch_shuffle=False,device='cuda'):
        super(Base_Batchfier).__init__()
        self.maxlen = maxlen
        self.minlen = minlen
        self.size = batch_size
        self.criteria = criteria
        self.seq_len = seq_len
        self.padding_index = padding_index
        self.epoch_shuffle = epoch_shuffle
        self.device = device
        # self.size = len(self.df) / num_buckets

    def truncate_small(self, df, criteria='lens'):
        lens = np.array(df[criteria])
        indices = np.nonzero((lens < self.minlen).astype(np.int64))[0]
        return df.drop(indices)

    def truncate_large(self, texts, lens):
        new_texts = []
        new_lens = []
        for i in range(len(texts)):
            text = texts[i]
            if len(text) > self.maxlen:
                new_texts.append(text[:self.maxlen])
                new_lens.append(self.maxlen)
            else:
                remainder = len(text) % self.seq_len
                l = lens[i]
                if remainder and remainder < 10:
                    text = text[:-remainder]
                    l = l - remainder
                new_texts.append(text)
                new_lens.append(l)
        return new_texts, new_lens

    def shuffle(self, df, num_buckets):
        dfs = []
        for bucket in range(num_buckets):
            new_df = df.iloc[bucket * self.size: (bucket + 1) * self.size]
            dfs.append(new_df)
        random.shuffle(dfs)
        df = pd.concat(dfs)
        return df

    def sort(self, df, criteria):
        return df.sort_values(criteria).reset_index(drop=True)



class Lyrics_Batchfier(Base_Batchfier):
    def __init__(self, filelist: list, batch_size: int = 32, seq_len=512, minlen=50, maxlen: int = 4096,
                 criteria: str = 'lens',
                 padding_index=70000, epoch_shuffle=False, device='cuda'):
        super(Lyrics_Batchfier,self).__init__(batch_size,seq_len,minlen,maxlen,criteria,
                                              padding_index,epoch_shuffle,device)
        self.filelist = filelist

    def iterator(self):
        for filename in self.filelist:
            cur_df = pd.read_pickle(filename)
            if self.epoch_shuffle:
                cur_df = self.truncate_small(cur_df)
                num_buckets = len(cur_df) // self.size + (len(cur_df) % self.size !=0)
                cur_df = self.sort(cur_df,self.criteria)
                cur_df = self.shuffle(cur_df, num_buckets)
            cur_pos = 0
            while cur_pos < len(cur_df):
                cur_batch = cur_df.iloc[cur_pos :cur_pos+self.size]
                cur_pos += self.size
                texts = cur_batch['texts'].to_list()
                lens = cur_batch['lens'].to_list()
                texts, lens = self.truncate_large(texts,lens)
                maxlen = max(lens)
                n_chunk = maxlen // self.seq_len + (maxlen % self.seq_len !=0)
                for chunk in range(n_chunk):
                    for i in range(len(texts)):
                        text = texts[i][chunk * self.seq_len: (chunk + 1) * self.seq_len]
                        text_len = max(min(lens[i], self.seq_len),0)
                        lens[i]-= self.seq_len
                        yield text, text_len

    def __iter__(self):
        return self.iterator()


    def __len__(self):
        ## hard coded should be fixed
        return 30000

    def collate(self, batch):
        texts = [torch.Tensor(item[0]).long() for item in batch]
        lens = torch.Tensor([item[1] for item in batch]).long()
        texts = torch.nn.utils.rnn.pad_sequence(texts, batch_first=True, padding_value=self.padding_index)
        return texts.to(self.device), lens.to(self.device), texts[:,1:].to(self.device)



class LyricsSampleBatchfier:
    def __init__(self, filelist: list, batch_size: int = 32,
                 n_samples=10000, prefix_len=5, token_len=100, device='cuda'):
        self.filelist = filelist
        self.batch_size = batch_size
        self.device = device
        self.n_samples = n_samples
        self.prefix_len = prefix_len
        self.token_len = token_len

    def __iter__(self):
        cur_df = pd.read_pickle(self.filelist[0])
        tot_generated = 0
        cur = 0
        textlen = self.token_len + self.prefix_len
        while tot_generated < self.n_samples:
            tot_generated += self.batch_size
            batch = cur_df.iloc[cur:cur + self.batch_size]
            cur += self.batch_size
            batch_text = [text[:textlen] for text in batch['texts'].tolist() if len(text) > textlen]
            short = self.batch_size - len(batch_text)
            while short != 0:
                if cur >= len(cur_df):
                    tot_generated = self.n_samples
                    break
                added = cur_df.iloc[cur:cur + short]
                added = [text[:textlen] for text in added['texts'].tolist() if len(text) > textlen]
                cur += short
                batch_text += added
                short = self.batch_size - len(batch_text)
            x = torch.LongTensor(batch_text).to(self.device)
            yield x, torch.LongTensor([self.prefix_len] * self.batch_size).to(self.device), x[:, 1:]


class BpttIterator:
    def __init__(self, dataset, batch_size, bptt_len, device='cuda', **kwargs):
        self.bptt_len = bptt_len
        self.dataset = dataset
        self.size = batch_size
        self.device = device
        self.iterations = 0
        self.data = self.prepair_dataset(dataset)

    def prepair_dataset(self, text):
        remainder= len(text) % self.size
        if remainder:
            text = text[:-remainder]
        data = np.array(text).reshape((self.size,-1))
        return data

    def __len__(self):
        return math.ceil((len(self.dataset) / self.size - 1)
                         / self.bptt_len)
    def len(self):
        return math.ceil((len(self.dataset) / self.size - 1)
                         / self.bptt_len)
    def __iter__(self):
        cur = 0
        data = self.data
        while cur < self.data.shape[1]:
            self.iterations += 1
            batch_text = data[:,cur:cur + self.bptt_len]
            x = torch.from_numpy(batch_text).to(self.device)
            cur+=self.bptt_len
            yield x, torch.LongTensor([batch_text.shape[1]]*self.size).to(self.device), x[:,1:]

class BpttIteratorWithPOS:
    def __init__(self, dataset, pos_dataset, batch_size, bptt_len, device='cuda', **kwargs):
        self.bptt_len = bptt_len
        self.dataset = dataset
        self.pos_dataset = pos_dataset
        self.size = batch_size
        self.device = device
        self.iterations = 0
        self.data = self.prepair_dataset(dataset)
        self.pos_data = self.prepair_dataset(pos_dataset)

    def prepair_dataset(self, text):
        # 两行为一条数据，因为包含source sentence 和 target sentence
        remainder= len(text) % (self.size * 2)
        if remainder:
            text = text[:-remainder]
        data = np.array(text).reshape((self.size * 2,-1))
        return data

    def __len__(self):
        return math.ceil((len(self.dataset) / self.size - 1)
                         / self.bptt_len)
    def len(self):
        return math.ceil((len(self.dataset) / self.size - 1)
                         / self.bptt_len)

    def __iter__(self):
        cur = 0
        data = self.data
        pos_data = self.pos_data
        while cur < self.data.shape[1]:
            self.iterations += 1
            batch_text = data[:,cur:cur + self.bptt_len]
            batch_pos_text = pos_data[:,cur:cur + self.bptt_len]
            x = torch.from_numpy(batch_text).to(self.device)
            pos_x = torch.from_numpy(batch_pos_text).to(self.device)
            cur+=self.bptt_len
            yield x, torch.LongTensor([batch_text.shape[1]]*self.size).to(self.device), pos_x, x[:,1:], torch.LongTensor([batch_text.shape[1]-1]*self.size).to(self.device), pos_x[:,1:]

"""
此时是seq2seq的问题，就要考虑BOS, EOS和padding的问题了，
输入的dataset和pos_dataset，每两个list为一组数据，source sentence 和 target sentence的前后都加上了bos和eos， 如果source sentence不需要bos可以在此删掉
ParaIterator要对一个batch做padding
ParaIteratorWithPOS的iter的输出也就和BpttIteratorWithPOS的一样 x， x_len, y, y_pos
"""
class ParaIteratorWithPOS(IterableDataset):
    def __init__(self, dataset, pos_dataset, batch_size, tokenizer, max_seq_len=384, device='cuda', **kwargs):
        self.max_seq_len = max_seq_len
        self.dataset = dataset
        self.pos_dataset = pos_dataset
        self.size = batch_size
        self.device = device
        self.iterations = 0
        self.tokenizer = tokenizer
        self.data = self.prepair_dataset(dataset)
        self.pos_data = self.prepair_dataset(pos_dataset)
    def prepair_dataset(self, text):
        remainder= len(text) % (self.size * 2)
        if remainder:
            text = text[:-remainder]
        
        data = np.array(text)
        return data
    def __len__(self):
        return len(self.data) 
    def len(self):
        return len(self.data) // (self.size * 2)

    def __iter__(self):
        """
        data [np.array]， [line_num, sentence len + 2]
        每两行要分成x和y，删除x的bos，并对一个batch做padding
        """
        cur_line = 0
        data = self.data
        pos_data = self.pos_data
        while cur_line < len(data):
            self.iterations += 1
            x = data[cur_line]
            pos_x = pos_data[cur_line]
            y = data[cur_line + 1]
            pos_y = pos_data[cur_line + 1]
            cur_line += 2
            yield x, len(x), pos_x, y, len(y), pos_y

    def collate(self, batch):
        """
        for batch padding
        """
     
        x = [item[0] for item in batch]
        x_l = [item[1] for item in batch]
        pos_x = [item[2] for item in batch]
        y = [item[3] for item in batch]
        y_l = [item[4] for item in batch]
        pos_y = [item[5] for item in batch]
    
        x_l = torch.LongTensor(x_l)
        y_l = torch.LongTensor(y_l)
     
        x_texts = torch.from_numpy(pad_sequences(x, maxlen=self.max_seq_len, dtype="long", 
                          value=self.tokenizer.padding_id, truncating="post", padding="post"))
        pos_x = torch.from_numpy(pad_sequences(pos_x, maxlen=self.max_seq_len, dtype="long", 
                          value=self.tokenizer.padding_id, truncating="post", padding="post"))
        y_texts = torch.from_numpy(pad_sequences(y, maxlen=self.max_seq_len, dtype="long", 
                          value=self.tokenizer.padding_id, truncating="post", padding="post"))
        pos_y = torch.from_numpy(pad_sequences(pos_y, maxlen=self.max_seq_len, dtype="long", 
                          value=self.tokenizer.padding_id, truncating="post", padding="post"))
        
        return x_texts.to(self.device), x_l.to(self.device), pos_x.to(self.device), y_texts.to(self.device), y_l.to(self.device), pos_y.to(self.device)


class BpttSamplingIterator:
    def __init__(self, dataset, batch_size, prefix_len, generate_len, device='cuda', **kwargs):
        self.prefix_len = prefix_len
        self.generate_len = generate_len
        self.pg_len = prefix_len + generate_len
        self.dataset = dataset
        self.size = batch_size
        self.device = device
        self.iterations = 0
        self.data = self.prepair_dataset(dataset)

    def prepair_dataset(self, text):
        remainder= len(text) % self.size
        if remainder:
            text = text[:-remainder]
        data = np.array(text).reshape((self.size,-1))
        return data

    def __len__(self):
        return self.data.shape[1] // self.pg_len
    def len(self):
        return self.data.shape[1] // self.pg_len

    def __iter__(self):
        cur = 0
        data = self.data
        while cur < self.data.shape[1]:
            self.iterations += 1
            batch_text = data[:,cur:cur + self.pg_len]
            x = torch.from_numpy(batch_text).to(self.device)
            cur+=self.pg_len
            yield x, torch.LongTensor([batch_text.shape[1]]*self.size).to(self.device), x[:,1:]

# TODO: SamplingIterator 和原来用作训练的Itertor有啥区别呢？
# 因为原来的Wiki103是LM任务，在训练和测试的时候输入是不一样的，测试的输入是前50个token，输出后100个token，yield的最后一项压根就没用到