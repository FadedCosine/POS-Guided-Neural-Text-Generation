import torch
import torch.nn as nn
import torch.nn.functional as F
from util.sampling import top_p_logits, top_k_top_p_filtering, pos_filtering
from .ops import gelu
import numpy as np
import logging
logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                    datefmt='%d-%m-%Y:%H:%M:%S')
logging.getLogger().setLevel(logging.DEBUG)
logger = logging.getLogger(__name__)
class Adaptive_Softmax(nn.Module):
    def __init__(self,vocab_size:int,hidden_dim:int,cutoffs:list,div_val:int):
        super(Adaptive_Softmax, self).__init__()
        self.n_clusters = len(cutoffs)
        self.head_size = cutoffs[0] + self.n_clusters
        self.cutoffs = [0] + cutoffs + [vocab_size]

        self.cluster_logit = nn.Linear(hidden_dim,self.n_clusters)
        self.head_size = cutoffs[0] + self.n_clusters

        self.projections = nn.ModuleList()
        self.logits = nn.ModuleList()
        self.proj_dims = [hidden_dim // (div_val**i) for i in range(self.n_clusters+1)]
        for i in range(self.n_clusters+1):
            n_vocabs = self.cutoffs[i] + self.cutoffs[i+1]
            self.projections.append(nn.Linear(hidden_dim,self.proj_dims[i],bias=False))
            self.logits.append(nn.Linear(self.proj_dims[i],n_vocabs))

    def forward(self, x,y):
        """
        :param x: final hidden state x.size() = [batch_size*seq_len,hidden_dim]
        :param y: target y.size() = [batch_size*seq_len]
        :return:
        """
        head_proj = self.projections[0](x)
        head_logit = torch.cat([self.logits[0](head_proj),self.cluster_logit(head_proj)],1)
        head_logprob = torch.log_softmax(head_logit, dim=1)

        nll = torch.zeros_like(y,
                               dtype=x.dtype, device=x.device)

        for i in range(len(self.cutoffs)-1):
            l,r = self.cutoffs[i], self.cutoffs[i+1]
            mask = (y >= l) & (y < r)
            indices = mask.nonzero().squeeze()
            if indices.numel() == 0:
                continue
            target_i = y[indices] - l
            head_logprob_i = head_logprob[indices]
            if i == 0:
                logprob_i = head_logprob_i.gather(1, target_i[:, None]).squeeze(1)
            else:
                tail_proj = self.projections[i](x[indices])
                tail_logit = self.logits[i](tail_proj)
                tail_logprob_i = torch.log_softmax(tail_logit, dim=1)
                logprob_i = head_logprob_i[:, -i] \
                            + tail_logprob_i.gather(1, target_i[:, None]).squeeze(1)
            nll[indices] = -logprob_i
        return nll


class Factorized_Softmax(nn.Module):
    def __init__(self,vocab_size:int,hidden_dim:int,cutoffs:list,padding_index:int, activation=gelu):
        super(Factorized_Softmax, self).__init__()
        self.padding_index = padding_index
        self.vocab_size = vocab_size
        self.n_clusters = len(cutoffs) + 1
        self.cutoffs = [0] + cutoffs + [vocab_size]
        self.cluster_logit = nn.Linear(hidden_dim, self.n_clusters, bias=False)
        # [hidden_dim, vocab_size]的参数矩阵
        self.logits = nn.Parameter(torch.Tensor(hidden_dim, vocab_size))
        self.transform = nn.ModuleList()
        self.layer_norm = nn.ModuleList()
        self.activation=activation
        for i in range(self.n_clusters):
            self.transform.append(nn.Sequential(nn.Linear(hidden_dim, hidden_dim)))
            self.layer_norm.append(nn.LayerNorm(hidden_dim))

    def hard_cluster_logit(self,x, top_w, ishard=True):
        def check_shorts(logits, top_w):
            if isinstance(top_w,int): #if case is top_k
                # print(((logits != 0).sum(dim=1) < top_w).sum())
                res = (logits != 0).sum(dim=1) < top_w
            elif isinstance(top_w,float): #if case is top_p
                res = logits.sum(dim=1) < top_w
            else:
                raise TypeError('type of top_w should be either int or float')
            return res

        logits = torch.zeros(x.size(0),self.vocab_size).to(x.device)
        cl = self.cluster_logit(x)
        cl_probs = torch.softmax(cl,1)

        if ishard:
            # cl : [batch_size * seq_len , n_clusters]
            _, target_cluster = torch.topk(cl,self.n_clusters, dim=1)
            # target_cluster : [batch_size * seq_len , n_clusters], 排序之后的idx
        else:
            cl = top_p_logits(cl,0.6)
            target_cluster = torch.multinomial(torch.softmax(cl,1) + 1e-6, self.n_clusters)
        idx = 0
        # target_cluster 中存着 batch_size * seq_len这些个位置的 n_clusters个所属cluster的可能
        while True:
            cs = check_shorts(logits,top_w)
            if cs.sum() == 0:
                break
            for i in range(self.n_clusters):
                l,r = self.cutoffs[i], self.cutoffs[i+1]
                indices = ((target_cluster[:,idx] == i) & cs).nonzero().squeeze(1)
                transformed = self.layer_norm[i](self.activation(self.transform[i](x[indices])))
                tail = torch.softmax(torch.matmul(transformed, self.logits[:,l:r]),1)
                logits[indices,l:r] = cl_probs[indices,i].unsqueeze(1) * tail
            idx+=1
        # 相较于soft_cluster_logit，hard的logits中并不包括所有token的概率，而是只包括了满足top k或者top p条件的token的概率
        return torch.log(logits)


    def soft_cluster_logit(self,x):
        logits = torch.zeros(x.size(0), self.vocab_size).to(x.device)
        cl = self.cluster_logit(x)
        cluster_prob = torch.softmax(cl,dim=1) # [ batch, n_cluster]
        for i in range(self.n_clusters):
            l,r = self.cutoffs[i], self.cutoffs[i+1]
            
            logits_weights = self.logits[:,l:r]
            transformed = self.layer_norm[i](self.activation(self.transform[i](x)))
            tail_logit = torch.matmul(transformed, logits_weights)
            tail_prob = torch.softmax(tail_logit,1)
            # print(cluster_prob[:,i].size(),tail_prob.size())
            # 每一次循环将c_t中的token概率赋值
            logits[:,l:r] = cluster_prob[:,i].unsqueeze(1) * tail_prob
        # 实际上logits中就包括了所有token的概率
        return torch.log(logits)

    def forward(self, x,y):
        """
        :param x: final hidden state x.size() = [batch_size*seq_len,hidden_dim]
        :param y: target y.size() = [batch_size*seq_len]
        :return:
        """
        ny = y.size(0)
        cl = self.cluster_logit(x)
        cluster_ll = torch.log_softmax(cl, dim=1)
        nll = torch.zeros_like(y,
                               dtype=x.dtype, device=x.device)

        for i in range(self.n_clusters):
            l,r = self.cutoffs[i], self.cutoffs[i+1]
            # 找到那些y在当前cluster当中的y
            mask = (y >= l) & (y < r)
            indices = mask.nonzero().squeeze(1)
            # 原来logits是[hidden_dim, vocab_size]，之后是[hidden_dim, r - l], 参数矩阵中的[l]列到[r-1]列，也就是，只是一个全连接的参数权重矩阵
            logits_weights = self.logits[:,l:r]
            if indices.numel() == 0:
                continue
            target_i = y[indices] - l
            
            # [indices_size, hidden_dim]
            transformed = self.layer_norm[i](self.activation(self.transform[i](x[indices])))
            # Decoupled Decoding， [indices_size,  r - l]
            # tail_logit得到的就是 P(x_t|c_t, x_{<t})
            tail_logit = torch.matmul(transformed, logits_weights)
            tail_logprob_i = torch.log_softmax(tail_logit, dim=1) # [b, vocab]
            # word_nll[indices] = -logprob_i
            nll[indices] = - cluster_ll[indices, i] - tail_logprob_i.gather(1,target_i[:,None]).squeeze(1)
        padding_mask = y == self.padding_index
        padding_indices = padding_mask.nonzero().squeeze(1)
        padding_size = padding_indices.size(0)
        nll[padding_indices] = 0
        return torch.sum(nll) / (ny-padding_size)

"""
V2的F2-softmax与V1的区别是，没有再对不同cluster的x的hidden_state在过一层全连接层
"""
class Factorized_SoftmaxV2(nn.Module):
    def __init__(self,vocab_size:int,hidden_dim:int,cutoffs:list,padding_index:int):
        super(Factorized_SoftmaxV2, self).__init__()
        self.padding_index = padding_index
        self.vocab_size = vocab_size
        self.n_clusters = len(cutoffs) + 1
        self.cutoffs = [0] + cutoffs + [vocab_size]
        self.cluster_logit = nn.Linear(hidden_dim, self.n_clusters, bias=False)
        self.logits = nn.Parameter(torch.Tensor(hidden_dim, vocab_size))

    def hard_cluster_logit(self,x, top_w, ishard=True):
        def check_shorts(logits, top_w):
            if isinstance(top_w,int): #if case is top_k
                # print(((logits != 0).sum(dim=1) < top_w).sum())
                res = (logits != 0).sum(dim=1) < top_w
            elif isinstance(top_w,float): #if case is top_p
                res = logits.sum(dim=1) < top_w
            else:
                raise TypeError('type of top_w should be either int or float')
            return res

        logits = torch.zeros(x.size(0),self.vocab_size).to(x.device)
        cl = self.cluster_logit(x)
        cl_probs = torch.softmax(cl,1)

        if ishard:
            _, target_cluster = torch.topk(cl,self.n_clusters, dim=1)
        else:
            cl = top_p_logits(cl, 0.9)
            target_cluster = torch.multinomial(torch.softmax(cl,1) + 1e-6, self.n_clusters)
        idx = 0
        while True:
            cs = check_shorts(logits,top_w)
            if cs.sum() == 0:
                break
            for i in range(self.n_clusters):
                l,r = self.cutoffs[i], self.cutoffs[i+1]
                indices = ((target_cluster[:,idx] == i) & cs).nonzero().squeeze(1)
         
                # if indices.size()[0] == 0:
                #     continue
                tail = torch.softmax(torch.matmul(x[indices], self.logits[:,l:r]),1)
                logits[indices,l:r] = cl_probs[indices,i].unsqueeze(1) * tail
            idx+=1
  
        return torch.log(logits)

    def soft_cluster_logit(self,x):
        logits = torch.zeros(x.size(0), self.vocab_size).to(x.device)
        cl = self.cluster_logit(x)
        cluster_prob = torch.softmax(cl,dim=1) # [ batch, n_cluster]
        for i in range(self.n_clusters):
            l,r = self.cutoffs[i], self.cutoffs[i+1]
            tail_prob = torch.softmax(torch.matmul(x, self.logits[:,l:r]),1)
            # print(cluster_prob[:,i].size(),tail_prob.size())
            logits[:,l:r] = cluster_prob[:,i].unsqueeze(1) * tail_prob
        return torch.log(logits)

    def forward(self, x,y):
        """
        :param x: final hidden state x.size() = [batch_size*seq_len,hidden_dim]
        :param y: target y.size() = [batch_size*seq_len]
        :return:
        """
        ny = y.size(0)
        cl = self.cluster_logit(x)
        cluster_ll = torch.log_softmax(cl, dim=1)
        nll = torch.zeros_like(y,
                               dtype=x.dtype, device=x.device)

        for i in range(self.n_clusters):
            l,r = self.cutoffs[i], self.cutoffs[i+1]
            mask = (y >= l) & (y < r)
            indices = mask.nonzero().squeeze(1)
            logits_weights = self.logits[:,l:r]
            if indices.numel() == 0:
                continue
            target_i = y[indices] - l
            tail_logit = torch.matmul(x[indices],logits_weights)
            tail_logprob_i = torch.log_softmax(tail_logit, dim=1) # [b,vocab]
            # word_nll[indices] = -logprob_i
            nll[indices] = - cluster_ll[indices, i] - tail_logprob_i.gather(1,target_i[:,None]).squeeze(1)
        return nll
        # return torch.sum(nll) / (ny-padding_size)

"""
p(x_t | x_{<t}) = p1(pos_t | x_{<t}) * p2(x_t | pos_t, x_{<t})
"""
class POS_Guided_Softmax(nn.Module):
    """
    实现同Factorized_SoftmaxV2，没有再对不同cluster的x的hidden_state在过一层全连接层

    """
    def __init__(self, vocab_size:int, hidden_dim:int, pos2token:list, token_in_pos_id:list, padding_index:int):
        super(POS_Guided_Softmax, self).__init__()
        self.padding_index = padding_index
        self.vocab_size = vocab_size
        self.pos2token = pos2token
        self.token_in_pos_id = token_in_pos_id
        self.n_clusters = len(pos2token)
        self.cluster_logit = nn.Linear(hidden_dim, self.n_clusters, bias=False)
        self.logits = nn.Parameter(torch.Tensor(hidden_dim, vocab_size))
    
    def hard_cluster_logit(self, x, top_w, ishard=True):
        """用于预测阶段的采样生成文本，top_w可以是top p也可以是top k，依据其类型而定

        Args:
            x ([torch.Tensor]): final hidden state , x.size() = [batch_size * seq_len, hidden_dim]
            top_w ([int, float]): top p 或者是 top k
            ishard (bool, optional): 影响target_pos的采样，如果为真，则使用topk按概率从高到低返回pos的index，
                                            否则，使用topp p=0.9，之后采样n_clusters个pos
        """
        def check_shorts(logits, top_w):
            """
            如果logits当中的概率非0的小于top_w【top k的情况】，则return True
            如果logits当中的概率之和的小于top_w【top p的情况】，则return True
            """
            if isinstance(top_w,int): #if case is top_k
                # print(((logits != 0).sum(dim=1) < top_w).sum())
                res = (logits != 0).sum(dim=1) < top_w
            elif isinstance(top_w,float): #if case is top_p
                res = logits.sum(dim=1) < top_w
            else:
                raise TypeError('type of top_w should be either int or float')
            return res
        logits = torch.zeros(x.size(0),self.vocab_size).to(x.device)
        # print("logits size is : ", logits.size())
        cl = self.cluster_logit(x)
        cl_probs = torch.softmax(cl,1)
        if ishard:
            _, target_pos = torch.topk(cl,self.n_clusters, dim=1)
        else:
            cl = top_p_logits(cl,0.9)
            target_pos = torch.multinomial(torch.softmax(cl,1) + 1e-6, self.n_clusters)
        # target_pos 中保存了 n_clusters 个 pos的采样结果, [batch_size*seq_len, n_clusters]
        idx = 0
        while True:
            cs = check_shorts(logits,top_w)
            if cs.sum() == 0:
                # 当前的logits满足top p和top k的条件时，这返回
                break
            for i in range(self.n_clusters):
                # 对输入的target_pos的[idx]的pos，计算此pos下voab的概率
                indices = ((target_pos[:,idx] == i) & cs).nonzero().squeeze(1)
                # tail： [pos_i_len, hidden_dim] * [hidden_dim, pos_i_vocab_size] = [pos_i_len, pos_i_vocab_size]
                tail = torch.softmax(torch.matmul(x[indices], self.logits[:, self.pos2token[i]]),1)
                logits[np.ix_(indices.cpu().numpy(), self.pos2token[i])] += cl_probs[indices,i].unsqueeze(1) * tail
                
            idx+=1
        return torch.log(logits)

    def pos_sampling(self, x, pos_top_w=None):
        """
        pos sampling，先采样出pos，再在此pos的token_list
        """
        #! pos-train训出来的模型预测pos 会把EOS的概率预测得很高，导致BOS之后直接生成EOS。
        logits = torch.zeros(x.size(0),self.vocab_size).to(x.device)
        pos_logits = self.cluster_logit(x)
        # top_k_top_p_filtering必须输入不经过softmax的logits
        filtered_pos_logits = F.softmax(top_k_top_p_filtering(pos_logits, pos_top_w), dim=-1)
        pos_prev = filtered_pos_logits.multinomial(num_samples=1).contiguous().squeeze(-1)
        # print("pos_prev is : ", pos_prev)
        for idx, pos in enumerate(pos_prev):
            tok_lists = self.pos2token[pos]
            # idx 是logits的第一个维度，即[idx]个token位置，tok_lists是当前token位置预测的pos所对应的token list
            tail_prob = torch.softmax(torch.matmul(x[idx], self.logits[:, tok_lists]),-1)
            # 下式不会再乘一个filtered_pos_logits，因为pos已经采样出来了
            logits[idx, tok_lists] = tail_prob
        
        # 或者换一种写法，遍历pos词表，同hard_cluster_logit
        # for i in range(self.n_clusters):
        #     indices = (pos_prev == i).nonzero().squeeze(1)
        #     tail = torch.softmax(torch.matmul(x[indices], self.logits[:, self.pos2token[i]]),1)
        #     logits[np.ix_(indices.cpu().numpy(), self.pos2token[i])] += filtered_pos_logits[indices,i].unsqueeze(1) * tail
                
        # filtered_tok_logits = pos_filtering(pos_prev, x, pos2word_dir=self.pos2token)
  
        return torch.log(logits)

    def soft_cluster_logit(self, x):
        """与hard_cluster_logit的区别是，hard的方法满足了top_w就停止，而soft的方法考虑所有pos的可能

        Args:
            x ([torch.Tensor]): final hidden state , x.size() = [batch_size * seq_len, hidden_dim]

        """
        logits = torch.zeros(x.size(0), self.vocab_size).to(x.device)
        cl = self.cluster_logit(x)
        cluster_prob = torch.softmax(cl,dim=1) # [ batch, n_cluster]
        for i in range(self.n_clusters):
            tail_prob = torch.softmax(torch.matmul(x, self.logits[:, self.pos2token[i]]),1)
            # print(cluster_prob[:,i].size(),tail_prob.size())
            logits[:, self.pos2token[i]] += cluster_prob[:,i].unsqueeze(1) * tail_prob
        return torch.log(logits)

    def forward(self, x, y, y_pos):
        """
        :param x: final hidden state x.size() = [batch_size * seq_len, hidden_dim]
        :param y: target y.size() = [batch_size * seq_len]
        :param y_pos: target's POS y_pos.size() = [batch_size * seq_len]
        :return:
        """
        ny = y.size(0)
        cl = self.cluster_logit(x)
        cluster_ll = torch.log_softmax(cl, dim=1)
        nll = torch.zeros_like(y,
                               dtype=x.dtype, device=x.device)
        for i in range(self.n_clusters):
            # 选出y_pos中pos是当前i的位置
            mask = (y_pos == i)
            indices = mask.nonzero().squeeze(1)
            # 选出当前pos的token_list对应的权重矩阵， [hidden_dim, pos_i_vocab_size]
            logits_weights = self.logits[:, self.pos2token[i]]
            if indices.numel() == 0:
                continue
            # 得到当前需要用于更新的那些y，在当前pos的词表中的id
            target_i = self.token_in_pos_id[i][y[indices]].long()
            
            # tail_logit : [pos_i_len, pos_i_vocab_size]
            tail_logit = torch.matmul(x[indices], logits_weights)
            tail_logprob_i = torch.log_softmax(tail_logit, dim=1) # [b,vocab]
           
            nll[indices] = - cluster_ll[indices, i] - tail_logprob_i.gather(1,target_i[:,None]).squeeze(1)
        return nll


class LinearTransform(nn.Module):
    def __init__(self,hidden_states:int,activation_fn):
        super(LinearTransform, self).__init__()
        self.linear = nn.Linear(hidden_states,hidden_states)
        self.ln = nn.LayerNorm(hidden_states)
        self.activation_fn = activation_fn

    def forward(self, x):
        x = self.activation_fn(self.linear(x))
        return self.ln(x)
