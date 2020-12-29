import torch
import torch.nn as nn

class Initializer:
    def __init__(self,mode:str,value:float,dropout:float):
        self.mode = mode
        self.value = value
        self.dropout = dropout

    def init_weight(self,weight):
        if self.mode == 'uniform':
            nn.init.uniform_(weight, -self.value, self.value)
        elif self.mode == 'normal':
            nn.init.normal_(weight, 0.0, self.value)

    def init_bias(self,bias):
        nn.init.constant_(bias, 0.0)

    def case_initialize(self,m):
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            if hasattr(m, 'weight') and m.weight is not None:
                self.init_weight(m.weight)
            if hasattr(m, 'bias') and m.bias is not None:
                self.init_bias(m.bias)

        elif classname.find('Multihead_Att') != -1:
            if hasattr(m, 'vec_u'):
                self.init_weight(m.vec_u)
            if hasattr(m, 'vec_v'):
                self.init_weight(m.vec_v)

        elif classname.find('Factorized_SoftmaxV2') != -1:
            if hasattr(m, 'logits'):
                self.init_weight(m.logits)

        elif classname.find('POS_Guided_Softmax') != -1:
            if hasattr(m, 'logits'):
                self.init_weight(m.logits)

        elif classname.find('Transformer_Decoder') != -1:
            if hasattr(m, 'rw_bias'):
                self.init_weight(m.rw_bias)
            if hasattr(m, 'rr_bias'):
                self.init_weight(m.rr_bias)

        elif classname.find('Two_Stream_Multihead_Att') != -1:
            if hasattr(m, 'vec_u'):
                self.init_weight(m.vec_u)
            if hasattr(m, 'vec_v'):
                self.init_weight(m.vec_v)

        elif classname.find('XLNet') != -1:
            if hasattr(m, 'vec_g'):
                self.init_weight(m.vec_g)

        elif classname.find('Embedding') != -1:
            if hasattr(m, 'weight'):
                self.init_weight(m.weight)

        elif classname.find('LayerNorm') != -1:
            if hasattr(m, 'weight'):
                nn.init.normal_(m.weight, 1.0, self.value)
            if hasattr(m, 'bias') and m.bias is not None:
                self.init_bias(m.bias)

    def initialize(self, m):
        m.apply(self.case_initialize)
