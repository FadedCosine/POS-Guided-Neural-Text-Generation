import torch
import torch.nn as nn


class Residual_FF(nn.Module):
    def __init__(self, hidden_dim:int, projection_dim:int, dropout:float, pre_lnorm=False):
        super(Residual_FF, self).__init__()

        self.hidden_dim = hidden_dim
        self.projection_dim = projection_dim
        self.dropout = dropout

        self.net = nn.Sequential(
            nn.Linear(hidden_dim, projection_dim), nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(projection_dim, hidden_dim),
            nn.Dropout(dropout),
        )

        self.layer_norm = nn.LayerNorm(hidden_dim)

        self.pre_lnorm = pre_lnorm

    def forward(self, x):
        if self.pre_lnorm:
            ##### layer normalization + positionwise feed-forward
            out = self.net(self.layer_norm(x))

            ##### residual connection
            output = out + x
        else:
            ##### positionwise feed-forward
            out = self.net(x)

            ##### residual connection + layer normalization
            output = self.layer_norm(x + out)

        return output

