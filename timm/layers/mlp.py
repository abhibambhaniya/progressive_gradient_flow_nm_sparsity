""" MLP module w/ dropout and configurable activation layer

Hacked together by / Copyright 2020 Ross Wightman
"""
from torch import nn as nn

from .helpers import to_2tuple

from ..sparse_pruning import sparse_functions as sf

class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, bias=True, drop=0., sparseConfig=None):
        super().__init__()
        self.sparseConfig=sparseConfig
        self.sparsity_type = sparseConfig.sparsity_type
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)

        if sparseConfig.sparsity_type is not None and sparseConfig.sparsity_type != 'DENSE' and 'FF' in sparseConfig.sparsity_loc:
            self.fc1 = sf.SparseLinear(in_features,hidden_features,bias=bias[0],sparseConfig=self.sparseConfig)
        else:
            print("Dense linear")
            self.fc1 = nn.Linear(in_features, hidden_features, bias=bias[0])

        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])

        if sparseConfig.sparsity_type is not None and sparseConfig.sparsity_type != 'DENSE' and 'FF' in sparseConfig.sparsity_loc:
            self.fc2 = sf.SparseLinear(hidden_features, out_features, bias=bias[1],sparseConfig=self.sparseConfig)
        else:
            print("Dense linear")
            self.fc2 = nn.Linear(hidden_features, out_features, bias=bias[1])

        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x, current_step=0,current_epoch=0):
        try:
            x = self.fc1(x,current_step_num=current_step,current_epoch=current_epoch)
        except:
            x = self.fc1(x)
#         x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        try:
            x = self.fc2(x,current_step_num=current_step,current_epoch=current_epoch)
        except:
            x = self.fc2(x)
#         x = self.fc2(x)
        x = self.drop2(x)
        return x


class GluMlp(nn.Module):
    """ MLP w/ GLU style gating
    See: https://arxiv.org/abs/1612.08083, https://arxiv.org/abs/2002.05202
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.Sigmoid, bias=True, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        assert hidden_features % 2 == 0
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Linear(hidden_features // 2, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def init_weights(self):
        # override init of fc1 w/ gate portion set to weight near zero, bias=1
        fc1_mid = self.fc1.bias.shape[0] // 2
        nn.init.ones_(self.fc1.bias[fc1_mid:])
        nn.init.normal_(self.fc1.weight[fc1_mid:], std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x, gates = x.chunk(2, dim=-1)
        x = x * self.act(gates)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class GatedMlp(nn.Module):
    """ MLP as used in gMLP
    """
    def __init__(
            self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU,
            gate_layer=None, bias=True, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        if gate_layer is not None:
            assert hidden_features % 2 == 0
            self.gate = gate_layer(hidden_features)
            hidden_features = hidden_features // 2  # FIXME base reduction on gate property?
        else:
            self.gate = nn.Identity()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.gate(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class ConvMlp(nn.Module):
    """ MLP using 1x1 convs that keeps spatial dims
    """
    def __init__(
            self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU,
            norm_layer=None, bias=True, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)

        self.fc1 = nn.Conv2d(in_features, hidden_features, kernel_size=1, bias=bias[0])
        self.norm = norm_layer(hidden_features) if norm_layer else nn.Identity()
        self.act = act_layer()
        self.drop = nn.Dropout(drop)
        self.fc2 = nn.Conv2d(hidden_features, out_features, kernel_size=1, bias=bias[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return x
