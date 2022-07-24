# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     disentangle
   Description :
   Author :       zouqi
   date：          2022/7/24
-------------------------------------------------
   Change Activity:
                   2022/7/24:
-------------------------------------------------
"""
__author__ = 'zouqi'

import re

import torch
import torch.nn as nn
from torch.nn import Parameter
from torch_geometric.nn.inits import glorot, zeros


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, num_layers=3, norm=False, shortcut=True):
        super(MLP, self).__init__()
        self.fcs = nn.ModuleList()
        self.shortcut = shortcut
        for idx in range(num_layers):
            in_ = in_dim if idx == 0 else out_dim
            out_ = out_dim
            self.fcs.append(nn.Linear(in_, out_))
            if norm:
                self.fcs.append(nn.BatchNorm1d(out_))
            self.fcs.append(nn.PReLU())
        if self.shortcut:
            self.linear_shortcut = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        x_ = self.fcs[0](x)
        for idx in range(1, len(self.fcs)):
            x_ = self.fcs[idx](x_)
        if self.shortcut:
            return x_ + self.linear_shortcut(x)
        else:
            return x_


class LinearDisentangleComponent(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super(LinearDisentangleComponent, self).__init__()
        self.weight = Parameter(torch.Tensor(in_dim, out_dim))
        self.bias = Parameter(torch.Tensor(out_dim))
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        zeros(self.bias)

    def forward(self, x):
        return torch.matmul(x, self.weight) + self.bias

    def __repr__(self):
        return self.__class__.__name__ + ' (in_features=' + str(self.weight.size(0)) + ', out_features=' + str(
            self.weight.size(1)) + ', bias=True)'


class LinearDisentangle(nn.Module):
    def __init__(self, in_dim, out_dim, M, dropout, base_ldc='linear'):
        super(LinearDisentangle, self).__init__()
        assert out_dim % M == 0
        self.disentangle_funs = nn.ModuleList()
        for _ in range(M):
            if base_ldc == 'linear':
                layer = LinearDisentangleComponent(in_dim, out_dim // M)
            elif base_ldc.startswith('mlp'):
                num_layers = int(re.findall(r'mlp(\d+)', base_ldc)[0])
                layer = MLP(in_dim, out_dim // M, num_layers)
            else:
                raise NotImplemented('Not implemented base linear disentangle function.', base_ldc)
            self.disentangle_funs.append(layer)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        out = []
        for i, comp_fun in enumerate(self.disentangle_funs):
            temp = comp_fun(x)
            temp = self.dropout(temp)
            out.append(temp)

        # Combine features from the K different components
        out = torch.cat(out, dim=-1)
        return out
