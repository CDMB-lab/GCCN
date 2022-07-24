# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     layers
   Description :
   Author :       zouqi
   date：          2022/7/24
-------------------------------------------------
   Change Activity:
                   2022/7/24:
-------------------------------------------------
"""
__author__ = 'zouqi'

import torch
from torch import nn
from torch.nn import functional as F

from custom_functions import squash
from denseconv import DenseGCNConv
from disentangle import LinearDisentangle
from sparsemax import Sparsemax


class PrimaryCapsuleLayer(nn.Module):
    def __init__(self, num_of_features, capsule_dimensions, dropout,
                 M, base_ldc):
        super(PrimaryCapsuleLayer, self).__init__()

        self.num_of_features = num_of_features
        self.capsule_dimensions = capsule_dimensions
        self.dropout = dropout
        self.M = M
        self.bn = nn.BatchNorm1d(self.number_of_features)

        self.encoder = LinearDisentangle(in_dim=num_of_features, out_dim=capsule_dimensions,
                                         M=M, dropout=dropout, base_ldc=base_ldc)

    def forward(self, x):
        # x.size: [batch_size, num_nodes, num_of_features]
        x_size = x.size()
        # BatchNorm1d requires 2-D tensor
        # temp.size: [batch_size, num_nodes * num_of_features]
        temp = self.bn(x.view(-1, x_size[-1]))
        temp = temp.view(x_size)

        if self.M > 0:
            # h.size: [batch_size, num_nodes, capsule_dimensions]
            h = self.encoder(temp)
        else:
            # h.size: [batch_size, num_nodes, capsule_dimensions=num_of_features]
            h = temp

        # out.size: [batch_size, num_nodes, capsule_dimensions]
        out = squash(h)
        return out


class DigitalCapsuleLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_in_units, num_out_units,
                 use_routing, num_iterations, dropout, use_residual=True):
        super(DigitalCapsuleLayer, self).__init__()

        self.num_prim_caps = num_in_units
        self.num_digit_caps = num_out_units
        self.in_cap_dim = in_dim
        self.out_cap_dim = out_dim
        self.use_routing = use_routing
        self.num_iterations = num_iterations
        self.dropout = dropout
        self.bn = nn.BatchNorm1d(self.in_cap_dim)
        self.softmax = Sparsemax(dim=2)
        self.use_residual = use_residual

        self.convs = nn.ModuleList()
        for _ in range(self.num_digit_caps):
            self.convs.append(DenseGCNConv(self.in_cap_dim, self.out_cap_dim, bias=True))

    def routing(self, x, adj):
        # x.size: [batch_size, num_prim_caps, in_cap_dim]
        x_size = x.size()

        # BatchNorm1d requires 2-D tensor
        temp = self.bn(x.view(-1, x_size[-1]))
        x = temp.view(x_size)

        u_hat = []
        for i, conv in enumerate(self.convs):
            # temp.size: [batch_size, num_prim_caps, out_cap_dim]
            temp = conv(x, adj)
            u_hat.append(temp)

        # u_hat.shape: [batch_size, num_prim_caps, num_digit_caps, out_cap_dim, 1]
        u_hat = torch.stack(u_hat, dim=2).unsqueeze(4)

        # detach u_hat during dynamic routing iterations to prevent gradients from flowing
        temp_u_hat = u_hat.detach()

        # b_ij.shape: [batch_size, num_prim_caps, num_digit_caps, 1, 1]
        b_ij = torch.zeros(x_size[0], self.num_prim_caps, self.num_digit_caps, 1, 1).to(x.device)
        # routing by agreement
        for t in range(self.num_iterations - 1):
            # c_ij.shape: [batch_size, num_prim_caps, num_digit_caps, 1, 1]
            # routing softmax for coupling coefﬁcients
            c_ij = self.softmax(b_ij)
            # s_ij.shape: [batch_size, num_prim_caps, num_digit_caps, out_cap_dim, 1]
            s_j = (c_ij * temp_u_hat).sum(dim=1, keepdims=True)
            # v is the output of the capsule layer
            v = squash(s_j, dim=-2)

            # u_product_v.shape: [batch_size, num_prim_caps, num_digit_caps, 1, 1]
            u_product_v = torch.matmul(temp_u_hat.transpose(-1, -2), v)
            # update b_ij
            b_ij = b_ij + u_product_v

        # calculate the output of the capsule layer
        c_ij = self.softmax(b_ij, dim=2)

        # s_ij.shape: [batch_size, num_prim_caps, num_digit_caps, out_cap_dim, 1]
        s_j = (c_ij * u_hat).sum(dim=1, keepdim=True)
        # Residual connection
        # x.shape: [batch_size, num_prim_nodes, prim_cap_dim]
        # use None to broadcast x to s_j
        if self.use_residual:
            s_j += torch.mean(x, dim=1)[:, None, None, :, None]

        # v is the output of the capsule layer
        # v.shape: [batch_size, num_prim_caps, num_digit_caps, out_cap_dim, 1]
        v = squash(s_j, dim=-1).squeeze(1).squeeze(-1)
        # c_ij.shape: [batch_size, num_prim_caps, num_digit_caps]
        c_ij = c_ij.squeeze(4).squeeze(3)

        # update the adjacent matrix
        # adj.shape: [batch_size, num_prim_caps, num_prim_caps]
        # => [batch_size, num_digit_caps, num_digit_caps]
        # digit_caps can be seen as the coarse grained version of prim_caps
        # as the number of digit_caps3 is much smaller than the number of prim_caps
        adj = torch.transpose(c_ij, 2, 1) @ adj @ c_ij
        return v, adj

    def no_routing(self, x, adj):
        # x.shape: [batch_size, num_prim_caps, in_cap_dim]
        x_size = x.size()

        # BatchNorm1d requires 2-D tensor
        temp = x.view(-1, x_size[-1])
        temp = self.bn(temp)
        x = temp.view(x_size)

        u_hat = []
        for i, conv in enumerate(self.convs):
            # temp.shap: [batch_size, num_prim_caps, out_cap_dim]
            temp = conv(x, adj)
            u_hat.append(temp)

        # u_hat.shape: [batch_size, num_digit_caps, num_prim_caps, out_cap_dim]
        u_hat = torch.stack(u_hat, dim=1)
        v = u_hat.mean(-2)
        v = squash(v, dim=-1)
        adj = v @ (v.permute(0, -1, -2))
        return v, adj

    def forward(self, x, adj):
        if self.use_routing:
            return self.routing(x, adj)
        else:
            return self.no_routing(x, adj)


class ReconstructionNet(nn.Module):
    def __init__(self, n_dim, n_classes, hidden):
        super(ReconstructionNet, self).__init__()
        self.n_dim = n_dim
        self.n_classes = n_classes
        self.fc1 = nn.Linear(n_dim * n_classes, hidden)

    def forward(self, first_capsule, class_capsule, y):
        mask = torch.zeros((class_capsule.size(0), self.n_classes), device=y.device)
        mask.scatter_(1, y.view(-1, 1), 1.)
        mask = mask.unsqueeze(2)
        class_capsule = class_capsule * mask
        class_capsule = class_capsule.view(-1, 1, self.n_dim * self.n_classes)

        # combine the first capsule and the class capsule (class-conditional)
        # N = first_capsule.size(1)
        class_capsule = F.relu(self.fc1(class_capsule))
        x = first_capsule + class_capsule
        x = torch.matmul(x, torch.transpose(x, 2, 1))
        return x
