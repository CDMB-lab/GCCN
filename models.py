# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     models
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

from layers import PrimaryCapsuleLayer, DigitalCapsuleLayer, ReconstructionNet


class HGCCN(nn.Module):
    def __init__(self, num_of_features, num_of_targets, num_prim_caps, num_digit_caps,
                 capsule_dimensions, use_routing, num_iterations, dropout, M, base_ldc,
                 use_residual, use_reconstruction, theta=0.1):
        super(HGCCN, self).__init__()

        self.num_of_features = num_of_features
        self.num_of_targets = num_of_targets
        self.capsule_dimensions = capsule_dimensions
        self.num_prim_caps = num_prim_caps
        self.num_digit_caps = num_digit_caps
        self.use_routing = use_routing
        self.num_iterations = num_iterations
        self.dropout = dropout
        self.M = M
        self.base_ldc = base_ldc
        self.use_residual = use_residual
        self.use_reconstruction = use_reconstruction
        self.theta = theta

        self._setup_layers()

    @staticmethod
    def cal_reconstruction_loss(pred_adj, adj):
        eps = 1e-7
        # Each entry in pred_adj cannot larger than 1
        pred_adj = torch.min(pred_adj, torch.ones(1, dtype=pred_adj.dtype).to(adj.device))
        # The diagonal entries in pred_adj should be 0
        pred_adj = pred_adj.masked_fill_(torch.eye(adj.size(1), adj.size(1)).bool().to(adj.device), 0)
        # Cross entropy loss
        link_loss = -adj * torch.log(pred_adj + eps) - (1 - adj) * torch.log(1 - pred_adj + eps)

        num_entries = pred_adj.size(0) * pred_adj.size(1) * pred_adj.size(2)

        link_loss = torch.sum(link_loss) / float(num_entries)
        return link_loss

    def _setup_primaryCapsuleLayer(self):
        self.primary_capsule = PrimaryCapsuleLayer(self.num_of_features, self.capsule_dimensions,
                                                   dropout=self.dropout, M=self.M, base_ldc=self.ldc_base)

    def _setup_digitalCapsuleLayer(self):
        self.digital_capsule = DigitalCapsuleLayer(in_dim=self.capsule_dimensions,
                                                   out_dim=self.capsule_dimensions,
                                                   num_in_units=self.primary_num,
                                                   num_out_units=self.capsule_num,
                                                   use_routing=True,
                                                   num_iterations=self.num_iterations,
                                                   dropout=self.dropout,
                                                   use_residual=self.use_residual)

    def _setup_classCapsuleLayer(self):
        self.class_capsule = DigitalCapsuleLayer(in_dim=self.capsule_dimensions,
                                                 out_dim=self.capsule_dimensions,
                                                 num_in_units=self.capsule_num,
                                                 num_out_units=self.num_of_targets,
                                                 use_routing=True,
                                                 num_iterations=self.num_iterations,
                                                 dropout=self.dropout,
                                                 use_residual=self.use_residual)

    def _setup_reconstructionNet(self):
        self.reconstruction_net = ReconstructionNet(n_dim=self.capsule_dimensions,
                                                    n_classes=self.num_of_targets,
                                                    hidden=self.capsule_dimensions)

    def _setup_layers(self):
        self._setup_primaryCapsuleLayer()
        self._setup_digitalCapsuleLayer()
        self._setup_classCapsuleLayer()
        if self.use_reconstruction:
            self._setup_reconstructionNet()

    def forward(self, x, adj, y):
        x_size = x.size()
        primary_out = self.primary_capsule(x)

        digital_out, digital_adj = self.digital_capsule(primary_out, adj)

        digital_adj = torch.min(digital_adj, torch.ones(1, dtype=digital_adj.dtype).to(digital_adj.device))
        # remove self-loops while aggregating capsule to higher capsule
        digital_adj = torch.min(digital_adj, torch.ones(1, dtype=digital_adj.dtype).to(digital_adj.device))
        digital_adj = digital_adj.masked_fill_(
            torch.eye(digital_adj.size(1), digital_adj.size(1)).bool().to(digital_adj.device), 0)

        class_out, _ = self.class_capsule(digital_out, digital_adj)

        if self.use_reconstruction:
            reconstruction_out = self.reconstruction_net(primary_out, class_out, y)
            reconstruction_loss = self.cal_reconstruction_loss(reconstruction_out, adj)
            out = F.softmax(torch.sqrt((class_out ** 2).sum(2)), dim=-1).view(x_size[0], self.num_of_targets)
            return out, reconstruction_loss, reconstruction_out
        else:
            out = F.softmax(torch.sqrt((class_out ** 2).sum(2)), dim=-1).view(x_size[0], self.num_of_targets)
            return out
