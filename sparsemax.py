# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     sparsemax
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


class Sparsemax(nn.Module):
    """Sparsemax function."""

    def __init__(self, dim=None):
        """Initialize sparsemax activation
        Args:
            dim (int, optional): The dimension over which to apply the sparsemax function.
        """
        super(Sparsemax, self).__init__()

        self.dim = -1 if dim is None else dim
        self.output = None
        self.grad_input = None

    def forward(self, input_):
        """Forward function.
        Args:
            input_ (torch.Tensor): Input tensor. First dimension should be the batch size
        Returns:
            torch.Tensor: [batch_size x number_of_logits] Output tensor
        """
        # Sparsemax currently only handles 2-dim tensors,
        # so we reshape to a convenient shape and reshape back after sparsemax
        input_ = input_.transpose(0, self.dim)
        original_size = input_.size()
        input_ = input_.reshape(input_.size(0), -1)
        input_ = input_.transpose(0, 1)
        dim = 1

        number_of_logits = input_.size(dim)

        # Translate input by max for numerical stability
        input_ = input_ - torch.max(input_, dim=dim, keepdim=True)[0].expand_as(input_)

        # Sort input in descending order.
        # (NOTE: Can be replaced with linear time selection method described here:
        # http://stanford.edu/~jduchi/projects/DuchiShSiCh08.html)
        zs = torch.sort(input=input_, dim=dim, descending=True)[0]
        range_ = torch.arange(start=1, end=number_of_logits + 1, step=1,
                              device=input_.device, dtype=input_.dtype).view(1, -1)
        range_ = range_.expand_as(zs)

        # Determine sparsity of projection
        bound = 1 + range_ * zs
        cumulative_sum_zs = torch.cumsum(zs, dim)
        is_gt = torch.gt(bound, cumulative_sum_zs).type(input_.type())
        k = torch.max(is_gt * range_, dim, keepdim=True)[0]

        # Compute threshold function
        zs_sparse = is_gt * zs

        # Compute taus
        taus = (torch.sum(zs_sparse, dim, keepdim=True) - 1) / k
        taus = taus.expand_as(input_)

        # Sparsemax
        self.output = torch.max(torch.zeros_like(input_), input_ - taus)

        # Reshape back to original shape
        output = self.output
        output = output.transpose(0, 1)
        output = output.reshape(original_size)
        output = output.transpose(0, self.dim)

        return output

    def backward(self, grad_output):
        """Backward function."""
        dim = 1

        nonzeros = torch.ne(self.output, 0)
        sum_ = torch.sum(grad_output * nonzeros, dim=dim) / torch.sum(nonzeros, dim=dim)
        self.grad_input = nonzeros * (grad_output - sum_.expand_as(grad_output))

        return self.grad_input
