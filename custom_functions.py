# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     custom_functions
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
from torch.nn import functional as F

from dataset import BIGDataset
from transform import FeatureExpander, NodeNorm


def squash(input_tensor, dim=-1, e=1, epsilon=1e-7):
    squared_norm = (input_tensor ** 2).sum(dim=dim, keepdim=True)
    safe_norm = torch.sqrt(squared_norm + epsilon)
    scale = squared_norm / (e + squared_norm)
    unit_vector = input_tensor / safe_norm
    return scale * unit_vector


def num_graphs(data):
    if data.batch is not None:
        return data.num_graphs
    else:
        return data.x.size(0)


def margin_loss(scores, target, loss_lambda=0.5):
    target = F.one_hot(target, scores.size(1))
    v_mag = scores

    zero = torch.zeros(1)
    zero = zero.to(target.device)
    m_plus = 0.9
    m_minus = 0.1

    max_l = torch.max(m_plus - v_mag, zero) ** 2
    max_r = torch.max(v_mag - m_minus, zero) ** 2
    T_c = target

    L_c = T_c * max_l + loss_lambda * (1.0 - T_c) * max_r
    L_c = L_c.sum(dim=1)
    L_c = L_c.mean()
    return L_c


def get_dataset(dataroot, name, p, radius=0.7, feat_str='deg+odeg30'):
    degree = feat_str.find("deg") >= 0
    onehot_maxdeg = re.findall("odeg(\d+)", feat_str)
    onehot_maxdeg = int(onehot_maxdeg[0]) if onehot_maxdeg else None
    k = re.findall("an{0,1}k(\d+)", feat_str)
    k = int(k[0]) if k else 0
    groupd = re.findall("groupd(\d+)", feat_str)
    groupd = int(groupd[0]) if groupd else 0
    remove_edges = re.findall("re(\w+)", feat_str)
    remove_edges = remove_edges[0] if remove_edges else 'none'
    edge_noises_add = re.findall("randa([\d\.]+)", feat_str)
    edge_noises_add = float(edge_noises_add[0]) if edge_noises_add else 0
    edge_noises_delete = re.findall("randd([\d\.]+)", feat_str)
    edge_noises_delete = float(
        edge_noises_delete[0]) if edge_noises_delete else 0
    centrality = feat_str.find("cent") >= 0

    max_node_num = 0
    if name == 'Dual':
        if p == 'p1':
            max_node_num = 69 + 210
        if p == 'p2':
            max_node_num = 59 + 210
        if p == 'p3':
            max_node_num = 199 + 210
        if p == 'p4':
            max_node_num = 109 + 210
        if p == 'p5':
            max_node_num = 279 + 210
        if p == 'p6':
            max_node_num = 139 + 210
        if p == 'p7':
            max_node_num = 199 + 210
        if p == 'p8':
            max_node_num = 129 + 210
        if p == 'p9':
            max_node_num = 249 + 210
        if p == 'p10':
            max_node_num = 229 + 210
    elif name == 'Gene':
        if p == 'p1':
            max_node_num = 69
        if p == 'p2':
            max_node_num = 59
        if p == 'p3':
            max_node_num = 199
        if p == 'p4':
            max_node_num = 109
        if p == 'p5':
            max_node_num = 279
        if p == 'p6':
            max_node_num = 139
        if p == 'p7':
            max_node_num = 199
        if p == 'p8':
            max_node_num = 129
        if p == 'p9':
            max_node_num = 249
        if p == 'p10':
            max_node_num = 229
    else:
        max_node_num = 210
    pre_transform = FeatureExpander(max_node_num=max_node_num,
                                    degree=degree, onehot_maxdeg=onehot_maxdeg, AK=k,
                                    centrality=centrality, remove_edges=remove_edges,
                                    edge_noises_add=edge_noises_add, edge_noises_delete=edge_noises_delete,
                                    group_degree=groupd).transform

    dataset = BIGDataset(dataroot, name=name, radius=radius, p=p,
                         use_node_attr=True, use_edge_attr=True, pre_transform=pre_transform)

    NN = NodeNorm('pr')
    for data in dataset:
        data.x = NN(data.x)

    dataset.data.edge_attr = None
    return dataset
