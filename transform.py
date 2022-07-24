# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     transform
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
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
from torch_geometric.nn.conv import MessagePassing
from torch_scatter import scatter_add
from torch_geometric.utils import degree
from torch_geometric.utils import to_dense_batch, to_dense_adj
from torch_geometric.utils import remove_self_loops, add_self_loops


class NodeNorm(torch.nn.Module):
    def __init__(self, node_norm_type="n", unbiased=False, eps=1e-5, power_root=2, **kwargs):
        super(NodeNorm, self).__init__()
        self.unbiased = unbiased
        self.eps = eps
        self.node_norm_type = node_norm_type
        self.power = 1 / power_root
        # print(f'------ ›››››››››  {self._get_name()}')

    def forward(self, x):
        # in GCN+Cora,
        # n v srv pr
        # 16 layer:  _19.8_  15.7 17.4 17.3
        # 32 layer:  20.3 _25.5_ 16.2 16.3

        if self.node_norm_type == "n":
            mean = torch.mean(x, dim=1, keepdim=True)
            std = (
                    torch.var(x, unbiased=self.unbiased, dim=1, keepdim=True) + self.eps
            ).sqrt()
            x = (x - mean) / std
        elif self.node_norm_type == "v":
            std = (
                    torch.var(x, unbiased=self.unbiased, dim=1, keepdim=True) + self.eps
            ).sqrt()
            x = x / std

        elif self.node_norm_type == "m":
            mean = torch.mean(x, dim=1, keepdim=True)
            x = x - mean
        elif self.node_norm_type == "srv":  # squre root of variance
            std = (
                    torch.var(x, unbiased=self.unbiased, dim=1, keepdim=True) + self.eps
            ).sqrt()
            x = x / torch.sqrt(std)
        elif self.node_norm_type == "pr":
            std = (
                    torch.var(x, unbiased=self.unbiased, dim=1, keepdim=True) + self.eps
            ).sqrt()
            x = x / torch.pow(std, self.power)
        return x

    def __repr__(self):
        original_str = super().__repr__()
        components = list(original_str)
        node_norm_type_str = f"node_norm_type={self.node_norm_type}"
        components.insert(-1, node_norm_type_str)
        new_str = "".join(components)
        return new_str


class FeatureExpander(MessagePassing):
    r"""Expand features.

    Args:
        degree (bool): whether to use degree feature.
        onehot_maxdeg (int): whether to use one_hot degree feature with
            with max degree capped. disableid with 0.
        AK (int): whether to use a^kx feature. disabled with 0.
        centrality (bool): whether to use centrality feature.
        remove_edges (strings): whether to remove edges, partially or totally.
        edge_noises_add (float): adding random edges (in ratio of current edges).
        edge_noises_delete (float): remove random ratio of edges.
        group_degree (int): group nodes to create super nodes, set 0 to disable.
    """

    def __init__(self, max_node_num, degree=True, onehot_maxdeg=0, AK=0,
                 centrality=False, remove_edges="none",
                 edge_noises_add=0, edge_noises_delete=0, group_degree=0):
        super(FeatureExpander, self).__init__('add', 'source_to_target')
        self.max_node_num = max_node_num
        self.degree = degree
        self.onehot_maxdeg = onehot_maxdeg
        self.AK = AK
        self.centrality = centrality
        self.remove_edges = remove_edges
        self.edge_noises_add = edge_noises_add
        self.edge_noises_delete = edge_noises_delete
        self.group_degree = group_degree
        assert remove_edges in ["none", "nonself", "all"], remove_edges

        self.edge_norm_diag = 1e-8  # edge norm is used, and set A diag to it

    def transform(self, data):
        if data.x is None:
            data.x = torch.ones([data.num_nodes, 1], dtype=torch.float)

        # Adding noises to edges before computing anything else.
        if self.edge_noises_delete > 0:
            num_edges_new = data.num_edges - int(
                data.num_edges * self.edge_noises_delete)
            idxs = torch.randperm(data.num_edges)[:num_edges_new]
            data.edge_index = data.edge_index[:, idxs]
        if self.edge_noises_add > 0:
            num_new_edges = int(data.num_edges * self.edge_noises_add)
            idx = torch.LongTensor(num_new_edges * 2).random_(0, data.num_nodes)
            new_edges = idx.reshape(2, -1)
            data.edge_index = torch.cat([data.edge_index, new_edges], 1)

        deg, deg_onehot = self.compute_degree(data.edge_index, data.num_nodes)
        akx = self.compute_akx(data.num_nodes, data.x, data.edge_index)
        cent = self.compute_centrality(data)
        data.x = torch.cat([data.x, deg, deg_onehot, akx, cent], -1)

        if self.remove_edges != "none":
            if self.remove_edges == "all":
                self_edge = None
            else:  # only keep self edge
                self_edge = torch.tensor(range(data.num_nodes)).view((1, -1))
                self_edge = torch.cat([self_edge, self_edge], 0)
            data.edge_index = self_edge

        # Reduce nodes by degree-based grouping
        if self.group_degree > 0:
            assert self.remove_edges == "all", "remove all edges"
            x_base = data.x
            deg_base = deg.view(-1)
            super_nodes = []
            for k in range(1, self.group_degree + 1):
                eq_idx = deg_base == k
                gt_idx = deg_base > k
                x_to_group = x_base[eq_idx]
                x_base = x_base[gt_idx]
                deg_base = deg_base[gt_idx]
                group_size = torch.zeros([1, 1]) + x_to_group.size(0)
                if x_to_group.size(0) == 0:
                    super_nodes.append(
                        torch.cat([group_size, data.x[:1] * 0], -1))
                else:
                    super_nodes.append(
                        torch.cat([group_size,
                                   x_to_group.mean(0, keepdim=True)], -1))
            if x_base.size(0) == 0:
                x_base = data.x[:1] * 0
            data.x = x_base
            data.xg = torch.cat(super_nodes, 0).view((1, -1))

        ### Create feature/adj/mask batch
        ##########################################################################
        features, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        adj_batch = to_dense_adj(edge_index)[0]
        adj_attr_batch = to_dense_adj(edge_index, edge_attr=edge_attr)[0].squeeze(-1)
        node_size = deg.size(0)

        if node_size > self.max_node_num:
            _, index = torch.topk(deg, k=self.max_node_num, dim=0)  # index: k*1
            mask = torch.zeros(node_size, 1)
            mask = mask.scatter(0, index, 1).bool().view(node_size)
            features = features[mask]
            adj_batch = adj_batch[mask][:, mask]
            adj_attr_batch = adj_attr_batch[mask][:, mask]

        features_batch = self.create_features_batch(features, self.max_node_num)
        adj_batch = self.create_adj_batch(adj_batch, self.max_node_num)
        adj_attr_batch = self.create_adj_batch(adj_attr_batch, self.max_node_num)

        data.x, data.adj, data.adj_attr = features_batch, adj_batch, adj_attr_batch
        del data.edge_index
        del data.edge_attr
        ##########################################################################
        return data

    def compute_degree(self, edge_index, num_nodes):
        row, col = edge_index
        deg = degree(row, num_nodes)
        deg = deg.view((-1, 1))

        if self.onehot_maxdeg is not None and self.onehot_maxdeg > 0:
            max_deg = torch.tensor(self.onehot_maxdeg, dtype=deg.dtype)
            deg_capped = torch.min(deg, max_deg).type(torch.int64)
            deg_onehot = F.one_hot(
                deg_capped.view(-1), num_classes=self.onehot_maxdeg + 1)
            deg_onehot = deg_onehot.type(deg.dtype)
        else:
            deg_onehot = self.empty_feature(num_nodes)

        if not self.degree:
            deg = self.empty_feature(num_nodes)

        return deg, deg_onehot

    def compute_centrality(self, data):
        if not self.centrality:
            return self.empty_feature(data.num_nodes)

        G = nx.Graph(data.edge_index.numpy().T.tolist())
        G.add_nodes_from(range(data.num_nodes))  # in case missing node ids
        closeness = nx.algorithms.closeness_centrality(G)
        betweenness = nx.algorithms.betweenness_centrality(G)
        pagerank = nx.pagerank_numpy(G)
        centrality_features = torch.tensor(
            [[closeness[i], betweenness[i], pagerank[i]] for i in range(
                data.num_nodes)])
        return centrality_features

    def compute_akx(self, num_nodes, x, edge_index, edge_weight=None):
        if self.AK is None or self.AK <= 0:
            return self.empty_feature(num_nodes)

        edge_index, norm = self.norm(
            edge_index, num_nodes, edge_weight, diag_val=self.edge_norm_diag)

        xs = []
        for k in range(1, self.AK + 1):
            x = self.propagate(edge_index, x=x, norm=norm)
            xs.append(x)
        return torch.cat(xs, -1)

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    @staticmethod
    def norm(edge_index, num_nodes, edge_weight, diag_val=1e-8, dtype=None):
        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1),),
                                     dtype=dtype,
                                     device=edge_index.device)
        edge_weight = edge_weight.view(-1)
        assert edge_weight.size(0) == edge_index.size(1)

        edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
        edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
        # Add edge_weight for loop edges.
        loop_weight = torch.full((num_nodes,),
                                 diag_val,
                                 dtype=edge_weight.dtype,
                                 device=edge_weight.device)
        edge_weight = torch.cat([edge_weight, loop_weight], dim=0)

        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    def empty_feature(self, num_nodes):
        return torch.zeros([num_nodes, 0])

    ### Create batch feature/adj/mask ###
    def cat_feature(self, feature, max_node_num):
        # pads the feature with zeros to the max node number
        padded_feature = nn.ConstantPad1d((0, 0, 0, max_node_num - feature.size(0)), 0)(feature)
        return padded_feature

    def create_features_batch(self, features, max_node_num):
        return self.cat_feature(features, max_node_num)

    def create_adj(self, adj, max_node_num):
        padded_adj = torch.zeros(max_node_num, max_node_num)
        padded_adj[:adj.size(0), :adj.size(1)] = adj
        return padded_adj

    def create_adj_batch(self, adj, max_node_num):
        # adj_batch = torch.stack([create_adj((adj_batch[i]==1).nonzero(), max_node_num) for i in range(batch_size)],
        # dim=0)
        adj_batch = self.create_adj(adj, max_node_num)
        return adj_batch
