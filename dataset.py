# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     dataset
   Description :
   Author :       zouqi
   date：          2022/7/24
-------------------------------------------------
   Change Activity:
                   2022/7/24:
-------------------------------------------------
"""
__author__ = 'zouqi'

import copy
from typing import Union, List, Tuple, Any, Optional
from os.path import join as opj

from torch_geometric.data.data import BaseData
from torch_geometric.data.separate import separate
from tqdm import tqdm
import pandas as pd
import torch

from torch_geometric.data import HeteroData, InMemoryDataset, Data


class BIGDataset(InMemoryDataset):
    def __init__(self, root: str, name='Dual', p='p1', radius=0.5,
                 use_node_attr=True, use_edge_attr=True, pre_transform=None, pre_filter=None):
        self.name = name
        self.radius = radius
        self.p = p
        super(BIGDataset, self).__init__(root, pre_transform=pre_transform, pre_filter=pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

        if self.data.x is not None and not use_node_attr:
            num_node_attributes = self.num_node_attributes
            self.data.x = self.data.x[:, num_node_attributes:]
        if self.data.edge_attr is not None and not use_edge_attr:
            num_edge_attributes = self.num_edge_attributes
            self.data.edge_attr = self.data.edge_attr[:, num_edge_attributes:]

    @property
    def raw_dir(self) -> str:
        return opj(self.root, 'raw', self.p)

    @property
    def processed_dir(self) -> str:
        return opj(self.root, 'processed', f'{self.name}-{self.p}-{self.radius}')

    @property
    def raw_file_names(self) -> Union[str, List[str], Tuple]:
        return ['gene', 'roi', 'gene-roi']

    @property
    def num_node_attributes(self) -> int:
        if self.data.x is None:
            return 0
        return self.data.x.size(1)

    @property
    def num_edge_attributes(self) -> int:
        if self.data.edge_attr is None:
            return 0
        return self.data.edge_attr.size(1)

    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple]:
        return 'data.pt'

    def download(self):
        pass

    def process(self):
        raw_Gene, raw_ROI, raw_Dual = self.raw_paths
        subjects = pd.read_csv(opj(self.raw_dir, 'subjects.csv'), sep=',', index_col=0)
        data_list = []
        for subject in tqdm(subjects['PTID']):

            g = HeteroData()

            if self.name == 'Gene' or self.name == 'Dual':
                gene_edges = pd.read_csv(opj(raw_Gene, 'edges', f'{subject}.csv'), sep=',', index_col=0)
                gene_edges = gene_edges[gene_edges['weight'] > self.radius]
                gene_node = pd.read_csv(opj(raw_Gene, 'nodes', f'{subject}.csv'), sep=',', index_col=0)

                g['gene'].x = torch.from_numpy(gene_node.values).float()
                g['gene', 'gg', 'gene'].edge_index = torch.from_numpy(
                    gene_edges[['source', 'target']].values).long().t().contiguous()
                g['gene', 'gg', 'gene'].edge_attr = torch.from_numpy(
                    gene_edges[['weight']].values).float().contiguous()

            if self.name == 'ROI' or self.name == 'Dual':
                roi_edges = pd.read_csv(opj(raw_ROI, 'edges', f'{subject}.csv'), sep=',', index_col=0)
                roi_edges = roi_edges[roi_edges['weight'] > self.radius]
                roi_node = pd.read_csv(opj(raw_ROI, 'nodes', f'{subject}.csv'), sep=',', index_col=0)

                g['roi'].x = torch.from_numpy(roi_node.values).float()
                g['roi', 'rr', 'roi'].edge_index = torch.from_numpy(
                    roi_edges[['source', 'target']].values).long().t().contiguous()
                g['roi', 'rr', 'roi'].edge_attr = torch.from_numpy(
                    roi_edges[['weight']].values).float().contiguous()

            if self.name == 'Dual':
                dual_edges = pd.read_csv(opj(raw_Dual, 'edges', f'{subject}.csv'), sep=',', index_col=0)
                dual_edges = dual_edges[dual_edges['weight'] > self.radius]
                g['gene', 'gr', 'roi'].edge_index = torch.from_numpy(
                    dual_edges[['source', 'target']].values).long().t().contiguous()
                g['gene', 'gr', 'roi'].edge_attr = torch.from_numpy(
                    dual_edges[['weight']].values).float().contiguous()

                g['roi', 'rg', 'gene'].edge_index = torch.from_numpy(
                    dual_edges[['target', 'source']].values).long().t().contiguous()
                g['roi', 'rg', 'gene'].edge_attr = torch.from_numpy(
                    dual_edges[['weight']].values).float().contiguous()

            g.y = torch.tensor([1 if subjects[subjects['PTID'] == subject]['Label'].iloc[0] == 'pMCI' else 0]).long()
            g = g.to_homogeneous(node_attrs=['x'], edge_attrs=['edge_attr'], add_node_type=False,
                                       add_edge_type=False)
            data_list.append(g)

        if self.pre_filter is not None:
            data_list = [g for g in data_list if self.pre_filter(g)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(g) for g in data_list]

        data, slices = self.collate(data_list)

        torch.save((data, slices), self.processed_paths[0])

    def get(self, idx: int) -> Union[Optional[BaseData], Any]:
        if self.len() == 1:
            return copy.copy(self.data)

        if not hasattr(self, '_data_list') or self._data_list is None:
            self._data_list = self.len() * [None]
        elif self._data_list[idx] is not None:
            return copy.copy(self._data_list[idx])

        data = separate(
            cls=self.data.__class__,
            batch=self.data,
            idx=idx,
            slice_dict=self.slices,
            decrement=False,
        )
        self._data_list[idx] = copy.copy(data)

        return data

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({len(self)})'
