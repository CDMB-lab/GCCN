# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     parameters
   Description :
   Author :       zouqi
   date：          2022/7/24
-------------------------------------------------
   Change Activity:
                   2022/7/24:
-------------------------------------------------
"""
__author__ = 'zouqi'


import argparse


def parameter_parser():
    parser = argparse.ArgumentParser(description='GCNN')
    str2bool = lambda x: x.lower() == 'true'

    ######## Common ########
    parser.add_argument('--data-root', type=str, default='/lab_data/data_cache/zouqi/codes/HGCCN/data')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lr_decay_factor', type=float, default=0.5)
    parser.add_argument('--lr_decay_step_size', type=int, default=100)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=2)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--model-path', type=str, default='/lab_data/data_cache/zouqi/codes/HGCCN/models')
    parser.add_argument('--logger', default='promote', choices=['epoch', 'promote'])
    parser.add_argument('--save-model', action='store_true')
    ######## Common ########

    ######## dataset ########
    parser.add_argument('--dataset', default='Dual', choices=['Gene', 'ROI', 'Dual'])
    parser.add_argument('--p', default='p5', choices=['p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7', 'p8', 'p9', 'p10'])
    parser.add_argument('--modality', default='FT', choices=['F', 'T', 'FT'])
    parser.add_argument('--radius', type=float, default=0.75)
    ######## dataset ########

    ######## model ########
    parser.add_argument('--capsule-dimensions', type=int, default=64)
    parser.add_argument('--use-routing', action='store_true')
    parser.add_argument('--num-iterations', type=int, default=3)
    parser.add_argument('--M', type=int, default=4)
    parser.add_argument('--num-capsules', type=int, default=6)
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--use-reconstruction', action='store_true')
    parser.add_argument('--base-ldc', type=str, default='linear')
    parser.add_argument('--theta', type=float, default=0.1)
    parser.add_argument('--use-residual', action='store_true')
    ######## model ########

    return parser.parse_args()
