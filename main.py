# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     main
   Description :
   Author :       zouqi
   date：          2022/7/24
-------------------------------------------------
   Change Activity:
                   2022/7/24:
-------------------------------------------------
"""
__author__ = 'zouqi'

import sys

from custom_functions import get_dataset
from parameters import parameter_parser
from train_eval_helpers import cross_validation
from utils import seed_everything, tab_printer, make_dir, logger

args = parameter_parser()
seed_everything(args.seed)
tab_printer(args)

if __name__ == '__main__':
    sys.stdout.flush()
    dataset = get_dataset(dataroot=args.data_root,
                          name=args.dataset,
                          p=args.p,
                          radius=args.radius)
    num_nodes = dataset[0].num_nodes
    num_features = dataset.num_features
    num_targets = 2
    print(f'Dataset: {args.dataset}, #subjects: {dataset}, #node: {num_nodes}, #features: {num_features}')

    make_dir(args.model_path)
    cross_validation(args=args, dataset=dataset, num_nodes=num_nodes, num_features=num_features,
                     num_targets=num_targets, folds=10, epochs=args.epochs, batch_size=args.batch_size,
                     lr=args.lr, lr_decay_factor=args.lr_decay_factor, lr_decay_step_size=args.lr_decay_step_size,
                     weight_decay=args.weight_decay, logger=logger)
