# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     utils
   Description :
   Author :       zouqi
   date：          2022/7/24
-------------------------------------------------
   Change Activity:
                   2022/7/24:
-------------------------------------------------
"""
__author__ = 'zouqi'

import os
import pickle
import shutil
import sys
import random
from texttable import Texttable

import numpy as np
import torch


def seed_everything(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_dir(dir_, keep=True):
    """
        create a directory, if the directory already exists,
         it will be processed according to the behavior specified by `keep`.
    """
    exists = os.path.exists(dir_)
    if keep and exists:
        return False
    if exists:
        shutil.rmtree(dir_)
    os.makedirs(dir_)


def logger(info):
    fold, epoch = info['fold'], info['epoch']
    if epoch == 1 or epoch % 1 == 0:
        train_acc, test_acc, test_roc, \
        test_precision, test_sensitivity, \
        test_specificity, test_f1 = info['train_acc'], info['test_acc'], info['test_roc'], \
                                    info['test_precision'], info['test_sensitivity'], \
                                    info['test_specificity'], info['test_f1']

        print('{:02d}/{:03d}: Train Acc: {:.3f}, Test Accuracy: {:.3f}, Test Roc: {:.3f}, Test Precision: {:.3f}, '
              'Test Sensitivity: {:.3f}, Test Specificity: {:.3f}, Test F1: {:.3f}'.format(
            fold, epoch, train_acc, test_acc, test_roc, test_precision,
            test_sensitivity, test_specificity, test_f1))
    sys.stdout.flush()


def tab_printer(args):
    """
    Function to print the logs in a nice tabular format.
    """
    args = vars(args)
    keys = sorted(args.keys())
    t = Texttable()
    t.add_rows([["Parameter", "Value"]])
    t.add_rows([[k.replace("_", " ").capitalize(), args[k]] for k in keys])
    print(t.draw())


def pickle_dump(data, dump_path):
    with open(dump_path, 'wb') as f:
        pickle.dump(data, f)

