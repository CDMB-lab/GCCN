# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     train_eval_helpers
   Description :
   Author :       zouqi
   date：          2022/7/24
-------------------------------------------------
   Change Activity:
                   2022/7/24:
-------------------------------------------------
"""
__author__ = 'zouqi'

import time
from os.path import join as opj
import pickle

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix, accuracy_score, precision_score
from torch import tensor
from torch.nn import functional as F
from sklearn.model_selection import StratifiedKFold
from torch.optim import Adam
from torch_geometric.loader import DenseDataLoader

from custom_functions import margin_loss, num_graphs
from models import GCCN
from utils import pickle_dump


def k_fold(dataset, folds):
    skf = StratifiedKFold(folds, shuffle=True, random_state=12345)

    test_indices, train_indices = [], []
    for _, idx in skf.split(torch.zeros(len(dataset)), dataset.data.y):
        test_indices.append(torch.from_numpy(idx))

    val_indices = [test_indices[i - 1] for i in range(folds)]

    for i in range(folds):
        train_mask = torch.ones(len(dataset), dtype=torch.uint8)
        train_mask[test_indices[i].long()] = 0
        train_mask[val_indices[i].long()] = 0
        train_indices.append(train_mask.nonzero().view(-1))

    return train_indices, test_indices, val_indices


def cross_validation(args, dataset, num_nodes, num_features, num_targets,
                     folds, epochs, batch_size, lr,
                     lr_decay_factor, lr_decay_step_size,
                     weight_decay, logger=None):
    if args.gpu != -1:
        device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')
    if device != 'cpu':
        torch.cuda.synchronize()

    val_losses, train_losses, train_accuracies, \
    test_losses, test_accuracies, test_precisions, test_sensitivities, \
    test_specificities, test_aucs, test_f1s, durations = [], [], [], [], [], [], [], [], [], [], []

    test_y_trues, test_y_preds = [], []
    for fold, (train_idx, test_idx, val_idx) in enumerate(zip(*k_fold(dataset, folds))):
        pickle_dump(train_idx, opj(args.model_path, f'fold_{fold}_train_idx.txt'))
        pickle_dump(test_idx, opj(args.model_path, f'fold_{fold}_test_idx.txt'))
        pickle_dump(val_idx, opj(args.model_path, f'fold_{fold}_val_idx.txt'))

        train_dataset = dataset[train_idx]
        test_dataset = dataset[test_idx]
        val_dataset = dataset[val_idx]

        # `DenseLoader` stacking all features in a new dimension
        # [batch_size, num_features, dim_features]
        train_loader = DenseDataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DenseDataLoader(test_dataset, batch_size=batch_size)
        val_loader = DenseDataLoader(val_dataset, batch_size=batch_size)

        model = GCCN(num_of_features=num_features, num_of_targets=num_targets, num_prim_caps=num_nodes,
                     num_digit_caps=args.num_capsules, capsule_dimensions=args.capsule_dimension,
                     use_routing=args.use_routing, num_iterations=args.num_iterations, dropout=args.dropout,
                     M=args.M, base_ldc=args.base_ldc, use_residual=args.use_residual,
                     use_reconstruction=args.use_reconstruction, theta=args.theta)

        optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        t_start = time.perf_counter()
        best_ = 0
        for epoch in range(1, epochs + 1):
            train_loss, train_accuracy = train(model, optimizer, train_loader, device)
            val_loss = eval_metrics(model, val_loader, device)[0]
            test_loss, test_accuracy, test_auc, test_precision, \
            test_sensitivity, test_specificity, test_f1, \
            test_y_true, test_y_pred = eval_metrics(model, test_loader, device)

            train_losses.append(train_loss)
            train_accuracies.append(train_accuracy)
            val_losses.append(val_loss)
            test_losses.append(test_loss)
            test_accuracies.append(test_accuracy)
            test_precisions.append(test_precision)
            test_sensitivities.append(test_sensitivity)
            test_specificities.append(test_specificity)
            test_aucs.append(test_auc)
            test_f1s.append(test_f1)

            eval_info = {
                'fold': fold,
                'epoch': epoch,
                'train_loss': train_losses[-1],
                'train_acc': train_accuracies[-1],
                'val_loss': val_losses[-1],
                'test_acc': test_accuracies[-1],
                'test_roc': test_aucs[-1],
                'test_precision': test_precisions[-1],
                'test_sensitivity': test_sensitivities[-1],
                'test_specificity': test_specificities[-1],
                'test_f1': test_f1s[-1]
            }

            if epoch % lr_decay_step_size == 0:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_decay_factor * param_group['lr']

            if args.logger == 'epoch':
                if logger is not None:
                    logger(eval_info)
            if test_accuracy > best_:
                best_ = test_accuracy

                if args.logger == 'promote':
                    if logger is not None:
                        logger(eval_info)

                if args.save_model:
                    torch.save(model.state_dict(), opj(args.model_path, f'best_model_{fold}.pth'))
                    with open(opj(args.model_path, 'fold_{}_test_y_true.txt'.format(fold)), 'wb') as f:
                        pickle.dump(test_y_true, f)
                    with open(opj(args.model_path, 'fold_{}_test_y_pred.txt'.format(fold)), 'wb') as f:
                        pickle.dump(test_y_pred, f)

            if device != 'cpu':
                torch.cuda.synchronize()

    t_end = time.perf_counter()
    durations.append(t_end - t_start)

    duration = tensor(durations)
    train_accuracies, train_losses, val_losses, \
    test_accuracies, test_precisions, \
    test_sensitivities, test_specificities,\
    test_aucs, test_f1s = tensor(train_accuracies), tensor(train_losses), tensor(val_losses), \
                          tensor(test_accuracies), tensor(test_precisions), \
                          tensor(test_sensitivities), tensor(test_specificities), \
                          tensor(test_aucs), tensor(test_f1s)

    train_accuracies = train_accuracies.view(folds, epochs)
    train_losses = train_losses.view(folds, epochs)
    val_losses = val_losses.view(folds, epochs)
    test_accuracies = test_accuracies.view(folds, epochs)
    test_precisions = test_precisions.view(folds, epochs)
    test_sensitivities = test_sensitivities.view(folds, epochs)
    test_specificities = test_specificities.view(folds, epochs)
    test_aucs = test_aucs.view(folds, epochs)
    test_f1s = test_f1s.view(folds, epochs)
    _, selected_epoch_rep = val_losses.min(dim=1)

    test_acc = test_accuracies[torch.arange(folds, dtype=torch.long), selected_epoch_rep]
    test_precision = test_precisions[torch.arange(folds, dtype=torch.long), selected_epoch_rep]
    test_sensitivity = test_sensitivities[torch.arange(folds, dtype=torch.long), selected_epoch_rep]
    test_specificity = test_specificities[torch.arange(folds, dtype=torch.long), selected_epoch_rep]
    test_auc = test_aucs[torch.arange(folds, dtype=torch.long), selected_epoch_rep]
    test_f1 = test_f1s[torch.arange(folds, dtype=torch.long), selected_epoch_rep]

    train_acc_mean = train_accuracies[:, -1].mean().item()
    test_acc_mean = test_acc.mean().item()
    test_acc_std = test_acc.std().item()
    test_precision_mean = test_precision.mean().item()
    test_precision_std = test_precision.std().item()
    test_sensitivity_mean = test_sensitivity.mean().item()
    test_sensitivity_std = test_sensitivity.std().item()
    test_specificity_mean = test_specificity.mean().item()
    test_specificity_std = test_specificity.std().item()
    test_roc_mean = test_auc.mean().item()
    test_roc_std = test_auc.std().item()
    test_f1_mean = test_f1.mean().item()
    test_f1_std = test_f1.std().item()
    duration_mean = duration.mean().item()

    print('Train Acc: {:.2f}, Test Acc: {:.2f} ({:.2f}), Test ROC: {:.2f} ({:.2f}), '
          'Test Precision: {:.2f} ({:.2f}), Test Sensitivity: {:.2f} ({:.2f}), Test Specificity: {:.2f} ({:.2f}), '
          'Test F1: {:.2f} ({:.2f}) Duration: {:.3f}'.format(
        train_acc_mean * 100, test_acc_mean * 100, test_acc_std * 100,
        test_roc_mean * 100, test_roc_std * 100,
        test_precision_mean * 100, test_precision_std * 100,
        test_sensitivity_mean * 100, test_sensitivity_std * 100,
        test_specificity_mean * 100, test_specificity_std * 100,
        test_f1_mean * 100, test_f1_std * 100, duration_mean))


def train(model, optimizer, loader, device):
    model.train()
    model = model.to(device)
    total_loss = 0.
    correct = 0
    total_ = len(loader.dataset)
    for batch_iter, data in enumerate(loader):
        optimizer.zero_grad()
        data = data.to(device)
        batch_x, batch_adj, batch_y = data.x, data.adj, data.y
        # out, primary_out, digital_out, class_out = model(batch_x, batch_adj)
        # loss = model.loss(primary_out, batch_adj, class_out, out, batch_y)
        out, recon_loss, recon_adj = model(batch_x, batch_adj, batch_y)
        LC = margin_loss(out, batch_y.view(-1))
        loss = LC + recon_loss

        pred = out.max(1)[1]
        correct += pred.eq(batch_y.view(-1)).sum().item()
        loss.backward()
        total_loss += loss.item() * num_graphs(data)

        optimizer.step()

    return total_loss / total_, correct / total_


def eval_metrics(model, loader, device):
    model.eval()

    y_true = []
    y_pred = []
    y_out = []
    losses = []
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            batch_x, batch_adj, batch_y = data.x, data.adj_attr, data.y

            # out, primary_out, digital_out, class_out = model(batch_x, batch_adj)
            # loss = model.loss(primary_out, batch_adj, class_out, out, batch_y)
            out, recon_loss, recon_adj = model(batch_x, batch_adj, batch_y)
            LC = margin_loss(out, batch_y.view(-1))
            loss = LC + recon_loss

            losses.append(loss.item())
            pred = out.max(1)[1]
            y_true.append(data.y.view(-1).cpu().numpy())
            y_pred.append(pred.cpu().numpy())
            y_out.append(out.cpu().numpy())

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    y_out = np.concatenate(y_out)

    auc = roc_auc_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    TN, FP, FN, TP = confusion_matrix(y_true, y_pred).ravel()
    loss = np.mean(losses)
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    sensitivity = TP / (TP + FN)
    specificity = TN / (TN + FP)

    return loss, accuracy, auc, precision, sensitivity, specificity, f1, y_true, y_out
