import os
import random
import time
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from sklearn.metrics import precision_score, recall_score, f1_score

def expand_label(n_classes, target):
    n_idx = target.shape[0]
    tar_one_hot = torch.zeros(n_idx, n_classes + 1)
    tar_one_hot[torch.arange(n_idx), target.long()] = 1
    return tar_one_hot[:, :n_classes]

def compute_mnist_ratio(num_classes, dataset, update=False):
    per_cls_pro = torch.zeros(num_classes).double()
    num = len(dataset)

    if update:
        per_cls_num = torch.zeros(num_classes).double()
        num = len(dataset)
        for i in range(num):
            labels = dataset[i][1].nonzero()
            for label in labels:
                per_cls_num[label] += 1
    else:
        per_cls_num = torch.Tensor([45174, 2287, 8606, 2442, 2243, 2791, 2464, 4321, 2098,
                                    2893, 1205, 1214, 481, 3844, 2241, 2818, 3041, 2068,
                                    1105, 1389, 1518, 668, 1324, 1798, 3924, 2749, 4861,
                                    2667, 1631, 1511, 2209, 1170, 2986, 1625, 1804, 1884,
                                    2511, 2343, 2368, 5968, 1771, 6518, 2537, 3097, 2493,
                                    5028, 1618, 1171, 1645, 1216, 1340, 1186, 821, 2202,
                                    1062, 2080, 8950, 3170, 3084, 2539, 8378, 2317, 3191,
                                    2475, 1290, 2180, 1471, 3322, 1089, 2003, 151, 3291,
                                    1671, 3734, 3159, 2530, 673, 1510, 128, 700]).double()

    for i in range(num_classes):
        per_cls_pro[i] = per_cls_num[i] / (num - per_cls_num[i])

    return per_cls_num, per_cls_pro

def adjust_learning_rate(args, optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * ((((epoch + 1) / 5) ** int(epoch < 5))
                    * (args.lr_decay ** int(epoch > 160))
                    * (args.lr_decay ** int(epoch > 180)))

    print('learning rate: ', lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def accuracy(args, output, target):
    batch_size = target.size(0)
    _, pred = output.topk(1, 1, True, True)
    pred = pred.squeeze()
    correct = pred.eq(target)

    class_correct = np.zeros([args.num_classes])
    class_predict = np.zeros([args.num_classes])

    for i in range(batch_size):
        labels = target[i].item()
        preds = pred[i].item()
        class_correct[labels] += correct[i].item()
        class_predict[preds] += 1
    return class_correct, class_predict

def accuracy_all(args, output, target):
    batch_size = target.size(0)
    _, pred = output.topk(1, 1, True, True)
    pred = pred.squeeze()
    correct = pred.eq(target)

    class_correct = np.zeros([args.num_classes])
    class_predict = np.zeros([args.num_classes])
    class_total = np.zeros([args.num_classes])

    for i in range(batch_size):
        labels = target[i].item()
        preds = pred[i].item()
        class_correct[labels] += correct[i].item()
        class_predict[preds] += 1
        class_total[labels] += 1
    return class_correct, class_predict, class_total

def accuracy_num(num, output, target):
    batch_size = target.size(0)
    pred = torch.argmax(output, dim=1)
    # pred = pred.squeeze()
    correct = pred.eq(target)

    class_correct = np.zeros([num])
    class_predict = np.zeros([num])
    class_total = np.zeros([num])

    for i in range(batch_size):
        labels = target[i].item()
        preds = pred[i].item()
        class_correct[labels] += correct[i].item()
        class_predict[preds] += 1
        class_total[labels] += 1
    return class_correct, class_predict, class_total

def accuracy_1(num, output, target):
    batch_size = target.size(0)
    _, pred = output.topk(1, 1, True, True)
    pred = pred.squeeze()
    correct = pred.eq(target)

    class_correct = np.zeros([num])
    class_predict = np.zeros([num])
    class_total = np.zeros([num])

    for i in range(batch_size):
        labels = target[i].item()
        preds = pred[i].item()
        class_correct[labels] += correct[i].item()
        class_predict[preds] += 1
        class_total[labels] += 1
    return class_correct, class_predict, class_total

def metrix(correct, predict, total):
    num = len(correct)
    avg = np.zeros([4 + num * 3])
    recall = correct / total.clip(min=1e-8)
    precision = correct / predict.clip(min=1e-8)
    f1 = (2 * recall * precision) / (recall + precision + 1e-8)
    avg[1] = recall.mean()
    avg[2] = precision.mean()
    avg[3] = f1.mean()
    avg[0] = 1 - avg[1]
    avg[4: 4 + num] = recall
    avg[4 + num: 4 + num * 2] = precision
    avg[4 + num * 2: 4 + num * 3] = f1
    return avg

def metrix_ImageNet(correct, predict, total, num_bound):
    avg = np.zeros([8])
    recall = correct / total.clip(min=1e-8)
    precision = correct / predict.clip(min=1e-8)
    f1 = (2 * recall * precision) / (recall + precision + 1e-8)
    avg[0] = 1 - recall.mean()
    avg[1] = recall.mean()
    avg[2] = precision.mean()
    avg[3] = f1.mean()
    avg[4] = recall.mean() 
    avg[5] = recall[num_bound['many']].mean() if len(num_bound['many']) > 0 else 0
    avg[6] = recall[num_bound['medium']].mean() if len(num_bound['medium']) > 0 else 0
    avg[7] = recall[num_bound['few']].mean() if len(num_bound['few']) > 0 else 0
    return avg

def find_indices(y, x1, x2):
    z1 = (y > x1).nonzero(as_tuple=True)[0]
    z2 = ((y <= x1) & (y > x2)).nonzero(as_tuple=True)[0]
    z3 = (y <= x2).nonzero(as_tuple=True)[0]

    return {"many": z1, "medium": z2, "few": z3}

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def save_csv(args, result, num, name):
    path = args.save_root + 'Record/'
    if not os.path.exists(path):
        os.makedirs(path)
    filename_correct = path + args.save_name + '_' + args.idx + '_' + name + '.csv'
    pd.DataFrame(data=result, index=range(num)).to_csv(filename_correct)

def save_avg(args, result, param):
    path = args.save_root + 'Record/'
    if not os.path.exists(path):
        os.makedirs(path)
    filename_dir = path + args.save_name + '_' + args.idx + '.csv'
    data_list = []
    result_list = []
    data_list.append(param)
    data_list.extend(result.tolist())
    result_list.append(data_list)

    if not os.path.exists(filename_dir):
        df = pd.DataFrame(data=result_list, columns=range(len(result_list[0])), index=range(len(result_list)))
        df.to_csv(filename_dir, index=False)
    else:
        data_pre = pd.read_csv(filename_dir).values.tolist()
        data_pre.append(data_list)
        df = pd.DataFrame(data=data_pre, columns=range(len(data_pre[0])), index=range(len(data_pre)))
        df.to_csv(filename_dir, index=False)

def save_checkpoint(args, state):
    path = args.save_root + 'checkpoint/'
    if not os.path.exists(path):
        os.makedirs(path)
    filename = path + args.save_name + '_' + args.idx + '_ckpt.pth.tar'
    torch.save(state, filename)

def save_checkpoint_param(args, state, param):
    path = args.save_root + 'checkpoint/'
    if not os.path.exists(path):
        os.makedirs(path)
    filename = path + args.save_name + '_' + param + '_' + args.idx + '_ckpt.pth.tar'
    torch.save(state, filename)

