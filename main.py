import os
import time
import torch
import random
import argparse
import torchvision
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.models as models
import os.path as path
from apex import amp
from torch.optim import lr_scheduler
from model import Resnet_LT
from model import ResNet_cifar
from utils.datasets import *
import utils.util as utils
from CIGB import *
from train import *

def get_args(parser):
    parser.add_argument('--dataroot', type=str, default='D:/data/CIFAR10/Data/')
    parser.add_argument('--save_root', type=str, default='./outputs/')
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--save_name', default='CIFAR10_Res32', type=str)
    parser.add_argument('--idx', default='1', type=str)
    parser.add_argument('--dataset', default='cifar10', type=str,
                        choices=('cifar10', 'cifar100', 'ImageNet-LT', 'iNaturelist2018'))
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=42, metavar='S',
                        help='random seed (default: 42)')
    parser.add_argument('--opt_level', default="O0", type=str,
                        help="Choose which accuracy to train. (default: 'O1')")
    parser.add_argument('--cpu_num', default=1, type=int)

    # Optimization
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--test_batch_size', type=int, default=-1)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr_decay', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=5e-3)
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--print-freq', '-p', default=10, type=int,
                        help='print frequency (default: 10)')
    # indices
    parser.add_argument('--divi_many', type=int, default=100)
    parser.add_argument('--divi_few', type=int, default=20)
    # PLM
    parser.add_argument('--pro_update', type=bool, default=False,
                        help='update the init positive ratio')
    parser.add_argument('--bins', type=int, default=10)
    parser.add_argument('--type', default='imbalance', type=str)
    parser.add_argument('--imb_factor', type=float, default=0.01)
    parser.add_argument('--cb_rate', default='0', type=float)
    parser.add_argument('--ib_rate', default='0', type=float)

    args = parser.parse_args()
    for arg in vars(args):
        print("{}={}".format(arg, getattr(args, arg)))
    return args

def get_model(args):
    if args.dataset == "ImageNet-LT" or args.dataset == "iNaturelist2018":
        print("=> creating model '{}'".format(args.model))
        if args.model == 'resnet50':
            model = Resnet_LT.resnet50(num_classes=args.num_classes)
        else:
            model = Resnet_LT.resnext50_32x4d(num_classes=args.num_classes)
    elif args.dataset == "cifar10" or args.dataset == "cifar100":
        print("=> creating model '{}'".format('resnet32'))
        model = ResNet_cifar.resnet32(num_class=args.num_classes)
    return model

def main():
    start = time.time()
    args = get_args(argparse.ArgumentParser())

    if args.test_batch_size == -1:
        args.test_batch_size = args.batch_size

    torch.set_num_threads(args.cpu_num)
    np.random.seed(42)
    random.seed(42)
    torch.manual_seed(args.seed)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))
    device = torch.device("cuda" if use_cuda else "cpu")
    print('device is: ', device)

    train_dataset, valid_dataset, per_cls_pro, per_cls_num = build_dataset(args)
    if args.dataset == "ImageNet-LT":
        num_classes = len(np.unique(train_dataset.targets))
        num_bound = utils.find_indices(per_cls_num, args.divi_many, args.divi_few)
        assert num_classes == args.num_classes
    else:
        num_bound = np.zeros([1])
    print(f'length of train dataset: {len(train_dataset)}')
    n_dataset = len(train_dataset)

    per_cls_weight = per_cls_pro.to(device)
    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=args.workers,
                              pin_memory=True
                              )
    valid_loader = DataLoader(valid_dataset,
                              batch_size=args.test_batch_size,
                              shuffle=False,
                              num_workers=args.workers,
                              pin_memory=True
                              )

    model1 = get_model(args)
    model2 = get_model(args)
    model1.to(device)
    model2.to(device)
    torch.backends.cudnn.benchmark = True

    loss_param1 = CE_param1()
    optimizer1 = optim.SGD(model1.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    optimizer2 = optim.SGD(model2.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler1 = lr_scheduler.CosineAnnealingLR(optimizer1, T_max=args.epochs)
    scheduler2 = lr_scheduler.CosineAnnealingLR(optimizer2, T_max=args.epochs)
    model1, optimizer1 = amp.initialize(model1, optimizer1, opt_level=args.opt_level)
    model2, optimizer2 = amp.initialize(model2, optimizer2, opt_level=args.opt_level)

    greb = GREB(positive_ratio=per_cls_pro, greb_rate=args.cb_rate, n_bins=args.bins)
    idm = IDM(n_dataset=n_dataset, n_classes=args.num_classes, c=1, idm_rate=args.ib_rate)
    best_error1 = 1
    avg_best = np.zeros([8])
    epoch_best = 0
    best_state = {}

    for epoch in range(args.epochs):
        train(args, train_loader, model1, optimizer1, scheduler1,
              model2, optimizer2, scheduler2, loss_param1, device, greb, idm, epoch)
        avg = test(args, valid_loader, model1, model2, device, num_bound)
        error = avg[0]
        is_best = error < best_error1
        best_error1 = min(error, best_error1)
        output_best = 'Best Prec@1: %.3f\n' % (best_error1)
        print(output_best)
        if is_best:
            epoch_best = epoch + 1
            avg_best = avg
            best_state = {'state_dict1': model1.state_dict(),
                          'state_dict2': model2.state_dict()}
    param = 'epoch' + str(epoch_best) + 'b' + str(args.batch_size) + '_ib' + str(args.ib_rate) + '_cb' + str(args.cb_rate)
    utils.save_avg(args, avg_best, param)
    utils.save_checkpoint_param(args, best_state, param)
    print('Finishing train')

    end = time.time()
    print('Time cost: %d' % ((end - start) / 60))
    print('=' * 50)

if __name__ == '__main__':
    main()
