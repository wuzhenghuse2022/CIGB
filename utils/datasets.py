import torch
import random
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torchvision
import numpy as np
import copy
import moco_loader as moco_loader
from randaugment import rand_augment_transform
from PIL import Image
import os.path
import pickle
from autoaug import CIFAR10Policy, Cutout

from ..imbalance_data.cifar100Imbalance import *
from ..imbalance_data.cifar10Imbalance import *
from ..imbalance_data.ImageNetlt import *

np.random.seed(6)

def normalize_param(net_version):
    if net_version == "cifar100":
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)

    if net_version == "cifar10":
        mean = (0.49139968, 0.48215827, 0.44653124)
        std = (0.24703233, 0.24348505, 0.26158768)

    if net_version == "ImageNet-LT":
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

    if net_version == "iNaturelist2018":
        mean = [0.466, 0.471, 0.380]
        std = [0.195, 0.194, 0.192]
    return mean, std

def get_transform(args):
    dataset = args.dataset
    if dataset == "cifar10":
        mean, std = normalize_param("cifar10")
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            CIFAR10Policy(),  # add AutoAug
            transforms.ToTensor(),
            Cutout(n_holes=1, length=16),
            transforms.Normalize(mean, std),
        ])

        transform_val = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        return transform_train,transform_val

    elif dataset == "cifar100":
        mean, std = normalize_param("cifar100")
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            CIFAR10Policy(),  # add AutoAug
            transforms.ToTensor(),
            Cutout(n_holes=1, length=16),
            transforms.Normalize(mean, std),
        ])

        transform_val = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        return transform_train, transform_val

    elif dataset == "ImageNet-LT":
        r = random.random()
        mean, std = normalize_param("ImageNet-LT")
        rgb_mean = (0.485, 0.456, 0.406)
        ra_params = dict(translate_const=int(224 * 0.45),img_mean=tuple([min(255, round(255 * x)) for x in rgb_mean]), )

        augmentation_randncls = [
            transforms.RandomResizedCrop(224, scale=(0.08, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.0)], p=1.0),
            rand_augment_transform('rand-n{}-m{}-mstd0.5'.format(2, 10), ra_params),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
        augmentation_sim = [
            transforms.RandomResizedCrop(224),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.0)], p=1.0),  # not strengthened
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([moco_loader.GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]
        transform_val = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)])

        if r < 0.5:
            transform_train = transforms.Compose(augmentation_randncls)
        else:
            transform_train = transforms.Compose(augmentation_sim)

        return transform_train, transform_val

    elif dataset == "iNaturelist2018":
        r = random.random()
        mean, std = normalize_param("ImageNet-LT")
        rgb_mean = (0.485, 0.456, 0.406)
        ra_params = dict(translate_const=int(224 * 0.45),
                         img_mean=tuple([min(255, round(255 * x)) for x in rgb_mean]), )

        augmentation_randncls = [
            transforms.RandomResizedCrop(224, scale=(0.08, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.0)
            ], p=1.0),
            rand_augment_transform('rand-n{}-m{}-mstd0.5'.format(2, 10), ra_params),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
        augmentation_sim = [
            transforms.RandomResizedCrop(224),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.0)  # not strengthened
            ], p=1.0),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([moco_loader.GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]
        transform_val = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)])

        if r < 0.5:
            transform_train = transforms.Compose(augmentation_randncls)
        else:
            transform_train = transforms.Compose(augmentation_sim)
        return transform_train, transform_val

def balanced_dataset(args):
    transform_train, transform_val = get_transform(args)
    if args.dataset == 'cifar10':
        trainset = torchvision.datasets.CIFAR10(transform=transform_train,
                                                root=args.dataroot,
                                                train=True,
                                                download=True)
        testset = torchvision.datasets.CIFAR10(transform=transform_train,
                                                root=args.dataroot,
                                                train=False,
                                                download=True)

        cls_num_list = torch.full((10,), 5000)
        r = 5000 / (50000 - 5000)
        cls_ratio_list = torch.full((10,), r)

        print("load cifar10")
        return trainset, testset, cls_ratio_list, cls_num_list

    if args.dataset == 'cifar100':
        trainset = torchvision.datasets.CIFAR100(transform=transform_train,
                                                root=args.dataroot,
                                                train=True,
                                                download=True)
        testset = torchvision.datasets.CIFAR100(transform=transform_train,
                                                root=args.dataroot,
                                                train=False,
                                                download=True)

        cls_num_list = torch.full((100,), 500)
        r = 500 / (50000 - 500)
        cls_ratio_list = torch.full((100,), r)

        print("load cifar100")
        return trainset, testset, cls_ratio_list, cls_num_list


def build_dataset(args):
    transform_train, transform_val = get_transform(args)
    if args.dataset == 'cifar10':
        trainset = Cifar10Imbanlance(transform=transform_train, imbanlance_rate=args.imb_factor,
                                     train=True, file_path=args.dataroot)
        testset = Cifar10Imbanlance(imbanlance_rate=args.imb_factor, train=False, transform=transform_val,
                                    file_path=args.dataroot)
        print("load cifar10")
        cls_ratio_list, cls_num_list = trainset.get_per_class_num()
        return trainset, testset, torch.Tensor(cls_ratio_list), torch.Tensor(cls_num_list)

    if args.dataset == 'cifar100':
        trainset = Cifar100Imbanlance(transform=transform_train, imbanlance_rate=args.imb_factor,
                                      train=True, file_path=args.dataroot)
        testset = Cifar100Imbanlance(imbanlance_rate=args.imb_factor, train=False, transform=transform_val,
                                     file_path=args.dataroot)
        print("load cifar100")
        cls_ratio_list, cls_num_list = trainset.get_per_class_num()
        return trainset, testset, torch.Tensor(cls_ratio_list), torch.Tensor(cls_num_list)

    if args.dataset == 'ImageNet-LT':
        dataroot = os.path.join(args.dataroot, 'ImageNet')
        dir_train_txt = os.path.join(args.dataroot, 'LT_data_txt/ImageNet_LT_train.txt')
        dir_test_txt = os.path.join(args.dataroot, 'LT_data_txt/ImageNet_LT_test.txt')
        trainset = LT_Dataset(dataroot, dir_train_txt, transform_train)
        testset = LT_Dataset(dataroot, dir_test_txt, transform_val)
        cls_ratio_list, cls_num_list = trainset.get_per_class_num()
        return trainset, testset, torch.Tensor(cls_ratio_list), torch.Tensor(cls_num_list)

    if args.dataset == 'iNaturelist2018':
        dataroot = os.path.join(args.dataroot, 'iNaturelist2018')
        dir_train_txt = os.path.join(dataroot, 'LT_data_txt/iNaturalist18_train.txt')
        dir_test_txt = os.path.join(dataroot, 'LT_data_txt/iNaturalist18_val.txt')
        trainset = LT_Dataset(dataroot, dir_train_txt, transform_train)
        testset = LT_Dataset(dataroot, dir_test_txt, transform_val)
        cls_ratio_list, cls_num_list = trainset.get_per_class_num()
        return trainset, testset, torch.Tensor(cls_ratio_list), torch.Tensor(cls_num_list)


