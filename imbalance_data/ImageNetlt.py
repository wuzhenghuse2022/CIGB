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
from torch.utils.data import Dataset, DataLoader
from PIL import Image

class LT_Dataset(Dataset):
    def __init__(self, root, txt, transform=None):
        self.img_path = []
        self.labels = []
        self.transform = transform
        with open(txt) as f:
            for line in f:
                self.img_path.append(os.path.join(root, line.split()[0]))
                self.labels.append(int(line.split()[1]))
        self.targets = self.labels  # Sampler needs to use targets

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        path = self.img_path[index]
        label = self.labels[index]
        with open(path, 'rb') as f:
            sample = Image.open(f).convert('RGB')
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, label

    def get_per_class_num(self):
        num_classes = len(np.unique(self.targets))
        cls_num_list = [0] * num_classes
        for label in self.targets:
            cls_num_list[label] += 1

        pro_per_cls = []
        for idx in range(num_classes):
            p = cls_num_list[idx] / (sum(cls_num_list) - cls_num_list[idx])
            pro_per_cls.append(p)
        return pro_per_cls, cls_num_list

