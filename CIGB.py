import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import os
import time
import numpy as np
import math

class CE_param1(nn.Module):
    def __init__(self):
        super(CE_param1, self).__init__()
        self._eps = 1e-8

    def forward(self, outputs, target, param1):
        output = outputs + torch.log(param1)
        loss = F.cross_entropy(output, target, reduction='none')
        return loss.mean()


class CE_param2(nn.Module):
    def __init__(self):
        super(CE_param2, self).__init__()

    def forward(self, outputs, target, param1, param2):
        output = outputs + torch.log(param1) + torch.log(param2)
        loss = F.cross_entropy(output, target, reduction='none')
        return loss.mean()

class CE_3param(nn.Module):
    def __init__(self):
        super(CE_3param, self).__init__()

    def forward(self, outputs, target, param1, param2):
        output = outputs * param1 + torch.log(param2)
        loss = F.cross_entropy(output, target, reduction='none')
        return loss.mean()

class IDM(nn.Module):
    def __init__(self, n_dataset, n_classes, c=1, idm_rate=1):
        super(IDM, self).__init__()
        self._eps = 1e-8
        self.c = c
        self.idm_rate = idm_rate
        self.n_classes = n_classes
        self.n_dataset = n_dataset
        self.pred_previous = np.zeros((n_dataset, n_classes))
        self.du_sum = torch.ones((n_dataset, n_classes))
        self.dl_sum = torch.ones((n_dataset, n_classes))
        self.Dvalue = torch.ones((n_dataset, n_classes))
        self.clear_init()

    def clear_init(self):
        self.pred = np.ones((1, self.n_classes))
        self.label = np.ones((1, self.n_classes))

    def record_pred(self, output, target):
        output = torch.softmax(output, dim=1)
        # label = F.one_hot(target.long(), self.n_classes)
        pred = output.detach().numpy()
        label = target.detach().numpy()

        self.pred = np.concatenate((self.pred, pred), axis=0)
        self.label = np.concatenate((self.label, label), axis=0)

    def update_ratio(self):
        pred = self.pred[1:, ]
        label = self.label[1:, ]

        (du, dl) = self._compute_diffculty(pred, label)
        du = torch.from_numpy(du).double()
        dl = torch.from_numpy(dl).double()

        self.du_sum += du
        self.dl_sum += dl

        self.compute_Dvalue()
        self.clear_init()

    def compute_Dvalue(self):
        d = torch.stack((self.du_sum, self.dl_sum), dim=-1)
        x = torch.tensor([1.0, 0.0], device=d.device)
        cos_value = torch.sum(d * x, dim=-1) / (torch.norm(d, dim=-1) * torch.norm(x))

        angle = torch.acos(cos_value) - torch.tensor(math.pi / 4, device=d.device)

        value = torch.sin(angle)
        self.Dvalue = torch.exp(value * self.idm_rate)

    def _compute_diffculty(self, input, label):
        index = label  
        pred_current = input
        dif = pred_current - self.pred_previous

        dif_true = dif * index
        dif_other = dif * (1 - index)
        dif_true_u = (dif_true < 0)  
        dif_true_l = (dif_true > 0)  
        dif_other_u = (dif_other > 0)  
        dif_other_l = (dif_other < 0)  

        dif_PSI = self._JS(pred_current, self.pred_previous)
        du = dif_PSI * dif_true_u + dif_PSI * dif_other_u
        dl = dif_PSI * dif_true_l + dif_PSI * dif_other_l

        self.pred_previous = pred_current
        return du, dl

    def generate_Dvalue(self, idx, batch_size):
        idx_begin = batch_size * idx
        idx_end = batch_size * (idx + 1)
        if idx_end > self.n_dataset:
            idx_end = self.n_dataset
        return self.Dvalue[idx_begin: idx_end]

    def _JS(self, p1, p2):
        q = (p1 + p2) / 2
        kl1 = self.compute_KL(p1, q)
        kl2 = self.compute_KL(p2, q)
        return 0.5 * (kl1 + kl2)

    def compute_KL(self, p, t):
        kl_div = p * np.log(p / t.clip(min=self._eps) + self._eps)
        return kl_div

    def _PSI(self, current, previous):
        PSI = (current - previous) * np.log(current / previous.clip(min=self._eps))
        return PSI

class GREB(nn.Module):
    def __init__(self, positive_ratio, greb_rate=1, n_bins=10, store_hist=True):
        super(GREB, self).__init__()
        self._eps = 1e-8
        self.greb_rate = greb_rate
        self.positive_ratio = positive_ratio
        self.greb_ratio_ideal = positive_ratio
        self.n_bins = n_bins
        self.n_classes = positive_ratio.shape[0]
        self.store_hist = store_hist
        self._clear_probability_histogram()
        self.greb_matrix = torch.ones([self.n_classes, self.n_classes])

    def _clear_probability_histogram(self):
        self.out_pos = np.ones((self.n_bins, self.n_classes))
        self.out_neg = np.ones((self.n_bins, self.n_classes))
        self.tar_pos = np.ones((self.n_bins, self.n_classes))
        self.tar_neg = np.ones((self.n_bins, self.n_classes))

    def record_histogram(self, outputs, target):
        if self.store_hist:
            (tar_pos, tar_neg) = self.target_hist(outputs, target)
            self.tar_pos += tar_pos
            self.tar_neg += tar_neg
            (out_pos, out_neg) = self.output_hist(outputs, target)
            self.out_pos += out_pos
            self.out_neg += out_neg

    def target_hist(self, outputs, target):
        n_classes = self.n_classes
        n_bins = self.n_bins
        value_range = [0.0, 1.0]

        outputs_percent = torch.sigmoid(outputs)
        hist_pos = np.zeros((n_bins, n_classes), int)
        hist_neg = np.zeros((n_bins, n_classes), int)

        outputs_percent = outputs_percent.detach().numpy()
        target = target.detach().numpy()

        for class_i in range(n_classes):
            target_class = target[:, class_i]
            outputs_class = outputs_percent[:, class_i]

            pos_indices = target_class > 0.5 
            neg_indices = ~pos_indices 

            hist_pos[:, class_i], _ = np.histogram(outputs_class[pos_indices], n_bins, value_range)
            hist_neg[:, class_i], _ = np.histogram((1 - outputs_class)[neg_indices], n_bins, value_range)
        return hist_pos, hist_neg

    def output_hist(self, outputs, target):
        n_classes = self.n_classes
        n_bins = self.n_bins
        value_range = [0.0, 1.0]

        outputs_percent = torch.where(torch.sigmoid(outputs) > 0.5, 1., 0.)

        hist_pos = np.zeros((n_bins, n_classes), int)
        hist_neg = np.zeros((n_bins, n_classes), int)

        outputs_percent = outputs_percent.detach().numpy()
        target = target.detach().numpy()

        for class_i in range(n_classes):
            target_class = target[:, class_i]  
            outputs_class = outputs_percent[:, class_i]  

            pos_indices = outputs_class > 0.5  
            neg_indices = ~pos_indices  

            hist_pos[:, class_i], _ = np.histogram(target_class[pos_indices], n_bins, value_range)
            hist_neg[:, class_i], _ = np.histogram((1 - target_class)[neg_indices], n_bins, value_range)

        return hist_pos, hist_neg

    def generate_mask(self):
        greb_weight = self.positive_ratio / self.greb_ratio_ideal
        return greb_weight

    def update_ratio(self):
        greb_tar = self.greb_compute_probabilities_difference(self.tar_pos, self.tar_neg)
        greb_out = self.greb_compute_probabilities_difference(self.out_pos, self.out_neg)
        greb_diff = torch.from_numpy(greb_out - greb_tar).double()

        self.greb_ratio_ideal = self.greb_ratio_ideal * torch.exp(self.greb_rate * greb_diff)
        self._clear_probability_histogram()

    def greb_compute_probabilities_difference(self, pos_pred, neg_pred):
        pos_pred = pos_pred / pos_pred.sum(0)
        neg_pred = neg_pred / neg_pred.sum(0)

        hist_diff_pred = self.compute_JS(pos_pred, neg_pred)
        return hist_diff_pred

    def _compute_probabilities_difference(self, pos_pred, pos_true, neg_pred, neg_true):
        pos_pred = pos_pred / pos_pred.sum(0)
        pos_true = pos_true / pos_true.sum(0)
        neg_pred = neg_pred / neg_pred.sum(0)
        neg_true = neg_true / neg_true.sum(0)

        hist_diff_pos = self.compute_JS(pos_true, pos_pred)
        hist_diff_neg = self.compute_JS(neg_true, neg_pred)
        return hist_diff_pos - hist_diff_neg

    def compute_JS(self, p1, p2):
        q = (p1 + p2) / 2
        kl1 = self.compute_KL(p1, q)
        kl2 = self.compute_KL(p2, q)
        return 0.5 * (kl1 + kl2)

    def compute_KL(self, p, t):
        t = np.where(t > 0, t, self._eps)
        kl_div = np.sum(p * np.log(p / t + self._eps), axis=0)
        return kl_div


if __name__ == '__main__':
    pred = torch.rand([128, 100])
    true = torch.randint(0, 100, [128])
    pos = torch.rand([100])

    t0 = time.time()
    t1 = time.time()
    print(t1-t0)
