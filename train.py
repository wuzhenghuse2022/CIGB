import os
import time
import torch
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
from apex import amp
from torch.optim import lr_scheduler
import utils.util as utils
import torch.nn.functional as F

def train(args, train_loader, model1, optimizer1, scheduler1,
          model2, optimizer2, scheduler2, loss_param, device, greb, idm, epoch):
    losses = utils.AverageMeter()
    model1.train()
    t1 = time.time()
    for i, (images, target) in enumerate(train_loader):
        optimizer1.zero_grad()
        target_cuda = target.to(device, non_blocking=True).long()
        images_cuda = images.to(device, non_blocking=True)
        outputs_cuda = model1(images_cuda)
        outputs = outputs_cuda.cpu()
        target_onehot = F.one_hot(target.long(), args.num_classes)
        greb.record_histogram(outputs, target_onehot)
        greb_weight = greb.generate_mask()
        greb_weigth = greb_weight.to(device)
        loss = loss_param(outputs_cuda, target_cuda, greb_weigth)

        with amp.scale_loss(loss, optimizer1) as scaled_loss:
            scaled_loss.backward()
        optimizer1.step()
        losses.update(loss.item(), images.size(0))
        if i % args.print_freq == 0:
            t2 = time.time()
            print('10 Iter cost: ', t2-t1)
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss1 {loss.val:.4f} ({loss.avg:.4f})\t'
                  .format(epoch, i, len(train_loader), loss=losses))
            t1 = time.time()
    greb.update_ratio()
    scheduler1.step()

    model2.train()
    for i, (images, target) in enumerate(train_loader):
        optimizer2.zero_grad()
        target_cuda = target.to(device, non_blocking=True).long()
        images_cuda = images.to(device, non_blocking=True)
        outputs_cuda = model2(images_cuda)
        outputs = outputs_cuda.cpu()
        target_onehot = F.one_hot(target.long(), args.num_classes)
        idm.record_pred(outputs, target_onehot)
        idm_weight = idm.generate_Dvalue(i, args.batch_size)
        idm_weight = idm_weight.to(device)
        loss = loss_param(outputs_cuda, target_cuda, idm_weight)

        with amp.scale_loss(loss, optimizer2) as scaled_loss:
            scaled_loss.backward()
        optimizer2.step()
        losses.update(loss.item(), images.size(0))
        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss2 {loss.val:.4f} ({loss.avg:.4f})\t'
                  .format(epoch, i, len(train_loader), loss=losses))
    idm.update_ratio()
    scheduler2.step()

def test(args, valid_loader, model1, model2, device, num_bound):
    model1.eval()
    model2.eval()
    with torch.no_grad():
        correct = np.zeros([args.num_classes])
        predict = np.zeros([args.num_classes])
        total = np.zeros([args.num_classes])

        for i, (images, target) in enumerate(valid_loader):
            images = images.to(device, non_blocking=True)
            output1 = model1(images)
            output2 = model2(images)
            output = (output1 + output2) / 2
            class_correct, class_predict, class_total = utils.accuracy_all(args, output.cpu(), target)
            correct += class_correct
            predict += class_predict
            total += class_total
        if args.dataset == "ImageNet-LT":
            avg = utils.metrix_ImageNet(correct, predict, total, num_bound)
        else:
            avg = utils.metrix(correct, predict, total)
        return avg

