# --------------------------------------------------------
# PyTorch WSDDN
# Copyright 2018. Seungkwan Lee
# Licensed under The MIT License [see LICENSE for details]
# Written by Seungkwan Lee
# Some parts of this implementation are based on code from Ross Girshick, Jiasen Lu, and Jianwei Yang
# --------------------------------------------------------
import torch
import numpy as np


def clip_gradient(model, clip_norm):
    totalnorm = 0
    for p in model.parameters():
        if p.requires_grad and p.grad is not None:
            modulenorm = p.grad.data.norm().item()
            totalnorm = totalnorm + modulenorm ** 2
    totalnorm = np.sqrt(totalnorm)

    norm = clip_norm / max(totalnorm, clip_norm)
    for p in model.parameters():
        if p.requires_grad and p.grad is not None:
            p.grad.mul_(norm)


def adjust_learning_rate(optimizer, decay=0.1):
    for param_group in optimizer.param_groups:
        param_group['lr'] = decay * param_group['lr']


def save_checkpoint(state, filename):
    torch.save(state, filename)
