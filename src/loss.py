import torch
import torch.nn as nn
import numpy as np
from torch import einsum
from chemical import aa2num
from kinematics import get_dih

def calc_c6d_loss(logit_s, label_s, mask_2d, eps=1e-6):
    loss_s = list()
    for i in range(len(logit_s)):
        loss = nn.CrossEntropyLoss(reduction='none')(logit_s[i], label_s[...,i]) # (B, L, L)
        loss = (mask_2d*loss).sum() / (mask_2d.sum() + eps)
        loss_s.append(loss)
    loss_s = torch.stack(loss_s)
    return loss_s
