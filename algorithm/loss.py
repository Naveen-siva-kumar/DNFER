'''
Aum Sri Sai Ram

Naveen
'''

import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import math
eps = 1e-8

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Sai_weighted_CCE(nn.Module):
    """
    Implementing Noise Robust CE Loss 
    """

    def __init__(self,  num_class=7, reduction="mean"):
        super(Sai_weighted_CCE, self).__init__()
        
        self.reduction = reduction
        self.num_class= num_class

    def forward(self, prediction, target_label, one_hot=True):

        if one_hot:
            y_true = F.one_hot(target_label.type(torch.LongTensor), num_classes=self.num_class).float().to(device)

        y_pred = F.softmax(prediction, dim=1)
        y_pred = torch.clamp(y_pred, eps, 1-eps)
        
        pred_tmp = torch.sum(y_true * y_pred, axis=-1).reshape(-1, 1)

        avg_post = torch.mean(y_pred, dim=0)
        
        avg_post = avg_post.reshape(-1, 1)
        
        std_post = torch.std(y_pred, dim=0)
        
        std_post = std_post.reshape(-1, 1)
        
        avg_post_ref = torch.matmul(y_true.type(torch.float), avg_post)
        
        pred_prun = torch.where((pred_tmp >= avg_post_ref ), pred_tmp, torch.zeros_like(pred_tmp)) #confident
        confident_idx = torch.where(pred_prun != 0.)[0]
        noisy_idx = torch.where(pred_prun == 0.)[0]
        if len(confident_idx) != 0:
            prun_targets = torch.argmax(torch.index_select(y_true, 0, confident_idx), dim=1)
            weighted_loss = F.cross_entropy(torch.index_select(prediction, 0, confident_idx), 
                            prun_targets, reduction=self.reduction)
        else:
            weighted_loss = F.cross_entropy(prediction, target_label)

        return weighted_loss, confident_idx, noisy_idx , avg_post.reshape(-1), std_post.reshape(-1)



# #Source: https://github.com/dbp1994/masters_thesis_codes/blob/main/BARE/losses.py
# class weighted_CCE(nn.Module):
#     """
#     Implementing Weighted Generalised Cross-Entropy (GCE) Loss (BARE) 
#     """

#     def __init__(self, k=1, num_class=7, reduction="mean"):
#         super(weighted_CCE, self).__init__()

#         self.k = k
#         self.reduction = reduction
#         self.num_class= num_class

#     def forward(self, prediction, target_label, one_hot=True):

#         if one_hot:
#             y_true = F.one_hot(target_label.type(torch.LongTensor), num_classes=self.num_class).float().to(device)

#         y_pred = F.softmax(prediction, dim=1)
#         y_pred = torch.clamp(y_pred, eps, 1-eps)
        
#         #print(y_pred.shape, y_true.shape)

#         pred_tmp = torch.sum(y_true * y_pred, axis=-1).reshape(-1, 1)

#         ## Compute batch statistics

#         # print("pred_tmp", pred_tmp)
#         # print("pred_tmp", pred_tmp.shape)

#         avg_post = torch.mean(y_pred, dim=0)
#         # print(avg_post)
#         # print(avg_post.shape)
#         avg_post = avg_post.reshape(-1, 1)
#         # print(avg_post.shape)

#         # med_post = torch.median(y_pred, dim=0,keepdim=True).values
#         # # # print(f"\nmed_post: {med_post}\n")
#         # med_post = med_post.reshape(-1, 1)
#         # # # print(f"\nmed_post: {med_post.shape}\n")
#         # med_post_ref = torch.matmul(y_true.type(torch.float), med_post)

#         std_post = torch.std(y_pred, dim=0)
#         # print(std_post)
#         std_post = std_post.reshape(-1, 1)
#         # print(std_post.shape)

#         avg_post_ref = torch.matmul(y_true.type(torch.float), avg_post)
#         # print("avg_post_ref", avg_post_ref)
#         # print("avg_post_ref", avg_post_ref.shape)

#         std_post_ref = torch.matmul(y_true.type(torch.float), std_post)
#         # print("std_post_ref", std_post_ref)
#         # print("std_post_ref", std_post_ref.shape)

#         # pred_prun = torch.where((torch.abs(pred_tmp - avg_post_ref) <= std_post_ref), pred_tmp, torch.zeros_like(pred_tmp))
#         # pred_prun = torch.where((pred_tmp >= avg_post_ref), pred_tmp, torch.zeros_like(pred_tmp))

#         pred_prun = torch.where((pred_tmp - avg_post_ref >= self.k * std_post_ref), pred_tmp, torch.zeros_like(pred_tmp))
#         # pred_prun = torch.where((pred_tmp >= med_post_ref), pred_tmp, torch.zeros_like(pred_tmp))


#         # prun_idx will tell us which examples are 
#         # 'trustworthy' for the given batch
#         prun_idx = torch.where(pred_prun != 0.)[0]

#         if len(prun_idx) != 0:
#             prun_targets = torch.argmax(torch.index_select(y_true, 0, prun_idx), dim=1)
#             weighted_loss = F.cross_entropy(torch.index_select(prediction, 0, prun_idx), 
#                             prun_targets, reduction=self.reduction)
#         else:
#             weighted_loss = F.cross_entropy(prediction, target_label)

#         return weighted_loss, prun_idx
 

        
def cross_entropy(logits, labels, reduction='mean'):
    """
    :param logits: shape: (N, C)
    :param labels: shape: (N, C)
    :param reduction: options: "none", "mean", "sum"
    :return: loss or losses
    """
    N, C = logits.shape
    assert labels.size(0) == N and labels.size(1) == C, f'label tensor shape is {labels.shape}, while logits tensor shape is {logits.shape}'

    log_logits = F.log_softmax(logits, dim=1)
    losses = -torch.sum(log_logits * labels, dim=1)  # (N)

    if reduction == 'none':
        return losses
    elif reduction == 'mean':
        return torch.sum(losses) / logits.size(0)
    elif reduction == 'sum':
        return torch.sum(losses)
    else:
        raise AssertionError('reduction has to be none, mean or sum')        
        
def kl_div(p, q):
    # p, q is in shape (batch_size, n_classes)
    return (p * p.log2() - p * q.log2()).sum(dim=1)


def symmetric_kl_div(p, q):
    return kl_div(p, q) + kl_div(q, p)


def js_div(p, q):
    # Jensen-Shannon divergence, value is in (0, 1)
    m = 0.5 * (p + q)
    return 0.5 * kl_div(p, m) + 0.5 * kl_div(q, m)
        

