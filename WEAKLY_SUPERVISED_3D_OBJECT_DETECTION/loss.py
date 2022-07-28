from turtle import forward
import torch.nn as nn
import torch
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, preds, target):
        pred = preds
        celoss = F.binary_cross_entropy(preds, target, reduction='sum')
        alpha = target * self.alpha + (1- target) * (1 - self.alpha)
        pt = torch.where(target == 1, pred, 1-pred)
        return torch.sum(alpha * (1 - pt) ** self.gamma * celoss)