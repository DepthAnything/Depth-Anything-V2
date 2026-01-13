import torch
from torch import nn


class SiLogLoss(nn.Module):
    def __init__(self, lambd=0.5):
        super().__init__()
        self.lambd = lambd

    def forward(self, pred, target, valid_mask):
        valid_mask = valid_mask.detach()
        diff_log = torch.log(target[valid_mask]) - torch.log(pred[valid_mask])
        loss = torch.sqrt(torch.pow(diff_log, 2).mean() -
                          self.lambd * torch.pow(diff_log.mean(), 2))

        return loss


class BerHuLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target, mask=None):
        if mask is not None:
            pred = pred[mask]
            target = target[mask]

        error = pred - target
        abs_error = torch.abs(error)

        c = 0.2 * torch.max(abs_error).item()

        l1_part = abs_error <= c
        l2_part = abs_error > c

        loss = torch.zeros_like(abs_error)
        loss[l1_part] = abs_error[l1_part]
        loss[l2_part] = (error[l2_part] ** 2 + c ** 2) / (2 * c)

        return torch.mean(loss)
