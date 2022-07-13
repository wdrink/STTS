#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Loss functions."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from slowfast.models.topk import batched_index_select

class SoftTargetCrossEntropy(nn.Module):
    """
    Cross entropy loss with soft target.
    """

    def __init__(self, reduction="mean"):
        """
        Args:
            reduction (str): specifies reduction to apply to the output. It can be
                "mean" (default) or "none".
        """
        super(SoftTargetCrossEntropy, self).__init__()
        self.reduction = reduction

    def forward(self, x, y):
        loss = torch.sum(-y * F.log_softmax(x, dim=-1), dim=-1)
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "none":
            return loss
        else:
            raise NotImplementedError


class SoftTargetCrossEntropyPruning(nn.Module):
    """
    Cross entropy loss with soft target.
    """

    def __init__(self, ratio_weight=2.0, pruning_loc=[0], keep_ratio=[0.5], clf_weight=1.0, reduction="mean"):
        """
        Args:
            reduction (str): specifies reduction to apply to the output. It can be
                "mean" (default) or "none".
        """
        super(SoftTargetCrossEntropyPruning, self).__init__()
        self.reduction = reduction
        self.clf_weight = clf_weight
        self.pruning_loc = pruning_loc
        self.keep_ratio = keep_ratio
        
        self.cls_loss = 0
        self.ratio_loss = 0

        self.ratio_weight = ratio_weight


    def forward(self, x, y):
        pred, out_pred_score = x
        cls_loss = torch.sum(-y * F.log_softmax(pred, dim=-1), dim=-1)
        if self.reduction == "mean":
            cls_loss = cls_loss.mean()
        elif self.reduction == "none":
            cls_loss = cls_loss
        else:
            raise NotImplementedError
        
        pred_loss = 0.0
        ratio = self.keep_ratio
        left_ratio = 1.
        for i, score in enumerate(out_pred_score):
            pos_ratio = score.mean(1)
            left_ratio = left_ratio * ratio[i]
            print(left_ratio, pos_ratio)
            pred_loss = pred_loss + ((pos_ratio - left_ratio) ** 2).mean()

        loss = self.clf_weight * cls_loss + self.ratio_weight * pred_loss / len(self.pruning_loc)
        return loss
        

_LOSSES = {
    "cross_entropy": nn.CrossEntropyLoss,
    "bce": nn.BCELoss,
    "bce_logit": nn.BCEWithLogitsLoss,
    "soft_cross_entropy": SoftTargetCrossEntropy,
    "soft_cross_entropy_pruning": SoftTargetCrossEntropyPruning,
    
}


def get_loss_func(loss_name):
    """
    Retrieve the loss given the loss name.
    Args (int):
        loss_name: the name of the loss to use.
    """
    if loss_name not in _LOSSES.keys():
        raise NotImplementedError("Loss {} is not supported".format(loss_name))
    return _LOSSES[loss_name]
