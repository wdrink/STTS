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
        

class DistillationLoss(torch.nn.Module):
    """
    This module wraps a standard criterion and adds an extra knowledge distillation loss by
    taking a teacher model prediction and using it as additional supervision.
    """
    def __init__(self, teacher_model, distill_type='hard', alpha=0.5, tau=1.0):
        super().__init__()
        self.teacher_model = teacher_model
        assert distill_type in ['none', 'soft', 'hard']
        self.distillation_type = distill_type
        self.alpha = alpha
        self.tau = tau

    def forward(self, inputs, outputs, labels):
        """
        Args:
            inputs: The original inputs that are feed to the teacher model
            outputs: the outputs of the model to be trained. It is expected to be
                either a Tensor, or a Tuple[Tensor, Tensor], with the original output
                in the first position and the distillation predictions as the second output
            labels: the labels for the base criterion
        """
        
        base_loss = torch.sum(-labels * F.log_softmax(outputs, dim=-1), dim=-1).mean()
       
        
        if self.distillation_type == 'none':
            return base_loss

        # don't backprop throught the teacher
        with torch.no_grad():
            teacher_outputs = self.teacher_model(inputs)

        if self.distillation_type == 'soft':
            T = self.tau
            distillation_loss = F.kl_div(
                F.log_softmax(outputs / T, dim=1),
                F.log_softmax(teacher_outputs / T, dim=1),
                reduction='sum',
                log_target=True
            ) * (T * T) / outputs.numel()
           
        elif self.distillation_type == 'hard':
            distillation_loss = torch.sum(-teacher_outputs * F.log_softmax(outputs, dim=-1), dim=-1).mean()

        loss = base_loss * (1 - self.alpha) + distillation_loss * self.alpha
        return loss


class MarginLoss(torch.nn.Module):
    """
    This module wraps a standard criterion and adds an extra knowledge distillation loss by
    taking a teacher model prediction and using it as additional supervision.
    """
    def __init__(self, margin=0.5, alpha1=2, alpha2=0.5):
        super().__init__()
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.margin = margin

    def forward(self, outputs, labels, bottom_outputs):
        """
        Args:
            inputs: The original inputs that are feed to the teacher model
            outputs: the outputs of the model to be trained. It is expected to be
                either a Tensor, or a Tuple[Tensor, Tensor], with the original output
                in the first position and the distillation predictions as the second output
            labels: the labels for the base criterion
        """
        
        base_loss = torch.sum(-labels * F.log_softmax(outputs, dim=-1), dim=-1).mean()

        if bottom_outputs is None:
            loss = base_loss
        
        else:
            gt_labels = labels.argmax(dim=1).unsqueeze(1)
            outputs = F.softmax(outputs, dim=-1)
            bottom_outputs = F.softmax(bottom_outputs, dim=-1)
            topk_prob = batched_index_select(outputs, dim=1, index=gt_labels)
            bottom_prob = batched_index_select(bottom_outputs, dim=1, index=gt_labels)
            margin_loss = bottom_prob - topk_prob + self.margin
            margin_loss = F.relu(margin_loss).mean()
            loss = base_loss * self.alpha1 + margin_loss * self.alpha2
            
        return loss


_LOSSES = {
    "cross_entropy": nn.CrossEntropyLoss,
    "bce": nn.BCELoss,
    "bce_logit": nn.BCEWithLogitsLoss,
    "soft_cross_entropy": SoftTargetCrossEntropy,
    "soft_cross_entropy_pruning": SoftTargetCrossEntropyPruning,
    "distill": DistillationLoss,
    "margin": MarginLoss
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
