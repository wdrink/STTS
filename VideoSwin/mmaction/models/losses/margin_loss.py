import torch
import torch.nn.functional as F
import torch.nn as nn
from ..builder import LOSSES


def batched_index_select(input, dim, index):
    for i in range(1, len(input.shape)):
        if i != dim:
            index = index.unsqueeze(i)
    expanse = list(input.shape)
    expanse[0] = -1
    expanse[dim] = -1
    index = index.expand(expanse)
    return torch.gather(input, dim, index)



@LOSSES.register_module()
class MarginLoss(nn.Module):
    """
    This module wraps a standard criterion and adds an extra knowledge distillation loss by
    taking a teacher model prediction and using it as additional supervision.
    """
    def __init__(self, margin=0.5, alpha1=2, alpha2=0.5, loss_weight=1.):
        super().__init__()
    
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.margin = margin

    def forward(self, cls_score, labels, bottom_outputs):
        """
        Args:
            inputs: The original inputs that are feed to the teacher model
            outputs: the outputs of the model to be trained. It is expected to be
                either a Tensor, or a Tuple[Tensor, Tensor], with the original output
                in the first position and the distillation predictions as the second output
            labels: the labels for the base criterion
        """
        
        base_loss = F.cross_entropy(cls_score, labels)

        if bottom_outputs is None:
            loss = base_loss
        
        else:
            labels = labels.unsqueeze(1)
            outputs = F.softmax(cls_score, dim=-1)
            bottom_outputs = F.softmax(bottom_outputs, dim=-1)
            topk_prob = batched_index_select(outputs, dim=1, index=labels)
            bottom_prob = batched_index_select(bottom_outputs, dim=1, index=labels)
            margin_loss = bottom_prob - topk_prob + self.margin
            margin_loss = F.relu(margin_loss).mean()
            loss = base_loss * self.alpha1 + margin_loss * self.alpha2
            
        return loss