import torch.nn.functional as F
import torch.nn as nn
from ..builder import LOSSES



@LOSSES.register_module()
class DistillationLoss(nn.Module):
    """
    This module wraps a standard criterion and adds an extra knowledge distillation loss by
    taking a teacher model prediction and using it as additional supervision.
    """
    def __init__(self, distill_type='hard', alpha=0.5, tau=1.0, loss_weight=0.):
        super().__init__()
        assert distill_type in ['none', 'soft', 'hard']
        self.distillation_type = distill_type
        self.alpha = alpha
        self.tau = tau

    def forward(self, cls_score, labels, teacher_outputs):
        """
        Args:
            inputs: The original inputs that are feed to the teacher model
            outputs: the outputs of the model to be trained. It is expected to be
                either a Tensor, or a Tuple[Tensor, Tensor], with the original output
                in the first position and the distillation predictions as the second output
            labels: the labels for the base criterion
        """
        if cls_score.size() == labels.size():
            # calculate loss for soft label
            lsm = F.log_softmax(cls_score, 1)
            loss_cls = -(labels * lsm).sum(1)
            base_loss = loss_cls.mean()

        else:
            # calculate loss for hard label
            base_loss = F.cross_entropy(cls_score, labels)

        # base_loss = F.cross_entropy(cls_score, labels)
        
        if self.distillation_type == 'none':
            return base_loss

        if self.distillation_type == 'soft':
            T = self.tau
            distillation_loss = F.kl_div(
                F.log_softmax(cls_score / T, dim=1),
                F.log_softmax(teacher_outputs / T, dim=1),
                reduction='sum',
                log_target=True
            ) * (T * T) / cls_score.numel()
           
        elif self.distillation_type == 'hard':
            distillation_loss = F.cross_entropy(cls_score, teacher_outputs.argmax(dim=1))

        loss = base_loss * (1 - self.alpha) + distillation_loss * self.alpha

        return loss

