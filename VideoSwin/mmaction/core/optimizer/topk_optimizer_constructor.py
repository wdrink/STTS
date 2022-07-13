import torch
from mmcv.runner import OPTIMIZER_BUILDERS, DefaultOptimizerConstructor
from mmcv.utils import SyncBatchNorm, _BatchNorm, _ConvNd


@OPTIMIZER_BUILDERS.register_module()
class TopkOptimizerConstructor(DefaultOptimizerConstructor):
    """Optimizer constructor in TSM model.

    This constructor builds optimizer in different ways from the default one.

    1. Parameters of the first conv layer have default lr and weight decay.
    2. Parameters of BN layers have default lr and zero weight decay.
    3. If the field "fc_lr5" in paramwise_cfg is set to True, the parameters
       of the last fc layer in cls_head have 5x lr multiplier and 10x weight
       decay multiplier.
    4. Weights of other layers have default lr and weight decay, and biases
       have a 2x lr multiplier and zero weight decay.
    """

    def add_params(self, params, model):
        """Add parameters and their corresponding lr and wd to the params.

        Args:
            params (list): The list to be modified, containing all parameter
                groups and their corresponding lr and wd configurations.
            model (nn.Module): The model to be trained with the optimizer.
        """
        train_topk_only = self.paramwise_cfg['train_topk_only']
        # Batchnorm parameters.
        bn_params = []
        # Non-batchnorm parameters.
        non_bn_params = []
        predictor = []

        for name, param in model.named_parameters():
            if 'predictor' in name:
                predictor.append(param)
            elif train_topk_only:
                continue    # frozen weights other than predictor
            elif "bn" in name:
                bn_params.append(param)
            else:
                non_bn_params.append(param)
        
        params.append({
            'params': predictor,
            'lr': self.base_lr,
            'weight_decay': self.base_wd
        })

        params.append({
            'params': bn_params,
            'lr': self.base_lr * 0.01,
            'weight_decay': 0.0
        })

        params.append({
            'params': non_bn_params,
            'lr': self.base_lr * 0.01,
            'weight_decay': self.base_wd
        })


    
        
