# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Optimizer."""

import torch

import slowfast.utils.lr_policy as lr_policy


def construct_optimizer(model, cfg):
    """
    Construct a stochastic gradient descent or ADAM optimizer with momentum.
    Details can be found in:
    Herbert Robbins, and Sutton Monro. "A stochastic approximation method."
    and
    Diederik P.Kingma, and Jimmy Ba.
    "Adam: A Method for Stochastic Optimization."

    Args:
        model (model): model to perform stochastic gradient descent
        optimization or ADAM optimization.
        cfg (config): configs of hyper-parameters of SGD or ADAM, includes base
        learning rate,  momentum, weight_decay, dampening, and etc.
    """
    train_topk_only = cfg.TRAIN.TRAIN_TOPK_ONLY
    # Batchnorm parameters.
    bn_params = []
    # Non-batchnorm parameters.
    non_bn_params = []
    zero_params = []
    predictor = []

    skip = {}
    if hasattr(model, "no_weight_decay"):
        skip = model.no_weight_decay()

    for name, m in model.named_modules():
        is_bn = isinstance(m, torch.nn.modules.batchnorm._NormBase)
        for p in m.parameters(recurse=False):
            if not p.requires_grad:
                continue
            if 'predictor' in name:
                predictor.append(p)
            elif train_topk_only:
                continue
            elif is_bn:
                bn_params.append(p)
            elif name in skip or (
                (len(p.shape) == 1 or name.endswith(".bias"))
                and cfg.SOLVER.ZERO_WD_1D_PARAM
            ):
                zero_params.append(p)
            else:
                non_bn_params.append(p)
    
    
    optim_params = [
        {"params": predictor, "weight_decay": cfg.SOLVER.WEIGHT_DECAY, 'name': 'predictor'},
        {"params": bn_params, "weight_decay": cfg.BN.WEIGHT_DECAY, 'name': 'backbone_bn'},
        {"params": non_bn_params, "weight_decay": cfg.SOLVER.WEIGHT_DECAY, 'name': 'backbone_nonbn'},
        {"params": zero_params, "weight_decay": 0.0, 'name': 'bacbone_zero'},
    ]

    optim_params = [x for x in optim_params if len(x["params"])]


    
    

    
    

    if cfg.SOLVER.OPTIMIZING_METHOD == "sgd":
        return torch.optim.SGD(
            optim_params,
            lr=cfg.SOLVER.BASE_LR,
            momentum=cfg.SOLVER.MOMENTUM,
            weight_decay=cfg.SOLVER.WEIGHT_DECAY,
            dampening=cfg.SOLVER.DAMPENING,
            nesterov=cfg.SOLVER.NESTEROV,
        )
    elif cfg.SOLVER.OPTIMIZING_METHOD == "adam":
        return torch.optim.Adam(
            optim_params,
            lr=cfg.SOLVER.BASE_LR,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=cfg.SOLVER.WEIGHT_DECAY,
        )
    elif cfg.SOLVER.OPTIMIZING_METHOD == "adamw":
        return torch.optim.AdamW(
            optim_params,
            lr=cfg.SOLVER.BASE_LR,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=cfg.SOLVER.WEIGHT_DECAY,
        )
    else:
        raise NotImplementedError(
            "Does not support {} optimizer".format(cfg.SOLVER.OPTIMIZING_METHOD)
        )


def get_epoch_lr(cur_epoch, cfg):
    """
    Retrieves the lr for the given epoch (as specified by the lr policy).
    Args:
        cfg (config): configs of hyper-parameters of ADAM, includes base
        learning rate, betas, and weight decays.
        cur_epoch (float): the number of epoch of the current training stage.
    """
    return lr_policy.get_lr_at_epoch(cfg, cur_epoch)


def set_lr(optimizer, new_lr, cfg):
    """
    Sets the optimizer lr to the specified value.
    Args:
        optimizer (optim): the optimizer using to optimize the current network.
        new_lr (float): the new learning rate to set.
    """
    if cfg.TRAIN.FINETUNE:
        for param_group in optimizer.param_groups:
            if param_group['name'] == 'predictor':
                param_group['lr'] = new_lr[0]
            else:
                param_group['lr'] = new_lr[1]
    else:
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr[0]

