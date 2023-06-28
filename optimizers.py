from __future__ import absolute_import
from box import Box

import torch
import os
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))


def init_optim(optim: str, params: dict, lr: float, weight_decay: float, momentum=0.9):
    if optim == 'adam':
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    elif optim == 'adamw':
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    elif optim == 'amsgrad':
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay, amsgrad=True)
    elif optim == 'sgd':
        return torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif optim == 'rmsprop':
        return torch.optim.RMSprop(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif optim == 'fusedsgd':
        from habana_frameworks.torch.hpex.optimizers import FusedSGD
        return FusedSGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif optim == 'fusedadamw':
        from habana_frameworks.torch.hpex.optimizers import FusedAdamW
        return FusedAdamW(params, lr=lr, weight_decay=weight_decay)
    else:
        raise KeyError("Unsupported optimizer: {}".format(optim))
    

def get_optimizer(config: Box, model: torch.nn.Module, state_dict: dict):
    optimizer = init_optim(
        optim=config.train.optimizer,
        params=model.parameters(),
        lr=config.train.lr,
        weight_decay=config.train.weight_decay)
    if state_dict:
        optimizer.load_state_dict(state_dict)
    return optimizer
