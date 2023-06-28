
from torch.optim import lr_scheduler
from box import Box
import torch


def init_scheduler(config: Box, optimizer: torch.optim.Optimizer, state_dict: dict) -> tuple:
    batch_schedulers = ("cos", "cosr", "plt")
    name = config.train.scheduler_name
    if name == 'cosr':
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(
                        optimizer,
                        T_0=config.train.restart_epoch,
                        T_mult=config.train.restart_multiplier,
                        eta_min=1e-16, last_epoch=-1)
    # elif name == 'exp':
    #     scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=config.gamma, last_epoch=-1)
    # elif name == 'cos':
    #     scheduler = lr_scheduler.CosineAnnealingLR(
    #                     optimizer,
    #                     T_max=config.max_epoch-config.warmup,
    #                     eta_min=1e-8, last_epoch=-1)
    # elif name == 'plt':
    #     scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=config.gamma, patience=config.lr_patience)
    # elif name == 'steps':  # steps case
    #     scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=config.stepsize, gamma=config.gamma,
    #                                             last_epoch=-1)
    else:
        scheduler = None
    if state_dict:
        scheduler.load_state_dict(state_dict)
    return scheduler, name in batch_schedulers
