from abc import ABCMeta, abstractmethod
from typing import List
from box import Box
import torch
from torch.nn import Module
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

class BaseTrainer(metaclass=ABCMeta):
    def __init__(self,
                 config: Box,
                 network: Module,
                 optimizer: torch.optim.Optimizer,
                 lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
                 is_batch_scheduler: bool,
                 device: torch.device,
                 train_loader: DataLoader,
                 val_loaders: List[DataLoader],
                 writer: SummaryWriter):
        self.config = config
        self.network = network
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.is_batch_scheduler = is_batch_scheduler
        self.device = device
        self.train_loader = train_loader
        self.val_loaders = val_loaders
        self.writer = writer

    @abstractmethod
    def save_model(self):
        raise NotImplementedError

    @abstractmethod
    def train_one_epoch(self):
        raise NotImplementedError

    @abstractmethod
    def train(self):
        raise NotImplementedError

    @abstractmethod
    def validate(self):
        raise NotImplementedError