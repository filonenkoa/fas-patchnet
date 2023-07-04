from abc import ABCMeta, abstractmethod

class BaseTrainer(metaclass=ABCMeta):
    def __init__(self, config, network, optimizer, lr_scheduler, is_batch_scheduler: bool, device, trainloader, valloader, writer):
        self.config = config
        self.network = network
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.is_batch_scheduler = is_batch_scheduler
        self.device = device
        self.trainloader = trainloader
        self.valloader = valloader
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