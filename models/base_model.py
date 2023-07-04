from box import Box
from torch import Tensor
from torch.nn import Module
from abc import ABCMeta, abstractmethod


class BaseModel(Module, metaclass=ABCMeta):
    def __init__(self, config: Box):
        super().__init__()
        self.config = config
        
    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        pass