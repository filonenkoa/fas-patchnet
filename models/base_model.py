from box import Box
from torch import Module, Tensor
from abc import ABCMeta, abstractmethod


class BaseModel(Module, metaclass=ABCMeta):
    def __init__(self, config: Box):
        self.config = config
        
    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        pass