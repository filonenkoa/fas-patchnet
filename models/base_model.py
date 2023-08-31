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
    
    @property
    @abstractmethod
    def can_reparameterize(self) -> bool:
        pass
    
    def reparameterize(self) -> Module:
        if not self.can_reparameterize:
            raise Exception("Reparameterization is not supported for the current model")
        return self._reparameterize()
    
    def _reparameterize(self) -> Module:
        pass