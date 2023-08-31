from box import Box
from torch import Tensor
from torch.nn import Module
import torch.nn.functional as F
from metrics.losses import PatchLoss

from models.base_model import BaseModel


class PatchnetModel(BaseModel):
    def __init__(self, config: Box, backbone: Module):
        super().__init__(config=config)
        self.backbone: Module = backbone
        self.patch_loss = PatchLoss(
            alpha1=self.config.loss.alpha1,
            alpha2=self.config.loss.alpha2,
            s=self.config.loss.s,
            m_l=self.config.loss.m_l,
            m_s=self.config.loss.m_s,
            descriptor_size=self.config.model.descriptor_size
            )

    def forward(self, x: Tensor) -> Tensor:
        x = self.get_descriptors(x)
        x = self.predict(x)
        return x
    
    def get_descriptors(self, x: Tensor) -> Tensor:
        return self.backbone(x)
    
    def compute_loss(self, descriptor1: Tensor, descriptor2: Tensor, label: Tensor):
        loss = self.patch_loss(descriptor1, descriptor2, label)
        return loss
    
    def predict(self, descriptor: Tensor) -> Tensor:
        return F.softmax(self.patch_loss.amsm_loss.s * self.patch_loss.amsm_loss.fc(descriptor), dim=1)
    
    @property
    def can_reparameterize(self) -> bool:
        return self.backbone.can_reparameterize

    def reparameterize(self) -> Module:
        self.backbone = self.backbone.reparameterize()