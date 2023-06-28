from box import Box
import torch
from torch import nn
from torchvision import models
from base_model import BaseModel


class FeatureExtractor(BaseModel):
    def __init__(self, config: Box):
        super().__init__(config=config)
        base_model = models.resnet18(pretrained=self.config.model.pretrained)
        self.nets = nn.Sequential(*(list(base_model.children())[:-1]))
        
    def forward(self, x):
        return self.nets(x)
