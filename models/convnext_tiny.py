from box import Box
from torch import nn
from torchvision import models


class FeatureExtractor(nn.Module):
    def __init__(self, config: Box):
        super().__init__(config=config)
        base_model = models.convnext_tiny(pretrained=self.config.model.pretrained)
        self.nets = nn.Sequential(*(list(base_model.children())[:-1]))
        
    def forward(self, x):
        return self.nets(x)
