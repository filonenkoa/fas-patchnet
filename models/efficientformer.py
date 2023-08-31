from pathlib import Path
import sys
import os
from os.path import dirname as dn
from box import Box

root_path = dn(dn(os.path.abspath(__file__)))
sys.path.append(root_path)

import torch
import torch.nn as nn

from modules.efficientformer.models.efficientformer_v2 import efficientformerv2_l, efficientformerv2_s0, efficientformerv2_s1
from base_model import BaseModel


class EfficientFormerV2Backbone(BaseModel):
    """ Utilize EfficientFormerV2 from
    https://github.com/snap-research/EfficientFormer
    """
    
    PRETRAINED_FILE_NAMES = {
        "S1": "eformer_s1_450.pth"
    }
    
    def __init__(self, config: Box, type: str):
        super().__init__(config)
        resolution = config.dataset.crop_size
        descriptor_size = config.model.descriptor_size
        drop_rate = config.model.dropout          
        
        if type.upper() == "L":            
            self.backbone = efficientformerv2_l(resolution=resolution,
                                                num_classes=1000,
                                                distillation=False,
                                                drop_rate=drop_rate)
        elif type.upper() == "S0":            
            self.backbone = efficientformerv2_s0(resolution=resolution,
                                                 num_classes=1000,
                                                 distillation=False,
                                                 drop_rate=drop_rate)
        elif type.upper() == "S1":            
            self.backbone = efficientformerv2_s1(resolution=resolution,
                                                 num_classes=1000,
                                                 distillation=False,
                                                 drop_rate=drop_rate)
        if self.config.model.pretrained:
            state_dict = torch.load(Path("weights", "efficientformerv2", self.PRETRAINED_FILE_NAMES[type.upper()]))
            del state_dict["model"]["dist_head.weight"]
            del state_dict["model"]["dist_head.bias"]
            self.backbone.load_state_dict(state_dict["model"])

        input_neurons = self.backbone.head.weight.shape[1]
        self.backbone.head = nn.Linear(input_neurons, descriptor_size, bias=False)
        nn.init.trunc_normal_(self.backbone.head.weight, std=0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)
    
    @property
    def can_reparameterize(self) -> bool:
        return False


def EFFICIENTFORMER_V2_S0(config: Box=None):
    return EfficientFormerV2Backbone(
        config=config,
        type="S0")
    

def EFFICIENTFORMER_V2_S1(config: Box=None):
    return EfficientFormerV2Backbone(
        config=config,
        type="S1")
