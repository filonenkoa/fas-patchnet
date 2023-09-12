import os
import sys
from dataclasses import dataclass
from os.path import dirname as dn
from pathlib import Path
from types import FunctionType

from box import Box
import torch
import torch.nn as nn
import torch.distributed as dist


root_path = dn(dn(os.path.abspath(__file__)))
sys.path.append(root_path)
from reporting import Severity, report
from utils.misc import replace_file_line
from base_model import BaseModel
from torch.nn import Module

try:
    replace_file_line(
    Path(root_path, "modules/apple_fastvit/models/fastvit.py"),
    "from models.modules.mobileone import MobileOneBlock",
    "from modules.apple_fastvit.models.modules.mobileone import MobileOneBlock")
    replace_file_line(
        Path(root_path, "modules/apple_fastvit/models/fastvit.py"),
        "from models.modules.replknet import ReparamLargeKernelConv",
        "from modules.apple_fastvit.models.modules.replknet import ReparamLargeKernelConv")
except Exception:
    pass

from modules.apple_fastvit.models.fastvit import (fastvit_s12, fastvit_t8,
                                                  fastvit_t12)
from modules.apple_fastvit.models.modules.mobileone import reparameterize_model

try:
    replace_file_line(
        Path(root_path, "modules/apple_fastvit/models/fastvit.py"),
        "from modules.apple_fastvit.models.modules.mobileone import MobileOneBlock",
        "from models.modules.mobileone import MobileOneBlock")
    replace_file_line(
        Path(root_path, "modules/apple_fastvit/models/fastvit.py"),
        "from modules.apple_fastvit.models.modules.replknet import ReparamLargeKernelConv",
        "from models.modules.replknet import ReparamLargeKernelConv")
except Exception:
    pass


@dataclass
class FastVitMeta:
    download_link: str
    pretrained_path: Path
    init_function: FunctionType


class FastVitBackbone(BaseModel):
    """ Utilize FastVit from
    https://github.com/apple/ml-fastvit
    """
    
    METADATA = {
        "T8":
            FastVitMeta(
                "https://docs-assets.developer.apple.com/ml-research/models/fastvit/image_classification_distilled_models/fastvit_t8.pth.tar",
                Path("weights/fastvit/fastvit_t8.pth.tar"),
                fastvit_t8),
        "T12": FastVitMeta(
                "https://docs-assets.developer.apple.com/ml-research/models/fastvit/image_classification_distilled_models/fastvit_t12.pth.tar",
                Path("weights/fastvit/fastvit_t12.pth.tar"),
                fastvit_t12),
        "S12": FastVitMeta(
                "https://docs-assets.developer.apple.com/ml-research/models/fastvit/image_classification_distilled_models/fastvit_s12.pth.tar",
                Path("weights/fastvit/fastvit_s12.pth.tar"),
                fastvit_s12),
    }
    
    def __init__(self, config: Box, type: str):
        super().__init__(config)
        descriptor_size = config.model.descriptor_size
        drop_rate = config.model.dropout
        drop_path_rate = config.model.drop_path_rate
        metadata = self.METADATA[type.upper()]
        

        self.backbone = metadata.init_function(num_classes=1000,
                                               drop_rate=drop_rate,
                                               drop_path_rate=drop_path_rate)
        if self.config.model.pretrained:
            if not metadata.pretrained_path.is_file() and self.config.local_rank == 0:
                report(f"Could not find a pretrained checkpoint at {metadata.pretrained_path.absolute().as_posix()}. Downloading", severity=Severity.WARN)
                import wget
                response = wget.download(metadata.download_link, metadata.pretrained_path.as_posix())
            if self.config.world_size > 1:
                dist.barrier()
            state_dict = torch.load(metadata.pretrained_path)
            self.backbone.load_state_dict(state_dict["state_dict"])

        input_neurons = self.backbone.head.weight.shape[1]
        self.backbone.head = nn.Linear(input_neurons, descriptor_size, bias=False)
        nn.init.trunc_normal_(self.backbone.head.weight, std=0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)
    
    @property
    def can_reparameterize(self) -> bool:
        return True
    
    def _reparameterize(self) -> Module:
        return reparameterize_model(self.backbone)


def FASTVIT_T8(config: Box=None):
    return FastVitBackbone(
        config=config,
        type="T8")
    

def FASTVIT_T12(config: Box=None):
    return FastVitBackbone(
        config=config,
        type="T12")
    
    
def FASTVIT_S12(config: Box=None):
    return FastVitBackbone(
        config=config,
        type="S12")