import copy
from pathlib import Path
import sys
from box import Box
from loguru import logger
import numpy as np
import torch
from ptflops import get_model_complexity_info
from torch.nn.parallel import DistributedDataParallel as DDP
from functools import partial

from pathlib import Path

from models.patchnet_model import PatchnetModel
from reporting import Severity, report
from utils.utils import test_inference_speed
sys.path.append(Path(__file__).resolve().parent.as_posix())
sys.path.append(Path(__file__).resolve().parents[1].as_posix())

from metrics.losses import PatchLoss
from models.base_model import BaseModel

def models_weights_difference_ratio(model_1: torch.nn.Module, model_2: torch.nn.Module) -> float:
    models_differ = 0
    total_items = 0
    for key_item_1, key_item_2 in zip(model_1.state_dict().items(), model_2.state_dict().items()):
        if torch.equal(key_item_1[1], key_item_2[1]):
            total_items += 1
        else:
            models_differ += 1
            total_items += 1
            if (key_item_1[0] == key_item_2[0]):
                pass
            else:
                print(f"Mismtach in keys found at {key_item_1[0]}. Are models the same?")
                raise Exception
    return models_differ / total_items


def get_backbone(config: Box) -> torch.nn.Module:
    if config.model.base == "cdcn":
        from CDCNs import FeatureExtractor as Backbone
    elif config.model.base == "convnext_tiny":
        from convnext_tiny import FeatureExtractor as Backbone
    elif config.model.base == "dc_cdn":
        from DC_CDN import FeatureExtractor as Backbone
    elif config.model.base == "resnet18":
        from resnet18 import FeatureExtractor as Backbone
    elif config.model.base == "swin_base":
        from swin_base import FeatureExtractor as Backbone
    elif config.model.base == "efficientformerv2_s0":
        from efficientformer import EFFICIENTFORMER_V2_S0 as Backbone
    elif config.model.base == "efficientformerv2_s1":
        from efficientformer import EFFICIENTFORMER_V2_S1 as Backbone
    elif config.model.base == "fastvit_t8":
        from fastvit import FASTVIT_T8 as Backbone
    elif config.model.base == "fastvit_t12":
        from fastvit import FASTVIT_T12 as Backbone
    elif config.model.base == "fastvit_s12":
        from fastvit import FASTVIT_S12 as Backbone
    else:
        raise NotImplementedError
    return Backbone(config)


def load_checkpoint(config: Box) -> dict:
    checkpoint = {}
    checkpoint_path = config.model.checkpoint_path
    if checkpoint_path != "":
        logger.info(f"Loading weights from {checkpoint_path}")
        assert Path(checkpoint_path).is_file(), f"Cannot find {checkpoint_path}"
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        if config.model.resume:
            if not config.train.load_optimizer:
                del checkpoint["optimizer"]
            if not config.train.load_scheduler:
                del checkpoint["scheduler"]
    return checkpoint


def random_input_constructor(input_res: int, dtype, device):
    return {"x": torch.rand((2, *input_res), requires_grad=False)}


def build_network(config: Box, state_dict: dict) -> PatchnetModel:
    logger.info(f"Rank {config.world_rank}. Initializing model")
    backbone: BaseModel = get_backbone(config)
    
    model = PatchnetModel(config=config, backbone=backbone)
    
    with torch.no_grad():
        test_model = copy.deepcopy(model)
        if config.world_rank == 0:
            test_model.eval()
            input_constructor = partial(random_input_constructor, dtype=next(test_model.parameters()).dtype, device=next(test_model.parameters()).device)
            macs, params = get_model_complexity_info(
                test_model,
                (3, config.dataset.crop_size, config.dataset.crop_size),
                as_strings=False,
                print_per_layer_stat=False,
                verbose=False, input_constructor=input_constructor)
            report(f"🧠 Model parameters: {params/1_000_000:.3f} M")
            report(f"💻 Model complexity: {macs/2_000_000_000:.3f} GMACs")
            
            raw_inference_speed = test_inference_speed(test_model, config.device, config.dataset.crop_size) * 1000
            report(f"Average inference time the original model: {raw_inference_speed:.4f} ms")
        
            
            if test_model.can_reparameterize:
                rep_model = copy.deepcopy(test_model)
                rep_model.reparameterize()
                macs, params = get_model_complexity_info(
                    rep_model,
                    (3, config.dataset.crop_size, config.dataset.crop_size),
                    as_strings=False,
                    print_per_layer_stat=False,
                    verbose=False, input_constructor=input_constructor)
                
                test_input = input_constructor((3,224,224))["x"]
                raw_output = test_model(test_input)
                reparameterized_output = rep_model(test_input)
                same_output = np.allclose(raw_output[0], reparameterized_output[0], atol=0.001)
                if not same_output:
                    report("Reparameterized model produces different outputs", Severity.WARN)
                    
                rep_inference_speed = test_inference_speed(rep_model, config.device, config.dataset.crop_size) * 1000
                
                del rep_model
                report(f"🧠 Model parameters: {params/1_000_000:.3f} M  after reparameterization")
                report(f"💻 Model complexity: {macs/2_000_000_000:.3f} GMACs after reparameterization")    
                report(f"Average inference time for original and reparameterized models: {raw_inference_speed:.4f} and {rep_inference_speed:.4f} ms")
        
    if state_dict.get("model") is not None:
        model_raw = copy.deepcopy(model)
        # model_raw = model_raw.to(config.device)
        model.load_state_dict(state_dict.get("model"), strict = config.model.resume_strict)
        difference_ratio = models_weights_difference_ratio(model_raw, model)
        logger.info(f"The difference between before and after weights loading is {difference_ratio*100:.4}%")
    model = model.to(config.device)   
    if config.world_size > 1:
        if config.device_name == "cuda":
            # model = DDP(model, device_ids=[config.local_rank], output_device=config.local_rank)
            model = DDP(model,
                        device_ids=[config.local_rank],
                        output_device=config.local_rank,
                        broadcast_buffers=False,
                        gradient_as_bucket_view=True,
                        static_graph=False)
        else:
            raise Exception(f"DDP does not work with device {config.device_name}")    
    return model
