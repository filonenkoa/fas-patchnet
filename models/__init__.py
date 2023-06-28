import copy
from pathlib import Path
from box import Box
from loguru import logger
import torch
from ptflops import get_model_complexity_info
from torch.nn.parallel import DistributedDataParallel as DDP


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


def get_backbone(config: Box) -> torch.Module:
    if config.mode.base == "cdcn":
        from CDCNs import FeatureExtractor as Backbone
    elif config.mode.base == "convnext_tiny":
        from convnext_tiny import FeatureExtractor as Backbone
    elif config.mode.base == "dc_cdn":
        from DC_CDN import FeatureExtractor as Backbone
    elif config.mode.base == "resnet18":
        from resnet18 import FeatureExtractor as Backbone
    elif config.mode.base == "swin_base":
        from swin_base import FeatureExtractor as Backbone
    elif config.mode.base == "efficientformerv2_s0":
        from efficientformer import EFFICIENTFORMER_V2_S0 as Backbone
    elif config.mode.base == "efficientformerv2_s1":
        from efficientformer import EFFICIENTFORMER_V2_S1 as Backbone
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


def build_network(config: Box, state_dict: dict):
    logger.info(f"Rank {config.world_rank}. Initializing model")
    backbone = get_backbone(config)
    
    with torch.no_grad():
        macs, params = get_model_complexity_info(
            backbone,
            (3, config.dataset.crop_size, config.dataset.crop_size),
            as_strings=False,
            print_per_layer_stat=False,
            verbose=False)
    if config.world_rank == 0:
        logger.info(f"🧠 Model parameters: {params/1_000_000:.3f} M")
        logger.info(f"💻 Model complexity: {macs/1_000_000_000:.3f} GMACs")
        
    if state_dict is not None:
        model_raw = copy.deepcopy(backbone)
        # model_raw = model_raw.to(config.device)
        backbone.load_state_dict(state_dict, strict = config.model.resume_strict)
        difference_ratio = models_weights_difference_ratio(model_raw, backbone)
        logger.info(f"The difference between before and after weights loading is {difference_ratio*100:.4}%")
        
    if config.world_size > 1:
        if config.device_name == "cuda":
            # model = DDP(model, device_ids=[config.local_rank], output_device=config.local_rank)
            backbone = DDP(backbone,
                        device_ids=[config.local_rank],
                        output_device=config.local_rank,
                        broadcast_buffers=False,
                        gradient_as_bucket_view=True,
                        static_graph=False)
        else:
            raise Exception(f"DDP does not work with device {config.device_name}")    
    return backbone
