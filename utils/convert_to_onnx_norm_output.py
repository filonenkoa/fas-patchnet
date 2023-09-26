"""
Sometimes, PatchNet may be too sure in its predictions, and outputs for the netrowk will be [1.000, 0.000] or [0.000, 1.000] for any input.
This script computes the actual unnormalized outputs, collects their statistics, and generates the rescaled normalizer instead of SoftMax.
"""
import argparse
import datetime
from pathlib import Path
import shutil
import sys
from typing import Dict

from box import Box
from loguru import logger
import numpy as np
import torch
from torch.nn import Module
from tqdm import tqdm



sys.path.append(Path(__file__).absolute().parent.parent.as_posix())
from torch.utils.data import DataLoader, ConcatDataset
from models.patchnet_model import PatchnetModel
from tool.train import initialize_datasets
from dataset.FAS_dataset import FASDataset, ConcatDatasetWithLabels
from utils.misc import read_cfg
from dataset.transforms import get_transforms
from models import build_network, load_checkpoint
from utils.convert_to_onnx import convert_to_onnx



class ModelWithRescaler(Module):
    def __init__(self, patchnet_model: PatchnetModel, stats: Box, device: torch.device, use_95_percentile: bool = True):
        super().__init__()
        self.register_module("patchnet_model", patchnet_model.to(device))
        # self.patchnet_model = patchnet_model
        self.patchnet_model.use_softmax = False
        # self.stats = stats
        self.register_buffer("min_val", torch.tensor([0, 0], dtype=torch.float32))
        if use_95_percentile:
            max_val = torch.tensor([stats.percentile95_spoof, stats.percentile95_live], device=device, dtype=torch.float32)
        else:
            max_val = torch.tensor([self.max_spoof, self.max_live], device=device, dtype=torch.float32)
        self.register_buffer("max_val", max_val)
        
    def forward(self, x):
        unnormalized_x = self.patchnet_model(x)
        normalized = (unnormalized_x - self.min_val) / (self.max_val - self.min_val)
        clamped = torch.clamp(normalized, min = 0.0, max = 1.0)
        return clamped
        
        

def get_config() -> Box:
    parser = argparse.ArgumentParser(description='Face anti-spoofing training: PatchNet. Output re-normalization.')
    parser.add_argument('--config', type=str, help='Name of the configuration (.yaml) file')
    parser.add_argument('--checkpoint', type=str, help='Path to the checkpoint file')
    parser.add_argument('--precomputed_predictions', type=str, default=None, help='Path to the precomputed outputs file')
    parser.add_argument('--datasets', nargs='+', help='Datasets used to compute statistics', required=True)
    parser.add_argument('-b', '--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('-n', '--num_workers', type=int, default=1, help='Number of CPU workers for the data loader')
    args = parser.parse_args()
    
    config_path = Path("config", args.config)
    assert config_path.is_file(), f"Configuration file {config_path} does not exist"
    config = read_cfg(cfg_file=config_path.as_posix())
    config.log_root = config.log_dir
    config.config_path = config_path
    config.model.checkpoint_path = args.checkpoint
    config.dataset.val_set = args.datasets
    config.batch_size = args.batch_size
    config.dataset.num_workers = args.num_workers
    config.precomputed_predictions_path = args.precomputed_predictions
    
    return config


def get_outputs(model: PatchnetModel, data_loader: DataLoader, log_dir: Path, device: torch.device) -> tuple:
    predictions_file = Path(log_dir, "predictions.npy")
    model.eval()
    model.use_softmax = False
    predictions = []

        
    for (img1, _) in tqdm(data_loader, desc="Getting predictions", total=len(data_loader)):
        img1 = img1.to(device)
        score = model(img1)
        predictions.append(score.cpu().detach().numpy())
    
    logger.info(f"Inference is done. Saving outputs to {predictions_file.as_posix()}")
    predictions = np.concatenate(predictions, axis=0)
    np.save(predictions_file.as_posix(), predictions)
    return predictions_file


def get_predictions_minmax(path: Path) -> Box:
    try:
        predictions = np.load(path.as_posix())
    except Exception as ex:
        logger.error(f"Could not load {path}, {ex}")
        return None
    
    logger.info("Computing min and max values")
    
    spoof_predictions = predictions[:, 0]
    live_predictions = predictions[:, 1]
    
    min_spoof = np.min(spoof_predictions)
    min_live = np.min(live_predictions)
    
    max_spoof = np.max(spoof_predictions)
    max_live = np.max(live_predictions)
    
    percentile95_spoof = np.percentile(spoof_predictions, 95)
    percentile95_live = np.percentile(live_predictions, 95)
    
    result = Box({
        "min_spoof": min_spoof,
        "min_live": min_live,
        "max_spoof": max_spoof,
        "max_live": max_live,
        "percentile95_spoof": percentile95_spoof,
        "percentile95_live": percentile95_live
        })
    
    return result


if __name__ == "__main__":
    config = get_config()
    config.log_root = config.log_dir
    current_datetime = datetime.datetime.now()
    config.log_dir = Path(config.log_root, "renormalize_output", f"{config.model.base}_{config.dataset.name}", str(current_datetime).replace(":", "-"))
    config.log_dir.mkdir(parents=True)
    logger.add(Path(config.log_dir, "log.log"))
    shutil.copy(config.config_path, Path(config.log_dir, "config.yaml"))
    config.device = torch.device("cuda")
    # torch.cuda.set_device(f"cuda:0")
    
    data_transforms = get_transforms(config, is_train=False)
   
    # config.all_dataset_names = [FASDataset.path_to_name(p) for p in config.dataset.val_set]
    
    datasets = ConcatDatasetWithLabels(initialize_datasets(config.dataset.val_set, data_transforms, config.dataset.smoothing, False))
    
    data_loader = DataLoader(
        dataset=datasets,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.dataset.num_workers,
        drop_last=False
    )
    
    config.world_rank = 0
    config.world_size = 1
    config.model.pretrained = False
    config.test_inference_speed = False
    
    state_dict = load_checkpoint(config)
    model = build_network(config, state_dict)
    model.to(config.device)
    
    if config.precomputed_predictions_path is None:
        outputs_path = get_outputs(model, data_loader, config.log_dir, config.device)
    else:
        outputs_path = Path(config.precomputed_predictions_path)
        assert outputs_path.is_file()
        
    assert outputs_path
    logger.info(f"Loading predictions from {outputs_path}")
    
    minmax = get_predictions_minmax(outputs_path)
    logger.info(minmax)
    # torch.cuda.set_device("cpu")
    norm_model = ModelWithRescaler(model, minmax, config.device, True).to(config.device)
    norm_model.eval()
    img1, _ = datasets[0]
    img2, _ = datasets[len(datasets)-1]
    img = torch.tensor(np.stack([img1, img2]), device=config.device)
    norm_output = norm_model(img)

    
    onnx_path = config.log_dir / Path(f"{Path(config.model.checkpoint_path).stem}.onnx")
    logger.info("Converting with dynamic batch size")
    convert_to_onnx(config, norm_model, onnx_path, batch_size=img.shape[0], sample=img)

    