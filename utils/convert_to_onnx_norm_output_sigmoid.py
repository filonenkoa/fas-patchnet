"""
Sometimes, PatchNet may be too sure in its predictions, and outputs for the netrowk will be [1.000, 0.000] or [0.000, 1.000] for any input.
This script computes the actual unnormalized outputs, collects their statistics, and generates the rescaled normalizer with two sigmoids.
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


class ModelWithRescalerSigmoid(Module):
    def __init__(self, patchnet_model: PatchnetModel, device: torch.device, scale=0.1, add_input_norm=True):
        super().__init__()
        self.register_module("patchnet_model", patchnet_model.to(device))
        self.scale = scale
        self.add_input_norm = add_input_norm
        self.patchnet_model.use_softmax = False
        
        if add_input_norm:
            mean = [0.406, 0.456, 0.485]
            std = [0.225, 0.224, 0.229]
            self.bn = torch.nn.BatchNorm2d(3, affine=False, track_running_stats=True, eps=1e-10)
            var = [s * s for s in std]

            self.bn.running_mean.data = torch.Tensor(mean)
            self.bn.running_var.data = torch.Tensor(var)

    def forward(self, x):
        if self.add_input_norm:
            x = self.bn(x)
        x = self.patchnet_model(x)
        x = x * self.scale
        x = torch.nn.functional.sigmoid(x)
        return x


def get_config() -> Box:
    parser = argparse.ArgumentParser(
        description="Face anti-spoofing training: PatchNet. Output re-normalization."
    )
    parser.add_argument(
        "--config", type=str, help="Name of the configuration (.yaml) file"
    )
    parser.add_argument("--checkpoint", type=str, help="Path to the checkpoint file")
    parser.add_argument(
        "--datasets",
        nargs="+",
        help="Datasets used to compute statistics",
        required=True,
    )
    parser.add_argument("-b", "--batch_size", type=int, default=1, help="Batch size. -1 means dynamic")
    parser.add_argument(
        "-n",
        "--num_workers",
        type=int,
        default=-1,
        help="Number of CPU workers for the data loader",
    )
    parser.add_argument(
        "--precomputed_outputs",
        type=str,
        default="",
        help="Path to the file with precomputed outputs generated by 'get_outputs' function",
    )
    
    args = parser.parse_args()

    config_path = Path("config", args.config)
    assert config_path.is_file(), f"Configuration file {config_path} does not exist"
    config = read_cfg(cfg_file=config_path.as_posix())
    config.log_root = config.log_dir
    config.config_path = config_path
    config.model.checkpoint_path = args.checkpoint
    config.dataset.val_set = args.datasets
    config.batch_size = abs(args.batch_size)
    assert config.batch_size != 0
    config.dynamic_batch = args.batch_size < 0
    config.dataset.num_workers = args.num_workers
    config.precomputed_outputs = (
        args.precomputed_outputs if args.precomputed_outputs != "" else None
    )

    return config


def get_outputs(
    model: PatchnetModel, data_loader: DataLoader, log_dir: Path, device: torch.device
) -> Path:
    predictions_file = Path(log_dir, "predictions.npy")
    model.eval()
    model.use_softmax = False
    predictions = []

    for img1, label in tqdm(
        data_loader, desc="Getting predictions", total=len(data_loader)
    ):
        img1, label = img1.to(device), label.to(device)
        score = model(img1)
        predictions.append(score.cpu().detach().numpy())

    logger.info(f"Inference is done. Saving outputs to {predictions_file.as_posix()}")
    predictions = np.concatenate(predictions, axis=0)
    np.save(predictions_file.as_posix(), predictions)
    return predictions_file


if __name__ == "__main__":
    config = get_config()
    ignore_normalization = True

    config.log_root = config.log_dir
    current_datetime = datetime.datetime.now()
    config.log_dir = Path(
        config.log_root,
        "renormalize_output_sigmoid",
        f"{config.model.base}_{config.dataset.name}",
        str(current_datetime).replace(":", "-"),
    )

    config.log_dir.mkdir(parents=True)
    logger.add(Path(config.log_dir, "log.log"))
    shutil.copy(config.config_path, Path(config.log_dir, "config.yaml"))
    config.device = torch.device("cuda")

    data_transforms = get_transforms(config, is_train=False, ignore_normalization=ignore_normalization)
    datasets = ConcatDatasetWithLabels(
        initialize_datasets(
            config.dataset.val_set, data_transforms, config.dataset.smoothing, False
        )
    )

    data_loader = DataLoader(
        dataset=datasets,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.dataset.num_workers,
        drop_last=False,
    )

    config.world_rank = 0
    config.world_size = 1
    config.model.pretrained = False
    config.test_inference_speed = False

    state_dict = load_checkpoint(config)
    model = build_network(config, state_dict)
    model.to(config.device)

    if not config.precomputed_outputs:
        outputs_path = get_outputs(model, data_loader, config.log_dir, config.device)
    else:
        outputs_path = Path(config.precomputed_outputs)

    for img, label in data_loader:
        img, label = img.to(config.device), label.to(config.device)
        if img.dtype == torch.uint8:
            img = img.to(torch.float32)
            img /= 255.0
        break

    logger.info(f"Loading predictions from {outputs_path}")

    norm_model = ModelWithRescalerSigmoid(model, config.device).to(config.device)
    norm_model.eval()
    
    
    norm_output = norm_model(img)
    suffix = "dynamic" if config.dynamic_batch else config.batch_size
    file_name_onnx = Path(f"{Path(config.model.checkpoint_path).stem}_{suffix}.onnx")
    onnx_path = config.log_dir / file_name_onnx
    logger.info(f"Converting with {suffix} batch size")
    convert_to_onnx(config, norm_model, onnx_path, batch_size=img.shape[0], sample=img, dynamic_batch=config.dynamic_batch)
