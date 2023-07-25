import datetime
import math
import os
import shutil
import sys
from typing import List, Tuple
from box import Box
import numpy as np
import torch
import warnings
import argparse
from pathlib import Path
from loguru import logger
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
from albumentations import Compose

from pathlib import Path

sys.path.append(Path(__file__).resolve().parents[1].as_posix())

from samplers import DistributedBalancedSampler
from engine.Patchnet_trainer import Trainer
from dataset.FAS_dataset import FASDataset
from utils.utils import read_cfg
from dataset.transforms import get_transforms
from models import build_network, load_checkpoint
from optimizers import get_optimizer
from schedulers import init_scheduler
from reporting import log


def get_config() -> Box:
    parser = argparse.ArgumentParser(description='Face anti-spoofing training: PatchNet')
    parser.add_argument('--config', type=str, help='Name of the configuration (.yaml) file')
    args = parser.parse_args()
    config_path = Path("config", args.config)
    assert config_path.is_file(), f"Configuration file {config_path} does not exist"
    config = read_cfg(cfg_file=config_path.as_posix())
    config.log_root = config.log_dir
    config.config_path = config_path
    return config


def init_logger(config: Box) -> None:
    logger.add(Path(config.log_dir, "log.log"))


def confirm_dataset_path(dataset_path: Path) -> Path:
    if not dataset_path.is_file():
        possible_markup_path = dataset_path / "markup.csv"
        if possible_markup_path.is_file():
            dataset_path = possible_markup_path
        else:
            raise Exception(f"Could not find markup file for {dataset_path}")
    return dataset_path


def initialize_datasets(dataset_paths: List[str], transforms: Compose, smoothing: bool, is_train: bool) -> List[FASDataset]:
    datasets = []
    for dataset_path in dataset_paths:
        dataset_path = confirm_dataset_path(Path(dataset_path))
        datasets.append(FASDataset(
            root_dir=dataset_path.parent,
            csv_path=dataset_path,
            transform=transforms,
            smoothing=smoothing,
            is_train=is_train
        ))
    return datasets
            

def get_train_set(config: Box, transforms: Compose) -> ConcatDataset:
    assert len(config.dataset.train_set) > 0
    if config.world_rank == 0:
        logger.info(f"Combining {config.dataset.train_set} train datasets")
    datasets = ConcatDataset(initialize_datasets(config.dataset.train_set, transforms, config.dataset.smoothing, True))
    return datasets
    

def get_val_sets(config: Box, transforms: Compose) -> List[FASDataset]:
    assert len(config.dataset.val_set) > 0
    max_datasets_per_rank = math.ceil(len(config.dataset.val_set) / config.world_size)
    lower_bound = config.world_rank * max_datasets_per_rank
    upper_bound = min((config.world_rank + 1) * max_datasets_per_rank, len(config.dataset.val_set))
    local_datasets = config.dataset.val_set[lower_bound:upper_bound]
    config.datasets_start_index = lower_bound
    config.all_dataset_names = [FASDataset.path_to_name(p) for p in config.dataset.val_set]
    log(f"Rank {config.world_rank}: Local validation datasets: {[FASDataset.path_to_name(ds) for ds in local_datasets]}")
    datasets = initialize_datasets(local_datasets, transforms, config.dataset.smoothing, False)
    return datasets


def get_dataloaders(config: Box) -> Tuple[DataLoader, List[DataLoader]]:
    train_transform = get_transforms(config, is_train=True)
    val_transform = get_transforms(config, is_train=False)
    
    train_dataset = get_train_set(config, train_transform)
    val_datasets = get_val_sets(config, val_transform)

    sampler = None
    if config.world_size > 1:
        if config.train.balanced_sampler:
            sampler = DistributedBalancedSampler(
                dataset=train_dataset,
                num_replicas=config.world_size,
                rank=config.world_rank,
                shuffle=True,
                seed=config.seed,
                drop_last=False)
        else:
            sampler = torch.utils.data.distributed.DistributedSampler(
                dataset = train_dataset,
                shuffle = True,
                seed=config.seed,
                drop_last=True
            )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config['train']['batch_size'],
        shuffle=True if sampler is None else False,
        num_workers=config.dataset.num_workers,
        sampler=sampler
    )
    val_loaders = []
    for ds in val_datasets:
        val_loaders.append(
            DataLoader(
                dataset=ds,
                batch_size=config['val']['batch_size'],
                shuffle=False,
                num_workers=config.dataset.num_workers_val,
                drop_last=False
                ))
    
    return train_loader, val_loaders
    

def init_libraries(config: Box) -> None:
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True


def init_communication(config: Box) -> None:
    if config.device_name == "hpu":  # Habana Gaudi (Synapse AI)       
        import habana_frameworks.torch.distributed.hccl
        from habana_frameworks.torch.distributed.hccl import initialize_distributed_hpu
        config.world_size, config.world_rank, config.local_rank = initialize_distributed_hpu()
        
        if config.local_rank == -1:
            config.world_size = 1
            config.world_rank = 0
            config.local_rank = 0
        
        if config.world_size > 0:
            # patch torch cuda functions that are being unconditionally invoked
            # in the multiprocessing data loader
            torch.cuda.current_device = lambda: None
            torch.cuda.set_device = lambda x: None
        
        # To improve resnet dist performance, decrease number of all_reduce calls to 1 by increasing bucket size to 230
        dist._DEFAULT_FIRST_BUCKET_BYTES = 230*1024*1024
        
        dist.init_process_group(
                backend="hccl",
                timeout=datetime.timedelta(seconds=config.dist_timeout),
                # init_method=f"tcp://{config.dist_url}",
                world_size=config.world_size,
                rank=config.world_rank
                )

    elif config.device_name == "cuda":
        world_size = os.environ.get("WORLD_SIZE")
        if not world_size:
            config.world_size = 1
            config.local_rank = 0
            config.world_rank = 0
            config.dist_url = "127.0.0.1:23001"
        else:
            config.world_size = int(os.environ["WORLD_SIZE"])
            config.local_rank = int(os.environ["LOCAL_RANK"])
            config.world_rank = int(os.environ["RANK"])
            config.dist_url = f'{os.environ["MASTER_ADDR"]}:{os.environ["MASTER_PORT"]}'
        
            dist.init_process_group(
                    backend="nccl",
                    timeout=datetime.timedelta(seconds=config.dist_timeout),
                    init_method=f"tcp://{config.dist_url}",
                    world_size=config.world_size,
                    rank=config.world_rank
                    )
        
        logger.info(f"DEVICE: {config.device_name}\tWORLD: {config.world_size}\tLOCAL_RANK: {config.local_rank}\tWORLD_RANK: {config.world_rank}")
    os.environ["ID"] = str(config.local_rank)
    os.environ["LOCAL_RANK"] = str(config.local_rank)


def init_device(config: Box):
    if config.device_name == "cpu":
         config.device = torch.device("cpu")
    elif config.device_name == "cuda":
        config.device = torch.device("cuda")
        torch.cuda.set_device(f"cuda:{config.local_rank}")
    elif config.device_name == "hpu":
        config.device = torch.device('hpu')
    else:
        raise Exception(f"Device {config.device_name} is not supported")
    logger.info(f"Rank {config.local_rank}. Set device {config.device}")


def main() -> None:
    config: Box = get_config()
    init_libraries(config)
    init_communication(config)
    
    if config.world_rank == 0:
        config.log_root = config.log_dir
        current_datetime = datetime.datetime.now()
        config.log_dir = Path(config.log_root, f"{config.model.base}_{config.dataset.name}", str(current_datetime).replace(":", "-"))
        config.log_dir.mkdir(parents=True)
        init_logger(config)
        shutil.copy(config.config_path, Path(config.log_dir, "config.yaml"))
    
    init_device(config)
    if config.local_rank == 0:
        logger.info(config)
    
    train_loader, val_loaders = get_dataloaders(config)
    
    # build model and engine
    state_dict = load_checkpoint(config)
    # model_state_dict =  state_dict["model"] if "model" in state_dict.keys() else None
    model = build_network(config, state_dict)
    model.to(config.device)
    optimizer = get_optimizer(config, model, state_dict.get("optimizer"))
    lr_scheduler, is_batch_scheduler = init_scheduler(config, optimizer, state_dict.get("scheduler"))
    
    writer = SummaryWriter(config.log_dir) if config.world_rank == 0 else None
    
    start_epoch = state_dict.get("epoch") if "epoch" in state_dict.keys() else 0
    
    trainer = Trainer(
        config=config,
        network=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        is_batch_scheduler=is_batch_scheduler,
        device=config.device,
        trainloader=train_loader,
        val_loaders=val_loaders,
        writer=writer,
        start_epoch=start_epoch
    )

    logger.info(f"Rank {config.world_rank}. Start training...")
    trainer.train()
    if writer is not None:
        writer.close()
        

if __name__ == "__main__":
    main()