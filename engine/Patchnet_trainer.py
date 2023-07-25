import os
from pathlib import Path
from random import randint
import sys
from typing import Dict, List
from box import Box
import torch
from torch.nn import Module
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
from containers import ClassificationMetrics, PredictionCounters
from engine.base_trainer import BaseTrainer
from metrics.meter import AvgMeter
from tqdm.auto import tqdm, trange
import time
import datetime

from pathlib import Path
sys.path.append(Path(__file__).resolve().parents[1].as_posix())

from metrics.metrics_counter import MetricsCounter
from reporting import report
from utils.utils import calc_acc


class Trainer(BaseTrainer):
    def __init__(self,
                 config: Box,
                 network: Module,
                 optimizer: torch.optim.Optimizer,
                 lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
                 is_batch_scheduler: bool,
                 device: torch.device,
                 trainloader: DataLoader,
                 val_loaders: List[DataLoader],
                 writer: SummaryWriter,
                 start_epoch: int = 0):
        super().__init__(config, network, optimizer, lr_scheduler, is_batch_scheduler, device, trainloader, val_loaders, writer)
        # self.network = self.network.to(device)
        
        self.train_loss_metric = AvgMeter(writer=writer, name='Loss/train', num_iter_per_epoch=len(self.train_loader), per_iter_vis=True)
        self.train_acc_metric = AvgMeter(writer=writer, name='Accuracy/train', num_iter_per_epoch=len(self.train_loader), per_iter_vis=True)

        self.val_loss_metric = AvgMeter(writer=writer, name='Loss/val', num_iter_per_epoch=len(self.val_loaders))
        self.val_acc_metric = AvgMeter(writer=writer, name='Accuracy/val', num_iter_per_epoch=len(self.val_loaders))
        
        self.start_epoch = start_epoch
        
        self.epoch_time = AvgMeter(writer=writer, name="Epoch time, s", num_iter_per_epoch=1)
        self.best_val_acer = 1.0
        self.best_epoch = -1
        
        self.metrics_counter = MetricsCounter()
        
        self.total_val_sets = len(self.config.dataset.val_set)
        
    def save_model(self, epoch, val_metrics: ClassificationMetrics):
        file_name = Path(self.config.log_dir, f"{epoch:04d}_{self.config.model.base}_{val_metrics.acer:.4f}.pth")
        
        model = self.network.module if dist.is_initialized() else self.network

        state = {
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            "scheduler": self.lr_scheduler.state_dict(),
            # "loss": self.loss.state_dict(),
            "val_loss": self.val_loss_metric.avg,
            "val_acc": self.val_acc_metric.avg
        }
        
        torch.save(state, file_name.as_posix())

    @staticmethod
    def update_prediction_counters(prediction_counters: PredictionCounters,
                                   prediction: torch.Tensor,
                                   target: torch.Tensor) -> None:
        predicted_class = torch.argmax(prediction, dim=1)
        # TODO: Think how to vectorize the computations
        for pred, t in zip(predicted_class, target):
            if pred == t:
                if pred == 0:
                    prediction_counters.tp += 1
                else:
                    prediction_counters.tn += 1
            else:
                if pred == 0:
                    prediction_counters.fp += 1
                else:
                    prediction_counters.fn += 1
        
    def train_one_epoch(self, epoch):
        prediction_counters_epoch = PredictionCounters()
        self.network.train()
        self.train_loss_metric.reset(epoch)
        self.train_acc_metric.reset(epoch)
        max_num_batches = len(self.train_loader)
        
        iterator = enumerate(self.train_loader)
        if self.config.local_rank == 0:
            iterator = tqdm(iterator, desc=f"Training epoch {epoch}", total=max_num_batches)
            
        for batch_index, (img1, img2, label) in iterator:
            if self.lr_scheduler is not None and self.is_batch_scheduler:
                self.lr_scheduler.step(epoch + (batch_index / max_num_batches))
            img1, img2, label = img1.to(self.device), img2.to(self.device), label.to(self.device)
            model = self.network.module if dist.is_initialized() else self.network
            feature1 = model.get_descriptors(img1)
            feature2 = model.get_descriptors(img2)
            self.optimizer.zero_grad()
            loss = model.compute_loss(feature1, feature2, label)
            loss.backward()
            self.optimizer.step()

            score1 = model.predict(feature1)
            score2 = model.predict(feature2)
            
            label_squeezed = label.squeeze().type(torch.int8)

            acc1 = calc_acc(score1, label_squeezed)
            acc2 = calc_acc(score2, label_squeezed)
            accuracy = (acc1 + acc2) / 2
            
            prediction_counters_batch = PredictionCounters()
            self.update_prediction_counters(prediction_counters_batch, score1, label_squeezed)
            prediction_counters_epoch += prediction_counters_batch
            
            # Update metrics
            # batch_metrics = self.metrics_counter(prediction_counters_batch)
            epoch_metrics = self.metrics_counter(prediction_counters_epoch)
            
            
            self.train_loss_metric.update(loss.item())
            self.train_acc_metric.update(accuracy)
            if self.config.world_rank == 0:
                lr = self.get_lr()
                text = (
                    f"E: {epoch}, "
                    f"loss: {self.train_loss_metric.avg:.4f}, "
                    f"acc: {self.train_acc_metric.avg*100:.2f}%, "
                    f"ACER: {epoch_metrics.acer*100:.4f}%, "
                    f"F1: {epoch_metrics.f1*100:.4f}%, "
                    f"F3: {epoch_metrics.f3*100:.4f}%, "
                    f"P: {epoch_metrics.precision*100:.4f}%, "
                    f"R: {epoch_metrics.recall*100:.4f}%, "
                    f"S: {epoch_metrics.specificity*100:.4f}%"
                    f"LR: {lr:.4E}"
                    )
                iterator.set_description(text)
                globiter = epoch * max_num_batches + batch_index
                self.writer.add_scalar("LR", lr, globiter) 
                   
        if self.config.world_rank == 0:
            epoch_report = f"\nEpoch {epoch}, train metrics:\n{epoch_metrics}"
            report(epoch_report, use_telegram=self.config.telegram_reports)
            self.writer.add_scalar("ACER/train", epoch_metrics.acer, epoch)
            self.writer.add_scalar("APCER/train", epoch_metrics.apcer, epoch)
            self.writer.add_scalar("BPCER/train", epoch_metrics.bpcer, epoch)
            self.writer.add_scalar("F1/train", epoch_metrics.f1, epoch)
            self.writer.add_scalar("F3/train", epoch_metrics.f3, epoch)
            self.writer.add_scalar("Precision/train", epoch_metrics.precision, epoch)
            self.writer.add_scalar("Recall/train", epoch_metrics.recall, epoch)
            self.writer.add_scalar("Specificity/train", epoch_metrics.specificity, epoch)
            
        if self.lr_scheduler is not None and not self.is_batch_scheduler:
            self.lr_scheduler.step()
        torch.cuda.empty_cache()
        
    def get_lr(self) -> float:
        current_lr = 0
        for param_group in self.optimizer.param_groups:
            current_lr = param_group['lr']
        return current_lr

            
    def train(self):
        if self.config.train.val_before_train:
            epoch = -1
            val_metrics = self.validate(epoch)
            if self.config.local_rank == 0:
                if val_metrics["total"].acer < self.best_val_acer:
                    report(f"Validation ACER improved from {self.best_val_acer*100:.4f}% to {val_metrics['total'].acer*100:.4f}%")
                    self.best_val_acer = val_metrics["total"].acer
                    self.best_epoch = epoch
        if dist.is_initialized():
            dist.barrier()
        
        iterator = range(self.start_epoch, self.config.train.num_epochs, 1)
        if self.config.local_rank == 0:
            iterator = tqdm(iterator)
        
        for epoch in iterator:
            epoch_start_time = time.time()
            
            if self.config.local_rank == 0:
                iterator.set_description(f"Epoch {epoch}")
            
            if hasattr(self.train_loader.sampler, "set_epoch"):
                self.train_loader.sampler.set_epoch(epoch)
            self.train_one_epoch(epoch)
            if dist.is_initialized():
                dist.barrier()
            val_metrics = self.validate(epoch)
            if self.config.world_rank == 0:
                if val_metrics["total"].acer < self.best_val_acer:
                    report(f"Validation ACER improved from {self.best_val_acer*100:.2f}% to {val_metrics['total'].acer*100:.2f}%")
                    self.best_val_acer = val_metrics["total"].acer
                    self.best_epoch = epoch
                
                self.save_model(epoch, val_metrics["total"])
                epoch_time = time.time() - epoch_start_time
                self.epoch_time.update(epoch_time)
                epoch_end_message = (
                    f"Epoch {epoch} time = {int(self.epoch_time.val)} seconds",
                    f"Best ACER = {self.best_val_acer*100:.4f}% at epoch {self.best_epoch}"
                    )
                report(epoch_end_message, use_telegram=self.config.telegram_reports)
    
    @staticmethod                 
    def estimate_time(start_time: float, cur_iter: int, max_iter: int):
        telapsed = time.time() - start_time
        testimated = (telapsed/cur_iter)*(max_iter)

        finishtime = start_time + testimated
        finishtime = datetime.datetime.fromtimestamp(finishtime).strftime("%H:%M:%S")  # in time

        lefttime = testimated-telapsed  # in seconds

        return (int(telapsed), int(lefttime), finishtime)
            
    def validate(self, epoch) -> Dict[str, ClassificationMetrics]: # TODO: make it work with multiple GPUs
        self.network.eval()
        # self.val_loss_metric.reset(epoch)
        # self.val_acc_metric.reset(epoch)
        prediction_counters = {val_loader.dataset.name: PredictionCounters() for val_loader in self.val_loaders}
        tqdm_desc = f"Rank {self.config.world_rank}: Validating epoch {epoch}"
        tqdm_position = self.config.world_rank + 1
        tqdm_pbar = tqdm(enumerate(self.val_loaders), desc=tqdm_desc, position=tqdm_position, total=len(self.val_loaders))
        
        if self.config.world_size > 1:  # collect prediction counters
            predictions_tensor = torch.zeros(self.total_val_sets * 4, dtype=torch.int64, device=self.config.device)
            
        num_batches = 0
        for val_loader in self.val_loaders:
            num_batches += len(val_loader)
        start_time = time.time()
        batch_index_acc = 0
        with torch.no_grad():
            for i, val_loader in tqdm_pbar:
                
                for batch_index, (img1, label) in enumerate(val_loader):
                    
                    img1, label = img1.to(self.device), label.to(self.device)
                    model = self.network.module if dist.is_initialized() else self.network
                    feature1 = model.get_descriptors(img1)
                    score1 = model.predict(feature1)
                    label_squeezed = label.squeeze().type(torch.int8)

                    prediction_counters_batch = PredictionCounters()
                    self.update_prediction_counters(prediction_counters_batch, score1, label_squeezed)
                    prediction_counters[val_loader.dataset.name] += prediction_counters_batch
                    batch_index_acc += 1
                    time_elapsed, time_left, time_eta = self.estimate_time(start_time, batch_index_acc, num_batches)
                    tqdm_pbar.set_description(tqdm_desc + f" {val_loader.dataset.name} {batch_index_acc}/{num_batches} ETA:{time_left}")
                    
                if self.config.world_size > 1:
                    position = (self.config.datasets_start_index + i) * 4
                    predictions_tensor[position:position + 4] = prediction_counters[val_loader.dataset.name].as_tensor()

        # Gather predictions
        if self.config.world_size > 1:
            dist.reduce(predictions_tensor, dst=0, op=dist.ReduceOp.SUM)
            for dataset_index, dataset_name in enumerate(self.config.all_dataset_names):
                lower_bound = dataset_index * 4
                dataset_tensor = predictions_tensor[lower_bound:lower_bound + 4]
                prediction_counters[dataset_name] = PredictionCounters.from_tensor(dataset_tensor)
        metrics = dict()
        if self.config.world_rank == 0:
            counters_sum = PredictionCounters()
            val_end_text = f"\nValidation epoch {epoch}"
            for dataset_name, counters in prediction_counters.items():
                metrics[dataset_name] = self.metrics_counter(counters)
                counters_sum += counters
                val_end_text += f"\nDataset {dataset_name}: {metrics[dataset_name]}"
                self.writer.add_scalar(f"ACER/{dataset_name}", metrics[dataset_name].acer, epoch)
                self.writer.add_scalar(f"APCER/{dataset_name}", metrics[dataset_name].apcer, epoch)
                self.writer.add_scalar(f"BPCER/{dataset_name}", metrics[dataset_name].bpcer, epoch)
                self.writer.add_scalar(f"F1/{dataset_name}", metrics[dataset_name].f1, epoch)
                self.writer.add_scalar(f"F3/{dataset_name}", metrics[dataset_name].f3, epoch)
                self.writer.add_scalar(f"Precision/{dataset_name}", metrics[dataset_name].precision, epoch)
                self.writer.add_scalar(f"Recall/{dataset_name}", metrics[dataset_name].recall, epoch)
                self.writer.add_scalar(f"Specificity/{dataset_name}", metrics[dataset_name].specificity, epoch)
                
                
            metrics["total"] = self.metrics_counter(counters_sum)
            val_end_text += f"\nCombined metrics: {metrics['total']}"
            
            self.writer.add_scalar(f"ACER/val", metrics["total"].acer, epoch)
            self.writer.add_scalar(f"APCER/val", metrics["total"].apcer, epoch)
            self.writer.add_scalar(f"BPCER/val", metrics["total"].bpcer, epoch)
            self.writer.add_scalar(f"F1/val", metrics["total"].f1, epoch)
            self.writer.add_scalar(f"F3/val", metrics["total"].f3, epoch)
            self.writer.add_scalar(f"Precision/val", metrics["total"].precision, epoch)
            self.writer.add_scalar(f"Recall/val", metrics["total"].recall, epoch)
            self.writer.add_scalar(f"Specificity/val", metrics["total"].specificity, epoch)    
                
            report(val_end_text, use_telegram=self.config.telegram_reports)
        
        if dist.is_initialized():
            dist.barrier()
        torch.cuda.empty_cache()
        return metrics
                