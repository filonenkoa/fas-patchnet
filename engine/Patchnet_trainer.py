import os
from pathlib import Path
from random import randint
import sys
from box import Box
import torch
from torch.nn import Module
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
from containers import ClassificationMetrics, PredictionCounters
from engine.base_trainer import BaseTrainer
from metrics.meter import AvgMeter
from tqdm.auto import tqdm, trange
import time
from utils.utils import calc_acc
from loguru import logger

from pathlib import Path
sys.path.append(Path(__file__).resolve().parents[1].as_posix())

from metrics.metrics_counter import MetricsCounter


class Trainer(BaseTrainer):
    def __init__(self,
                 config: Box,
                 network: Module,
                 optimizer: torch.optim.Optimizer,
                 lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
                 is_batch_scheduler: bool,
                 device: torch.device,
                 trainloader: DataLoader,
                 valloader: DataLoader,
                 writer: SummaryWriter,
                 start_epoch: int = 0):
        super().__init__(config, network, optimizer, lr_scheduler, is_batch_scheduler, device, trainloader, valloader, writer)
        # self.network = self.network.to(device)
        
        self.train_loss_metric = AvgMeter(writer=writer, name='Loss/train', num_iter_per_epoch=len(self.trainloader), per_iter_vis=True)
        self.train_acc_metric = AvgMeter(writer=writer, name='Accuracy/train', num_iter_per_epoch=len(self.trainloader), per_iter_vis=True)

        self.val_loss_metric = AvgMeter(writer=writer, name='Loss/val', num_iter_per_epoch=len(self.valloader))
        self.val_acc_metric = AvgMeter(writer=writer, name='Accuracy/val', num_iter_per_epoch=len(self.valloader))
        
        self.start_epoch = start_epoch
        
        self.epoch_time = AvgMeter(writer=writer, name="Epoch time, s", num_iter_per_epoch=1)
        self.best_val_acer = 1.0
        self.best_epoch = -1
        
        self.metrics_counter = MetricsCounter()
        
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
        max_num_batches = len(self.trainloader)
        
        iterator = enumerate(self.trainloader)
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
            batch_metrics = self.metrics_counter(prediction_counters_batch)
            epoch_metrics = self.metrics_counter(prediction_counters_epoch)
            
            
            self.train_loss_metric.update(loss.item())
            self.train_acc_metric.update(accuracy)
            if self.config.world_rank == 0:
                text = (
                    f"E: {epoch}, "
                    f"loss: {self.train_loss_metric.val:.4f}|{self.train_loss_metric.avg:.4f}, "
                    f"acc: {self.train_acc_metric.val*100:.2f}%|{self.train_acc_metric.avg*100:.2f}%, "
                    f"ACER: {batch_metrics.acer*100:.2f}%|{epoch_metrics.acer*100:.2f}%, "
                    f"F1: {batch_metrics.f1*100:.2f}%|{epoch_metrics.f1*100:.2f}%, "
                    f"F3: {batch_metrics.f3*100:.2f}%|{epoch_metrics.f3*100:.2f}%, "
                    f"P: {batch_metrics.precision*100:.2f}%|{epoch_metrics.precision*100:.2f}%, "
                    f"R: {batch_metrics.recall*100:.2f}%|{epoch_metrics.recall*100:.2f}%, "
                    f"S: {batch_metrics.specificity*100:.2f}%|{epoch_metrics.specificity*100:.2f}%"
                    )
                iterator.set_description(text)
                   
        if self.config.world_rank == 0:
            logger.info(f"Epoch {epoch}, train metrics:\n{epoch_metrics}")
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
            
    def train(self):
        if self.config.train.val_before_train and self.config.world_rank == 0:
            epoch = -1
            val_metrics = self.validate(epoch)
            if val_metrics.acer < self.best_val_acer:
                logger.info(f"Validation ACER improved from {self.best_val_acer:.4f} to {val_metrics.acer:.4f}")
                self.best_val_acer = val_metrics.acer
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
            
            if hasattr(self.trainloader.sampler, "set_epoch"):
                self.trainloader.sampler.set_epoch(epoch)
            self.train_one_epoch(epoch)
            if dist.is_initialized():
                dist.barrier()
            if self.config.world_rank == 0:
                val_metrics = self.validate(epoch)
                if val_metrics.acer < self.best_val_acer:
                    logger.info(f"Validation ACER improved from {self.best_val_acer*100:.2f}% to {val_metrics.acer*100:.2f}%")
                    self.best_val_acer = val_metrics.acer
                    self.best_epoch = epoch
                
                self.save_model(epoch, val_metrics)
                epoch_time = time.time() - epoch_start_time
                self.epoch_time.update(epoch_time)
                logger.info(f"Epoch {epoch} time = {int(self.epoch_time.val)} seconds")
                logger.info(f"Best ACER = {self.best_val_acer:.4f} at epoch {self.best_epoch}")
            
            
    def validate(self, epoch) -> ClassificationMetrics: # TODO: make it work with multiple GPUs
        self.network.eval()
        self.val_loss_metric.reset(epoch)
        self.val_acc_metric.reset(epoch)
        prediction_counters_epoch = PredictionCounters()
        
        with torch.no_grad():
            for (img1, img2, label) in tqdm(self.valloader, desc=f"Validating epoch {epoch}"):
                img1, img2, label = img1.to(self.device), img2.to(self.device), label.to(self.device)
                model = self.network.module if dist.is_initialized() else self.network
                feature1 = model.get_descriptors(img1)
                feature2 = model.get_descriptors(img2)
                loss = model.compute_loss(feature1, feature2, label)

                score1 = model.predict(feature1)
                # score2 = F.softmax(self.loss.amsm_loss.s * self.loss.amsm_loss.fc(feature2.squeeze()), dim=1)
                
                label_squeezed = label.squeeze().type(torch.int8)

                acc1 = calc_acc(score1, label_squeezed)
                # acc2 = calc_acc(score2, label.squeeze().type(torch.int32))
                # accuracy = (acc1 + acc2) / 2
                accuracy = acc1
                
                prediction_counters_batch = PredictionCounters()
                self.update_prediction_counters(prediction_counters_batch, score1, label_squeezed)
                prediction_counters_epoch += prediction_counters_batch

                # Update metrics
                self.val_loss_metric.update(loss.item())
                self.val_acc_metric.update(accuracy)

        if self.config.world_rank == 0:
            metrics = self.metrics_counter(prediction_counters_epoch)
            logger.info("\nValidation epoch {} =============================".format(epoch))
            logger.info("Epoch: {:3}, loss: {:.5}, acc: {:.5}\n".format(epoch, self.val_loss_metric.avg, self.val_acc_metric.avg))
            logger.info(metrics)
            logger.info("=================================================")

            self.writer.add_scalar("ACER/val", metrics.acer, epoch)
            self.writer.add_scalar("APCER/val", metrics.apcer, epoch)
            self.writer.add_scalar("BPCER/val", metrics.bpcer, epoch)
            self.writer.add_scalar("F1/val", metrics.f1, epoch)
            self.writer.add_scalar("F3/val", metrics.f3, epoch)
            self.writer.add_scalar("Precision/val", metrics.precision, epoch)
            self.writer.add_scalar("Recall/val", metrics.recall, epoch)
            self.writer.add_scalar("Specificity/val", metrics.specificity, epoch)
        torch.cuda.empty_cache()
        return metrics
                