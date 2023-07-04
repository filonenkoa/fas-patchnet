import os
from pathlib import Path
from random import randint
from box import Box
import torch
from torch.nn import Module
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
from engine.base_trainer import BaseTrainer
from metrics.meter import AvgMeter
from tqdm.auto import tqdm, trange
import time
from utils.utils import calc_acc
from loguru import logger


class Trainer(BaseTrainer):
    def __init__(self,
                 config: Box,
                 network: Module,
                 optimizer: torch.optim.Optimizer,
                 loss: Module,
                 lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
                 is_batch_scheduler: bool,
                 device: torch.device,
                 trainloader: DataLoader,
                 valloader: DataLoader,
                 writer: SummaryWriter):
        super().__init__(config, network, optimizer, loss, lr_scheduler, is_batch_scheduler, device, trainloader, valloader, writer)
        # self.network = self.network.to(device)
        
        self.train_loss_metric = AvgMeter(writer=writer, name='Loss/train', num_iter_per_epoch=len(self.trainloader), per_iter_vis=True)
        self.train_acc_metric = AvgMeter(writer=writer, name='Accuracy/train', num_iter_per_epoch=len(self.trainloader), per_iter_vis=True)

        self.val_loss_metric = AvgMeter(writer=writer, name='Loss/val', num_iter_per_epoch=len(self.valloader))
        self.val_acc_metric = AvgMeter(writer=writer, name='Accuracy/val', num_iter_per_epoch=len(self.valloader))
        
        self.epoch_time = AvgMeter(writer=writer, name="Epoch time, s", num_iter_per_epoch=1)
        self.best_val_acc = 0.0
        self.best_epoch = -1
        
    def save_model(self, epoch):
        file_name = Path(self.config.log_dir, f"{epoch:04d}_{self.config.model.base}_{self.val_acc_metric.avg:.4f}.pth")

        state = {
            'epoch': epoch,
            'model': self.network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            "scheduler": self.lr_scheduler.state_dict(),
            "loss": self.loss.state_dict(),
            "val_loss": self.val_loss_metric.avg,
            "val_acc": self.val_acc_metric.avg
        }
        
        torch.save(state, file_name.as_posix())
        
    def train_one_epoch(self, epoch):

        self.network.train()
        self.train_loss_metric.reset(epoch)
        self.train_acc_metric.reset(epoch)
        max_num_batches = len(self.trainloader)
        if self.config.local_rank == 0:
            iterator = tqdm(enumerate(self.trainloader), desc=f"Training epoch {epoch}", total=max_num_batches)
        else:
            iterator = enumerate(self.trainloader)
        for batch_index, (img1, img2, label) in iterator:
            if self.lr_scheduler is not None and self.is_batch_scheduler:
                self.lr_scheduler.step(epoch + (batch_index / max_num_batches))
            img1, img2, label = img1.to(self.device), img2.to(self.device), label.to(self.device)
            feature1 = self.network(img1)
            feature2 = self.network(img2)
            self.optimizer.zero_grad()
            loss = self.loss(feature1, feature2, label)
            loss.backward()
            self.optimizer.step()

            score1 = F.softmax(self.loss.amsm_loss.s * self.loss.amsm_loss.fc(feature1.squeeze()), dim=1)
            score2 = F.softmax(self.loss.amsm_loss.s * self.loss.amsm_loss.fc(feature2.squeeze()), dim=1)

            acc1 = calc_acc(score1, label.squeeze().type(torch.int32))
            acc2 = calc_acc(score2, label.squeeze().type(torch.int32))
            accuracy = (acc1 + acc2) / 2
            
            # Update metrics
            self.train_loss_metric.update(loss.item())
            self.train_acc_metric.update(accuracy)
            if self.config.world_rank == 0:
                text = (
                    f"E: {epoch:04d}, "
                    f"loss: {self.train_loss_metric.val:.5f}|{self.train_loss_metric.avg:.5f}, "
                    f"acc: {self.train_acc_metric.val*100:.3f}%|{self.train_acc_metric.avg*100:.3f}%"
                    )
                iterator.set_description(text)
        
        if self.lr_scheduler is not None and not self.is_batch_scheduler:
            self.lr_scheduler.step()
        torch.cuda.empty_cache()
            
    def train(self):
        if self.config.train.val_before_train and self.config.world_rank == 0:
            epoch = -1
            self.validate(epoch)
            if self.val_acc_metric.avg > self.best_val_acc:
                logger.info(f"Validation accuracy improved from {self.best_val_acc:.4f} to {self.val_acc_metric.avg:.4f}")
                self.best_val_acc = self.val_acc_metric.avg
                self.best_epoch = epoch
        if dist.is_initialized():
            dist.barrier() 
        for epoch in trange(self.config.train.num_epochs):
            epoch_start_time = time.time()
            if hasattr(self.trainloader.sampler, "set_epoch"):
                self.trainloader.sampler.set_epoch(epoch)
            self.train_one_epoch(epoch)
            if dist.is_initialized():
                dist.barrier()
            if self.config.world_rank == 0:
                self.validate(epoch)
                if self.val_acc_metric.avg > self.best_val_acc:
                    logger.info(f"Validation accuracy improved from {self.best_val_acc:.4f} to {self.val_acc_metric.avg:.4f}")
                    self.best_val_acc = self.val_acc_metric.avg
                    self.best_epoch = epoch
                
                self.save_model(epoch)
                epoch_time = time.time() - epoch_start_time
                self.epoch_time.update(epoch_time)
                logger.info(f"Epoch {epoch} time = {int(self.epoch_time.val)} seconds")
                logger.info(f"Best accuracy =  {self.best_val_acc:.4f} at epoch {self.best_epoch}")
            
            
    def validate(self, epoch): # TODO: make it work with multiple GPUs
        self.network.eval()
        self.val_loss_metric.reset(epoch)
        self.val_acc_metric.reset(epoch)
        
        with torch.no_grad():
            for (img1, img2, label) in tqdm(self.valloader, desc=f"Validating epoch {epoch}"):
                img1, img2, label = img1.to(self.device), img2.to(self.device), label.to(self.device)
                feature1 = self.network(img1)
                feature2 = self.network(img2)
                loss = self.loss(feature1, feature2, label)

                score1 = F.softmax(self.loss.amsm_loss.s * self.loss.amsm_loss.fc(feature1.squeeze()), dim=1)
                # score2 = F.softmax(self.loss.amsm_loss.s * self.loss.amsm_loss.fc(feature2.squeeze()), dim=1)

                acc1 = calc_acc(score1, label.squeeze().type(torch.int32))
                # acc2 = calc_acc(score2, label.squeeze().type(torch.int32))
                # accuracy = (acc1 + acc2) / 2
                accuracy = acc1

                # Update metrics
                self.val_loss_metric.update(loss.item())
                self.val_acc_metric.update(accuracy)
        if self.config.world_rank == 0:
            logger.info("Validation epoch {} =============================".format(epoch))
            logger.info("Epoch: {:3}, loss: {:.5}, acc: {:.5}".format(epoch, self.val_loss_metric.avg, self.val_acc_metric.avg))
            logger.info("=================================================")
        torch.cuda.empty_cache()
        return self.val_loss_metric.avg
                