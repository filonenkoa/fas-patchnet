import os
from pathlib import Path
from random import randint
from box import Box
import torch
from torch import Module
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
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
                 device: torch.device,
                 trainloader: DataLoader,
                 valloader: DataLoader,
                 writer: SummaryWriter):
        super().__init__(config, network, optimizer, loss, lr_scheduler, device, trainloader, valloader, writer)
        # self.network = self.network.to(device)
        
        self.train_loss_metric = AvgMeter(writer=writer, name='Loss/train', num_iter_per_epoch=len(self.trainloader), per_iter_vis=True)
        self.train_acc_metric = AvgMeter(writer=writer, name='Accuracy/train', num_iter_per_epoch=len(self.trainloader), per_iter_vis=True)

        self.val_loss_metric = AvgMeter(writer=writer, name='Loss/val', num_iter_per_epoch=len(self.valloader))
        self.val_acc_metric = AvgMeter(writer=writer, name='Accuracy/val', num_iter_per_epoch=len(self.valloader))
        
        self.epoch_time = AvgMeter(writer=writer, name="Epoch time, s", num_iter_per_epoch=1)
        self.best_val_acc = 0.0
        self.best_epoch = -1
        
    def save_model(self, epoch):
        file_name = Path(self.config.log_dir, f"{epoch:04d}_{self.config.model.base}_{self.val_acc_metric.val:.4d}.pth")

        state = {
            'epoch': epoch,
            'model': self.network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "val_loss": self.val_loss_metric.val,
            "val_acc": self.val_acc_metric.val
        }
        
        torch.save(state, file_name.as_posix())
        
    def train_one_epoch(self, epoch):

        self.network.train()
        self.train_loss_metric.reset(epoch)
        self.train_acc_metric.reset(epoch)

        for i, (img1, img2, label) in enumerate(self.trainloader):
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
                print('Epoch: {:3}, iter: {:5}, loss: {:.5}, acc: {:.5}'.\
                    format(epoch, epoch * len(self.trainloader) + i, \
                    self.train_loss_metric.avg, self.train_acc_metric.avg))
        
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
            
    def train(self):
        for epoch in trange(self.config.train.num_epochs):
            epoch_start_time = time.time()
            self.train_one_epoch(epoch)
            val_loss = self.validate(epoch)
            if self.val_acc_metric.val > self.best_val_acc:
                if self.config.world_rank == 0:
                    logger.info(f"Validation accuracy improved from {self.best_val_acc:.4d} to {self.val_acc_metric.val:.4d}")
                self.best_val_acc = self.val_acc_metric.val
                self.best_epoch = epoch
                
            self.save_model(epoch)
            epoch_time = time.time() - epoch_start_time
            self.epoch_time.update(epoch_time)
            if self.config.world_rank == 0:
                logger.info(f"Epoch {epoch} time = {self.epoch_time.val} seconds")
                logger.info(f"Best accuracy =  {self.best_val_acc:.4d} at epoch {self.best_epoch}")
            
            
    def validate(self, epoch):
        self.network.eval()
        self.val_loss_metric.reset(epoch)
        self.val_acc_metric.reset(epoch)

        seed = randint(0, len(self.valloader)-1)
        
        with torch.no_grad():
            for i, (img1, img2, label) in enumerate(self.valloader):
                img1, img2, label = img1.to(self.device), img2.to(self.device), label.to(self.device)
                feature1 = self.network(img1)
                feature2 = self.network(img2)
                loss = self.loss(feature1, feature2, label)

                score1 = F.softmax(self.loss.amsm_loss.s * self.loss.amsm_loss.fc(feature1.squeeze()), dim=1)
                score2 = F.softmax(self.loss.amsm_loss.s * self.loss.amsm_loss.fc(feature2.squeeze()), dim=1)

                acc1 = calc_acc(score1, label.squeeze().type(torch.int32))
                acc2 = calc_acc(score2, label.squeeze().type(torch.int32))
                accuracy = (acc1 + acc2) / 2

                # Update metrics
                self.val_loss_metric.update(loss.item())
                self.val_acc_metric.update(accuracy)
        if self.config.world_rank == 0:
            logger.info("Validation epoch {} =============================".format(epoch))
            logger.info("Epoch: {:3}, loss: {:.5}, acc: {:.5}".format(epoch, self.val_loss_metric.avg, self.val_acc_metric.avg))
            logger.info("=================================================")

        return self.val_loss_metric.avg
                