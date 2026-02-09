#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
"""
Trainer for the DiscreteVAE model.

This script trains the DiscreteVAE model which encodes audio to 25Hz discrete tokens.

Usage:
    python minimaxspeech/trainers/vq_vae_trainer.py \
        --config configs/vq_vae_config_libritts.yaml

    # Multi-GPU with DDP
    export CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7'
    torchrun --nproc_per_node=8 minimaxspeech/trainers/vq_vae_trainer.py \
        --config configs/vq_vae_config_libritts.yaml
"""
import argparse
import logging
import os

import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from minimaxspeech.datasets.vae_dataset import VAEDataset
from minimaxspeech.modules.vq_vae.dvae import DiscreteVAE, TorchMelSpectrogram
from minimaxspeech.utils.commons.logger import setup_logger


class VQVAETrainer:
    def __init__(self, config: dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.setup_distributed()
        setup_logger(os.path.join(self.config.trainer.output_dir, 'logs'))
        if self.local_rank == 0:
            self.writer = SummaryWriter(os.path.join(self.config.trainer.output_dir, 'tensorboard'))
        else:
            self.writer = None

        self.start_epoch = 0
        self.global_step = 0
        self.best_loss = float('inf')

        self.setup_model()
        self.load_checkpoint()
        self.setup_optimizer()
        self.setup_scheduler()
        self.setup_dataset()

    def setup_distributed(self):
        # Setup distributed training
        self.distributed = int(os.environ.get('WORLD_SIZE', 1)) > 1
        if self.distributed:
            dist.init_process_group(backend='nccl')
            self.local_rank = int(os.environ.get('LOCAL_RANK', 0))
            torch.cuda.set_device(self.local_rank)
            logging.info(f"Distributed training")
        else:
            self.local_rank = 0
            logging.info(f"Single card training")

    def setup_model(self):
        self.torch_mel_spectrogram_vq_vae = TorchMelSpectrogram(
            mel_norm_file=self.config.trainer.mel_norm_file,
            sampling_rate=self.config.dataset.sample_rate
        ).to(self.device)
        self.model = DiscreteVAE(**self.config.model).to(self.device)
        logging.info(f"Model initialized and ready to train")

    def load_checkpoint(self):
        # Load pretrained decoder if available
        if self.config.trainer.resume:
            logging.info(f"Loading pretrained decoder from {self.config.trainer.checkpoint}")
            checkpoint = torch.load(self.config.trainer.checkpoint, map_location='cpu')
            
            if 'vq_vae' in checkpoint: # self-host checkpoint
                self.model.load_state_dict(checkpoint['vq_vae'])
                self.start_epoch = checkpoint['epoch'] + 1
                self.global_step = checkpoint['global_step']
            else: # xtts2 checkpoint
                self.model.load_state_dict(checkpoint)
                self.start_epoch = 0
                self.global_step = 0
            
            if 'optimizer' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            if 'scheduler' in checkpoint:
                self.scheduler.load_state_dict(checkpoint['scheduler'])
            logging.info(f"Resumed: start_epoch={self.start_epoch}, global_step={self.global_step}")

        # Wrap models with DDP if distributed
        if self.distributed:
            self.model = DDP(self.model, device_ids=[self.local_rank])

    def setup_optimizer(self):
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.trainer.learning_rate
        )

    def setup_scheduler(self):
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.config.trainer.epochs,
            eta_min=self.config.trainer.min_lr
        )

    def setup_dataset(self):
        # Setup dataset and dataloader
        self.train_dataset = VAEDataset(
            data_file_list=self.config.dataset.train_file_list,
            sample_rate=self.config.dataset.sample_rate,
            max_duration=self.config.dataset.max_duration
        )
        self.val_dataset = VAEDataset(
            data_file_list=self.config.dataset.valid_file_list,
            sample_rate=self.config.dataset.sample_rate,
            max_duration=self.config.dataset.max_duration
        )
        if self.distributed:
            self.train_sampler = DistributedSampler(self.train_dataset)
            self.val_sampler = DistributedSampler(self.val_dataset)
        else:
            self.train_sampler = None
            self.val_sampler = None
        
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.dataset.batch_size,
            shuffle=(self.train_sampler is None),
            sampler=self.train_sampler,
            num_workers=self.config.dataset.num_workers,
            pin_memory=True,
            drop_last=False,
            collate_fn=self.train_dataset.collate_fn
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config.dataset.batch_size,
            shuffle=(self.val_sampler is None),
            sampler=self.val_sampler,
            num_workers=self.config.dataset.num_workers,
            pin_memory=True,
            drop_last=False,
            collate_fn=self.val_dataset.collate_fn
        )

    def train_step(self, batch):
        """Single training step."""
        waveforms, filenames, lengths = batch
        waveforms = waveforms.to(self.device)

        mel_spectrograms = self.torch_mel_spectrogram_vq_vae(waveforms)
        remainder = mel_spectrograms.shape[-1] % 4
        if remainder:
            # if the mel spectrogram is not divisible by 4 then input.shape != output.shape 
            mel_spectrograms = mel_spectrograms[:, :, :-remainder]
        
        recon_loss, commitment_loss, out = self.model(mel_spectrograms)
        recon_loss = recon_loss.mean()
        total_loss = recon_loss + commitment_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return {
            'recon_loss': recon_loss.item(),
            'commitment_loss': commitment_loss.item(),
            'total_loss': total_loss.item()
        }

    def train(self):
        """Training loop"""
        logging.info(f"Start training: epoch={self.start_epoch}, global_step={self.global_step}")
        for epoch in range(self.start_epoch, self.config.trainer.epochs):
            self.current_epoch = epoch
            if self.distributed:
                self.train_sampler.set_epoch(epoch)
            local_step = 0
            for batch in self.train_loader:
                train_losses = self.train_step(batch)
                local_step += 1
                self.global_step += 1

                # Log training loss
                if self.global_step % self.config.trainer.log_interval == 0 and self.local_rank == 0:
                    lr = float(self.optimizer.param_groups[0]["lr"])
                    logging.info(f"Epoch: {epoch}, Global Step: {self.global_step}, "
                                 f"LR: {lr:.2e}, Train Losses: {train_losses}")
                    self.writer.add_scalar("train/lr", lr, self.global_step)
                    for k, v in train_losses.items():
                        self.writer.add_scalar(f'train/{k}', v, self.global_step)

                # Validation
                if self.global_step % self.config.trainer.val_interval == 0:
                    val_losses = self.validate()
                    if self.local_rank == 0:
                        logging.info(f"Epoch: {epoch}, Global Step: {self.global_step}, Valid Losses: {val_losses}")

                        for key in val_losses:
                            self.writer.add_scalar(f'val/{key}', val_losses[key], self.global_step)

                        if val_losses['total_loss'] < self.best_loss:
                            self.best_loss = val_losses['total_loss']
                            # self.save_checkpoint(self.model, self.optimizer, self.global_step, epoch, self.config)
                
                # Save checkpoint
                if self.global_step % self.config.trainer.save_interval == 0 and self.local_rank == 0:
                    self.save_checkpoint(self.model, self.optimizer, self.global_step, epoch, self.config)

            # Update scheduler
            self.scheduler.step()

    def validate(self):
        """Validation loop."""
        self.model.eval()
        val_losses = {'total_loss': 0, 'recon_loss': 0, 'commitment_loss': 0}
        valid_step = 0
        for batch in self.val_loader:
            waveforms, filenames, lengths = batch
            waveforms = waveforms.to(self.device)

            mel_spectrograms = self.torch_mel_spectrogram_vq_vae(waveforms)
            remainder = mel_spectrograms.shape[-1] % 4
            if remainder:
                # if the mel spectrogram is not divisible by 4 then input.shape != output.shape 
                mel_spectrograms = mel_spectrograms[:, :, :-remainder]

            recon_loss, commitment_loss, out = self.model(mel_spectrograms)
            
            recon_loss = recon_loss.mean()
            commitment_loss = commitment_loss.mean()
            total_loss = recon_loss + commitment_loss
            val_losses['total_loss'] += total_loss.item()
            val_losses['recon_loss'] += recon_loss.item()
            val_losses['commitment_loss'] += commitment_loss.item()
            valid_step += 1
        
        val_losses['total_loss'] /= valid_step
        val_losses['recon_loss'] /= valid_step
        val_losses['commitment_loss'] /= valid_step
        self.model.train()
        return val_losses

    def save_checkpoint(self, model, optimizer, global_step, epoch, config):
        checkpoint = {
            'vq_vae': model.module.state_dict() if isinstance(model, DDP) else model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'global_step': global_step,
            'epoch': epoch,
            'config': config
        }
        output_file = os.path.join(self.config.trainer.output_dir, f'checkpoint_{global_step:06d}.pth')
        torch.save(checkpoint, output_file)
        logging.info(f"Saved checkpoint to {output_file}")

    def destroy(self):
        if self.writer is not None:
            self.writer.close()
        if self.distributed:
            dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()
    config = OmegaConf.load(args.config)
    trainer = VQVAETrainer(config)
    trainer.train()
    trainer.destroy()
