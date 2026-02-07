#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
"""
Trainer for the DAC_FLOW_VAE model.

This script trains the DAC_FLOW_VAE model which encodes audio with flow-based VAE.

Usage:
    python minimaxspeech/trainers/flow_vae_trainer.py \
        --config configs/flow_vae_config.yaml

    # Multi-GPU with DDP
    export CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7'
    torchrun --nproc_per_node=8 minimaxspeech/trainers/flow_vae_trainer.py \
        --config configs/flow_vae_config.yaml
"""
import argparse
import logging
import os
from collections import defaultdict

import torch
import torch.distributed as dist
from audiotools import AudioSignal
from omegaconf import OmegaConf
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from minimaxspeech.datasets.vae_dataset import VAEDataset
from minimaxspeech.modules.flow_vae.discriminator import Discriminator
from minimaxspeech.modules.flow_vae.flow_vae import DAC_FLOW_VAE
from minimaxspeech.modules.flow_vae.loss import L1Loss, MultiScaleSTFTLoss, MelSpectrogramLoss, GANLoss
from minimaxspeech.utils.commons.logger import setup_logger


class FlowVAETrainer:
    def __init__(self, config: dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.setup_distributed()
        setup_logger(os.path.join(self.config.trainer.output_dir, 'logs'))
        if self.local_rank == 0:
            self.writer = SummaryWriter(os.path.join(self.config.trainer.output_dir, 'tensorboard'))

        self.setup_model()
        self.load_checkpoint()
        self.setup_losses()
        self.setup_optimizer()
        self.setup_scheduler()
        self.setup_dataset()

        self.start_epoch = 0
        self.global_step = 0
        self.best_loss = float('inf')

    def setup_distributed(self):
        # Setup distributed training
        self.distributed = int(os.environ.get('WORLD_SIZE', 1)) > 1
        if self.distributed:
            dist.init_process_group(backend='nccl')
            self.local_rank = int(os.environ.get('LOCAL_RANK', 0))
            torch.cuda.set_device(self.local_rank)
            logging.info(f"Distributed training enabled")
        else:
            self.local_rank = 0
            logging.info(f"Single card training")

    def setup_model(self):
        self.flow_vae = DAC_FLOW_VAE(**self.config.model.flow_vae).to(self.device)
        logging.info(f"DAC_FLOW_VAE model initialized and ready to train")

        # DAC(https://arxiv.org/abs/2306.06546) shows that discriminator is important for audio reconstruction
        self.discriminator = Discriminator(**self.config.model.discriminator).to(self.device)
        logging.info(f"Discriminator model initialized and ready to train")

    def setup_losses(self):
        self.waveform_loss = L1Loss()
        self.stft_loss = MultiScaleSTFTLoss(**self.config.loss.stft_loss)
        self.mel_loss = MelSpectrogramLoss(**self.config.loss.mel_loss)
        self.gan_loss = GANLoss(self.discriminator)


    def load_checkpoint(self):
        # Load pretrained model if available
        if self.config.trainer.resume:
            logging.info(f"Loading checkpoint from {self.config.trainer.checkpoint}")
            checkpoint = torch.load(self.config.trainer.checkpoint, map_location='cpu')
            
            self.flow_vae.load_state_dict(checkpoint['flow_vae'])
            self.discriminator.load_state_dict(checkpoint['discriminator'])
            self.start_epoch = checkpoint['epoch'] + 1
            self.global_step = checkpoint['global_step']
            
            logging.info(f"Resumed: start_epoch={self.start_epoch}, global_step={self.global_step}")

        # Wrap models with DDP if distributed
        if self.distributed:
            self.flow_vae = DDP(self.flow_vae, device_ids=[self.local_rank], find_unused_parameters=True)
            self.discriminator = DDP(self.discriminator, device_ids=[self.local_rank], find_unused_parameters=True)

    def setup_optimizer(self):
        self.optimizer_g = torch.optim.AdamW(
            self.flow_vae.parameters(),
            lr=self.config.trainer.learning_rate,
            betas=(0.9, 0.999),
            weight_decay=self.config.trainer.weight_decay
        )
        self.optimizer_d = torch.optim.AdamW(
            self.discriminator.parameters(),
            lr=self.config.trainer.learning_rate,
            betas=(0.9, 0.999),
            weight_decay=self.config.trainer.weight_decay
        )

    def setup_scheduler(self):
        self.scheduler_g = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer_g,
            T_max=self.config.trainer.epochs,
            eta_min=self.config.trainer.min_lr
        )
        self.scheduler_d = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer_d,
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
            drop_last=True,
            collate_fn=self.train_dataset.collate_fn
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config.dataset.batch_size,
            shuffle=False,
            sampler=self.val_sampler,
            num_workers=self.config.dataset.num_workers,
            pin_memory=True,
            drop_last=False,
            collate_fn=self.val_dataset.collate_fn
        )

    def train_step(self, batch):
        """
        Single training step.
        Reference: https://github.com/descriptinc/descript-audio-codec/blob/c7cfc5d2647e26471dc394f95846a0830e7bec34/scripts/train.py#L226
        """
        audios, filenames, lengths = batch
        audios = audios.to(self.device)
        
        """Discriminator"""
        # Forward pass: DAC_FLOW_VAE returns (reconstructed_audio, flow_kl_loss)
        audios = AudioSignal(audios, self.config.dataset.sample_rate)
        audios_hat, loss_kl = self.flow_vae(audios.audio_data, lengths)

        audios_hat = AudioSignal(audios_hat, self.config.dataset.sample_rate)
        assert audios.audio_data.size(-1) == audios_hat.audio_data.size(-1), f"{audios.shape}, {audios_hat.shape}"

        loss_disc = self.gan_loss.discriminator_loss(audios_hat, audios)
        self.optimizer_d.zero_grad()
        loss_disc.backward()
        torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), self.config.trainer.grad_clip)
        self.optimizer_d.step()

        """Generator"""
        for p in self.discriminator.parameters():
            p.requires_grad = False
        loss_stft = self.stft_loss(audios, audios_hat)
        loss_mel = self.mel_loss(audios, audios_hat)
        loss_waveform = self.waveform_loss(audios, audios_hat)
        loss_gen, loss_feat = self.gan_loss.generator_loss(audios, audios_hat)
        loss_weights = self.config.loss.loss_weights
        total_loss = (loss_mel * loss_weights.mel_loss + 
                      loss_gen * loss_weights.gen_loss + 
                      loss_feat * loss_weights.feat_loss + 
                      loss_kl * loss_weights.kl_loss)
        self.optimizer_g.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.flow_vae.parameters(), self.config.trainer.grad_clip)
        self.optimizer_g.step()
        for p in self.discriminator.parameters():
            p.requires_grad = True

        return {
            'total_loss': total_loss.item(),
            'disc_loss': loss_disc.item(),
            'stft_loss': loss_stft.item(),
            'mel_loss': loss_mel.item(),
            'waveform_loss': loss_waveform.item(),
            'gen_loss': loss_gen.item(),
            'feat_loss': loss_feat.item(),
            'kl_loss': loss_kl.item()
        }

    def train(self):
        """Training loop"""
        logging.info(f"Start training: epoch={self.start_epoch}, global_step={self.global_step}")
        for epoch in range(self.start_epoch, self.config.trainer.epochs):
            self.current_epoch = epoch
            if self.distributed:
                self.train_sampler.set_epoch(epoch)
            
            self.flow_vae.train()
            local_step = 0
            for batch_idx, batch in enumerate(self.train_loader):
                train_losses = self.train_step(batch)
                local_step += 1
                self.global_step += 1

                # Log training loss
                if self.global_step % self.config.trainer.log_interval == 0 and self.local_rank == 0:
                    lr = float(self.optimizer_g.param_groups[0]["lr"])
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
                
                # Save checkpoint
                if self.global_step % self.config.trainer.save_interval == 0 and self.local_rank == 0:
                    self.save_checkpoint(
                        self.flow_vae, self.optimizer_g, self.scheduler_g,
                        self.discriminator, self.optimizer_d, self.scheduler_d,
                        self.global_step, epoch, self.config
                    )

            # Update scheduler
            self.scheduler_g.step()
            self.scheduler_d.step()

    def validate(self):
        """Validation loop."""
        self.flow_vae.eval()
        val_losses = defaultdict(float)
        valid_step = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                audios, filenames, lengths = batch
                audios = audios.to(self.device)

                # Forward pass
                audios = AudioSignal(audios, self.config.dataset.sample_rate)
                audios_hat, loss_kl = self.flow_vae(audios.audio_data, lengths)

                audios_hat = AudioSignal(audios_hat, self.config.dataset.sample_rate)

                val_losses['kl_loss'] += loss_kl.item()
                val_losses['stft_loss'] += self.stft_loss(audios, audios_hat).item()
                val_losses['mel_loss'] += self.mel_loss(audios, audios_hat).item()
                val_losses['waveform_loss'] += self.waveform_loss(audios, audios_hat).item()
                valid_step += 1
        
        for key in val_losses:
            val_losses[key] /= valid_step
        self.flow_vae.train()
        return val_losses

    def save_checkpoint(
            self, flow_vae, optimizer_g, scheduler_g,
            discriminator, optimizer_d, scheduler_d, 
            global_step, epoch, config
        ):
        checkpoint = {
            'flow_vae': flow_vae.module.state_dict() if isinstance(flow_vae, DDP) else flow_vae.state_dict(),
            'optimizer_g': optimizer_g.state_dict(),
            'scheduler_g': scheduler_g.state_dict(),
            'discriminator': discriminator.module.state_dict() \
                if isinstance(discriminator, DDP) else discriminator.state_dict(),
            'optimizer_d': optimizer_d.state_dict(),
            'scheduler_d': scheduler_d.state_dict(),
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
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    args = parser.parse_args()
    
    config = OmegaConf.load(args.config)
    
    os.makedirs(config.trainer.output_dir, exist_ok=True)
    
    trainer = FlowVAETrainer(config)
    trainer.train()
    trainer.destroy()
