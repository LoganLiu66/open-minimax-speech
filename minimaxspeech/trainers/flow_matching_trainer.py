#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
"""
Trainer for the ConditionalCFM (Conditional Flow Matching) model.

Usage:
    python minimaxspeech/trainers/flow_matching_trainer.py \
        --config configs/flow_matching_config.yaml

    # Multi-GPU with DDP
    export CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7'
    torchrun --nproc_per_node=8 minimaxspeech/trainers/flow_matching_trainer.py \
        --config configs/flow_matching_config.yaml
"""
import argparse
import logging
import math
import os

import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from minimaxspeech.datasets.gpt_dataset import GPTDataset
from minimaxspeech.modules.flow_matching.flow_matching import MaskedDiffWithXvec
from minimaxspeech.modules.flow_vae.flow_vae import DAC_FLOW_VAE
from minimaxspeech.modules.gpt.gpt import GPT
from minimaxspeech.modules.vq_vae.dvae import DiscreteVAE, TorchMelSpectrogram
from minimaxspeech.utils.commons.logger import setup_logger


class FlowMatchingTrainer:
    def __init__(self, config: dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.setup_distributed()
        setup_logger(os.path.join(self.config.trainer.output_dir, 'logs'))
        self.writer = SummaryWriter(os.path.join(self.config.trainer.output_dir, 'tensorboard'))

        self.start_epoch = 0
        self.global_step = 0
        self.best_loss = float('inf')

        self.setup_model()
        self.setup_optimizer()
        self.setup_scheduler()
        self.load_checkpoint()
        self.setup_dataset()

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
        self.mel_extractor = TorchMelSpectrogram(
            mel_norm_file=self.config.trainer.mel_norm_file,
            sampling_rate=self.config.dataset.sample_rate,
        ).to(self.device)

        """Setup ConditionalCFM model."""
        self.model = MaskedDiffWithXvec(**self.config.model.flow_matching).to(self.device)
        logging.info(f"ConditionalCFM model initialized and ready to train")

        self.vq_vae = DiscreteVAE(**self.config.model.vq_vae).to(self.device)
        logging.info(f"VQVAE model initialized")

        self.gpt = GPT(**self.config.model.gpt).to(self.device)
        logging.info(f"GPT model initialized")

        self.flow_vae = DAC_FLOW_VAE(**self.config.model.flow_vae).to(self.device)
        self.downsample_rate = self.flow_vae.downsample_rate  # store before potential DDP wrapping
        logging.info(f"FlowVAE model initialized")

    def load_checkpoint(self):
        self.vq_vae.load_state_dict(torch.load(self.config.trainer.vq_vae_checkpoint, map_location="cpu"))
        self.vq_vae.eval()

        checkpoint = torch.load(self.config.trainer.gpt_checkpoint, map_location="cpu")
        if 'gpt' in checkpoint: # self-host checkpoint
            self.gpt.load_state_dict(checkpoint['gpt'])
        else: # from xtts2 checkpoint
            new_dict = {k[4:]: v for k, v in checkpoint['model'].items() if k.startswith('gpt')}
            self.gpt.load_state_dict(new_dict)
        self.gpt.eval()

        self.flow_vae.load_state_dict(
            torch.load(self.config.trainer.flow_vae_checkpoint, map_location="cpu")['flow_vae']
        )
        self.flow_vae.eval()

        # Load pretrained model if available
        if self.config.trainer.resume:
            checkpoint_path = self.config.trainer.checkpoint
            logging.info(f"Loading checkpoint from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            self.model.load_state_dict(checkpoint['flow_matching'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.scheduler.load_state_dict(checkpoint['scheduler'])
            self.start_epoch = checkpoint['epoch'] + 1
            self.global_step = checkpoint['global_step']
            logging.info(f"Resumed: start_epoch={self.start_epoch}, global_step={self.global_step}")

        # Wrap only the trainable model with DDP if distributed
        # Frozen models (vq_vae, gpt, flow_vae) don't need DDP
        if self.distributed:
            self.model = DDP(self.model, device_ids=[self.local_rank])

    def setup_optimizer(self):
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.trainer.learning_rate,
            betas=(0.9, 0.999),
            weight_decay=self.config.trainer.weight_decay
        )

    def setup_scheduler(self):
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.config.trainer.epochs,
            eta_min=self.config.trainer.min_lr
        )

    def setup_dataset(self):
        # Setup dataset and dataloader
        self.train_dataset = GPTDataset(
            data_file_list=self.config.dataset.train_file_list,
            sample_rate=self.config.dataset.sample_rate,
            min_conditioning_duration=self.config.dataset.min_conditioning_duration,
            max_conditioning_duration=self.config.dataset.max_conditioning_duration,
            min_audio_duration=self.config.dataset.min_audio_duration,
            max_audio_duration=self.config.dataset.max_audio_duration,
            min_text_length=self.config.dataset.min_text_length,
            max_text_length=self.config.dataset.max_text_length,
            use_masking_gt_prompt_approach=self.config.dataset.use_masking_gt_prompt_approach,
            tokenizer_file=self.config.dataset.tokenizer_file,
            is_eval=False
        )
        self.val_dataset = GPTDataset(
            data_file_list=self.config.dataset.valid_file_list,
            sample_rate=self.config.dataset.sample_rate,
            min_conditioning_duration=self.config.dataset.min_conditioning_duration,
            max_conditioning_duration=self.config.dataset.max_conditioning_duration,
            min_audio_duration=self.config.dataset.min_audio_duration,
            max_audio_duration=self.config.dataset.max_audio_duration,
            min_text_length=self.config.dataset.min_text_length,
            max_text_length=self.config.dataset.max_text_length,
            use_masking_gt_prompt_approach=self.config.dataset.use_masking_gt_prompt_approach,
            tokenizer_file=self.config.dataset.tokenizer_file,
            is_eval=True
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
        """Single training step."""
        wavs = batch["waveform"].to(self.device)
        wav_lengths = batch["wav_lengths"].to(self.device)
        conds = batch["cond"].to(self.device)

        with torch.no_grad():
            mels = self.mel_extractor(wavs)
            token = self.vq_vae.get_codebook_indices(mels)
            token_len = torch.ceil((wav_lengths + 1) / 1024).long()

            cond_mels = self.mel_extractor(conds)
            embedding = self.gpt.get_style_emb(cond_mels)
            embedding = torch.mean(embedding, dim=2) # (b, d, 32) -> (b, d)

            feat = self.flow_vae.encode(wavs).transpose(1, 2) # (b, d, t) -> (b, t, d)
            feat_len = wav_lengths // self.downsample_rate
        
        # Forward pass: compute flow matching loss
        loss = self.model(
            token=token,
            token_len=token_len,
            embedding=embedding,
            feat=feat,
            feat_len=feat_len
        )

        self.optimizer.zero_grad()
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.trainer.grad_clip)
        self.optimizer.step()

        return {"total_loss": loss.item()}

    def train(self):
        """Training loop"""
        logging.info(f"Start training: epoch={self.start_epoch}, global_step={self.global_step}")
        for epoch in range(self.start_epoch, self.config.trainer.epochs):
            self.current_epoch = epoch
            if self.distributed:
                self.train_sampler.set_epoch(epoch)
            
            self.model.train()
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
                            self.save_checkpoint(
                                self.model, self.optimizer, self.scheduler,
                                self.global_step, epoch, self.config
                            )
                
                # Save checkpoint
                if self.global_step % self.config.trainer.save_interval == 0 and self.local_rank == 0:
                    self.save_checkpoint(
                        self.model, self.optimizer, self.scheduler,
                        self.global_step, epoch, self.config
                    )

            # Update scheduler
            self.scheduler.step()

    def validate(self):
        """Validation loop."""
        self.model.eval()
        val_losses = {'total_loss': 0}
        valid_step = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                wavs = batch["waveform"].to(self.device)
                wav_lengths = batch["wav_lengths"].to(self.device)
                conds = batch["cond"].to(self.device)

                with torch.no_grad():
                    mels = self.mel_extractor(wavs)
                    token = self.vq_vae.get_codebook_indices(mels)
                    token_len = torch.ceil((wav_lengths + 1) / 1024).long()

                    cond_mels = self.mel_extractor(conds)
                    embedding = self.gpt.get_style_emb(cond_mels)
                    embedding = torch.mean(embedding, dim=2) # (b, d, 32) -> (b, d)

                    feat = self.flow_vae.encode(wavs).transpose(1, 2) # (b, d, t) -> (b, t, d)
                    feat_len = wav_lengths // self.downsample_rate
                
                # Forward pass: compute flow matching loss
                loss = self.model(
                    token=token,
                    token_len=token_len,
                    embedding=embedding,
                    feat=feat,
                    feat_len=feat_len
                )
                
                val_losses['total_loss'] += loss.item()
                valid_step += 1
        
        val_losses['total_loss'] /= valid_step
        self.model.train()
        return val_losses

    def save_checkpoint(self, model, optimizer, scheduler, global_step, epoch, config):
        checkpoint = {
            'flow_matching': model.module.state_dict() if isinstance(model, DDP) else model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
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
    
    trainer = FlowMatchingTrainer(config)
    trainer.train()
    trainer.destroy()
