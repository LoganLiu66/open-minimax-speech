#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
"""
Trainer for UnifiedVoice (GPT2-style AR model).

Usage:
    python minimaxspeech/trainers/gpt_trainer.py \
        --config configs/gpt_config_libritts.yaml

DDP:
    export CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7'
    torchrun --nproc_per_node=8 minimaxspeech/trainers/gpt_trainer.py \
        --config configs/gpt_config_libritts.yaml

Reference: 
- https://github.com/anhnh2002/XTTSv2-Finetuning-for-New-Languages/blob/main/TTS/tts/layers/xtts/trainer/gpt_trainer.py#L69
- https://github.com/anhnh2002/XTTSv2-Finetuning-for-New-Languages/blob/main/recipes/ljspeech/xtts_v2/train_gpt_xtts.py
- https://github.com/anhnh2002/XTTSv2-Finetuning-for-New-Languages/blob/main/train_gpt_xtts.py
- https://github.com/neonbjb/tortoise-tts/blob/main/tortoise/models/autoregressive.py
- https://github.com/JarodMica/index-tts/blob/main/trainers/train_gpt_v2.py
- https://github.com/coqui-ai/TTS/blob/dev/TTS/tts/layers/xtts/gpt.py
- https://github.com/coqui-ai/TTS/blob/dev/TTS/tts/layers/xtts/trainer/gpt_trainer.py
"""

import argparse
import logging
import os
from typing import Optional

import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter

from minimaxspeech.datasets.gpt_dataset import GPTDataset
from minimaxspeech.modules.gpt.gpt import GPT
from minimaxspeech.modules.vq_vae.dvae import DiscreteVAE, TorchMelSpectrogram
from minimaxspeech.utils.commons.logger import setup_logger


class GPT2Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.setup_distributed()

        setup_logger(os.path.join(self.config.trainer.output_dir, "logs"))
        self.writer: Optional[SummaryWriter] = None
        if self.local_rank == 0:
            self.writer = SummaryWriter(os.path.join(self.config.trainer.output_dir, "tensorboard"))

        self.start_epoch = 0
        self.current_epoch = 0
        self.global_step = 0
        self.best_loss = float("inf")

        self.setup_models()
        self.setup_optimizer()
        self.setup_scheduler()
        self.setup_dataset()
        self.load_checkpoint()

    def setup_distributed(self):
        self.distributed = int(os.environ.get("WORLD_SIZE", 1)) > 1
        if self.distributed:
            dist.init_process_group(backend="nccl")
            self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
            torch.cuda.set_device(self.local_rank)
            logging.info("Distributed training enabled.")
        else:
            self.local_rank = 0
            logging.info("Single GPU/CPU training.")

    def setup_models(self):
        self.mel_extractor = TorchMelSpectrogram(
            mel_norm_file=self.config.model.vq_vae.mel_norm_file,
            sampling_rate=self.config.dataset.sample_rate,
        ).to(self.device)

        # VQVAE tokenizer (frozen)
        self.vq_vae = DiscreteVAE(**self.config.model.vq_vae.model).to(self.device)
        
        # GPT https://github.com/coqui-ai/TTS/blob/dev/TTS/tts/layers/xtts/trainer/gpt_trainer.py#L91
        self.model = GPT(**self.config.model.gpt).to(self.device)

    def setup_optimizer(self):
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.trainer.learning_rate,
            betas=self.config.trainer.betas,
            weight_decay=self.config.trainer.weight_decay
        )

    def setup_scheduler(self):
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.config.trainer.epochs,
            eta_min=self.config.trainer.min_lr
        )

    def setup_dataset(self):
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
            self.train_sampler = DistributedSampler(self.train_dataset, shuffle=True)
            self.val_sampler = DistributedSampler(self.val_dataset, shuffle=False)
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
            collate_fn=GPTDataset.collate_fn,
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config.dataset.batch_size,
            shuffle=False,
            sampler=self.val_sampler,
            num_workers=self.config.dataset.num_workers,
            pin_memory=True,
            drop_last=False,
            collate_fn=GPTDataset.collate_fn,
        )

    def load_checkpoint(self):
        logging.info(f"Initializing VQVAE from checkpoint {self.config.model.vq_vae.checkpoint}")
        ckpt = torch.load(self.config.model.vq_vae.checkpoint, map_location="cpu")
        if isinstance(ckpt, dict) and "model" in ckpt:
            self.vq_vae.load_state_dict(ckpt["model"], strict=True)
        else:
            self.vq_vae.load_state_dict(ckpt, strict=False)
        self.vq_vae.eval()
        for p in self.vq_vae.parameters():
            p.requires_grad = False
        logging.info("VQVAE initialized.")

        # Load pretrained decoder if available
        if self.config.trainer.resume:
            logging.info(f"Loading GPT from checkpoint {self.config.trainer.checkpoint}")
            checkpoint = torch.load(self.config.trainer.checkpoint, map_location='cpu')
            
            if 'gpt' in checkpoint: # self-host checkpoint
                self.model.load_state_dict(checkpoint['gpt'])
                self.start_epoch = checkpoint['epoch'] + 1
                self.global_step = checkpoint['global_step']
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                self.scheduler.load_state_dict(checkpoint['scheduler'])
            else: # from xtts2 checkpoint
                new_dict = {k[4:]: v for k, v in checkpoint['model'].items() if k.startswith('gpt')}
                self.model.load_state_dict(new_dict)
                self.start_epoch = 0
                self.global_step = 0
            
            logging.info(f"Resumed: start_epoch={self.start_epoch}, global_step={self.global_step}")

        # Wrap models with DDP if distributed
        if self.distributed:
            self.model = DDP(self.model, device_ids=[self.local_rank])

    @torch.no_grad()
    def _compute_mels(self, waveforms: torch.Tensor) -> torch.Tensor:
        # waveforms: (B, 1, T) or (B, T)
        mels = self.mel_extractor(waveforms)  # (B, 80, S)
        remainder = int(mels.shape[-1] % 4)
        if remainder:
            mels = mels[:, :, :-remainder]
        return mels

    @torch.no_grad()
    def _compute_mel_codes(self, mels: torch.Tensor) -> torch.LongTensor:
        # VQVAE expects (B, 80, S)
        return self.vq_vae.get_codebook_indices(mels)

    @torch.no_grad()
    def _compute_conditioning_latent(self, mels: torch.Tensor) -> torch.Tensor:
        model = self.model.module if isinstance(self.model, DDP) else self.model
        return model.get_conditioning(mels)

    def train_step(self, batch):
        wavs = batch["waveform"].to(self.device)
        wav_lengths = batch["wav_lengths"].to(self.device)
        text_tokens = batch["text_tokens"].to(self.device)
        text_lengths = batch["text_lengths"].to(self.device)
        conds = batch["cond"].to(self.device)
        if batch["cond_lens"] is not None:
            cond_lens = batch["cond_lens"].to(self.device)
        else:
            cond_lens = None
        if batch["cond_idxs"] is not None:
            cond_idxs = batch["cond_idxs"].to(self.device)
        else:
            cond_idxs = None

        with torch.no_grad():
            mels = self._compute_mels(wavs)  # (B, 80, S), wavs.shape[-1] // 256 // 4 * 4 = mels.shape[-1]
            mel_codes = self._compute_mel_codes(mels)  # (B, M), mels.shape[-1] // 4 = mel_codes.shape[1]
            # cond_latent = self._compute_conditioning_latent(mels)  # (B, D)
            
            cond_mels = self._compute_mels(conds)  # (B, 80, S)
            # ref_cond_latent = self._compute_conditioning_latent(ref_mels)  # (B, D)

        loss_text, loss_mel, _ = self.model(
            text_inputs=text_tokens,
            text_lengths=text_lengths,
            audio_codes=mel_codes,
            wav_lengths=wav_lengths,
            cond_mels=cond_mels,
            cond_idxs=cond_idxs,
            cond_lens=cond_lens,
            return_attentions=False,
            return_latent=False
        )
        text_w = self.config.trainer.text_loss_weight
        mel_w = self.config.trainer.mel_loss_weight
        total_loss = loss_text * text_w + loss_mel * mel_w

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return {
            "loss_text": float(loss_text.item()),
            "loss_mel": float(loss_mel.item()),
            "total_loss": float(total_loss.item()),
        }

    @torch.no_grad()
    def validate(self) -> dict:
        self.model.eval()
        val_losses = {'total_loss': 0, 'loss_text': 0, 'loss_mel': 0}
        valid_step = 0

        for batch in self.val_loader:
            wavs = batch["waveform"].to(self.device)
            wav_lengths = batch["wav_lengths"].to(self.device)
            text_tokens = batch["text_tokens"].to(self.device)
            text_lengths = batch["text_lengths"].to(self.device)
            conds = batch["cond"].to(self.device)
            if batch["cond_lens"] is not None:
                cond_lens = batch["cond_lens"].to(self.device)
            else:
                cond_lens = None
            if batch["cond_idxs"] is not None:
                cond_idxs = batch["cond_idxs"].to(self.device)
            else:
                cond_idxs = None

            mels = self._compute_mels(wavs)  # (B, 80, S)
            mel_codes = self._compute_mel_codes(mels)  # (B, M)
            cond_mels = self._compute_mels(conds)  # (B, 80, S)

            loss_text, loss_mel, _ = self.model(
                text_inputs=text_tokens,
                text_lengths=text_lengths,
                audio_codes=mel_codes,
                wav_lengths=wav_lengths,
                cond_mels=cond_mels,
                cond_idxs=cond_idxs,
                cond_lens=cond_lens,
                return_attentions=False,
                return_latent=False
            )
            text_w = self.config.trainer.text_loss_weight
            mel_w = self.config.trainer.mel_loss_weight
            loss = loss_text * text_w + loss_mel * mel_w

            val_losses['total_loss'] += loss.item()
            val_losses['loss_text'] += loss_text.item()
            val_losses['loss_mel'] += loss_mel.item()
            valid_step += 1

        val_losses['total_loss'] /= valid_step
        val_losses['loss_text'] /= valid_step
        val_losses['loss_mel'] /= valid_step
        self.model.train()
        return val_losses

    def save_checkpoint(self, model, optimizer, global_step, epoch, config):
        checkpoint = {
            'gpt': model.module.state_dict() if isinstance(model, DDP) else model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'global_step': global_step,
            'epoch': epoch,
            'config': config
        }
        output_file = os.path.join(self.config.trainer.output_dir, f'checkpoint_{global_step:06d}.pth')
        torch.save(checkpoint, output_file)
        logging.info(f"Saved checkpoint to {output_file}")

    def train(self):
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
                    logging.info(f"Epoch: {epoch}, Global Step: {self.global_step}, LR: {lr:.2e}, Train Losses: {train_losses}")
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

            self.scheduler.step()

    def destroy(self):
        if self.writer is not None:
            self.writer.close()
        if self.distributed:
            dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)
    trainer = GPT2Trainer(cfg)
    trainer.train()
    trainer.destroy()
