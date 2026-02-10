#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
"""
Reference:
- https://github.com/FireRedTeam/FireRedTTS/blob/fireredtts-1s/fireredtts/modules/flowmatching/__init__.py
- https://github.com/zai-org/GLM-TTS/blob/main/flow/flow.py
- https://github.com/FunAudioLLM/CosyVoice/blob/main/cosyvoice/flow/flow_matching.py
- https://github.com/shivammehta25/Matcha-TTS/blob/main/matcha/models/components/flow_matching.py
"""
import logging
import random
from abc import ABC
from typing import Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig

from minimaxspeech.modules.common.mask import make_pad_mask
from minimaxspeech.modules.common.transformer.encoder import ConformerEncoder
from minimaxspeech.modules.flow_matching.estimator import ConditionalDecoder
from minimaxspeech.modules.flow_matching.length_regulator import InterpolateRegulator


class BASECFM(torch.nn.Module, ABC):
    """
    Reference: https://github.com/shivammehta25/Matcha-TTS/blob/main/matcha/models/components/flow_matching.py
    """
    def __init__(
        self,
        n_feats,
        cfm_params,
        n_spks=1,
        spk_emb_dim=128,
    ):
        super().__init__()
        self.n_feats = n_feats
        self.n_spks = n_spks
        self.spk_emb_dim = spk_emb_dim
        self.solver = cfm_params.solver
        if hasattr(cfm_params, "sigma_min"):
            self.sigma_min = cfm_params.sigma_min
        else:
            self.sigma_min = 1e-4

        self.estimator = None

    @torch.inference_mode()
    def forward(self, mu, mask, n_timesteps, temperature=1.0, spks=None, cond=None):
        """Forward diffusion

        Args:
            mu (torch.Tensor): output of encoder
                shape: (batch_size, n_feats, mel_timesteps)
            mask (torch.Tensor): output_mask
                shape: (batch_size, 1, mel_timesteps)
            n_timesteps (int): number of diffusion steps
            temperature (float, optional): temperature for scaling noise. Defaults to 1.0.
            spks (torch.Tensor, optional): speaker ids. Defaults to None.
                shape: (batch_size, spk_emb_dim)
            cond: Not used but kept for future purposes

        Returns:
            sample: generated mel-spectrogram
                shape: (batch_size, n_feats, mel_timesteps)
        """
        z = torch.randn_like(mu) * temperature
        t_span = torch.linspace(0, 1, n_timesteps + 1, device=mu.device)
        return self.solve_euler(z, t_span=t_span, mu=mu, mask=mask, spks=spks, cond=cond)

    def solve_euler(self, x, t_span, mu, mask, spks, cond):
        """
        Fixed euler solver for ODEs.
        Args:
            x (torch.Tensor): random noise
            t_span (torch.Tensor): n_timesteps interpolated
                shape: (n_timesteps + 1,)
            mu (torch.Tensor): output of encoder
                shape: (batch_size, n_feats, mel_timesteps)
            mask (torch.Tensor): output_mask
                shape: (batch_size, 1, mel_timesteps)
            spks (torch.Tensor, optional): speaker ids. Defaults to None.
                shape: (batch_size, spk_emb_dim)
            cond: Not used but kept for future purposes
        """
        t, _, dt = t_span[0], t_span[-1], t_span[1] - t_span[0]

        # I am storing this because I can later plot it by putting a debugger here and saving it to a file
        # Or in future might add like a return_all_steps flag
        sol = []

        for step in range(1, len(t_span)):
            dphi_dt = self.estimator(x, mask, mu, t, spks, cond)

            x = x + dt * dphi_dt
            t = t + dt
            sol.append(x)
            if step < len(t_span) - 1:
                dt = t_span[step + 1] - t

        return sol[-1]

    def compute_loss(self, x1, mask, mu, spks=None, cond=None):
        """Computes diffusion loss

        Args:
            x1 (torch.Tensor): Target
                shape: (batch_size, n_feats, mel_timesteps)
            mask (torch.Tensor): target mask
                shape: (batch_size, 1, mel_timesteps)
            mu (torch.Tensor): output of encoder
                shape: (batch_size, n_feats, mel_timesteps)
            spks (torch.Tensor, optional): speaker embedding. Defaults to None.
                shape: (batch_size, spk_emb_dim)

        Returns:
            loss: conditional flow matching loss
            y: conditional flow
                shape: (batch_size, n_feats, mel_timesteps)
        """
        b, _, t = mu.shape

        # random timestep
        t = torch.rand([b, 1, 1], device=mu.device, dtype=mu.dtype)
        # sample noise p(x_0)
        z = torch.randn_like(x1)

        y = (1 - (1 - self.sigma_min) * t) * z + t * x1
        u = x1 - (1 - self.sigma_min) * z

        loss = F.mse_loss(self.estimator(y, mask, mu, t.squeeze(), spks), u, reduction="sum") / (
            torch.sum(mask) * u.shape[1]
        )
        return loss, y


class ConditionalCFM(BASECFM):
    """
    Reference: https://github.com/FunAudioLLM/CosyVoice/blob/main/cosyvoice/flow/flow_matching.py
    """
    def __init__(self, in_channels, cfm_params, estimator, n_spks=1, spk_emb_dim=64):
        super().__init__(
            n_feats=in_channels,
            cfm_params=cfm_params,
            n_spks=n_spks,
            spk_emb_dim=spk_emb_dim,
        )
        self.t_scheduler = cfm_params.t_scheduler
        self.training_cfg_rate = cfm_params.training_cfg_rate
        self.inference_cfg_rate = cfm_params.inference_cfg_rate
        in_channels = in_channels + (spk_emb_dim if n_spks > 0 else 0)
        # Just change the architecture of the estimator here
        self.estimator = ConditionalDecoder(**estimator)
        
    @torch.inference_mode()
    def forward(self, mu, mask, n_timesteps, temperature=1.0, spks=None, cond=None, prompt_len=0, cache=torch.zeros(1, 80, 0, 2)):
        """Forward diffusion

        Args:
            mu (torch.Tensor): output of encoder
                shape: (batch_size, n_feats, mel_timesteps)
            mask (torch.Tensor): output_mask
                shape: (batch_size, 1, mel_timesteps)
            n_timesteps (int): number of diffusion steps
            temperature (float, optional): temperature for scaling noise. Defaults to 1.0.
            spks (torch.Tensor, optional): speaker ids. Defaults to None.
                shape: (batch_size, spk_emb_dim)
            cond: Not used but kept for future purposes

        Returns:
            sample: generated mel-spectrogram
                shape: (batch_size, n_feats, mel_timesteps)
        """

        z = torch.randn_like(mu).to(mu.device).to(mu.dtype) * temperature
        cache_size = cache.shape[2]
        # fix prompt and overlap part mu and z
        if cache_size != 0:
            z[:, :, :cache_size] = cache[:, :, :, 0]
            mu[:, :, :cache_size] = cache[:, :, :, 1]
        z_cache = torch.concat([z[:, :, :prompt_len], z[:, :, -34:]], dim=2)
        mu_cache = torch.concat([mu[:, :, :prompt_len], mu[:, :, -34:]], dim=2)
        cache = torch.stack([z_cache, mu_cache], dim=-1)

        t_span = torch.linspace(0, 1, n_timesteps + 1, device=mu.device, dtype=mu.dtype)
        if self.t_scheduler == 'cosine':
            t_span = 1 - torch.cos(t_span * 0.5 * torch.pi)
        return self.solve_euler(z, t_span=t_span, mu=mu, mask=mask, spks=spks, cond=cond), cache

    def solve_euler(self, x, t_span, mu, mask, spks, cond, streaming=False):
        """
        Fixed euler solver for ODEs.
        Args:
            x (torch.Tensor): random noise
            t_span (torch.Tensor): n_timesteps interpolated
                shape: (n_timesteps + 1,)
            mu (torch.Tensor): output of encoder
                shape: (batch_size, n_feats, mel_timesteps)
            mask (torch.Tensor): output_mask
                shape: (batch_size, 1, mel_timesteps)
            spks (torch.Tensor, optional): speaker ids. Defaults to None.
                shape: (batch_size, spk_emb_dim)
            cond: Not used but kept for future purposes
        """
        t, _, dt = t_span[0], t_span[-1], t_span[1] - t_span[0]
        t = t.unsqueeze(dim=0)

        # I am storing this because I can later plot it by putting a debugger here and saving it to a file
        # Or in future might add like a return_all_steps flag
        sol = []

        # Do not use concat, it may cause memory format changed and trt infer with wrong results!
        # NOTE when flow run in amp mode, x.dtype is float32, which cause nan in trt fp16 inference, so set dtype=spks.dtype
        x_in = torch.zeros([2, 80, x.size(2)], device=x.device, dtype=spks.dtype)
        mask_in = torch.zeros([2, 1, x.size(2)], device=x.device, dtype=spks.dtype)
        mu_in = torch.zeros([2, 80, x.size(2)], device=x.device, dtype=spks.dtype)
        t_in = torch.zeros([2], device=x.device, dtype=spks.dtype)
        spks_in = torch.zeros([2, 80], device=x.device, dtype=spks.dtype)
        cond_in = torch.zeros([2, 80, x.size(2)], device=x.device, dtype=spks.dtype)
        for step in range(1, len(t_span)):
            # Classifier-Free Guidance inference introduced in VoiceBox
            x_in[:] = x
            mask_in[:] = mask
            mu_in[0] = mu
            t_in[:] = t.unsqueeze(0)
            spks_in[0] = spks
            cond_in[0] = cond
            dphi_dt = self.forward_estimator(
                x_in, mask_in,
                mu_in, t_in,
                spks_in,
                cond_in,
                streaming
            )
            dphi_dt, cfg_dphi_dt = torch.split(dphi_dt, [x.size(0), x.size(0)], dim=0)
            dphi_dt = ((1.0 + self.inference_cfg_rate) * dphi_dt - self.inference_cfg_rate * cfg_dphi_dt)
            x = x + dt * dphi_dt
            t = t + dt
            sol.append(x)
            if step < len(t_span) - 1:
                dt = t_span[step + 1] - t

        return sol[-1].float()

    def forward_estimator(self, x, mask, mu, t, spks, cond, streaming=False):
        if isinstance(self.estimator, torch.nn.Module):
            return self.estimator(x, mask, mu, t, spks, cond, streaming=streaming)
        else:
            [estimator, stream], trt_engine = self.estimator.acquire_estimator()
            # NOTE need to synchronize when switching stream
            torch.cuda.current_stream().synchronize()
            with stream:
                estimator.set_input_shape('x', (2, 80, x.size(2)))
                estimator.set_input_shape('mask', (2, 1, x.size(2)))
                estimator.set_input_shape('mu', (2, 80, x.size(2)))
                estimator.set_input_shape('t', (2,))
                estimator.set_input_shape('spks', (2, 80))
                estimator.set_input_shape('cond', (2, 80, x.size(2)))
                data_ptrs = [x.contiguous().data_ptr(),
                             mask.contiguous().data_ptr(),
                             mu.contiguous().data_ptr(),
                             t.contiguous().data_ptr(),
                             spks.contiguous().data_ptr(),
                             cond.contiguous().data_ptr(),
                             x.data_ptr()]
                for i, j in enumerate(data_ptrs):
                    estimator.set_tensor_address(trt_engine.get_tensor_name(i), j)
                # run trt engine
                assert estimator.execute_async_v3(torch.cuda.current_stream().cuda_stream) is True
                torch.cuda.current_stream().synchronize()
            self.estimator.release_estimator(estimator, stream)
            return x

    def compute_loss(self, x1, mask, mu, spks=None, cond=None, streaming=False):
        """Computes diffusion loss

        Args:
            x1 (torch.Tensor): Target
                shape: (batch_size, n_feats, mel_timesteps)
            mask (torch.Tensor): target mask
                shape: (batch_size, 1, mel_timesteps)
            mu (torch.Tensor): output of encoder
                shape: (batch_size, n_feats, mel_timesteps)
            spks (torch.Tensor, optional): speaker embedding. Defaults to None.
                shape: (batch_size, spk_emb_dim)

        Returns:
            loss: conditional flow matching loss
            y: conditional flow
                shape: (batch_size, n_feats, mel_timesteps)
        """
        b, _, t = mu.shape

        # random timestep
        t = torch.rand([b, 1, 1], device=mu.device, dtype=mu.dtype)
        if self.t_scheduler == 'cosine':
            t = 1 - torch.cos(t * 0.5 * torch.pi)
        # sample noise p(x_0)
        z = torch.randn_like(x1)

        y = (1 - (1 - self.sigma_min) * t) * z + t * x1
        u = x1 - (1 - self.sigma_min) * z

        # during training, we randomly drop condition to trade off mode coverage and sample fidelity
        if self.training_cfg_rate > 0:
            cfg_mask = torch.rand(b, device=x1.device) > self.training_cfg_rate
            mu = mu * cfg_mask.view(-1, 1, 1)
            spks = spks * cfg_mask.view(-1, 1)
            cond = cond * cfg_mask.view(-1, 1, 1)

        pred = self.estimator(y, mask, mu, t.squeeze(), spks, cond, streaming=streaming)
        loss = F.mse_loss(pred * mask, u * mask, reduction="sum") / (torch.sum(mask) * u.shape[1])
        return loss, y


class MaskedDiffWithXvec(torch.nn.Module):
    """
    Reference: https://github.com/FunAudioLLM/CosyVoice/blob/f08872a82fb18b1632c99f4e1d177da08c33d137/cosyvoice/flow/flow.py#L25
    """
    def __init__(
        self,
        encoder: DictConfig,
        length_regulator: DictConfig,
        decoder: DictConfig,
        input_size: int = 512,
        output_size: int = 256,
        spk_embed_dim: int = 192,
        vocab_size: int = 4096,
        input_frame_rate: int = 50,
        only_mask_loss: bool = True,
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.vocab_size = vocab_size
        self.input_frame_rate = input_frame_rate
        logging.info(f"input frame rate={self.input_frame_rate}")
        self.input_embedding = nn.Embedding(vocab_size, input_size)
        self.spk_embed_affine_layer = torch.nn.Linear(spk_embed_dim, output_size)
        self.encoder = ConformerEncoder(**encoder)
        self.encoder_proj = torch.nn.Linear(self.encoder.output_size(), output_size)
        self.length_regulator = InterpolateRegulator(**length_regulator)
        self.decoder = ConditionalCFM(**decoder)
        self.only_mask_loss = only_mask_loss

    def forward(self, token, token_len, embedding, feat, feat_len) -> Dict[str, Optional[torch.Tensor]]:
        """Flow Matching

        Args:
            token (torch.Tensor):
                shape: (batch_size, n_feats, mel_timesteps)
            token_len (torch.Tensor):
                shape: (batch_size,)
            embedding (torch.Tensor, optional):
                shape: (batch_size, spk_emb_dim)
            feat (torch.Tensor): Target
                shape: (batch_size, mel_timesteps, n_feats)
        """
        # xvec projection
        embedding = F.normalize(embedding, dim=1)
        embedding = self.spk_embed_affine_layer(embedding) # (b, spk_embed_dim) -> (b, output_size)

        # concat text and prompt_text
        mask = (~make_pad_mask(token_len, max_len=token.size(-1))).float().unsqueeze(-1).to(token.device)
        token = self.input_embedding(torch.clamp(token, min=0)) * mask # (b, t) -> (b, t, input_size)

        # text encode
        h, h_lengths = self.encoder(token, token_len)
        h = self.encoder_proj(h) # (b, t, input_size) -> (b, t, output_size)
        h, h_lengths = self.length_regulator(h, feat_len) # (b, t, output_size) -> (b, t', output_size)

        # get conditions
        # according to minimax paper: During the training process, information from the beginning of the current sentence is utilized as a prompt with a certain probability
        conds = torch.zeros(feat.shape, device=token.device)
        for i, j in enumerate(feat_len):
            if random.random() < 0.5:
                continue
            index = random.randint(0, int(0.3 * j))
            conds[i, :index] = feat[i, :index]
        conds = conds.transpose(1, 2) # (b, t', output_size) -> (b, output_size, t')

        mask = (~make_pad_mask(feat_len)).to(h)
        # NOTE this is unnecessary, feat/h already same shape
        loss, _ = self.decoder.compute_loss(
            feat.transpose(1, 2).contiguous(),
            mask.unsqueeze(1),
            h.transpose(1, 2).contiguous(),
            embedding,
            cond=conds
        )
        return loss

    @torch.inference_mode()
    def inference(
        self,
        token,
        token_len,
        prompt_token,
        prompt_token_len,
        prompt_feat,
        prompt_feat_len,
        embedding,
        flow_cache
    ):
        assert token.shape[0] == 1
        # xvec projection
        embedding = F.normalize(embedding, dim=1)
        embedding = self.spk_embed_affine_layer(embedding)

        # concat speech token and prompt speech token
        token_len1, token_len2 = prompt_token.shape[1], token.shape[1]
        token, token_len = torch.concat([prompt_token, token], dim=1), prompt_token_len + token_len
        mask = (~make_pad_mask(token_len)).unsqueeze(-1).to(embedding)
        token = self.input_embedding(torch.clamp(token, min=0)) * mask

        # text encode
        h, h_lengths = self.encoder(token, token_len)
        h = self.encoder_proj(h)
        mel_len1, mel_len2 = prompt_feat.shape[1], int(token_len2 / self.input_frame_rate * 22050 / 256)
        h, h_lengths = self.length_regulator.inference(h[:, :token_len1], h[:, token_len1:], mel_len1, mel_len2, self.input_frame_rate)

        # get conditions
        conds = torch.zeros([1, mel_len1 + mel_len2, self.output_size], device=token.device).to(h.dtype)
        conds[:, :mel_len1] = prompt_feat
        conds = conds.transpose(1, 2)

        mask = (~make_pad_mask(torch.tensor([mel_len1 + mel_len2]))).to(h)
        feat, flow_cache = self.decoder(
            mu=h.transpose(1, 2).contiguous(),
            mask=mask.unsqueeze(1),
            spks=embedding,
            cond=conds,
            n_timesteps=10,
            prompt_len=mel_len1,
            cache=flow_cache
        )
        feat = feat[:, :, mel_len1:]
        assert feat.shape[2] == mel_len2
        return feat.float(), flow_cache
    

if __name__ == "__main__":
    from omegaconf import OmegaConf
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = OmegaConf.load("./configs/flow_matching_config.yaml")
    model = MaskedDiffWithXvec(**config.flow_matching).to(device)
    
    batch_size = 3
    # input
    speech_token = torch.randint(0, 4096, (batch_size, 100)).to(device)
    speech_token_len = torch.randint(10, 100, (batch_size,)).to(device)
    speech_token_len[0] = 100
    embedding = torch.randn(batch_size, 1024).to(device)

    # target
    speech_feat = torch.randn(batch_size, 200, 256).to(device)
    speech_feat_len = speech_token_len * 2

    from torchinfo import summary
    summary(model, input_data=(speech_token, speech_token_len, embedding, speech_feat, speech_feat_len), depth=5)