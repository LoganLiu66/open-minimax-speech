#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
"""
Reference:
- https://github.com/descriptinc/descript-audio-codec/blob/main/dac/model/dac.py
- https://github.com/facebookresearch/dacvae/blob/main/dacvae/model/dacvae.py
- https://github.com/fmu2/flow-VAE/blob/main/static_flow_vae.py
"""
import math
import os
from typing import List, Optional, Union

import numpy as np
import torch
from audiotools import AudioSignal
from audiotools.ml import BaseModel
from huggingface_hub import hf_hub_download
from torch import nn

from minimaxspeech.modules.flow_vae.bottleneck import VAEBottleneck
from minimaxspeech.modules.flow_vae.flow import ResidualCouplingBlock
from minimaxspeech.modules.flow_vae.layers import Snake1d, WNConv1d, WNConvTranspose1d
from minimaxspeech.modules.flow_vae.quantize import ResidualVectorQuantize
from minimaxspeech.modules.flow_vae.base import CodecMixin


def init_weights(m):
    if isinstance(m, nn.Conv1d):
        nn.init.trunc_normal_(m.weight, std=0.02)
        nn.init.constant_(m.bias, 0)


class ResidualUnit(nn.Module):
    def __init__(self, dim: int = 16, dilation: int = 1):
        super().__init__()
        pad = ((7 - 1) * dilation) // 2
        self.block = nn.Sequential(
            Snake1d(dim),
            WNConv1d(dim, dim, kernel_size=7, dilation=dilation, padding=pad),
            Snake1d(dim),
            WNConv1d(dim, dim, kernel_size=1),
        )

    def forward(self, x):
        y = self.block(x)
        pad = (x.shape[-1] - y.shape[-1]) // 2
        if pad > 0:
            x = x[..., pad:-pad]
        return x + y


class EncoderBlock(nn.Module):
    def __init__(self, dim: int = 16, stride: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            ResidualUnit(dim // 2, dilation=1),
            ResidualUnit(dim // 2, dilation=3),
            ResidualUnit(dim // 2, dilation=9),
            Snake1d(dim // 2),
            WNConv1d(
                dim // 2,
                dim,
                kernel_size=2 * stride,
                stride=stride,
                padding=math.ceil(stride / 2),
            ),
        )

    def forward(self, x):
        return self.block(x)


class Encoder(nn.Module):
    def __init__(
        self,
        d_model: int = 64,
        strides: list = [2, 4, 8, 8],
        d_latent: int = 64,
    ):
        super().__init__()
        # Create first convolution
        self.block = [WNConv1d(1, d_model, kernel_size=7, padding=3)]

        # Create EncoderBlocks that double channels as they downsample by `stride`
        for stride in strides:
            d_model *= 2
            self.block += [EncoderBlock(d_model, stride=stride)]

        # Create last convolution
        self.block += [
            Snake1d(d_model),
            WNConv1d(d_model, d_latent, kernel_size=3, padding=1),
        ]

        # Wrap black into nn.Sequential
        self.block = nn.Sequential(*self.block)
        self.enc_dim = d_model

    def forward(self, x):
        return self.block(x)


class DecoderBlock(nn.Module):
    def __init__(self, input_dim: int = 16, output_dim: int = 8, stride: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            Snake1d(input_dim),
            WNConvTranspose1d(
                input_dim,
                output_dim,
                kernel_size=2 * stride,
                stride=stride,
                padding=math.ceil(stride / 2),
            ),
            ResidualUnit(output_dim, dilation=1),
            ResidualUnit(output_dim, dilation=3),
            ResidualUnit(output_dim, dilation=9),
        )

    def forward(self, x):
        return self.block(x)


class Decoder(nn.Module):
    def __init__(
        self,
        input_channel,
        channels,
        rates,
        d_out: int = 1,
    ):
        super().__init__()

        # Add first conv layer
        layers = [WNConv1d(input_channel, channels, kernel_size=7, padding=3)]

        # Add upsampling + MRF blocks
        for i, stride in enumerate(rates):
            input_dim = channels // 2**i
            output_dim = channels // 2 ** (i + 1)
            layers += [DecoderBlock(input_dim, output_dim, stride)]

        # Add final conv layer
        layers += [
            Snake1d(output_dim),
            WNConv1d(output_dim, d_out, kernel_size=7, padding=3),
            nn.Tanh(),
        ]

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class DAC(BaseModel, CodecMixin):
    """
    Reference: https://github.com/facebookresearch/dacvae/blob/main/dacvae/model/dacvae.py
    """
    def __init__(
        self,
        encoder_dim: int = 64,
        encoder_rates: List[int] = [2, 4, 8, 8],
        latent_dim: int = None,
        decoder_dim: int = 1536,
        decoder_rates: List[int] = [8, 8, 4, 2],
        n_codebooks: int = 9,
        codebook_size: int = 1024,
        codebook_dim: Union[int, list] = 8,
        quantizer_dropout: bool = False,
        sample_rate: int = 44100,
    ):
        super().__init__()

        self.encoder_dim = encoder_dim
        self.encoder_rates = encoder_rates
        self.decoder_dim = decoder_dim
        self.decoder_rates = decoder_rates
        self.sample_rate = sample_rate

        if latent_dim is None:
            latent_dim = encoder_dim * (2 ** len(encoder_rates))

        self.latent_dim = latent_dim

        self.hop_length = np.prod(encoder_rates)
        self.encoder = Encoder(encoder_dim, encoder_rates, latent_dim)

        self.n_codebooks = n_codebooks
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.quantizer = ResidualVectorQuantize(
            input_dim=latent_dim,
            n_codebooks=n_codebooks,
            codebook_size=codebook_size,
            codebook_dim=codebook_dim,
            quantizer_dropout=quantizer_dropout,
        )

        self.decoder = Decoder(
            latent_dim,
            decoder_dim,
            decoder_rates,
        )
        self.sample_rate = sample_rate
        self.apply(init_weights)

        self.delay = self.get_delay()

    def preprocess(self, audio_data, sample_rate):
        if sample_rate is None:
            sample_rate = self.sample_rate
        assert sample_rate == self.sample_rate

        length = audio_data.shape[-1]
        right_pad = math.ceil(length / self.hop_length) * self.hop_length - length
        audio_data = nn.functional.pad(audio_data, (0, right_pad))

        return audio_data

    def encode(
        self,
        audio_data: torch.Tensor,
        n_quantizers: int = None,
    ):
        """Encode given audio data and return quantized latent codes

        Parameters
        ----------
        audio_data : Tensor[B x 1 x T]
            Audio data to encode
        n_quantizers : int, optional
            Number of quantizers to use, by default None
            If None, all quantizers are used.

        Returns
        -------
        dict
            A dictionary with the following keys:
            "z" : Tensor[B x D x T]
                Quantized continuous representation of input
            "codes" : Tensor[B x N x T]
                Codebook indices for each codebook
                (quantized discrete representation of input)
            "latents" : Tensor[B x N*D x T]
                Projected latents (continuous representation of input before quantization)
            "vq/commitment_loss" : Tensor[1]
                Commitment loss to train encoder to predict vectors closer to codebook
                entries
            "vq/codebook_loss" : Tensor[1]
                Codebook loss to update the codebook
            "length" : int
                Number of samples in input audio
        """
        z = self.encoder(audio_data)
        z, codes, latents, commitment_loss, codebook_loss = self.quantizer(z, n_quantizers)
        return z, codes, latents, commitment_loss, codebook_loss

    def decode(self, z: torch.Tensor):
        """Decode given latent codes and return audio data

        Parameters
        ----------
        z : Tensor[B x D x T]
            Quantized continuous representation of input
        length : int, optional
            Number of samples in output audio, by default None

        Returns
        -------
        dict
            A dictionary with the following keys:
            "audio" : Tensor[B x 1 x length]
                Decoded audio data.
        """
        return self.decoder(z)

    def forward(
        self,
        audio_data: torch.Tensor,
        sample_rate: int = None,
        n_quantizers: int = None,
    ):
        """Model forward pass

        Parameters
        ----------
        audio_data : Tensor[B x 1 x T]
            Audio data to encode
        sample_rate : int, optional
            Sample rate of audio data in Hz, by default None
            If None, defaults to `self.sample_rate`
        n_quantizers : int, optional
            Number of quantizers to use, by default None.
            If None, all quantizers are used.

        Returns
        -------
        dict
            A dictionary with the following keys:
            "z" : Tensor[B x D x T]
                Quantized continuous representation of input
            "codes" : Tensor[B x N x T]
                Codebook indices for each codebook
                (quantized discrete representation of input)
            "latents" : Tensor[B x N*D x T]
                Projected latents (continuous representation of input before quantization)
            "vq/commitment_loss" : Tensor[1]
                Commitment loss to train encoder to predict vectors closer to codebook
                entries
            "vq/codebook_loss" : Tensor[1]
                Codebook loss to update the codebook
            "length" : int
                Number of samples in input audio
            "audio" : Tensor[B x 1 x length]
                Decoded audio data.
        """
        length = audio_data.shape[-1]
        audio_data = self.preprocess(audio_data, sample_rate)
        z, codes, latents, commitment_loss, codebook_loss = self.encode(
            audio_data, n_quantizers
        )

        x = self.decode(z)
        return {
            "audio": x[..., :length],
            "z": z,
            "codes": codes,
            "latents": latents,
            "vq/commitment_loss": commitment_loss,
            "vq/codebook_loss": codebook_loss,
        }


class DAC_VAE(DAC):
    def __init__(
        self,
        encoder_dim: int = 64,
        encoder_rates: List[int] = [2, 4, 8, 8],
        latent_dim: Optional[int] = None,
        decoder_dim: int = 1536,
        decoder_rates: List[int] = [8, 8, 4, 2],
        n_codebooks: int = 9,
        codebook_size: int = 1024,
        codebook_dim: Union[int, list] = 8,
        quantizer_dropout: bool = False,
        sample_rate: int = 44100,
    ):
        super().__init__(
            encoder_dim=encoder_dim,
            encoder_rates=encoder_rates,
            latent_dim=latent_dim,
            decoder_dim=decoder_dim,
            decoder_rates=decoder_rates,
            n_codebooks=n_codebooks,
            codebook_size=codebook_size,
            codebook_dim=codebook_dim,
            quantizer_dropout=quantizer_dropout,
            sample_rate=sample_rate,
        )
        self.quantizer = VAEBottleneck(
            input_dim=self.latent_dim,
            codebook_size=codebook_size,
            codebook_dim=codebook_dim,
        )

    @classmethod
    def load(cls, path):
        if not os.path.exists(path) and path.startswith("facebook/"):
            path = hf_hub_download(repo_id=path, filename="weights.pth")
        return super().load(path)

    def _pad(self, wavs):
        length = wavs.size(-1)
        if length % self.hop_length:
            p1d = (0, self.hop_length - (length % self.hop_length))
            return torch.nn.functional.pad(wavs, p1d, "reflect")
        else:
            return wavs

    def encode(
        self,
        audio_data: torch.Tensor,
    ):
        z = self.encoder(self._pad(audio_data))
        mean, scale = self.quantizer.in_proj(z).chunk(2, dim=1)
        encoded_frames, _ = self.quantizer._vae_sample(mean, scale)
        return encoded_frames

    def decode(self, encoded_frames: torch.Tensor):
        emb = self.quantizer.out_proj(encoded_frames)
        return self.decoder(emb)


class DAC_FLOW_VAE(DAC):
    def __init__(
        self,
        encoder_dim: int = 64,
        encoder_rates: List[int] = [2, 4, 8, 8],
        latent_dim: Optional[int] = None,
        decoder_dim: int = 1536,
        decoder_rates: List[int] = [8, 8, 4, 2],
        sample_rate: int = 44100,
        n_flows: int = 4,
        flow_dim: int = 128,
        flow_hidden_dim: int = 128,
        flow_layers: int = 4,
        flow_kernel_size: int = 5,
        flow_dilation_rate: int = 1,
        flow_mean_only: bool = True,
        spk_emb_dim: int = 0
    ):
        super().__init__(
            encoder_dim=encoder_dim,
            encoder_rates=encoder_rates,
            latent_dim=latent_dim,
            decoder_dim=decoder_dim,
            decoder_rates=decoder_rates,
            sample_rate=sample_rate,
        )
        self.downsample_rate = math.prod(encoder_rates)
        self.flow = ResidualCouplingBlock(
            channels=flow_dim,
            hidden_channels=flow_hidden_dim,
            kernel_size=flow_kernel_size,
            dilation_rate=flow_dilation_rate,
            n_layers=flow_layers,
            n_flows=n_flows,
            gin_channels=spk_emb_dim,
            mean_only=flow_mean_only
        )
        self.quantizer = None

    def sequence_mask(self, length, max_length=None):
        if max_length is None:
            max_length = length.max()
        x = torch.arange(max_length, dtype=length.dtype, device=length.device)
        mask = x.unsqueeze(0) < length.unsqueeze(1)
        return mask.unsqueeze(1)

    def kl_loss(self, z0: torch.Tensor, logdet: torch.Tensor, mask: torch.Tensor):
        """A flow model to reversibly transform a normal distribution into a standard normal distribution.
        Args:
            z0: Output of the flow model [B, C, T]
            logdet: Log-determinant of the Jacobian [B]
            mask: Sequence mask [B, 1, T]
        """
        # 1. Calculate log-likelihood of standard normal distribution: -0.5 * (log(2*pi) + z0^2)
        # Note: log(2*pi) is a constant during optimization; the term z0^2 is the primary driver.
        prior_ll = -0.5 * (np.log(2 * np.pi) + z0**2) 
        
        # 2. Apply mask and sum across all dimensions
        prior_ll = torch.sum(prior_ll * mask)
        
        # logdet is typically already handled with respect to the mask inside the Flow layers
        logdet_sum = torch.sum(logdet) 
        
        # 3. Calculate Negative Log-Likelihood (NLL)
        # Loss = -(prior_log_likelihood + logdet)
        loss = -(prior_ll + logdet_sum)
        
        # Average the loss by the number of valid elements (sum of mask)
        return loss / torch.sum(mask)

    def encode(self, audio_data: torch.Tensor):
        z = self.encoder(audio_data)
        return z

    def forward(self, audios: torch.Tensor, audio_lengths: torch.Tensor, spk_emb: torch.Tensor = None):
        z = self.encode(audios) # [B, 1, L]
        z_mask = self.sequence_mask(audio_lengths // self.downsample_rate).to(z.device)
        z0, log_det = self.flow(z, z_mask, spk_emb)
        audio_data_hat = self.decode(z)
        return audio_data_hat, self.kl_loss(z0, log_det, z_mask)



if __name__ == "__main__":
    from omegaconf import OmegaConf
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = OmegaConf.load("./configs/flow_vae_config.yaml")
    model = DAC_FLOW_VAE(**config.model.flow_vae).to(device)
    
    length = 512 * 40 + 511
    batch_size = 2
    audios = torch.randn(batch_size, 1, length).to(model.device)
    audio_lengths = torch.tensor([length] * batch_size).to(model.device)

    from torchinfo import summary
    summary(model, input_data=(audios, audio_lengths), depth=5)

    audio_hats, loss_kl = model(audios, audio_lengths)
    print(audios.shape, audio_hats.shape)
    z = model.encode(audios)
    print(z.shape) # z.size(-1) = audios.shape[-1] // 512