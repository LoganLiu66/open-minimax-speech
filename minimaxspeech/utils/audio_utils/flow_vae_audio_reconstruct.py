#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
"""
Reference:
- https://github.com/descriptinc/descript-audio-codec/blob/main/scripts/get_samples.py

Usage:
    python minimaxspeech/utils/audio_utils/flow_vae_audio_reconstruct.py \
        --config configs/flow_vae_config.yaml \
        --ckpt_file output/flow_vae/checkpoint_090000.pth \
        --input data/dac/evaluate/valid \
        --output data/dac/evaluate/valid_recon_flow_vae
"""
import os
from pathlib import Path

import argbind
import torch
from audiotools import AudioSignal
from audiotools.core import util
from audiotools.ml.decorators import Tracker
from omegaconf import OmegaConf

from minimaxspeech.modules.flow_vae.flow_vae import DAC_FLOW_VAE


@torch.no_grad()
def process(signal, model, sr):
    signal = signal.resample(sr).to(model.device)
    z = model.encode(signal.audio_data)

    # Decode audio signal
    recons = model.decode(z)
    recons = AudioSignal(recons, signal.sample_rate)
    recons = recons.normalize(signal.loudness())
    return recons.cpu()


@argbind.bind(without_prefix=True)
@torch.no_grad()
def get_samples(
    config: str = "configs/flow_vae_config.yaml",
    ckpt_file: str = "output/flow_vae/checkpoint_090000.pth",
    input: str = "samples/input",
    output: str = "samples/output"
):
    os.makedirs(output, exist_ok=True)
    tracker = Tracker(log_file=f"{output}/eval.txt")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = OmegaConf.load(config)
    sr = config.model.flow_vae.sample_rate
    model = DAC_FLOW_VAE(**config.model.flow_vae).to(device)
    model.load_state_dict(torch.load(ckpt_file)['flow_vae'])
    model.eval()

    audio_files = util.find_audio(input)

    global process
    process = tracker.track("process", len(audio_files))(process)

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)

    with tracker.live:
        for i in range(len(audio_files)):
            signal = AudioSignal(audio_files[i])
            recons = process(signal, model, sr)
            recons.write(output / audio_files[i].name)

        tracker.done("test", f"N={len(audio_files)}")


if __name__ == "__main__":
    args = argbind.parse_args()
    with argbind.scope(args):
        get_samples()