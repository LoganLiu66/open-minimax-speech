#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
"""
Reference:
- https://github.com/descriptinc/descript-audio-codec/blob/main/scripts/get_samples.py

Usage:
    python minimaxspeech/utils/audio_utils/dac_audio_reconstruct.py \
        --input data/dac/evaluate/valid \
        --output data/dac/evaluate/valid_recon
"""
import os
from pathlib import Path

import argbind
import torch
from audiotools import AudioSignal
from audiotools.core import util
from audiotools.ml.decorators import Tracker

import dac


@torch.no_grad()
def process(signal, model):
    signal = signal.resample(44100).to(model.device)
    x = model.preprocess(signal.audio_data, signal.sample_rate)
    z, codes, latents, _, _ = model.encode(x)

    # Decode audio signal
    recons = model.decode(z)
    recons = AudioSignal(recons, signal.sample_rate)
    recons = recons.normalize(signal.loudness())
    return recons.cpu()


@argbind.bind(without_prefix=True)
@torch.no_grad()
def get_samples(
    input: str = "samples/input",
    output: str = "samples/output"
):
    os.makedirs(output, exist_ok=True)
    tracker = Tracker(log_file=f"{output}/eval.txt")
    model_path = dac.utils.download(model_type="44khz")
    model = dac.DAC.load(model_path).to('cuda')

    audio_files = util.find_audio(input)

    global process
    process = tracker.track("process", len(audio_files))(process)

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)

    with tracker.live:
        for i in range(len(audio_files)):
            signal = AudioSignal(audio_files[i])
            recons = process(signal, model)
            recons.write(output / audio_files[i].name)

        tracker.done("test", f"N={len(audio_files)}")


if __name__ == "__main__":
    args = argbind.parse_args()
    with argbind.scope(args):
        get_samples()