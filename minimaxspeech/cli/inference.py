#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
"""
Usage:
    python minimaxspeech/cli/inference.py \
        --config configs/minimaxspeech_config.yaml \
        --prompt_audio data/LJ001-0001.wav \
        --text "Hello, how are you?" \
        --lang en \
        --output_file output/audio.wav
"""
import argparse

import torch
import torchaudio
from omegaconf import OmegaConf

from minimaxspeech.cli.model import MiniMaxSpeechModel
from minimaxspeech.cli.frontend import MiniMaxSpeechFrontend
from minimaxspeech.utils.commons.logger import setup_logger


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/minimaxspeech_config.yaml")
    parser.add_argument("--prompt_audio", type=str, default="data/LJ001-0001.wav")
    parser.add_argument("--text", type=str, default="Hello, how are you?")
    parser.add_argument("--lang", type=str, default="en")
    parser.add_argument("--output_file", type=str, default="output/audio.wav")
    return parser.parse_args()


class MiniMaxSpeech:
    def __init__(self, config: dict):
        self.config = config
        self.sampling_rate = config.model.vq_vae.sample_rate
        self.model = MiniMaxSpeechModel(config)
        self.frontend = MiniMaxSpeechFrontend(config.frontend.vocab_file)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def generate(self, prompt_audio: str, text: str, lang: str='en'):
        y, sr = torchaudio.load(prompt_audio) # [1, L]
        if sr != self.sampling_rate:
            y = torchaudio.functional.resample(y, orig_freq=sr, new_freq=self.sampling_rate)
        text_tokens = self.frontend.encode(text, lang=lang)
        y = y.to(self.device)
        text_tokens = torch.tensor(text_tokens).to(self.device)
        
        audio = self.model.generate(y, text_tokens)
        return audio


if __name__ == "__main__":
    args = get_args()
    logger = setup_logger("output/logs")
    logger.info("Starting MiniMaxSpeech inference")
    config = OmegaConf.load(args.config)
    minimaxspeech = MiniMaxSpeech(config)
    audio = minimaxspeech.generate(args.prompt_audio, args.text, lang=args.lang)
    torchaudio.save(args.output_file, audio.cpu(), 22050)
    logger.info(f"saving audio to {args.output_file}")