import logging

from omegaconf import OmegaConf
import torch
import torchaudio

from minimaxspeech.cli.model import MiniMaxSpeechModel
from minimaxspeech.cli.frontend import MiniMaxSpeechFrontend
from minimaxspeech.utils.commons.logger import setup_logger

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
    setup_logger("output/logs")
    config = OmegaConf.load("configs/minimaxspeech_config.yaml")
    minimaxspeech = MiniMaxSpeech(config)
    audio = minimaxspeech.generate("data/LJ001-0001.wav", "Hello world!", lang="en")
    audio = audio[0].cpu()
    torchaudio.save("output/audio.wav", audio, 22050)