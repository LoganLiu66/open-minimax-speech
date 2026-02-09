#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
import json
import logging
import random
from typing import Any, Dict

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset

from minimaxspeech.modules.tokenizer.xtts_tokenizer import VoiceBpeTokenizer


class GPTDataset(Dataset):
    def __init__(
        self,
        data_file_list: str,
        sample_rate: int = 22050,
        tokenizer_file: str = "checkpoints/dvae/tokenizer.json",
        min_conditioning_duration: float = 3.0,
        max_conditioning_duration: float = 6.0,
        min_audio_duration: float = 1,
        max_audio_duration: float = 30.0,
        min_text_length: int = 1,
        max_text_length: int = 300,
        use_masking_gt_prompt_approach: bool = True,
        is_eval: bool = False
    ):
        self.sampling_rate = sample_rate
        self.min_conditioning_duration = min_conditioning_duration
        self.max_conditioning_duration = max_conditioning_duration
        self.min_audio_duration = min_audio_duration
        self.max_audio_duration = max_audio_duration
        self.min_text_length = min_text_length
        self.max_text_length = max_text_length
        self.use_masking_gt_prompt_approach = use_masking_gt_prompt_approach
        self.is_eval = is_eval

        self.text_tokenizer = VoiceBpeTokenizer(vocab_file=tokenizer_file)

        self.metadata_list = self.load_file_list(data_file_list)
        logging.info(f"Loaded {len(self.metadata_list)} items from {data_file_list}")

    def load_file_list(self, data_list_file: str):
        metadata_list = []
        with open(data_list_file, 'r') as f:
            for line in f.readlines()[:]:
                meta_data = json.loads(line.strip())
                metadata_list.append({
                    'audio_file': meta_data['audio_file'],
                    'sid': meta_data['sid'],
                    'lang': meta_data['lang'],
                    'text': meta_data['text'],
                    'ref_audio_file': meta_data['ref_audio_file']
                })
        return metadata_list

    def __len__(self) -> int:
        return len(self.metadata_list)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        try:
            return self.getitem(idx)
        except Exception as e:
            # logging.info(f"Error getting item {self.metadata_list[idx]['audio_file']}: {e}")
            return self.__getitem__(random.randint(0, len(self.metadata_list) - 1))
    
    def getitem(self, idx):
        audio_info = self.metadata_list[idx]
        wav, wav_len = self.load_audio(audio_info["audio_file"])
        
        # Reference: https://github.com/coqui-ai/TTS/blob/dev/TTS/tts/layers/xtts/trainer/dataset.py#L117
        # Load conditioning audio
        if self.use_masking_gt_prompt_approach:
            # get a slice from GT to condition the model
            cond, _, cond_idxs = self.get_prompt_slice(
                audio_info["audio_file"],
                int(self.max_conditioning_duration * self.sampling_rate),
                int(self.min_conditioning_duration * self.sampling_rate),
                self.is_eval
            )
            # if use masking do not use cond_len
            cond_len = np.nan
        else:
            cond, cond_len, _ = self.get_prompt_slice(
                audio_info["ref_audio_file"],
                int(self.max_conditioning_duration * self.sampling_rate),
                int(self.min_conditioning_duration * self.sampling_rate),
                self.is_eval
            )
            # if do not use masking use cond_len
            cond_idxs = np.nan

        text = audio_info["text"]
        text_tokens = self.text_tokenizer.encode(text, audio_info['lang'])
        text_len = len(text_tokens)

        if text_len > self.max_text_length or text_len < self.min_text_length:
            raise ValueError(f"Text length {text_len} is out of {self.min_text_length} to {self.max_text_length}")

        return {
            "waveform": wav,  # (1, Tvar)
            "wav_length": wav_len,
            "text_tokens": text_tokens,
            "text_length": text_len,
            "cond": cond,
            "cond_lens": cond_len,
            "cond_idxs": cond_idxs,
            "audio_file": audio_info["audio_file"],
            "ref_audio_file": audio_info["ref_audio_file"],
        }

    def load_audio(self, audio_file):
        wav, sr = torchaudio.load(audio_file)
        if wav.dim() != 1:
            wav = wav[:1, :]  # force mono
        if sr != self.sampling_rate:
            wav = torchaudio.functional.resample(wav, orig_freq=sr, new_freq=self.sampling_rate)

        duration = wav.size(-1) / self.sampling_rate
        if duration > self.max_audio_duration or duration < self.min_audio_duration:
            raise ValueError(f"Audio length {duration} out of {self.min_audio_duration} to {self.max_audio_duration}")
        return wav, wav.size(-1)
        
    def get_prompt_slice(self, gt_path, max_sample_length, min_sample_length, is_eval=False):
        rel_clip, wav_len = self.load_audio(gt_path)
        # if eval uses a middle size sample when it is possible to be more reproducible
        if is_eval:
            sample_length = int((min_sample_length + max_sample_length) / 2)
        else:
            sample_length = random.randint(min_sample_length, max_sample_length)
        gap = rel_clip.shape[-1] - sample_length
        if gap < 0:
            sample_length = rel_clip.shape[-1] // 2
        gap = rel_clip.shape[-1] - sample_length

        # if eval start always from the position 0 to be more reproducible
        if is_eval:
            rand_start = 0
        else:
            rand_start = random.randint(0, gap)

        rand_end = rand_start + sample_length
        rel_clip = rel_clip[:, rand_start:rand_end]
        rel_clip = F.pad(rel_clip, pad=(0, max_sample_length - rel_clip.shape[-1]))
        cond_idxs = [rand_start, rand_end]
        return rel_clip, rel_clip.shape[-1], cond_idxs

    @staticmethod
    def collate_fn(batch):
        # waveforms: pad to max T
        batch = {k: [dic[k] for dic in batch] for k in batch[0]}

        # conditioning
        conds = torch.stack(batch["cond"])
        if batch["cond_lens"][0] is np.nan:
            cond_lens = None
        else:
            cond_lens = torch.tensor(batch["cond_lens"], dtype=torch.long)
        if batch["cond_idxs"][0] is np.nan:
            cond_idxs = None
        else:
            cond_idxs = torch.tensor(batch["cond_idxs"], dtype=torch.long)


        # waveform padding
        wav_lengths = torch.tensor(batch["wav_length"], dtype=torch.long)
        max_wav_len = wav_lengths.max()
        wavs_padded = []
        for w in batch["waveform"]:
            pad = max_wav_len - w.size(-1)
            if pad > 0:
                w = F.pad(w, (0, pad), mode="constant", value=0.0)
            wavs_padded.append(w)
        wavs = torch.stack(wavs_padded, dim=0)  # (B, 1, T)

        # text padding
        text_lengths = torch.tensor(batch["text_length"], dtype=torch.long)
        max_text_len = text_lengths.max()
        texts_padded = []
        for t in batch["text_tokens"]:
            pad = max_text_len - len(t)
            if pad > 0:
                t = t + [0] * pad
            texts_padded.append(t)
        text_tokens = torch.tensor(texts_padded, dtype=torch.long)  # (B, L)

        audio_files = batch["audio_file"]
        ref_audio_files = batch["ref_audio_file"]

        return {
            "waveform": wavs,
            "wav_lengths": wav_lengths,
            "text_tokens": text_tokens,
            "text_lengths": text_lengths,
            "cond": conds,
            "cond_lens": cond_lens,
            "cond_idxs": cond_idxs,
            "audio_files": audio_files,
            "ref_audio_files": ref_audio_files,
        }


