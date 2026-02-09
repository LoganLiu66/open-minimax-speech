import json
import logging
import random

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset


class VAEDataset(Dataset):
    def __init__(self, data_file_list, sample_rate=22050, max_duration=15.0):
        self.sampling_rate = sample_rate
        self.max_duration = max_duration
        self.downsample_rate = 512
        self.max_samples = int(sample_rate * max_duration)

        self.metadata_list = self.load_file_list(data_file_list)
        logging.info(f"Loaded {len(self.metadata_list)} audio files from {data_file_list}")

    def load_file_list(self, data_list_file: str):
        metadata_list = []
        with open(data_list_file, 'r') as f:
            for line in f.readlines()[:]:
                meta_data = json.loads(line.strip())
                metadata_list.append({
                    'audio_file': meta_data['audio_file'],
                    'sid': meta_data['sid'],
                    'lang': meta_data['lang'],
                    'text': meta_data['text']
                })
        return metadata_list

    def __len__(self):
        return len(self.metadata_list)

    def __getitem__(self, idx):
        try:
            return self.getitem(idx)
        except Exception as e:
            logging.info(f"Error getting item {idx}: {e}")
            return self.__getitem__(random.randint(0, len(self.metadata_list) - 1))
    
    def getitem(self, idx):
        """
        Returns:
            waveform: with a shape of [1, L]
            filepath: absolute path
        """
        audio_info = self.metadata_list[idx]
        filepath = audio_info['audio_file']
        
        y, sr = torchaudio.load(filepath) # [1, L]
        if sr != self.sampling_rate:
            y = torchaudio.functional.resample(y, orig_freq=sr, new_freq=self.sampling_rate)

        length = y.size(-1) 
        if length > self.max_samples:
            start = np.random.randint(low=0, high=y.size(-1) - self.max_samples + 1)
            y = y[:, start : start + self.max_samples]

        length = y.size(-1) // self.downsample_rate * self.downsample_rate
        y = y[:1, :length]
        
        item = {
            'waveform': y,
            'filepath': filepath,
            'length': length
        }
        return item

    def collate_fn(self, batch):
        lengths = [item['length'] for item in batch]
        max_len = max(lengths)
        waveforms = [F.pad(item['waveform'], (0, max_len - item['length']), mode='constant', value=0) for item in batch]
        filepaths = [item['filepath'] for item in batch]

        waveforms = torch.stack(waveforms)
        lengths = torch.tensor(lengths)
        return waveforms, filepaths, lengths