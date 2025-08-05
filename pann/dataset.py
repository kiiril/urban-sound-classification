from torch.utils.data import Dataset
from pathlib import Path
import pandas as pd
import torch
import librosa
import numpy as np
import torchaudio
import torchaudio.transforms as T

class UrbanSound8KWav(Dataset):
    def __init__(self, root, folds, target_sr=32000):
        root = Path(root)
        meta = pd.read_csv(root/'UrbanSound8K.csv')
        self.items = meta[meta.fold.isin(folds)][['slice_file_name','fold','classID']].values
        self.audio_root = root/'audio'
        self.target_sr = target_sr
        
    def __len__(self): return len(self.items)
    
    def __getitem__(self, idx):
        file_name, fold, label = self.items[idx]
        path = self.audio_root/f'fold{fold}'/file_name
        
        waveform, orig_sr = torchaudio.load(str(path))
        
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        if orig_sr != self.target_sr:
            resampler = T.Resample(orig_freq=orig_sr, new_freq=self.target_sr)
            waveform = resampler(waveform)
        
        waveform = waveform.squeeze(0).float()
        
        return waveform, int(label)