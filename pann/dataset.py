from torch.utils.data import Dataset
from pathlib import Path
import pandas as pd
import torch
import librosa
import numpy as np

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
        
        wav, sr = librosa.load(path, sr=self.target_sr, mono=True)
        if wav.dtype != np.float32:
            wav = wav.astype(np.float32)
        
        return torch.from_numpy(wav), int(label)