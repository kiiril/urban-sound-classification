from torch.utils.data import Dataset
from pathlib import Path
import pandas as pd
import torch, torchaudio

class UrbanSound8K(Dataset):
    def __init__(self, root, folds):
        root = Path(root)
        meta = pd.read_csv(root/'UrbanSound8K.csv')
        self.items = meta[meta.fold.isin(folds)][['slice_file_name','fold','classID']].values
        self.audio_root = root/'audio'
        
    def __len__(self): return len(self.items)
    
    def __getitem__(self, idx):
        file_name, fold, label = self.items[idx]
        feat = wav_to_fbank(self.audio_root/f'fold{fold}'/file_name)
        return feat, label
        

def wav_to_fbank(wav_name, mel_bins=128, target_length=512):
    waveform, sr = torchaudio.load(wav_name)

    fbank = torchaudio.compliance.kaldi.fbank(
        waveform, htk_compat=True, sample_frequency=sr, use_energy=False,
        window_type='hanning', num_mel_bins=mel_bins, dither=0.0,
        frame_shift=10)

    n_frames = fbank.shape[0]

    p = target_length - n_frames
    if p > 0:
        m = torch.nn.ZeroPad2d((0, 0, 0, p))
        fbank = m(fbank)
    elif p < 0:
        fbank = fbank[0:target_length, :]

    fbank = (fbank - (-4.2677393)) / (4.5689974 * 2)
    return fbank