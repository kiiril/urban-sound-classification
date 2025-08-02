from torch.utils.data import Dataset
from pathlib import Path
import pandas as pd
import torch, torchaudio, torchaudio.transforms as T

SAMPLE_RATE = 16000
TARGET_LENGTH_SECONDS = 4.0
TARGET_LENGTH_SAMPLES = int(SAMPLE_RATE * TARGET_LENGTH_SECONDS)  # 64,000 samples

def standardize_audio_length(waveform, target_samples, training=True):
    """Standardize audio length with random crop for training, center crop for eval."""
    if waveform.shape[1] <= target_samples:
        # Pad with zeros if shorter
        pad_length = target_samples - waveform.shape[1]
        return torch.nn.functional.pad(waveform, (0, pad_length), mode='constant', value=0.0)
    
    else:
        if training:
            max_start = waveform.shape[1] - target_samples
            start_idx = torch.randint(0, max_start + 1, (1,)).item()
        else:
            start_idx = (waveform.shape[1] - target_samples) // 2
        return waveform[:, start_idx:start_idx + target_samples]

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
    waveform, orig_sr = torchaudio.load(wav_name)
    
    target_sr = 16000
    
    resampler = T.Resample(orig_freq=orig_sr, new_freq=target_sr)
    
    if orig_sr != target_sr:
        waveform = resampler(waveform)

    fbank = torchaudio.compliance.kaldi.fbank(
        waveform, htk_compat=True, sample_frequency=target_sr, use_energy=False,
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