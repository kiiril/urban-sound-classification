from pathlib import Path
import sys
import csv
import os
import wget
import argparse
import torch, torchaudio, timm
import numpy as np
from model import ASTModel


os.environ['TORCH_HOME'] = './pretrained_models'

AUDIOSET_MODEL_URL = 'https://www.dropbox.com/s/cv4knew8mvbrnvq/audioset_0.4593.pth?dl=1'
CHECKPOINT_PATH = './pretrained_models/audio_mdl.pth'
LABEL_PATH = './metadata/class_labels_indices.csv'
TARGET_LEN = 1024
MEL_BINS = 128


def make_features(wav_name, mel_bins=MEL_BINS, target_length=TARGET_LEN):
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


def load_label(label_csv):
    with open(label_csv, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        lines = list(reader)
    labels = []
    ids = []  # Each label has a unique id such as "/m/068hy"
    for i1 in range(1, len(lines)):
        id = lines[i1][1]
        label = lines[i1][2]
        ids.append(id)
        labels.append(label)
    return labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/", help="Folder with WAV files")
    parser.add_argument("--out_csv", default="results.csv", help="CSV to save predictions")
    parser.add_argument("--model_path", default=CHECKPOINT_PATH,
                        help="Checkpoint path (will auto-download if missing)")
    args = parser.parse_args()
    
    # ensure checkpoint
    checkpoint_path = args.model_path
    if not os.path.exists(checkpoint_path):
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        print(f'[*INFO] downloading checkpoint to {checkpoint_path}')
        wget.download(AUDIOSET_MODEL_URL, out=checkpoint_path)
        print()
        
    # collect WAV files
    wav_files = sorted([str(p) for p in Path(args.data_dir).rglob("*.wav")])
    if not wav_files:
        print(f"No .wav files found in {args.data_dir}")
        return
    print(f"[*] Found {len(wav_files)} wav files")
    
    labels = load_label(LABEL_PATH)
    num_labels = len(labels)
    
    ast_mdl = ASTModel(label_dim=num_labels, input_tdim=TARGET_LEN, imagenet_pretrain=False, audioset_pretrain=False)
    checkpoint = torch.load(checkpoint_path, map_location='cuda')
    audio_model = torch.nn.DataParallel(ast_mdl, device_ids=[0])
    audio_model.load_state_dict(checkpoint)
    audio_model = audio_model.to(torch.device("cuda:0"))
    audio_model.eval()
    
    with open(args.out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["file", "label", "score"])
        
        with torch.no_grad():
            for i, path in enumerate(wav_files, 1):
                try:
                    feats = make_features(path)
                    input_tdim = feats.shape[0]
                    feats_data = feats.expand(1, input_tdim, 128)   
                    
                    output = audio_model.forward(feats_data)
                    output = torch.sigmoid(output)
                    result_output = output.data.cpu().numpy()[0]
                    sorted_indexes = np.argsort(result_output)[::-1]
                    
                    top_index = sorted_indexes[0]
                    writer.writerow([path, labels[top_index], result_output[top_index]])
                except Exception as e:
                    print(f"[!] Error processing {path}: {e}")
                    
    print(f"[*] Done. Results saved to {args.out_csv}")
        

if __name__ == "__main__":
    main()