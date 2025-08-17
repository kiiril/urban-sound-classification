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


def make_features(wav_name, mel_bins, target_length=1024):
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio_path", required=True, help="Path to your wav/flac")
    parser.add_argument("--model_path", default="./pretrained_models/audio_mdl.pth",
                        help="Checkpoint path (will auto-download if missing)")
    args = parser.parse_args()

    checkpoint_path = args.model_path or CHECKPOINT_PATH
    if not os.path.exists(checkpoint_path):
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        print(f'[*INFO] downloading checkpoint to {checkpoint_path}')
        wget.download(AUDIOSET_MODEL_URL, out=checkpoint_path)
        print()

    audio_path = args.audio_path
    feats = make_features(audio_path, mel_bins=128)

    input_tdim = feats.shape[0]

    ast_mdl = ASTModel(label_dim=527, input_tdim=input_tdim, imagenet_pretrain=False, audioset_pretrain=False)
    checkpoint = torch.load(checkpoint_path, map_location='cuda')
    audio_model = torch.nn.DataParallel(ast_mdl, device_ids=[0])
    audio_model.load_state_dict(checkpoint)
    audio_model = audio_model.to(torch.device("cuda:0"))
    audio_model.eval()    
    
    feats_data = feats.expand(1, input_tdim, 128)   

    with torch.no_grad():
        output = audio_model.forward(feats_data)
        output = torch.sigmoid(output)
    result_output = output.data.cpu().numpy()[0]

    labels = load_label(LABEL_PATH)
    
    sorted_indexes = np.argsort(result_output)[::-1]

    print('[*INFO] predict results:')
    for k in range(10):
        print('{}: {:.4f}'.format(np.array(labels)[sorted_indexes[k]],
                                  result_output[sorted_indexes[k]]))