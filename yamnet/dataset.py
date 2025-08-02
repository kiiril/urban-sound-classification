from pathlib import Path
import pandas as pd
import numpy as np
import tensorflow as tf
import soundfile as sf
import params as yamnet_params
import resampy


SAMPLE_RATE = 16000


def load_wav_16k(path: str) -> np.ndarray:
    params = yamnet_params.Params()
    wav_data, sr = sf.read(path, dtype=np.int16)
    waveform = wav_data / 32768.0  # Convert to [-1.0, +1.0]
    waveform = waveform.astype('float32')
    
    if len(waveform.shape) > 1:
        waveform = np.mean(waveform, axis=1)
    if sr != params.sample_rate:
        waveform = resampy.resample(waveform, sr, params.sample_rate)
      
    return waveform

def build_lists(root: str, folds):
    root = Path(root)
    meta = pd.read_csv(root/"UrbanSound8K.csv")
    sub = meta[meta["fold"].isin(folds)]
    files = [str(root / "audio" / f"fold{fold}" / fname)
             for fname, fold in zip(sub["slice_file_name"], sub["fold"])]
    labels = sub["classID"].astype(int).tolist()
    return files, labels

def _tf_load(path, label):
    wav = tf.numpy_function(load_wav_16k, [path], tf.float32)
    wav.set_shape([None])   # 1-D waveform, variable length
    return wav, label

def make_dataset(files, labels, batch_size, training: bool):
    ds = tf.data.Dataset.from_tensor_slices((files, labels))
    if training:
        ds = ds.shuffle(len(files), reshuffle_each_iteration=True)
    ds = ds.map(_tf_load, num_parallel_calls=tf.data.AUTOTUNE)

    # (optional) simple gain augmentation during training
    # avoid for fair comparison
    # if training:
    #     def aug(w, y):
    #         gain = tf.random.uniform([], 0.8, 1.2)
    #         w = tf.clip_by_value(w * gain, -1.0, 1.0)
    #         return w, y
    #     ds = ds.map(aug, num_parallel_calls=tf.data.AUTOTUNE)

    # YAMNet handles variable-length; we just pad per batch.
    ds = ds.padded_batch(batch_size, padded_shapes=([None], []))
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds
    