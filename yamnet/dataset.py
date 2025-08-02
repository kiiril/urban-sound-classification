from pathlib import Path
import pandas as pd
import numpy as np
import tensorflow as tf
import soundfile as sf
import params as yamnet_params
import resampy


SAMPLE_RATE = 16000
TARGET_LENGTH_SECONDS = 4.0
TARGET_LENGTH_SAMPLES = int(SAMPLE_RATE * TARGET_LENGTH_SECONDS)  # 64,000 samples

def standardize_audio_length(waveform, target_samples, training=True):
    """Standardize audio length with random crop for training, center crop for eval."""
    n = len(waveform)
    if n <= target_samples:
        pad_length = target_samples - n
        return np.pad(waveform, (0, pad_length), mode='constant', constant_values=0.0)
    else:
        if training:
            start_idx = np.random.randint(0, n - target_samples + 1)
        else:
            start_idx = (n - target_samples) // 2
    return waveform[start_idx:start_idx + target_samples]


def load_wav_16k(path: str, training: bool = True) -> np.ndarray:
    params = yamnet_params.Params()
    wav_data, sr = sf.read(path, dtype=np.int16)
    waveform = wav_data / 32768.0  # Convert to [-1.0, +1.0]
    waveform = waveform.astype('float32')
    
    if len(waveform.shape) > 1:
        waveform = np.mean(waveform, axis=1)
    if sr != params.sample_rate:
        waveform = resampy.resample(waveform, sr, params.sample_rate)

    waveform = standardize_audio_length(waveform, TARGET_LENGTH_SAMPLES, training)
      
    return waveform

def build_lists(root: str, folds):
    root = Path(root)
    meta = pd.read_csv(root/"UrbanSound8K.csv")
    sub = meta[meta["fold"].isin(folds)]
    files = [str(root / "audio" / f"fold{fold}" / fname)
             for fname, fold in zip(sub["slice_file_name"], sub["fold"])]
    labels = sub["classID"].astype(int).tolist()
    return files, labels

def _tf_load_train(path, label):
    wav = tf.numpy_function(lambda p: load_wav_16k(p.decode(), training=True), [path], tf.float32)
    wav.set_shape([TARGET_LENGTH_SAMPLES])  # Fixed shape now
    return wav, label

def _tf_load_eval(path, label):
    wav = tf.numpy_function(lambda p: load_wav_16k(p.decode(), training=False), [path], tf.float32)
    wav.set_shape([TARGET_LENGTH_SAMPLES])  # Fixed shape now
    return wav, label

def make_dataset(files, labels, batch_size, training: bool):
    ds = tf.data.Dataset.from_tensor_slices((files, labels))
    if training:
        ds = ds.shuffle(len(files), reshuffle_each_iteration=True)
    
    if training:
        ds = ds.map(_tf_load_train, num_parallel_calls=tf.data.AUTOTUNE)
    else:
        ds = ds.map(_tf_load_eval, num_parallel_calls=tf.data.AUTOTUNE)

    # (optional) simple gain augmentation during training
    # avoid for fair comparison
    # if training:
    #     def aug(w, y):
    #         gain = tf.random.uniform([], 0.8, 1.2)
    #         w = tf.clip_by_value(w * gain, -1.0, 1.0)
    #         return w, y
    #     ds = ds.map(aug, num_parallel_calls=tf.data.AUTOTUNE)

    
    # Since all samples are now fixed length, we can use regular batching
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds
    