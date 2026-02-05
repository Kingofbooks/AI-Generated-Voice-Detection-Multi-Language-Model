import librosa
import numpy as np

def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=16000)
    if len(y) < 16000:
        raise ValueError("Audio too short")
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    mfcc_mean = mfcc.mean(axis=1)
    mfcc_std = mfcc.std(axis=1)

    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr).mean()
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr).mean()
    zero_crossing = librosa.feature.zero_crossing_rate(y).mean()
    spectral_flatness = librosa.feature.spectral_flatness(y=y).mean()
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr).mean()
    rms = librosa.feature.rms(y=y).mean()

    features = np.concatenate([
        mfcc_mean,
        mfcc_std,
        [spectral_centroid, spectral_rolloff, zero_crossing, spectral_flatness, spectral_bandwidth, rms]
    ])

    return features
