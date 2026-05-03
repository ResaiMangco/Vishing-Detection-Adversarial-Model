from __future__ import annotations

import torchaudio
import torch

from dataclasses import dataclass
from typing import Optional
from torch import nn


# ===[[ Spectrogram-based Input Transformations ]]===

@dataclass(frozen=True)
class LogMelSpectrogramConfig:
    sample_rate: int = 16000
    n_fft: int = 1024
    win_length: Optional[int] = None
    hop_length: int = 256
    n_mels: int = 64
    f_min: float = 20.0
    f_max: float = 7600.0
    power: float = 2.0
    log_epsilon: float = 1e-6
    normalize: bool = True


class WaveformToLogMelSpectrogram(nn.Module):
    def __init__(self, config: Optional[LogMelSpectrogramConfig] = None) -> None:
        """Initializes the waveform to log-mel transformation module"""

        super().__init__()
        self.config = config or LogMelSpectrogramConfig()
        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.config.sample_rate,
            n_fft=self.config.n_fft,
            win_length=self.config.win_length,
            hop_length=self.config.hop_length,
            f_min=self.config.f_min,
            f_max=self.config.f_max,
            n_mels=self.config.n_mels,
            power=self.config.power,
            center=True,
            pad_mode="reflect",
            norm="slaney",
            mel_scale="htk",
        )

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        """Converts raw waveforms to log-mel spectrograms"""

        # Compress/expand the input waveforms to [batch, time] for consistency
        if waveforms.ndim == 1:
            waveforms = waveforms.unsqueeze(0)
        elif waveforms.ndim == 3 and waveforms.shape[1] == 1:
            waveforms = waveforms[:, 0, :]
        elif waveforms.ndim != 2:
            raise ValueError(
                "Expected waveforms with shape [time], [batch, time], or [batch, 1, time]."
            )
        
        # Generate log-mel spectrogram from the input waveforms
        mel_spectrogram = self.mel_spectrogram(waveforms)
        log_mel_spectrogram = torch.log(mel_spectrogram.clamp_min(self.config.log_epsilon))

        # Normalize log-mel spectrograms if required
        if self.config.normalize:
            mean = log_mel_spectrogram.mean(dim=(-2, -1), keepdim=True)
            std = log_mel_spectrogram.std(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
            log_mel_spectrogram = (log_mel_spectrogram - mean) / std

        # Add a channel (dimension) for CNN input compatibility 
        return log_mel_spectrogram.unsqueeze(1)

class SpecAugment(nn.Module):
    def __init__(self, freq_mask_param: int = 15, time_mask_param: int = 35, num_masks: int = 2):
        super().__init__()
        self.freq_masking = torchaudio.transforms.FrequencyMasking(freq_mask_param)
        self.time_masking = torchaudio.transforms.TimeMasking(time_mask_param)
        self.num_masks = num_masks

    def forward(self, spectrogram: torch.Tensor) -> torch.Tensor:
        for _ in range(self.num_masks):
            spectrogram = self.freq_masking(spectrogram)
            spectrogram = self.time_masking(spectrogram)
        return spectrogram
# ===[[ Prosodic-based Input Transformations ]]===

import librosa
import numpy as np

PROSODIC_FEATURE_COLUMNS = [
    # Pitch
    "f0_mean", "f0_std", "f0_range", "f0_slope",
    "f0_q25", "f0_q75", "f0_continuity",
    # Energy
    "rms_mean", "rms_std", "rms_max",
    # Rhythm
    "voiced_ratio", "pause_ratio",
    # Delta
    "f0_delta_mean", "f0_delta_std",
    "rms_delta_mean", "rms_delta_std",
    # MFCCs
    *[f"mfcc{i}_mean" for i in range(13)],
    *[f"mfcc{i}_std" for i in range(13)],
    # Spectral shape
    "spectral_centroid_mean", "spectral_centroid_std",
    "spectral_bandwidth_mean", "spectral_bandwidth_std",
    "spectral_rolloff_mean", "spectral_rolloff_std",
    "spectral_flatness_mean", "spectral_flatness_std",
    # Spectral flux
    "spectral_flux_mean", "spectral_flux_std",
    # Spectral entropy
    "spectral_entropy_mean", "spectral_entropy_std",
    # Zero crossing rate
    "zcr_mean", "zcr_std",
    # Voice quality
    "jitter", "shimmer", "harmonics_to_noise_ratio", "cpps",
    # Formants
    "f1_mean", "f1_std",
    "f2_mean", "f2_std",
    "f3_mean", "f3_std",
]

# Update your config class if needed (no change required)
@dataclass(frozen=True)
class ProsodicFeatureConfig:
    sr: Optional[int] = None
    fmin: float = librosa.note_to_hz('C2')
    fmax: float = librosa.note_to_hz('C7')


import parselmouth
import numpy as np
import librosa
from scipy.stats import skew, kurtosis


def spectral_entropy(stft_mag):
    power = stft_mag ** 2
    power_norm = power / (power.sum(axis=0, keepdims=True) + 1e-8)
    entropy = -np.sum(power_norm * np.log(power_norm + 1e-8), axis=0)
    return entropy


def extract_prosodic_features(
    file_path: str,
    config: Optional[ProsodicFeatureConfig] = None,
) -> dict:
    config = config or ProsodicFeatureConfig()
    y, sr = librosa.load(file_path, sr=config.sr)

    # Pitch
    f0, voiced_flag, _ = librosa.pyin(
        y, fmin=config.fmin, fmax=config.fmax, sr=sr, fill_na=0.0
    )
    voiced_flag = voiced_flag.astype(bool)
    f0_voiced = f0[voiced_flag]

    pitch_feats = {
        "f0_mean":       np.nanmean(f0_voiced) if len(f0_voiced) else 0.0,
        "f0_std":        np.nanstd(f0_voiced)  if len(f0_voiced) else 0.0,
        "f0_range":      np.nanmax(f0_voiced) - np.nanmin(f0_voiced) if len(f0_voiced) else 0.0,
        "f0_slope":      np.polyfit(np.arange(len(f0_voiced)), f0_voiced, 1)[0] if len(f0_voiced) > 1 else 0.0,
        "f0_q25":        np.nanquantile(f0_voiced, 0.25) if len(f0_voiced) else 0.0,
        "f0_q75":        np.nanquantile(f0_voiced, 0.75) if len(f0_voiced) else 0.0,
        "f0_continuity": np.mean(np.diff(voiced_flag.astype(int)) != 0) if len(voiced_flag) > 1 else 0.0,
    }

    # Energy
    rms = librosa.feature.rms(y=y)[0]
    energy_feats = {
        "rms_mean": np.mean(rms),
        "rms_std":  np.std(rms),
        "rms_max":  np.max(rms),
    }

    # Rhythm
    voiced_ratio = np.mean(voiced_flag) if len(voiced_flag) else 0.0
    energy_threshold = 0.01 * np.max(rms) if np.max(rms) > 0 else 0.0
    pause_ratio = np.mean(rms < energy_threshold)

    # Deltas
    f0_delta = np.diff(f0_voiced) if len(f0_voiced) > 1 else np.array([0.0])
    rms_delta = np.diff(rms)      if len(rms) > 1      else np.array([0.0])
    delta_feats = {
        "f0_delta_mean":  np.mean(f0_delta),
        "f0_delta_std":   np.std(f0_delta),
        "rms_delta_mean": np.mean(rms_delta),
        "rms_delta_std":  np.std(rms_delta),
    }

    # MFCCs
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_feats = {}
    for i in range(13):
        mfcc_feats[f"mfcc{i}_mean"] = np.mean(mfccs[i])
        mfcc_feats[f"mfcc{i}_std"]  = np.std(mfccs[i])

    # Spectral shape
    centroid  = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    rolloff   = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    flatness  = librosa.feature.spectral_flatness(y=y)[0]
    spectral_feats = {
        "spectral_centroid_mean":   np.mean(centroid),
        "spectral_centroid_std":    np.std(centroid),
        "spectral_bandwidth_mean":  np.mean(bandwidth),
        "spectral_bandwidth_std":   np.std(bandwidth),
        "spectral_rolloff_mean":    np.mean(rolloff),
        "spectral_rolloff_std":     np.std(rolloff),
        "spectral_flatness_mean":   np.mean(flatness),
        "spectral_flatness_std":    np.std(flatness),
    }

    # Spectral flux
    stft = np.abs(librosa.stft(y))
    spectral_flux = np.sqrt(np.sum(np.diff(stft, axis=1) ** 2, axis=0))
    spectral_flux_feats = {
        "spectral_flux_mean": np.mean(spectral_flux),
        "spectral_flux_std":  np.std(spectral_flux),
    }

    # Spectral entropy
    spec_entropy = spectral_entropy(stft)
    spectral_entropy_feats = {
        "spectral_entropy_mean": np.mean(spec_entropy),
        "spectral_entropy_std":  np.std(spec_entropy),
    }

    # Zero crossing rate
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    zcr_feats = {
        "zcr_mean": np.mean(zcr),
        "zcr_std":  np.std(zcr),
    }

    # Voice quality
    sound = parselmouth.Sound(y, sr)
    try:
        point_process = parselmouth.praat.call(sound, "To PointProcess (periodic, cc)", 75, 600)
        jitter   = parselmouth.praat.call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
        shimmer  = parselmouth.praat.call([sound, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        harmonicity = sound.to_harmonicity()
        hnr = parselmouth.praat.call(harmonicity, "Get mean", 0, 0) if harmonicity else 0.0
    except Exception as e:
        print(f"Voice quality failed: {e}")
        jitter, shimmer, hnr = 0.0, 0.0, 0.0

    # CPP
    try:
        cpp_obj = parselmouth.praat.call(sound, "To PowerCepstrogram", 60, 0.002, 5000, 50)
        cpps = parselmouth.praat.call(cpp_obj, "Get CPPS", "yes", 0.02, 0.0, 60, 330, 0.05, "Parabolic", 0.001, 0, "Exponential decay", "Robust")
    except Exception as e:
        print(f"CPP failed: {e}")
        cpps = 0.0

    # Formants with pre-emphasis
    try:
        sound_preemph = parselmouth.praat.call(sound, "Filter (pre-emphasis)", 50)
        formant = parselmouth.praat.call(sound_preemph, "To Formant (burg)", 0.0, 5, 5500, 0.025, 50)
        f1_mean = parselmouth.praat.call(formant, "Get mean", 1, 0, 0)
        f1_std  = parselmouth.praat.call(formant, "Get standard deviation", 1, 0, 0)
        f2_mean = parselmouth.praat.call(formant, "Get mean", 2, 0, 0)
        f2_std  = parselmouth.praat.call(formant, "Get standard deviation", 2, 0, 0)
        f3_mean = parselmouth.praat.call(formant, "Get mean", 3, 0, 0)
        f3_std  = parselmouth.praat.call(formant, "Get standard deviation", 3, 0, 0)
    except Exception as e:
        print(f"Formant failed: {e}")
        f1_mean = f1_std = f2_mean = f2_std = f3_mean = f3_std = 0.0

    voice_quality_feats = {
        "jitter": jitter,
        "shimmer": shimmer,
        "harmonics_to_noise_ratio": hnr,
        "cpps": cpps,
        "f1_mean": f1_mean, "f1_std": f1_std,
        "f2_mean": f2_mean, "f2_std": f2_std,
        "f3_mean": f3_mean, "f3_std": f3_std,
    }

    features = {
        **pitch_feats,
        **energy_feats,
        "voiced_ratio": voiced_ratio,
        "pause_ratio":  pause_ratio,
        **delta_feats,
        **mfcc_feats,
        **spectral_feats,
        **spectral_flux_feats,
        **spectral_entropy_feats,
        **zcr_feats,
        **voice_quality_feats,
    }

    for k in features:
        if np.isnan(features[k]) or np.isinf(features[k]):
            features[k] = 0.0

    return features