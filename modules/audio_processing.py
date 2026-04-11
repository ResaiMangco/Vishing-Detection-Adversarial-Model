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


# ===[[ Prosodic-based Input Transformations ]]===

import librosa
import numpy as np


PROSODIC_FEATURE_COLUMNS = [
    # Pitch
    "f0_mean", "f0_std", "f0_range", "f0_slope",
    # Energy
    "rms_mean", "rms_std", "rms_max",
    # Speaking rate
    "zcr_mean", "zcr_std",
    # Rhythm & voicing
    "tempo", "voiced_ratio",
    # Pitch dynamics
    "f0_delta_mean", "f0_delta_std",
    # Energy dynamics
    "rms_delta_mean", "rms_delta_std",
    # MFCCs
    *[f"mfcc{i}_mean" for i in range(13)],
    *[f"mfcc{i}_std" for i in range(13)],
    # MFCC deltas
    *[f"mfcc{i}_delta_mean" for i in range(13)],
    *[f"mfcc{i}_delta_std" for i in range(13)],
    # Spectral shape
    "spectral_centroid_mean", "spectral_centroid_std",
    "spectral_bandwidth_mean", "spectral_bandwidth_std",
    "spectral_rolloff_mean", "spectral_rolloff_std",
    "spectral_flatness_mean", "spectral_flatness_std",
    # Spectral flux
    "spectral_flux_mean", "spectral_flux_std",
    # Voice quality
    "jitter", "shimmer", "harmonics_to_noise_ratio",
    # Chroma
    "chroma_mean", "chroma_std",
]

@dataclass(frozen=True)
class ProsodicFeatureConfig:
    sr: Optional[int] = None
    fmin: float = librosa.note_to_hz('C2')
    fmax: float = librosa.note_to_hz('C7')


def extract_prosodic_features(
    file_path: str,
    config: Optional[ProsodicFeatureConfig] = None,
) -> dict:
    """Extracts prosodic features (pitch, energy, rate, tempo, voicing) from an audio file."""
    config = config or ProsodicFeatureConfig()
    y, sr = librosa.load(file_path, sr=config.sr)

    # Pitch (F0) via yin
    f0 = librosa.yin(y, fmin=config.fmin, fmax=config.fmax)
    voiced_flag = (f0 > config.fmin) & (f0 < config.fmax)
    f0_voiced = f0[voiced_flag]

    pitch_feats = {
        "f0_mean":  np.nanmean(f0_voiced) if len(f0_voiced) else 0,
        "f0_std":   np.nanstd(f0_voiced)  if len(f0_voiced) else 0,
        "f0_range": np.nanmax(f0_voiced) - np.nanmin(f0_voiced) if len(f0_voiced) else 0,
        "f0_slope": np.polyfit(np.arange(len(f0_voiced)), f0_voiced, 1)[0] if len(f0_voiced) > 1 else 0,
    }

    # Energy (RMS)
    rms = librosa.feature.rms(y=y)[0]
    energy_feats = {
        "rms_mean": np.mean(rms),
        "rms_std":  np.std(rms),
        "rms_max":  np.max(rms),
    }

    # Speaking rate (zero-crossing rate)
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    rate_feats = {
        "zcr_mean": np.mean(zcr),
        "zcr_std":  np.std(zcr),
    }

    # Tempo via autocorrelation
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    ac = np.correlate(onset_env, onset_env, mode='full')
    ac = ac[len(ac) // 2:]
    hop_length = 512
    min_lag = int(60.0 / 300 * sr / hop_length)
    max_lag = int(60.0 / 30 * sr / hop_length)
    ac_search = ac[min_lag:max_lag]
    if len(ac_search) > 0:
        best_lag = min_lag + np.argmax(ac_search)
        tempo = 60.0 * sr / (best_lag * hop_length)
    else:
        tempo = 0.0

    # Voiced / unvoiced ratio
    voiced_ratio = np.sum(voiced_flag) / len(voiced_flag) if len(voiced_flag) else 0

    # Delta features
    f0_delta = np.diff(f0_voiced) if len(f0_voiced) > 1 else np.array([0.0])
    rms_delta = np.diff(rms) if len(rms) > 1 else np.array([0.0])
    delta_feats = {
        "f0_delta_mean":  np.mean(f0_delta),
        "f0_delta_std":   np.std(f0_delta),
        "rms_delta_mean": np.mean(rms_delta),
        "rms_delta_std":  np.std(rms_delta),
    }

    # MFCCs (13 coefficients → mean & std)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_feats = {}
    for i in range(13):
        mfcc_feats[f"mfcc{i}_mean"] = np.mean(mfccs[i])
        mfcc_feats[f"mfcc{i}_std"]  = np.std(mfccs[i])

    # NEW — MFCC deltas (temporal evolution of MFCCs)
    mfcc_deltas = librosa.feature.delta(mfccs)
    mfcc_delta_feats = {}
    for i in range(13):
        mfcc_delta_feats[f"mfcc{i}_delta_mean"] = np.mean(mfcc_deltas[i])
        mfcc_delta_feats[f"mfcc{i}_delta_std"]  = np.std(mfcc_deltas[i])

    # Spectral shape features
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    flatness = librosa.feature.spectral_flatness(y=y)[0]
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

    # NEW — Spectral flux (how fast spectrum changes frame to frame)
    stft = np.abs(librosa.stft(y))
    spectral_flux = np.sqrt(np.sum(np.diff(stft, axis=1) ** 2, axis=0))
    spectral_flux_feats = {
        "spectral_flux_mean": np.mean(spectral_flux),
        "spectral_flux_std":  np.std(spectral_flux),
    }

    # NEW — Voice quality features (jitter, shimmer, HNR)
    # Jitter — cycle-to-cycle pitch period variation
    if len(f0_voiced) > 1:
        periods = 1.0 / (f0_voiced + 1e-8)
        period_diffs = np.abs(np.diff(periods))
        jitter = np.mean(period_diffs) / (np.mean(periods) + 1e-8)
    else:
        jitter = 0.0

    # Shimmer — cycle-to-cycle amplitude variation
    if len(rms) > 1:
        rms_diffs = np.abs(np.diff(rms))
        shimmer = np.mean(rms_diffs) / (np.mean(rms) + 1e-8)
    else:
        shimmer = 0.0

    # HNR — Harmonics to Noise Ratio
    if len(f0_voiced) > 0:
        try:
            # Use autocorrelation-based HNR estimation
            frame_length = 2048
            hop = 512
            frames = librosa.util.frame(y, frame_length=frame_length, hop_length=hop)
            hnr_values = []
            for frame in frames.T:
                ac_frame = np.correlate(frame, frame, mode='full')
                ac_frame = ac_frame[len(ac_frame) // 2:]
                ac_frame = ac_frame / (ac_frame[0] + 1e-8)
                peak = np.max(ac_frame[1:]) if len(ac_frame) > 1 else 0
                peak = np.clip(peak, 0, 1 - 1e-8)
                hnr = 10 * np.log10(peak / (1 - peak + 1e-8))
                hnr_values.append(hnr)
            hnr = np.mean(hnr_values)
        except Exception:
            hnr = 0.0
    else:
        hnr = 0.0

    voice_quality_feats = {
        "jitter":                   jitter,
        "shimmer":                  shimmer,
        "harmonics_to_noise_ratio": hnr,
    }

    # NEW — Chroma features (harmonic structure)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_feats = {
        "chroma_mean": np.mean(chroma),
        "chroma_std":  np.std(chroma),
    }

    return {
        **pitch_feats,
        **energy_feats,
        **rate_feats,
        "tempo": tempo,
        "voiced_ratio": voiced_ratio,
        **delta_feats,
        **mfcc_feats,
        **mfcc_delta_feats,      
        **spectral_feats,
        **spectral_flux_feats,    
        **voice_quality_feats,    
        **chroma_feats,           
    }