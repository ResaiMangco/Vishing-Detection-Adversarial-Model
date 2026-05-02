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
    "f0_skew", "f0_kurtosis", "f0_q25", "f0_q75",
    # Energy
    "rms_mean", "rms_std", "rms_max",
    "rms_skew", "rms_kurtosis",
    # Rhythm
    "tempo", "voiced_ratio", "pause_ratio",
    # Delta
    "f0_delta_mean", "f0_delta_std",
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
    "f1_mean", "f1_std",
    "f2_mean", "f2_std",
    "f3_mean", "f3_std",
    # Chroma
    "chroma_mean", "chroma_std",
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

def extract_prosodic_features(
    file_path: str,
    config: Optional[ProsodicFeatureConfig] = None,
) -> dict:
    """Improved prosodic feature extraction for ASVspoof5 / Vishing detection."""
    config = config or ProsodicFeatureConfig()
    y, sr = librosa.load(file_path, sr=config.sr)

    # === Better Pitch with pyin ===
    f0, voiced_flag, voiced_prob = librosa.pyin(
        y, fmin=config.fmin, fmax=config.fmax, sr=sr, fill_na=0.0
    )
    voiced_flag = voiced_flag.astype(bool)
    f0_voiced = f0[voiced_flag]

    # Pitch stats
    pitch_feats = {
        "f0_mean": np.nanmean(f0_voiced) if len(f0_voiced) else 0.0,
        "f0_std": np.nanstd(f0_voiced) if len(f0_voiced) else 0.0,
        "f0_range": np.nanmax(f0_voiced) - np.nanmin(f0_voiced) if len(f0_voiced) else 0.0,
        "f0_slope": np.polyfit(np.arange(len(f0_voiced)), f0_voiced, 1)[0] if len(f0_voiced) > 1 else 0.0,
        "f0_skew": skew(f0_voiced) if len(f0_voiced) > 2 else 0.0,
        "f0_kurtosis": kurtosis(f0_voiced) if len(f0_voiced) > 2 else 0.0,
        "f0_q25": np.nanquantile(f0_voiced, 0.25) if len(f0_voiced) else 0.0,
        "f0_q75": np.nanquantile(f0_voiced, 0.75) if len(f0_voiced) else 0.0,
    }

    # Energy (RMS)
    rms = librosa.feature.rms(y=y)[0]
    energy_feats = {
        "rms_mean": np.mean(rms),
        "rms_std": np.std(rms),
        "rms_max": np.max(rms),
        "rms_skew": skew(rms) if len(rms) > 2 else 0.0,
        "rms_kurtosis": kurtosis(rms) if len(rms) > 2 else 0.0,
    }

    # Tempo (more reliable)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    tempo = float(tempo) if np.isscalar(tempo) else float(tempo[0]) if len(tempo) > 0 else 0.0

    # Voiced / Pause ratio
    voiced_ratio = np.mean(voiced_flag) if len(voiced_flag) else 0.0
    # Simple pause ratio (frames with very low energy)
    energy_threshold = 0.01 * np.max(rms) if np.max(rms) > 0 else 0.0
    pause_ratio = np.mean(rms < energy_threshold)

    # Delta features (keep similar)
    f0_delta = np.diff(f0_voiced) if len(f0_voiced) > 1 else np.array([0.0])
    rms_delta = np.diff(rms) if len(rms) > 1 else np.array([0.0])
    delta_feats = {
        "f0_delta_mean": np.mean(f0_delta),
        "f0_delta_std": np.std(f0_delta),
        "rms_delta_mean": np.mean(rms_delta),
        "rms_delta_std": np.std(rms_delta),
    }

    # MFCCs + deltas (unchanged - keep your existing code block)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_feats = {}
    for i in range(13):
        mfcc_feats[f"mfcc{i}_mean"] = np.mean(mfccs[i])
        mfcc_feats[f"mfcc{i}_std"] = np.std(mfccs[i])

    delta_width = min(9, mfccs.shape[1] if mfccs.shape[1] % 2 == 1 else mfccs.shape[1] - 1)
    delta_width = max(3, delta_width)  # minimum width is 3
    mfcc_deltas = librosa.feature.delta(mfccs, width=delta_width)
    mfcc_delta_feats = {}
    for i in range(13):
        mfcc_delta_feats[f"mfcc{i}_delta_mean"] = np.mean(mfcc_deltas[i])
        mfcc_delta_feats[f"mfcc{i}_delta_std"] = np.std(mfcc_deltas[i])

    # Spectral features (keep your existing)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    flatness = librosa.feature.spectral_flatness(y=y)[0]
    spectral_feats = {
        "spectral_centroid_mean": np.mean(centroid),
        "spectral_centroid_std": np.std(centroid),
        "spectral_bandwidth_mean": np.mean(bandwidth),
        "spectral_bandwidth_std": np.std(bandwidth),
        "spectral_rolloff_mean": np.mean(rolloff),
        "spectral_rolloff_std": np.std(rolloff),
        "spectral_flatness_mean": np.mean(flatness),
        "spectral_flatness_std": np.std(flatness),
    }

    stft = np.abs(librosa.stft(y))
    spectral_flux = np.sqrt(np.sum(np.diff(stft, axis=1) ** 2, axis=0))
    spectral_flux_feats = {
        "spectral_flux_mean": np.mean(spectral_flux),
        "spectral_flux_std": np.std(spectral_flux),
    }

    # === Voice Quality + Formants with Parselmouth (biggest improvement) ===
    sound = parselmouth.Sound(y, sr)
    try:
        point_process = parselmouth.praat.call(sound, "To PointProcess (periodic, cc)", 75, 600)
        jitter = parselmouth.praat.call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
        shimmer = parselmouth.praat.call([sound, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        
        harmonicity = sound.to_harmonicity()
        hnr = parselmouth.praat.call(harmonicity, "Get mean", 0, 0) if harmonicity else 0.0
    except Exception:
        jitter, shimmer, hnr = 0.0, 0.0, 0.0

    # Formants (Burg method - very useful for spoof detection)
    try:
        formant = parselmouth.praat.call(sound, "To Formant (burg)", 0.0, 5, 5500, 0.025, 50)
        f1_mean = parselmouth.praat.call(formant, "Get mean", 1, 0, 0)
        f1_std = parselmouth.praat.call(formant, "Get standard deviation", 1, 0, 0)
        f2_mean = parselmouth.praat.call(formant, "Get mean", 2, 0, 0)
        f2_std = parselmouth.praat.call(formant, "Get standard deviation", 2, 0, 0)
        f3_mean = parselmouth.praat.call(formant, "Get mean", 3, 0, 0)
        f3_std = parselmouth.praat.call(formant, "Get standard deviation", 3, 0, 0)
    except Exception:
        f1_mean = f1_std = f2_mean = f2_std = f3_mean = f3_std = 0.0

    voice_quality_feats = {
        "jitter": jitter,
        "shimmer": shimmer,
        "harmonics_to_noise_ratio": hnr,
        "f1_mean": f1_mean, "f1_std": f1_std,
        "f2_mean": f2_mean, "f2_std": f2_std,
        "f3_mean": f3_mean, "f3_std": f3_std,
    }

    # Chroma (unchanged)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_feats = {
        "chroma_mean": np.mean(chroma),
        "chroma_std": np.std(chroma),
    }

    # Combine all
    features = {
        **pitch_feats,
        **energy_feats,
        "tempo": tempo,
        "voiced_ratio": voiced_ratio,
        "pause_ratio": pause_ratio,
        **delta_feats,
        **mfcc_feats,
        **mfcc_delta_feats,
        **spectral_feats,
        **spectral_flux_feats,
        **voice_quality_feats,
        **chroma_feats,
    }

    # Safe NaN handling
    for k in features:
        if np.isnan(features[k]) or np.isinf(features[k]):
            features[k] = 0.0

    return features


# ===[[ wavlm-based Input Transformations ]]===
from torch.utils.data import Dataset
TARGET_SR = 16000
MAX_DURATION = 6.0
class WaveFormExtract(Dataset):
    def __init__(self, metadata_df, audio_root: str):
        self.df = metadata_df.reset_index(drop=True)
        self.audio_root = audio_root
        self.label_map = {"bonafide": 0, "spoof": 1}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        audio_path = f"{self.audio_root}/{row['FULL_FILE_PATH']}"  # Adjust column if needed

        waveform, sr = torchaudio.load(audio_path)
        if sr != TARGET_SR:
            waveform = torchaudio.functional.resample(waveform, sr, TARGET_SR)

        if waveform.shape[0] > 1:  # stereo -> mono
            waveform = waveform.mean(dim=0, keepdim=True)

        # Fixed length: truncate or pad
        max_samples = int(MAX_DURATION * TARGET_SR)
        if waveform.shape[1] > max_samples:
            waveform = waveform[:, :max_samples]
        else:
            pad = max_samples - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, pad))

        label = self.label_map.get(row['LABEL'], 1)  # default spoof
        return waveform.squeeze(0), torch.tensor(label, dtype=torch.long)  # [time], label

