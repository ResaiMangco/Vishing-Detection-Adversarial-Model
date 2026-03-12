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
