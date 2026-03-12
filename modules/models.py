from __future__ import annotations

import torch

from dataclasses import dataclass

from typing import Optional
from torch import nn
from modules.audio_processing import LogMelSpectrogramConfig, WaveformToLogMelSpectrogram


# ===[[ Spectrogram-based Model ]]===

@dataclass(frozen=True)
class SpectrogramCNNConfig:
    num_classes: int = 2
    base_channels: int = 32
    dropout: float = 0.30
    embedding_dim: int = 128


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers(inputs)


class SpectrogramCNN(nn.Module):
    """CNN architecture for Spectrogram-based Model"""
    def __init__(self, config: Optional[SpectrogramCNNConfig] = None) -> None:
        super().__init__()

        self.config = config or SpectrogramCNNConfig()
        channels = self.config.base_channels

        # Define feature extractor, pooling layer, and classifier head
        self.feature_extractor = nn.Sequential(
            ConvBlock(1, channels),
            ConvBlock(channels, channels * 2),
            ConvBlock(channels * 2, channels * 4),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels * 4, self.config.embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=self.config.dropout),
            nn.Linear(self.config.embedding_dim, self.config.num_classes),
        )

    def forward(self, spectrograms: torch.Tensor) -> torch.Tensor:
        features = self.feature_extractor(spectrograms)
        pooled_features = self.pool(features)
        return self.classifier(pooled_features)


class SpectrogramClassifier(nn.Module):
    """Spectrogram-based Model with raw waveform transformer and CNN architecture"""
    def __init__(
        self,
        transformer_config: Optional[LogMelSpectrogramConfig] = None,
        model_config: Optional[SpectrogramCNNConfig] = None,
    ) -> None:
        super().__init__()

        self.transformer = WaveformToLogMelSpectrogram(transformer_config)
        self.classifier = SpectrogramCNN(model_config)

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        spectrograms = self.transformer(waveforms)
        return self.classifier(spectrograms)

    @torch.no_grad()
    def predict_proba(self, waveforms: torch.Tensor) -> torch.Tensor:
        logits = self.forward(waveforms)
        return torch.softmax(logits, dim=-1)

    @torch.no_grad()
    def predict(self, waveforms: torch.Tensor) -> torch.Tensor:
        probabilities = self.predict_proba(waveforms)
        return probabilities.argmax(dim=-1)


# ===[[ Prosodic-based Model ]]===