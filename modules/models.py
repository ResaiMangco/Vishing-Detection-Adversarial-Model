from __future__ import annotations

import torch
import torchaudio

from dataclasses import dataclass

from typing import Optional
from torch import nn
from modules.audio_processing import LogMelSpectrogramConfig, SpecAugmentConfig, WaveformToLogMelSpectrogram


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
            ConvBlock(channels * 2, channels * 4)
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
        spec_augment_config: Optional[SpecAugmentConfig] = None,
    ) -> None:
        super().__init__()

        self.transformer = WaveformToLogMelSpectrogram(transformer_config)
        self.classifier = SpectrogramCNN(model_config)

        aug = spec_augment_config or SpecAugmentConfig()
        self.freq_masks = nn.ModuleList(
            [torchaudio.transforms.FrequencyMasking(freq_mask_param=aug.freq_mask_param) for _ in range(aug.num_freq_masks)]
        )
        self.time_masks = nn.ModuleList(
            [torchaudio.transforms.TimeMasking(time_mask_param=aug.time_mask_param) for _ in range(aug.num_time_masks)]
        )

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        spectrograms = self.transformer(waveforms)
        if self.training:
            for mask in self.freq_masks:
                spectrograms = mask(spectrograms)
            for mask in self.time_masks:
                spectrograms = mask(spectrograms)
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

@dataclass(frozen=True)
class ProsodyMLPConfig:
    input_dim: int = 49
    num_classes: int = 2
    hidden_dims: tuple = (128, 64)
    dropout: float = 0.5
    noise_std: float = 0.1


class ProsodyMLP(nn.Module):
    """MLP architecture for Prosodic-based Model"""
    def __init__(self, config: Optional[ProsodyMLPConfig] = None) -> None:
        super().__init__()
        self.config = config or ProsodyMLPConfig()

        layers = []
        prev_dim = self.config.input_dim
        for hidden_dim in self.config.hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(self.config.dropout))
            prev_dim = hidden_dim

        self.hidden = nn.Sequential(*layers)
        self.head = nn.Linear(prev_dim, self.config.num_classes)

        # Residual projection when input_dim != first hidden dim
        first_h = self.config.hidden_dims[0] if self.config.hidden_dims else prev_dim
        self.residual_proj = (
            nn.Linear(self.config.input_dim, first_h, bias=False)
            if self.config.input_dim != first_h
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Gaussian noise augmentation during training
        if self.training and self.config.noise_std > 0:
            x = x + torch.randn_like(x) * self.config.noise_std

        # Apply first block with residual connection
        block_size = 4  # Linear + BN + GELU + Dropout
        first_block = self.hidden[:block_size]
        rest = self.hidden[block_size:]

        h = first_block(x) + self.residual_proj(x)
        h = rest(h)
        return self.head(h)