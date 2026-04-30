from __future__ import annotations

import torch

from dataclasses import dataclass

from typing import Optional
from torch import nn
from modules.audio_processing import LogMelSpectrogramConfig, WaveformToLogMelSpectrogram, SpecAugment
import torchaudio

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
        # Skip connection to match channels when they differ
        self.skip = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.AvgPool2d(kernel_size=2, stride=2),
        ) if in_channels != out_channels else nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=2)
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers(inputs) + self.skip(inputs)

class SpectrogramCNN(nn.Module):
    """CNN architecture for Spectrogram-based Model - Improved"""
    def __init__(self, config: Optional[SpectrogramCNNConfig] = None) -> None:
        super().__init__()

        self.config = config or SpectrogramCNNConfig()
        channels = self.config.base_channels

        # Deeper feature extractor
        self.feature_extractor = nn.Sequential(
            ConvBlock(1, channels),           # Input will be 1 channel for now
            ConvBlock(channels, channels * 2),
            ConvBlock(channels * 2, channels * 4),
            ConvBlock(channels * 4, channels * 8),   # extra
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels * 8, self.config.embedding_dim),  # Updated from *4 to *8
            nn.ReLU(inplace=True),
            nn.Dropout(p=self.config.dropout),
            nn.Linear(self.config.embedding_dim, self.config.num_classes),
        )

    def forward(self, spectrograms: torch.Tensor) -> torch.Tensor:
        features = self.feature_extractor(spectrograms)
        pooled_features = self.pool(features)
        return self.classifier(pooled_features)



class SpectrogramClassifier(nn.Module):
    def __init__(
        self,
        transformer_config=None,
        model_config=None,
        use_spec_augment: bool = True,
        freq_mask_param: int = 15,
        time_mask_param: int = 35,
    ) -> None:
        super().__init__()
        self.transformer = WaveformToLogMelSpectrogram(transformer_config)
        self.spec_augment = SpecAugment(freq_mask_param, time_mask_param) if use_spec_augment else None
        self.classifier = SpectrogramCNN(model_config)

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        spectrograms = self.transformer(waveforms)
        if self.training and self.spec_augment is not None:
            spectrograms = self.spec_augment(spectrograms)
        return self.classifier(spectrograms)


# ===[[ Prosodic-based Model ]]===

@dataclass(frozen=True)
class ProsodyMLPConfig:
    input_dim: int = 85
    num_classes: int = 2
    hidden_dims: tuple = (512, 256, 128, 64)
    dropout: float = 0.4
    noise_std: float = 0.2
    use_attention: bool = True 


class ProsodyMLP(nn.Module):
    """MLP architecture for Prosodic-based Model"""
    def __init__(self, config: Optional[ProsodyMLPConfig] = None) -> None:
        super().__init__()
        self.config = config or ProsodyMLPConfig()

        # Feature attention
        self.feature_attention = nn.Sequential(
            nn.Linear(self.config.input_dim, self.config.input_dim),
            nn.Sigmoid()
        )

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

        # Apply feature attention
        if self.config.use_attention:
            attention_weights = self.feature_attention(x)
            x = x * attention_weights
            
        # Apply first block with residual connection
        block_size = 4  # Linear + BN + GELU + Dropout
        first_block = self.hidden[:block_size]
        rest = self.hidden[block_size:]

        h = first_block(x) + self.residual_proj(x)
        h = rest(h)
        return self.head(h)
    



# ===[[ wavlm-based Model ]]===
class WavLM_SpoofDetector(nn.Module):
    def __init__(self, num_classes=2, freeze_wavlm=False):
        super().__init__()
        self.wavlm = WavLMModel.from_pretrained("microsoft/wavlm-base")
        
        if freeze_wavlm:
            for param in self.wavlm.parameters():
                param.requires_grad = False
                
        hidden_dim = self.wavlm.config.hidden_size  # 768 for base
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, waveform):
        # waveform: [batch, time] at 16kHz
        outputs = self.wavlm(input_values=waveform)
        hidden_states = outputs.last_hidden_state  # [batch, seq_len, 768]
        
        # Mean pooling (simple & effective)
        pooled = hidden_states.mean(dim=1)  # [batch, 768]
        
        logits = self.classifier(pooled)
        return logits