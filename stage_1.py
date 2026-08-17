from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class StageOneConfig:
    code_dim: int = 512
    hand_points: int = 778
    object_points: int = 2048


class FullPointEncoder(nn.Module):
    def __init__(self, channels: int, points: int, code_dim: int = 512):
        super().__init__()
        self.channels = channels
        self.points = points
        self.code_dim = code_dim
        self.conv1 = nn.Sequential(
            nn.Conv1d(channels, 128, 1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(128, 256, 1),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(256, code_dim, 1),
            nn.BatchNorm1d(code_dim),
            nn.LeakyReLU(),
        )
        self.fc1 = nn.Sequential(nn.Linear(points, 1024), nn.LeakyReLU())
        self.fc2 = nn.Sequential(nn.Linear(1024, 512), nn.LeakyReLU())
        self.fc3 = nn.Sequential(nn.Linear(512, 256), nn.LeakyReLU())
        self.collapse = nn.Linear(256, 1)
        self.mean = nn.Sequential(nn.Linear(code_dim, code_dim), nn.Softsign())
        self.logvar = nn.Sequential(nn.Linear(code_dim, code_dim), nn.Softsign())

    def forward(self, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if values.ndim != 3 or values.shape[1:] != (self.points, self.channels):
            raise ValueError(
                f"Expected [B,{self.points},{self.channels}], received {tuple(values.shape)}"
            )
        value = torch.nan_to_num(values.float()).transpose(1, 2).contiguous()
        value = self.conv1(value)
        value = self.fc1(value)
        value = self.conv2(value)
        value = self.fc2(value)
        value = self.conv3(value)
        value = self.fc3(value)
        value = self.collapse(value).squeeze(-1)
        return self.mean(value), self.logvar(value)


class DecoderResidualLayer(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.first = nn.Linear(input_dim, hidden_dim)
        self.first_norm = nn.BatchNorm1d(hidden_dim)
        self.second = nn.Linear(hidden_dim, output_dim)
        self.second_norm = nn.BatchNorm1d(output_dim)
        self.skip = nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        residual = self.skip(value)
        value = self.activation(self.first_norm(self.first(value)))
        value = self.second_norm(self.second(value))
        return self.activation(value + residual)


class FullPointDecoder(nn.Module):
    def __init__(self, channels: int, points: int, code_dim: int = 512):
        super().__init__()
        self.channels = channels
        self.points = points
        self.regularization = nn.BatchNorm1d(code_dim)
        self.first = DecoderResidualLayer(code_dim, code_dim * 2)
        self.second = DecoderResidualLayer(code_dim * 2, code_dim * 4)
        self.output = nn.Linear(code_dim * 4, points * channels)

    def forward(self, code: torch.Tensor) -> torch.Tensor:
        value = self.regularization(code)
        value = self.first(value)
        value = self.second(value)
        return self.output(value).reshape(-1, self.points, self.channels)


class FullPointAutoencoder(nn.Module):
    def __init__(self, channels: int, points: int, code_dim: int = 512):
        super().__init__()
        self.encoder = FullPointEncoder(channels, points, code_dim)
        self.decoder = FullPointDecoder(channels, points, code_dim)

    def encode(self, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.encoder(values)

    def forward(self, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, logvar = self.encode(values)
        return self.decoder(mean), mean, logvar


class FullAfferentEncoder(nn.Module):
    def __init__(self, config: StageOneConfig = StageOneConfig()):
        super().__init__()
        self.hand_autoencoder = FullPointAutoencoder(4, config.hand_points, config.code_dim)
        self.object_autoencoder = FullPointAutoencoder(3, config.object_points, config.code_dim)

    def distributions(
        self,
        hand: torch.Tensor,
        object_surface: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hand_mean, hand_logvar = self.hand_autoencoder.encode(hand)
        object_mean, object_logvar = self.object_autoencoder.encode(object_surface)
        return hand_mean, hand_logvar, object_mean, object_logvar

    def reconstructions(
        self,
        hand: torch.Tensor,
        object_surface: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hand_output, hand_mean, hand_logvar = self.hand_autoencoder(hand)
        object_output, object_mean, object_logvar = self.object_autoencoder(object_surface)
        return hand_output, object_output, hand_mean, hand_logvar, object_mean, object_logvar

    def forward(
        self,
        hand: torch.Tensor,
        object_surface: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hand_mean, _, object_mean, _ = self.distributions(hand, object_surface)
        return hand_mean, object_mean


class SimpleRawPointEncoder(nn.Module):
    def __init__(self, channels: int, code_dim: int = 512):
        super().__init__()
        self.channels = channels
        self.net = nn.Sequential(
            nn.Conv1d(channels, 128, 1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Conv1d(128, 256, 1),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Conv1d(256, code_dim, 1),
            nn.BatchNorm1d(code_dim),
            nn.LeakyReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Softsign(),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if values.ndim != 3 or values.shape[2] != self.channels:
            raise ValueError(f"Expected [B,N,{self.channels}], received {tuple(values.shape)}")
        values = torch.nan_to_num(values.float()).transpose(1, 2).contiguous()
        return self.net(values).squeeze(2)


class SimpleRawAfferentEncoder(nn.Module):
    def __init__(self, code_dim: int = 512):
        super().__init__()
        self.hand_encoder = SimpleRawPointEncoder(4, code_dim)
        self.object_encoder = SimpleRawPointEncoder(3, code_dim)

    def forward(
        self,
        hand: torch.Tensor,
        object_surface: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.hand_encoder(hand), self.object_encoder(object_surface)
