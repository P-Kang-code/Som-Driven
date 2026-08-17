import math
from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class StageTwoConfig:
    latent_dim: int = 14
    code_dim: int = 512
    condition_dim: int = 256
    model_dim: int = 384
    depth: int = 6
    heads: int = 8
    head_dim: int = 64
    feedforward_multiplier: int = 4
    timesteps: int = 1000
    sampling_timesteps: int = 50
    condition_dropout: float = 0.1
    attention_dropout: float = 0.0
    feedforward_dropout: float = 0.1
    ddim_eta: float = 0.0


def cosine_beta_schedule(timesteps: int) -> torch.Tensor:
    grid = torch.linspace(0, timesteps, timesteps + 1, dtype=torch.float64)
    cumulative = torch.cos(((grid / timesteps) + 0.008) / 1.008 * math.pi * 0.5).square()
    cumulative = cumulative / cumulative[0]
    return (1 - cumulative[1:] / cumulative[:-1]).clamp(0, 0.999).float()


def gather(values: torch.Tensor, time: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    return values.gather(0, time).reshape(time.shape[0], *((1,) * (len(shape) - 1)))


class TimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        half = dim // 2
        frequency = torch.exp(
            -math.log(10000) * torch.arange(half, dtype=torch.float32) / max(half - 1, 1)
        )
        self.register_buffer("frequency", frequency)
        self.net = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        phase = time.float().unsqueeze(1) * self.frequency.unsqueeze(0)
        return self.net(torch.cat((phase.sin(), phase.cos()), dim=1))


class Attention(nn.Module):
    def __init__(
        self,
        query_dim: int,
        context_dim: int,
        heads: int,
        head_dim: int,
        dropout: float,
    ):
        super().__init__()
        self.heads = heads
        self.head_dim = head_dim
        inner = heads * head_dim
        self.query_norm = nn.LayerNorm(query_dim)
        self.context_norm = nn.LayerNorm(context_dim)
        self.to_query = nn.Linear(query_dim, inner, bias=False)
        self.to_key_value = nn.Linear(context_dim, inner * 2, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Sequential(nn.Linear(inner, query_dim, bias=False), nn.LayerNorm(query_dim))

    def forward(self, value: torch.Tensor, context: torch.Tensor | None = None) -> torch.Tensor:
        context = value if context is None else context
        query = self.to_query(self.query_norm(value))
        key, content = self.to_key_value(self.context_norm(context)).chunk(2, dim=2)
        batch, query_count = query.shape[:2]
        context_count = key.shape[1]
        query = query.reshape(batch, query_count, self.heads, self.head_dim).transpose(1, 2)
        key = key.reshape(batch, context_count, self.heads, self.head_dim).transpose(1, 2)
        content = content.reshape(batch, context_count, self.heads, self.head_dim).transpose(1, 2)
        score = torch.matmul(query, key.transpose(2, 3)) * self.head_dim ** -0.5
        weight = self.dropout(score.softmax(dim=3))
        output = torch.matmul(weight, content).transpose(1, 2).reshape(batch, query_count, -1)
        return self.output(output)


class FeedForward(nn.Module):
    def __init__(self, dim: int, multiplier: int, dropout: float):
        super().__init__()
        hidden = dim * multiplier
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden * 2),
            SwiGLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.net(value)


class SwiGLU(nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        content, gate = value.chunk(2, dim=2)
        return content * torch.nn.functional.silu(gate)


class TransformerLayer(nn.Module):
    def __init__(self, config: StageTwoConfig):
        super().__init__()
        self.self_attention = Attention(
            config.model_dim,
            config.model_dim,
            config.heads,
            config.head_dim,
            config.attention_dropout,
        )
        self.cross_attention = Attention(
            config.model_dim,
            config.condition_dim,
            config.heads,
            config.head_dim,
            config.attention_dropout,
        )
        self.feedforward = FeedForward(
            config.model_dim,
            config.feedforward_multiplier,
            config.feedforward_dropout,
        )

    def forward(self, value: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        value = value + self.self_attention(value)
        value = value + self.cross_attention(value, context)
        return value + self.feedforward(value)


class ConditionalDenoiser(nn.Module):
    def __init__(self, config: StageTwoConfig):
        super().__init__()
        self.config = config
        self.hand_token = nn.Sequential(
            nn.Linear(config.code_dim, config.condition_dim),
            nn.SiLU(),
            nn.Linear(config.condition_dim, config.condition_dim),
        )
        self.object_token = nn.Sequential(
            nn.Linear(config.code_dim, config.condition_dim),
            nn.SiLU(),
            nn.Linear(config.condition_dim, config.condition_dim),
        )
        self.type_embedding = nn.Parameter(torch.randn(2, config.condition_dim) * 0.02)
        self.task_embedding = nn.Parameter(torch.randn(2, config.condition_dim) * 0.02)
        self.time_embedding = TimeEmbedding(config.model_dim)
        self.latent_token = nn.Linear(config.latent_dim, config.model_dim)
        self.query = nn.Parameter(torch.randn(1, 1, config.model_dim) * 0.02)
        self.layers = nn.ModuleList(TransformerLayer(config) for _ in range(config.depth))
        self.output = nn.Sequential(nn.LayerNorm(config.model_dim), nn.Linear(config.model_dim, config.latent_dim))

    def condition_tokens(
        self,
        hand_code: torch.Tensor,
        object_code: torch.Tensor,
        task_code: torch.Tensor,
    ) -> torch.Tensor:
        hand = self.hand_token(hand_code) + self.type_embedding[0]
        object_value = self.object_token(object_code) + self.type_embedding[1]
        task = self.task_embedding[task_code.long().reshape(-1).clamp(0, 1)]
        context = torch.stack((hand, object_value, task), dim=1)
        if self.training and self.config.condition_dropout > 0:
            keep = (
                torch.rand(context.shape[0], 1, 1, device=context.device)
                >= self.config.condition_dropout
            )
            context = torch.where(keep, context, torch.zeros_like(context))
        return context

    def forward(
        self,
        latent: torch.Tensor,
        time: torch.Tensor,
        hand_code: torch.Tensor,
        object_code: torch.Tensor,
        task_code: torch.Tensor,
    ) -> torch.Tensor:
        context = self.condition_tokens(hand_code, object_code, task_code)
        time_token = self.time_embedding(time).unsqueeze(1)
        latent_token = self.latent_token(latent).unsqueeze(1)
        query = self.query.expand(latent.shape[0], -1, -1)
        value = torch.cat((time_token, latent_token, query), dim=1)
        for layer in self.layers:
            value = layer(value, context)
        return self.output(value[:, -1])


class TokenDiffusion(nn.Module):
    def __init__(self, config: StageTwoConfig = StageTwoConfig()):
        super().__init__()
        self.config = config
        beta = cosine_beta_schedule(config.timesteps)
        alpha = 1 - beta
        alpha_bar = torch.cumprod(alpha, dim=0)
        self.register_buffer("alpha_bar", alpha_bar)
        self.register_buffer("sqrt_alpha_bar", alpha_bar.sqrt())
        self.register_buffer("sqrt_one_minus_alpha_bar", (1 - alpha_bar).sqrt())
        self.denoiser = ConditionalDenoiser(config)

    def training_loss(
        self,
        target: torch.Tensor,
        mask: torch.Tensor,
        hand_code: torch.Tensor,
        object_code: torch.Tensor,
        task_code: torch.Tensor,
    ) -> torch.Tensor:
        time = torch.randint(0, self.config.timesteps, (target.shape[0],), device=target.device)
        noise = torch.randn_like(target)
        noised = (
            gather(self.sqrt_alpha_bar, time, target.shape) * target
            + gather(self.sqrt_one_minus_alpha_bar, time, target.shape) * noise
        )
        prediction = self.denoiser(noised, time, hand_code, object_code, task_code)
        losses = []
        weights = []
        for start, end, weight in ((0, 9, 0.35), (9, 12, 0.40), (12, 14, 0.25)):
            group_mask = mask[:, start:end]
            if bool((group_mask > 0).any()):
                denominator = group_mask.sum().clamp_min(1)
                losses.append(
                    ((prediction[:, start:end] - target[:, start:end]).square() * group_mask).sum()
                    / denominator
                )
                weights.append(weight)
        weight_tensor = prediction.new_tensor(weights)
        weight_tensor = weight_tensor / weight_tensor.sum().clamp_min(0.00000001)
        loss = (torch.stack(losses) * weight_tensor).sum()
        pose_valid = mask[:, :12].amax(dim=1) > 0.5
        if bool(pose_valid.any()):
            rotation = prediction[pose_valid, :9].reshape(-1, 3, 3)
            identity = torch.eye(3, device=rotation.device, dtype=rotation.dtype).expand(rotation.shape[0], -1, -1)
            orthogonality = (rotation.transpose(1, 2) @ rotation - identity).square().mean()
            determinant = (torch.det(rotation) - 1).square().mean()
            loss = loss + 0.002 * (orthogonality + determinant)
        return loss

    @torch.no_grad()
    def sample(
        self,
        hand_code: torch.Tensor,
        object_code: torch.Tensor,
        task_code: torch.Tensor,
    ) -> torch.Tensor:
        value = torch.randn(hand_code.shape[0], self.config.latent_dim, device=hand_code.device)
        times = torch.linspace(
            self.config.timesteps - 1,
            0,
            self.config.sampling_timesteps,
            device=hand_code.device,
        ).long()
        for position, time_value in enumerate(times):
            time = torch.full(
                (value.shape[0],),
                int(time_value.item()),
                device=value.device,
                dtype=torch.long,
            )
            prediction = self.denoiser(value, time, hand_code, object_code, task_code)
            if position == len(times) - 1:
                value = prediction
            else:
                next_time = times[position + 1]
                alpha = self.alpha_bar[time_value]
                alpha_next = self.alpha_bar[next_time]
                noise = (value - alpha.sqrt() * prediction) / (1 - alpha).sqrt().clamp_min(0.00000001)
                sigma = self.config.ddim_eta * (
                    (1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)
                ).clamp_min(0).sqrt()
                direction = (1 - alpha_next - sigma.square()).clamp_min(0).sqrt() * noise
                gaussian = torch.randn_like(value) if float(sigma.item()) > 0 else 0
                value = alpha_next.sqrt() * prediction + direction + sigma * gaussian
        return value
