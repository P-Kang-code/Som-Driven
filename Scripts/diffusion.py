from __future__ import annotations
from typing import Any, Optional, Tuple, Union
from dataclasses import dataclass
import math
import torch
from torch import nn
import torch.nn.functional as F
from collections import namedtuple
from inspect import isfunction
from einops import rearrange, repeat
from torch import einsum
from rotary_embedding_torch import RotaryEmbedding
from tqdm.auto import tqdm


def exists(x: Any) -> bool:
    return x is not None


def default(val: Any, d: Any) -> Any:
    if exists(val):
        return val
    return d() if isfunction(d) else d


def extract(a: torch.Tensor, t: torch.Tensor, x_shape: Tuple[int, ...]) -> torch.Tensor:
    b = t.shape[0]
    out = a.gather(-1, t)
    return out.reshape(b, *(1,) * (len(x_shape) - 1))


def linear_beta_schedule(timesteps: int) -> torch.Tensor:
    scale = 1000.0 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos((x / timesteps + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - alphas_cumprod[1:] / alphas_cumprod[:-1]
    return torch.clip(betas, 0, 0.999)


class LayerNorm(nn.Module):

    def __init__(self, dim: int, eps: float = 1e-05, stable: bool = False):
        super().__init__()
        self.eps = eps
        self.stable = stable
        self.g = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.stable:
            x = x / x.amax(dim=-1, keepdim=True).detach().clamp(min=self.eps)
        var = torch.var(x, dim=-1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=-1, keepdim=True)
        return (x - mean) * (var + self.eps).rsqrt() * self.g


class MLP(nn.Module):

    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        *,
        expansion_factor: float = 2.0,
        depth: int = 2,
        norm: bool = False,
    ):
        super().__init__()
        hidden_dim = int(expansion_factor * dim_out)
        norm_fn = lambda: nn.LayerNorm(hidden_dim) if norm else nn.Identity()
        layers = [nn.Sequential(nn.Linear(dim_in, hidden_dim), nn.SiLU(), norm_fn())]
        for _ in range(depth - 1):
            layers.append(nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), norm_fn()))
        layers.append(nn.Linear(hidden_dim, dim_out))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.float())


class RelPosBias(nn.Module):

    def __init__(self, heads: int = 8, num_buckets: int = 32, max_distance: int = 128):
        super().__init__()
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.relative_attention_bias = nn.Embedding(num_buckets, heads)

    @staticmethod
    def _relative_position_bucket(
        relative_position: torch.Tensor, num_buckets: int = 32, max_distance: int = 128
    ) -> torch.Tensor:
        n = -relative_position
        n = torch.max(n, torch.zeros_like(n))
        max_exact = num_buckets // 2
        is_small = n < max_exact
        val_if_large = (
            max_exact
            + (
                torch.log(n.float() / max_exact).clamp(min=0)
                / math.log(max_distance / max_exact)
                * (num_buckets - max_exact)
            ).long()
        )
        val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_buckets - 1))
        return torch.where(is_small, n, val_if_large)

    def forward(self, i: int, j: int, *, device: torch.device) -> torch.Tensor:
        q_pos = torch.arange(i, dtype=torch.long, device=device)
        k_pos = torch.arange(j, dtype=torch.long, device=device)
        rel_pos = rearrange(k_pos, "j -> 1 j") - rearrange(q_pos, "i -> i 1")
        rp_bucket = self._relative_position_bucket(
            rel_pos, num_buckets=self.num_buckets, max_distance=self.max_distance
        )
        values = self.relative_attention_bias(rp_bucket)
        return rearrange(values, "i j h -> h i j")


class SwiGLU(nn.Module):

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, gate = x.chunk(2, dim=-1)
        return x * F.silu(gate)


def FeedForward(
    dim: int,
    out_dim: Optional[int] = None,
    mult: int = 4,
    dropout: float = 0.0,
    post_activation_norm: bool = False,
) -> nn.Sequential:
    out_dim = default(out_dim, dim)
    inner_dim = int(mult * dim)
    return nn.Sequential(
        LayerNorm(dim),
        nn.Linear(dim, inner_dim * 2, bias=False),
        SwiGLU(),
        LayerNorm(inner_dim) if post_activation_norm else nn.Identity(),
        nn.Dropout(dropout),
        nn.Linear(inner_dim, out_dim, bias=False),
    )


class SinusoidalPosEmb(nn.Module):

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / max(half_dim - 1, 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x.float()[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1), value=0.0)
        return emb


class Attention(nn.Module):

    def __init__(
        self,
        dim: int,
        kv_dim: Optional[int] = None,
        *,
        out_dim: Optional[int] = None,
        dim_head: int = 64,
        heads: int = 8,
        dropout: float = 0.0,
        causal: bool = False,
        rotary_emb: Optional[nn.Module] = None,
        pb_relax_alpha: float = 128.0,
    ):
        super().__init__()
        self.pb_relax_alpha = pb_relax_alpha
        self.scale = dim_head ** (-0.5) * pb_relax_alpha ** (-1)
        self.heads = heads
        self.dim_head = dim_head
        inner_dim = dim_head * heads
        kv_dim = default(kv_dim, dim)
        self.causal = causal
        self.norm = LayerNorm(dim)
        self.context_norm = LayerNorm(kv_dim) if kv_dim != dim else None
        self.dropout = nn.Dropout(dropout)
        self.null_kv = nn.Parameter(torch.randn(2, heads, dim_head))
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(kv_dim, inner_dim * 2, bias=False)
        self.rotary_emb = rotary_emb
        out_dim = default(out_dim, dim)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, out_dim, bias=False), LayerNorm(out_dim))

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        attn_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        b, _, device = (*x.shape[:2], x.device)
        x_norm = self.norm(x)
        if context is None:
            context = x_norm
        else:
            context = context.float()
            if self.context_norm is not None:
                context = self.context_norm(context)
        q = self.to_q(x_norm)
        k, v = self.to_kv(context).chunk(2, dim=-1)
        q = rearrange(q, "b n (h d) -> b h n d", h=self.heads)
        k = rearrange(k, "b n (h d) -> b h n d", h=self.heads)
        v = rearrange(v, "b n (h d) -> b h n d", h=self.heads)
        q = q * self.scale
        if exists(self.rotary_emb):
            q, k = map(self.rotary_emb.rotate_queries_or_keys, (q, k))
        nk = repeat(self.null_kv[0], "h d -> b h 1 d", b=b)
        nv = repeat(self.null_kv[1], "h d -> b h 1 d", b=b)
        k = torch.cat((nk, k), dim=-2)
        v = torch.cat((nv, v), dim=-2)
        sim = einsum("b h i d, b h j d -> b h i j", q, k)
        if exists(attn_bias):
            sim = sim + attn_bias.unsqueeze(0)
        max_neg_value = -torch.finfo(sim.dtype).max
        if exists(mask):
            mask = F.pad(mask.bool(), (1, 0), value=True)
            mask = rearrange(mask, "b j -> b 1 1 j")
            sim = sim.masked_fill(~mask, max_neg_value)
        if self.causal:
            i, j = sim.shape[-2:]
            causal_mask = torch.ones((i, j), dtype=torch.bool, device=device).triu(j - i + 1)
            sim = sim.masked_fill(causal_mask, max_neg_value)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        sim = sim * self.pb_relax_alpha
        attn = sim.softmax(dim=-1)
        attn = self.dropout(attn)
        out = einsum("b h i j, b h j d -> b h i d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class CausalTransformer(nn.Module):

    def __init__(
        self,
        dim: int,
        depth: int,
        dim_in_out: Optional[int] = None,
        cross_attn: bool = True,
        point_feature_dim: Optional[int] = None,
        dim_head: int = 64,
        heads: int = 8,
        ff_mult: int = 4,
        norm_in: bool = False,
        norm_out: bool = True,
        attn_dropout: float = 0.0,
        ff_dropout: float = 0.3,
        final_proj: bool = True,
        normformer: bool = False,
        rotary_emb: bool = True,
        causal_self_attn: bool = True,
        **_: Any,
    ):
        super().__init__()
        self.init_norm = LayerNorm(dim) if norm_in else nn.Identity()
        self.rel_pos_bias = RelPosBias(heads=heads)
        rotary = (
            RotaryEmbedding(dim=min(32, dim_head))
            if rotary_emb and RotaryEmbedding is not None
            else None
        )
        rotary_cross = (
            RotaryEmbedding(dim=min(32, dim_head))
            if rotary_emb and RotaryEmbedding is not None
            else None
        )
        self.cross_attn = cross_attn
        context_dim = default(point_feature_dim, dim)
        self.layers = nn.ModuleList()
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        Attention(
                            dim=dim,
                            causal=causal_self_attn,
                            dim_head=dim_head,
                            heads=heads,
                            dropout=attn_dropout,
                            rotary_emb=rotary,
                        ),
                        (
                            Attention(
                                dim=dim,
                                kv_dim=context_dim,
                                causal=False,
                                dim_head=dim_head,
                                heads=heads,
                                dropout=attn_dropout,
                                rotary_emb=rotary_cross,
                            )
                            if cross_attn
                            else nn.Identity()
                        ),
                        FeedForward(
                            dim=dim,
                            out_dim=dim,
                            mult=ff_mult,
                            dropout=ff_dropout,
                            post_activation_norm=normformer,
                        ),
                    ]
                )
            )
        self.norm = LayerNorm(dim, stable=True) if norm_out else nn.Identity()
        self.project_out = nn.Linear(dim, dim, bias=False) if final_proj else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        time_emb: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        del time_emb
        n, device = (x.shape[1], x.device)
        x = self.init_norm(x)
        attn_bias = self.rel_pos_bias(n, n + 1, device=device)
        for self_attn, cross_attn, ff in self.layers:
            x = x + self_attn(x, attn_bias=attn_bias)
            if self.cross_attn and context is not None:
                x = x + cross_attn(x, context=context)
            x = x + ff(x)
        return self.project_out(self.norm(x))


class DiffusionNet(nn.Module):

    def __init__(
        self,
        dim: int,
        depth: int,
        dim_in_out: Optional[int] = None,
        num_timesteps: Optional[int] = None,
        num_time_embeds: int = 1,
        cond: bool = True,
        **kwargs: Any,
    ):
        super().__init__()
        self.num_time_embeds = num_time_embeds
        self.model_dim = dim
        self.depth = depth
        self.cond = cond
        self.cross_attn = bool(kwargs.pop("cross_attn", True))
        self.cond_dropout = float(kwargs.pop("cond_dropout", 0.0) or 0.0)
        self.dim_in_out = default(dim_in_out, dim)
        self.point_feature_dim = kwargs.pop("point_feature_dim", self.dim_in_out)
        self.to_time_embeds = nn.Sequential(
            nn.Embedding(num_timesteps, self.model_dim * num_time_embeds)
            if exists(num_timesteps)
            else nn.Sequential(
                SinusoidalPosEmb(self.model_dim),
                MLP(self.model_dim, self.model_dim * num_time_embeds),
            )
        )
        self.input_proj = (
            nn.Linear(self.dim_in_out, self.model_dim)
            if self.dim_in_out != self.model_dim
            else nn.Identity()
        )
        self.output_proj = (
            nn.Linear(self.model_dim, self.dim_in_out)
            if self.dim_in_out != self.model_dim
            else nn.Identity()
        )
        self.learned_query = nn.Parameter(torch.randn(self.model_dim))
        self.causal_transformer = CausalTransformer(
            dim=self.model_dim,
            depth=self.depth,
            cross_attn=self.cross_attn and self.cond,
            point_feature_dim=self.point_feature_dim,
            **kwargs,
        )

    def _prepare_context(
        self,
        cond_feature: Optional[torch.Tensor],
        batch_size: int,
        device: torch.device,
        pass_cond: int,
    ) -> Optional[torch.Tensor]:
        if not (self.cond and self.cross_attn):
            return None
        if pass_cond == 0 or cond_feature is None:
            return None
        context = cond_feature
        if context.ndim == 2:
            context = context.unsqueeze(1)
        context = context.to(device=device, dtype=torch.float32)
        if self.training and self.cond_dropout > 0.0 and (pass_cond < 0):
            drop_mask = (torch.rand(batch_size, device=device) < self.cond_dropout).view(
                batch_size, 1, 1
            )
            context = torch.where(drop_mask, torch.zeros_like(context), context)
        return context

    def forward(
        self,
        data: torch.Tensor,
        diffusion_timesteps: torch.Tensor,
        cond_feature: Optional[torch.Tensor] = None,
        pass_cond: int = -1,
    ) -> torch.Tensor:
        batch_size = data.shape[0]
        device = data.device
        time_embed = self.to_time_embeds(diffusion_timesteps)
        time_embed = time_embed.view(batch_size, self.num_time_embeds, self.model_dim)
        data_token = self.input_proj(data.float()).unsqueeze(1)
        learned_queries = repeat(self.learned_query, "d -> b 1 d", b=batch_size)
        tokens = torch.cat([time_embed, data_token, learned_queries], dim=1)
        context = self._prepare_context(cond_feature, batch_size, device, pass_cond)
        tokens = self.causal_transformer(tokens, context=context)
        pred = tokens[:, -1, :]
        return self.output_proj(pred)


ModelPrediction = namedtuple("ModelPrediction", ["pred_noise", "pred_x_start"])


class BaseDiffusionModel(nn.Module):

    def __init__(
        self,
        latent_dim: int = 1024,
        model_dim: Optional[int] = None,
        depth: int = 4,
        point_feature_dim: Optional[int] = None,
        timesteps: int = 1000,
        sampling_timesteps: Optional[int] = None,
        beta_schedule: str = "cosine",
        sample_pc_size: int = 128,
        perturb_pc: Optional[str] = "partial",
        crop_percent: float = 0.5,
        loss_type: str = "l2",
        objective: str = "pred_x0",
        data_scale: float = 1.0,
        data_shift: float = 0.0,
        p2_loss_weight_gamma: float = 0.0,
        p2_loss_weight_k: float = 1.0,
        ddim_sampling_eta: float = 1.0,
        cond_dropout: float = 0.0,
        num_time_embeds: int = 1,
        learned_sinusoidal_t: bool = False,
        **model_kwargs: Any,
    ):
        super().__init__()
        del learned_sinusoidal_t
        self.latent_dim = latent_dim
        self.model = DiffusionNet(
            dim=default(model_dim, latent_dim),
            depth=depth,
            dim_in_out=latent_dim,
            num_timesteps=None,
            num_time_embeds=num_time_embeds,
            cond=True,
            point_feature_dim=default(point_feature_dim, latent_dim),
            cond_dropout=cond_dropout,
            **model_kwargs,
        )
        self.objective = objective
        if beta_schedule == "linear":
            betas = linear_beta_schedule(timesteps)
        elif beta_schedule == "cosine":
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f"Unknown beta_schedule: {beta_schedule}")
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        (timesteps_,) = betas.shape
        self.num_timesteps = int(timesteps_)
        self.pc_size = sample_pc_size
        self.perturb_pc = perturb_pc
        self.crop_percent = crop_percent
        self.loss_fn = F.l1_loss if loss_type == "l1" else F.mse_loss
        self.sampling_timesteps = default(sampling_timesteps, timesteps_)
        self.ddim_sampling_eta = ddim_sampling_eta
        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))
        register_buffer("data_scale", torch.tensor(float(data_scale)))
        register_buffer("data_shift", torch.tensor(float(data_shift)))
        register_buffer("betas", betas)
        register_buffer("alphas_cumprod", alphas_cumprod)
        register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        register_buffer("log_one_minus_alphas_cumprod", torch.log(1.0 - alphas_cumprod))
        register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod))
        register_buffer("sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1))
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        register_buffer("posterior_variance", posterior_variance)
        register_buffer(
            "posterior_log_variance_clipped", torch.log(posterior_variance.clamp(min=1e-20))
        )
        register_buffer(
            "posterior_mean_coef1", betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        register_buffer(
            "posterior_mean_coef2",
            (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod),
        )
        register_buffer(
            "p2_loss_weight",
            (p2_loss_weight_k + alphas_cumprod / (1 - alphas_cumprod)) ** (-p2_loss_weight_gamma),
        )

    @property
    def device(self) -> torch.device:
        return self.betas.device

    def predict_start_from_noise(
        self, x_t: torch.Tensor, t: torch.Tensor, noise: torch.Tensor
    ) -> torch.Tensor:
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_start(
        self, x_t: torch.Tensor, t: torch.Tensor, x0: torch.Tensor
    ) -> torch.Tensor:
        # Invert x0 = A * x_t - B * noise; DDIM adds the recovered noise.
        return (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / extract(
            self.sqrt_recipm1_alphas_cumprod, t, x_t.shape
        )

    def model_predictions(
        self,
        model_input: Union[torch.Tensor, Tuple[torch.Tensor, Optional[torch.Tensor]]],
        t: torch.Tensor,
    ) -> ModelPrediction:
        if isinstance(model_input, tuple):
            x, cond = model_input
        else:
            x, cond = (model_input, None)
        model_output = self.model(x, t, cond, pass_cond=1)
        if self.objective == "pred_noise":
            pred_noise = model_output
            x_start = self.predict_start_from_noise(x, t, model_output)
        elif self.objective == "pred_x0":
            pred_noise = self.predict_noise_from_start(x, t, model_output)
            x_start = model_output
        else:
            raise ValueError(f"unknown objective {self.objective}")
        return ModelPrediction(pred_noise, x_start)

    @torch.no_grad()
    def ddim_sample(
        self,
        dim: Optional[int] = None,
        batch_size: int = 1,
        noise: Optional[torch.Tensor] = None,
        clip_denoised: bool = True,
        traj: bool = False,
        cond: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, list]]:
        dim = default(dim, self.model.dim_in_out)
        x_t = default(noise, lambda: torch.randn(batch_size, dim, device=self.device))
        trajectories = [] if traj else None
        total_timesteps = self.num_timesteps
        sampling_timesteps = self.sampling_timesteps
        eta = self.ddim_sampling_eta
        times = torch.linspace(
            0.0, total_timesteps, steps=sampling_timesteps + 2, device=self.device
        )[:-1]
        times = list(reversed(times.long().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))
        for time, time_next in tqdm(time_pairs, desc="sampling loop time step"):
            alpha = self.alphas_cumprod_prev[time]
            alpha_next = self.alphas_cumprod_prev[time_next]
            time_cond = torch.full((batch_size,), time, device=self.device, dtype=torch.long)
            pred = self.model_predictions((x_t, cond) if cond is not None else x_t, time_cond)
            pred_noise, x_start = (pred.pred_noise, pred.pred_x_start)
            if clip_denoised:
                x_start = x_start.clamp(-1.0, 1.0)
            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma**2).sqrt()
            noise_term = torch.randn_like(x_t) if time_next > 0 else 0.0
            x_t = x_start * alpha_next.sqrt() + c * pred_noise + sigma * noise_term
            if traj:
                trajectories.append(x_t.clone())
        return (x_t, trajectories) if traj else x_t


@dataclass
class PoseTokenDiffusionConfig:
    pose_dim: int = 14
    hand_code_dim: int = 512
    object_code_dim: int = 1
    cond_token_dim: int = 256
    model_dim: int = 384
    depth: int = 6
    timesteps: int = 1000
    sampling_timesteps: int = 50
    beta_schedule: str = "cosine"
    objective: str = "pred_x0"
    loss_type: str = "l2"
    cond_dropout: float = 0.1
    attn_dropout: float = 0.0
    ff_dropout: float = 0.1
    dim_head: int = 64
    heads: int = 8
    ff_mult: int = 4
    num_time_embeds: int = 1
    data_scale: float = 1.0
    data_shift: float = 0.0
    ddim_sampling_eta: float = 0.0
    use_task_token: bool = True


class TokenMLP(nn.Module):

    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim_in, dim_out), nn.SiLU(), nn.Linear(dim_out, dim_out))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.float())


class PoseTokenDiffusionModel(nn.Module):

    def __init__(self, config: PoseTokenDiffusionConfig):
        super().__init__()
        self.config = config
        self.hand_tokenizer = TokenMLP(config.hand_code_dim, config.cond_token_dim)
        self.object_tokenizer = TokenMLP(config.object_code_dim, config.cond_token_dim)
        self.token_type_embed = nn.Parameter(torch.randn(2, config.cond_token_dim) * 0.02)
        self.task_type_embed = nn.Parameter(torch.randn(2, config.cond_token_dim) * 0.02)
        self.diffusion = BaseDiffusionModel(
            latent_dim=config.pose_dim,
            model_dim=config.model_dim,
            depth=config.depth,
            point_feature_dim=config.cond_token_dim,
            timesteps=config.timesteps,
            sampling_timesteps=config.sampling_timesteps,
            beta_schedule=config.beta_schedule,
            objective=config.objective,
            loss_type=config.loss_type,
            cond_dropout=config.cond_dropout,
            attn_dropout=config.attn_dropout,
            ff_dropout=config.ff_dropout,
            dim_head=config.dim_head,
            heads=config.heads,
            ff_mult=config.ff_mult,
            num_time_embeds=config.num_time_embeds,
            data_scale=config.data_scale,
            data_shift=config.data_shift,
            ddim_sampling_eta=config.ddim_sampling_eta,
        )

    @property
    def device(self) -> torch.device:
        return self.diffusion.device

    def encode_condition_tokens(
        self,
        hand_code: torch.Tensor,
        object_code: torch.Tensor,
        task_code: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        hand_token = self.hand_tokenizer(hand_code) + self.token_type_embed[0]
        object_token = self.object_tokenizer(object_code) + self.token_type_embed[1]
        tokens = [hand_token, object_token]
        if self.config.use_task_token:
            if task_code is None:
                task_code = torch.zeros(
                    hand_code.shape[0], device=hand_code.device, dtype=torch.long
                )
            task_code = (
                task_code.to(device=hand_code.device, dtype=torch.long).reshape(-1).clamp(0, 1)
            )
            tokens.append(self.task_type_embed[task_code])
        return torch.stack(tokens, dim=1)

    @torch.no_grad()
    def generate_latent(self, hand_code, object_code, *, noise, task_code):
        tokens = self.encode_condition_tokens(hand_code, object_code, task_code)
        return self.diffusion.ddim_sample(
            dim=self.config.pose_dim, batch_size=hand_code.shape[0], noise=noise, cond=tokens
        )
