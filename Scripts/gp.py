from __future__ import annotations
from typing import Any, Dict, List, Mapping, Tuple
from dataclasses import dataclass
from pathlib import Path
import math
import json
import numpy as np
from safetensors.torch import load_file
import torch
import gpytorch
import linear_operator

POSE_DIM = 12
POSE_ROT_DIM = 9
_PROJECTION_CACHE = {}
GP_MAX_CHOLESKY_SIZE = 4096
GP_CHOLESKY_JITTER = 0.0001
GP_MAX_CG_ITERATIONS = 10000
GP_CG_TOLERANCE = 0.05


@dataclass(frozen=True)
class GPFeatureConfig:
    hand_projection_dim: int = 24
    object_projection_dim: int = 24
    projection_seed: int = 42
    geometry_hand_points: int = 32
    geometry_object_points: int = 128
    geometry_chunk_size: int = 256
    pressure_threshold: float = 0.0

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "GPFeatureConfig":
        value = value or {}
        return cls(
            hand_projection_dim=int(value.get("hand_projection_dim", 24)),
            object_projection_dim=int(value.get("object_projection_dim", 24)),
            projection_seed=int(value.get("projection_seed", 42)),
            geometry_hand_points=int(value.get("geometry_hand_points", 32)),
            geometry_object_points=int(value.get("geometry_object_points", 128)),
            geometry_chunk_size=int(value.get("geometry_chunk_size", 256)),
            pressure_threshold=float(value.get("pressure_threshold", 0.0)),
        )


def _projection_matrix(
    input_dim: int, output_dim: int, *, seed: int, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    if output_dim <= 0 or output_dim >= input_dim:
        return torch.eye(input_dim, device=device, dtype=dtype)
    key = (input_dim, output_dim, seed, str(device), str(dtype))
    cached = _PROJECTION_CACHE.get(key)
    if cached is not None:
        return cached
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    matrix = torch.randn(input_dim, output_dim, generator=generator, dtype=torch.float32)
    matrix = matrix / math.sqrt(float(output_dim))
    matrix = matrix.to(device=device, dtype=dtype)
    _PROJECTION_CACHE[key] = matrix
    return matrix


def _compress_code(code: torch.Tensor, output_dim: int, seed: int) -> torch.Tensor:
    if output_dim <= 0 or output_dim >= code.shape[1]:
        return code.float()
    projection = _projection_matrix(
        int(code.shape[1]), int(output_dim), seed=int(seed), device=code.device, dtype=code.dtype
    )
    return code.float() @ projection.float()


def deterministic_diffusion_noise(
    sample_indices: torch.Tensor, latent_dim: int, *, seed: int, device: torch.device
) -> torch.Tensor:
    """Independent N(0, 1) components, reproducible per (seed, sample index).

    Each sample uses a separate Philox key instead of small offsets to a shared
    uniform variate. Generate float32 noise on CPU, then transfer it so the same
    sample has identical noise across batch orders, batch sizes and devices.
    This local generator does not consume NumPy's or PyTorch's global RNG state.
    """
    latent_dim = int(latent_dim)
    if latent_dim < 0:
        raise ValueError("latent_dim must be non-negative")
    indices = sample_indices.detach().to(device="cpu", dtype=torch.int64).reshape(-1).tolist()
    noise = np.empty((len(indices), latent_dim), dtype=np.float32)
    uint64_mask = (1 << 64) - 1
    seed_key = (int(seed) & uint64_mask) << 64
    for row, sample_index in enumerate(indices):
        key = seed_key | (int(sample_index) & uint64_mask)
        generator = np.random.Generator(np.random.Philox(key=key))
        noise[row] = generator.standard_normal(latent_dim, dtype=np.float32)
    return torch.from_numpy(noise).to(device=device)


def _batch_project_to_so3(rot9: torch.Tensor) -> torch.Tensor:
    matrices = rot9.reshape(-1, 3, 3).float()
    u, _, vh = torch.linalg.svd(matrices)
    projected = u @ vh
    determinant_sign = torch.where(
        torch.det(projected) < 0,
        -torch.ones(projected.shape[0], device=projected.device, dtype=projected.dtype),
        torch.ones(projected.shape[0], device=projected.device, dtype=projected.dtype),
    )
    correction = torch.diag_embed(
        torch.stack(
            [
                torch.ones_like(determinant_sign),
                torch.ones_like(determinant_sign),
                determinant_sign,
            ],
            dim=1,
        )
    )
    return u @ correction @ vh


def project_latent_rotation(latent14: torch.Tensor) -> torch.Tensor:
    rotation = _batch_project_to_so3(latent14[:, :POSE_ROT_DIM]).reshape(-1, POSE_ROT_DIM)
    return torch.cat([rotation, latent14[:, POSE_ROT_DIM:]], dim=1)


def _skew(vector: torch.Tensor) -> torch.Tensor:
    x, y, z = vector.unbind(dim=-1)
    zero = torch.zeros_like(x)
    return torch.stack(
        [
            torch.stack([zero, -z, y], dim=-1),
            torch.stack([z, zero, -x], dim=-1),
            torch.stack([-y, x, zero], dim=-1),
        ],
        dim=-2,
    )


def so3_exp_map(rotvec: torch.Tensor) -> torch.Tensor:
    theta2 = (rotvec * rotvec).sum(dim=-1, keepdim=True)
    theta = torch.sqrt(theta2.clamp_min(1e-16))
    small = theta2 < 1e-08
    a = torch.where(
        small,
        1.0 - theta2 / 6.0 + theta2 * theta2 / 120.0,
        torch.sin(theta) / theta.clamp_min(1e-08),
    )
    b = torch.where(
        small,
        0.5 - theta2 / 24.0 + theta2 * theta2 / 720.0,
        (1.0 - torch.cos(theta)) / theta2.clamp_min(1e-08),
    )
    k = _skew(rotvec)
    eye = torch.eye(3, device=rotvec.device, dtype=rotvec.dtype).expand(k.shape)
    return eye + a[..., None] * k + b[..., None] * (k @ k)


def so3_log_map(rotation: torch.Tensor) -> torch.Tensor:
    trace = rotation[..., 0, 0] + rotation[..., 1, 1] + rotation[..., 2, 2]
    cos_theta = ((trace - 1.0) * 0.5).clamp(-1.0 + 1e-06, 1.0 - 1e-06)
    theta = torch.acos(cos_theta)
    vee = torch.stack(
        [
            rotation[..., 2, 1] - rotation[..., 1, 2],
            rotation[..., 0, 2] - rotation[..., 2, 0],
            rotation[..., 1, 0] - rotation[..., 0, 1],
        ],
        dim=-1,
    )
    scale = theta / (2.0 * torch.sin(theta).clamp_min(1e-06))
    small = theta.abs() < 0.0001
    scale = torch.where(small, torch.full_like(scale, 0.5), scale)
    return vee * scale.unsqueeze(-1)


def _sample_evenly(points: torch.Tensor, count: int) -> torch.Tensor:
    if points.shape[1] <= count:
        return points
    indices = (
        torch.linspace(0, points.shape[1] - 1, steps=count, device=points.device).round().long()
    )
    return points.index_select(1, indices)


def _geometry_features(
    hand_batch: torch.Tensor,
    object_batch: torch.Tensor,
    pred14: torch.Tensor,
    config: GPFeatureConfig,
) -> torch.Tensor:
    outputs: List[torch.Tensor] = []
    chunk_size = max(1, int(config.geometry_chunk_size))
    for start in range(0, hand_batch.shape[0], chunk_size):
        hand = hand_batch[start : start + chunk_size].float()
        obj = object_batch[start : start + chunk_size].float()
        pred = pred14[start : start + chunk_size].float()
        points = hand[..., :3]
        pressure = hand[..., 3].clamp_min(0.0)
        k = min(max(1, int(config.geometry_hand_points)), points.shape[1])
        top_pressure, top_idx = torch.topk(pressure, k=k, dim=1, largest=True, sorted=False)
        gather_idx = top_idx.unsqueeze(-1).expand(-1, -1, 3)
        contact_world = torch.gather(points, 1, gather_idx)
        rotation = _batch_project_to_so3(pred[:, :POSE_ROT_DIM])
        translation = pred[:, POSE_ROT_DIM:POSE_DIM]
        contact_local = (contact_world - translation[:, None, :]) @ rotation
        object_local = _sample_evenly(obj[..., :3], max(1, int(config.geometry_object_points)))
        nearest = torch.cdist(contact_local, object_local).amin(dim=2)
        weights = top_pressure + 1e-06
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-06)
        weighted_distance = (nearest * weights).sum(dim=1)
        positive_fraction = (pressure > float(config.pressure_threshold)).float().mean(dim=1)
        weighted_centroid = (contact_local * weights.unsqueeze(-1)).sum(dim=1)
        weighted_spread = torch.sqrt(
            ((contact_local - weighted_centroid[:, None, :]).square() * weights.unsqueeze(-1))
            .sum(dim=1)
            .clamp_min(1e-12)
        )
        scalar_features = torch.stack(
            [
                nearest.mean(dim=1),
                nearest.amin(dim=1),
                nearest.amax(dim=1),
                weighted_distance,
                pressure.mean(dim=1),
                pressure.amax(dim=1),
                positive_fraction,
                pressure.sum(dim=1) / float(max(pressure.shape[1], 1)),
            ],
            dim=1,
        )
        features = torch.cat(
            [
                contact_local.mean(dim=1),
                contact_local.std(dim=1),
                weighted_centroid,
                weighted_spread,
                scalar_features,
            ],
            dim=1,
        )
        outputs.append(features)
    return torch.cat(outputs, dim=0)


def build_gp_features(
    hand_code: torch.Tensor,
    object_code: torch.Tensor,
    hand_batch: torch.Tensor,
    object_batch: torch.Tensor,
    pred_latent14: torch.Tensor,
    feature_config: Mapping[str, Any] | GPFeatureConfig | None = None,
) -> torch.Tensor:
    config = (
        feature_config
        if isinstance(feature_config, GPFeatureConfig)
        else GPFeatureConfig.from_mapping(feature_config)
    )
    hand_small = _compress_code(hand_code, config.hand_projection_dim, config.projection_seed)
    object_small = _compress_code(
        object_code, config.object_projection_dim, config.projection_seed + 1
    )
    pred_valid = project_latent_rotation(pred_latent14)
    geometry = _geometry_features(hand_batch, object_batch, pred_valid, config)
    return torch.cat([hand_small, object_small, pred_valid, geometry], dim=1)


def apply_gp_residual_to_latent(
    pred14: torch.Tensor, residual8: torch.Tensor, residual_mask8: torch.Tensor
) -> torch.Tensor:
    out = project_latent_rotation(pred14)
    pose_valid = residual_mask8[:, 0] > 0.5
    if pose_valid.any():
        base_rotation = out[pose_valid, :POSE_ROT_DIM].reshape(-1, 3, 3)
        delta_rotation = so3_exp_map(residual8[pose_valid, :3])
        corrected_rotation = delta_rotation @ base_rotation
        out[pose_valid, :POSE_ROT_DIM] = corrected_rotation.reshape(-1, POSE_ROT_DIM)
        out[pose_valid, POSE_ROT_DIM:POSE_DIM] += residual8[pose_valid, 3:6]
    force_valid = residual_mask8[:, 6] > 0.5
    if force_valid.any():
        out[force_valid, POSE_DIM : POSE_DIM + 2] += residual8[force_valid, 6:8]
    return out


GP_GROUPS: Dict[str, Tuple[int, int]] = {
    "rotation": (0, 3),
    "translation": (3, 6),
    "force_torque": (6, 8),
}


class GroupedResidualGPModel(gpytorch.models.ExactGP):

    def __init__(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        likelihood: gpytorch.likelihoods.MultitaskGaussianLikelihood,
        *,
        coregionalization_rank: int = 1,
    ):
        super().__init__(train_x, train_y, likelihood)
        num_tasks = int(train_y.shape[1])
        feature_dim = int(train_x.shape[1])
        self.num_tasks = num_tasks
        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ConstantMean(), num_tasks=num_tasks
        )
        data_kernel = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(nu=1.5, ard_num_dims=feature_dim)
        )
        self.covar_module = gpytorch.kernels.MultitaskKernel(
            data_kernel,
            num_tasks=num_tasks,
            rank=max(1, min(int(coregionalization_rank), num_tasks)),
        )

    def forward(self, x: torch.Tensor) -> gpytorch.distributions.MultitaskMultivariateNormal:
        return gpytorch.distributions.MultitaskMultivariateNormal(
            self.mean_module(x), self.covar_module(x)
        )


def _posterior(
    model: GroupedResidualGPModel, x: torch.Tensor
) -> gpytorch.distributions.MultitaskMultivariateNormal:
    with gpytorch.settings.max_cholesky_size(
        GP_MAX_CHOLESKY_SIZE
    ), gpytorch.settings.fast_pred_var(), gpytorch.settings.cholesky_jitter(
        GP_CHOLESKY_JITTER
    ), linear_operator.settings.max_cg_iterations(
        GP_MAX_CG_ITERATIONS
    ), linear_operator.settings.cg_tolerance(
        GP_CG_TOLERANCE
    ):
        return model(x)


def load_grouped_gp_models(
    *, model_dir: Path, device: torch.device
) -> Tuple[
    List[GroupedResidualGPModel],
    List[gpytorch.likelihoods.MultitaskGaussianLikelihood],
    Dict[str, Any],
]:
    from gpytorch.constraints import GreaterThan
    from gpytorch.likelihoods import MultitaskGaussianLikelihood

    stats = load_file(str(model_dir / "gp_normalization.safetensors"))
    stats.update(json.loads((model_dir / "gp.json").read_text(encoding="utf-8")))
    models, likelihoods = [], []
    for name, (start, end) in GP_GROUPS.items():
        tensors = load_file(str(model_dir / f"gp_{name}.safetensors"), device=str(device))
        state = {key.removeprefix("state."): value for key, value in tensors.items()
                 if key.startswith("state.")}
        model_rank = int(state["covar_module.task_covar_module.covar_factor"].shape[-1])
        likelihood_rank = int(state["likelihood.task_noise_covar_factor"].shape[-1])
        likelihood = MultitaskGaussianLikelihood(
            num_tasks=end - start, rank=likelihood_rank,
            noise_constraint=GreaterThan(0.0001),
        ).to(device)
        model = GroupedResidualGPModel(
            tensors["reference_inputs"], tensors["reference_residuals"], likelihood,
            coregionalization_rank=model_rank,
        ).to(device)
        model.load_state_dict(state)
        model.eval()
        likelihood.eval()
        models.append(model)
        likelihoods.append(likelihood)
    return models, likelihoods, stats


@torch.no_grad()
def correct_batch(diffusion, models, stats, data, config, device):
    hand_code = data["hand_code"].to(device)
    object_code = data["object_code"].to(device)
    pose_mask = data["mask_pose"].to(device).reshape(-1, 1)
    force_mask = data["mask_force"].to(device).reshape(-1, 1)
    noise = deterministic_diffusion_noise(
        data["index"], 14, seed=config["fixed_noise_seed"], device=device
    )
    pred = diffusion.generate_latent(
        hand_code, object_code, noise=noise, task_code=(force_mask[:, 0] > 0.5).long()
    )
    features = build_gp_features(
        hand_code,
        object_code,
        data["hand_som"].to(device),
        data["object_data"].to(device),
        pred,
        config["features"],
    )
    features = (features - stats["x_center"].to(device)) / stats["x_scale"].to(device)
    mean_parts, gate_parts = ([], [])
    for model, (name, (start, end)) in zip(models, GP_GROUPS.items()):
        posterior = _posterior(model, features)
        mean = posterior.mean
        std = posterior.variance.clamp_min(1e-09).sqrt()
        policy = stats["group_policies"][name]
        if policy["gate_mode"] == "none":
            gate = torch.ones(std.shape[0], 1, device=device)
        else:
            gate = torch.sigmoid(
                (float(policy["std_threshold"]) - std.mean(dim=1))
                / max(float(policy["gate_temperature"]), 0.0001)
            ).unsqueeze(1)
        gate = gate * (float(policy["alpha"]) if policy["enabled"] else 0.0)
        mean_parts.append(mean)
        gate_parts.append(gate.expand(-1, end - start))
    mask = torch.cat([pose_mask.expand(-1, 6), force_mask.expand(-1, 2)], dim=1)
    gate = torch.cat(gate_parts, dim=1)
    residual = torch.cat(mean_parts, dim=1) * stats["y_scale"].to(device) + stats["y_center"].to(
        device
    )
    limit = config["max_correction_sigma"] * stats["y_scale"].to(device)
    residual = torch.maximum(torch.minimum(residual, limit), -limit) * gate * mask
    corrected = apply_gp_residual_to_latent(pred, residual, mask)
    saved = {key: data[key] for key in ("hand_som", "object_data", "mask_pose")}
    saved["gp_gate"] = (gate * mask).cpu()
    saved["gp_residual8"] = residual.cpu()
    return (pred.cpu(), corrected.cpu(), saved)
