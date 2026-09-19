from __future__ import annotations
from typing import Any, Dict, List, Mapping, Tuple
from dataclasses import dataclass
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from typing import Iterable
from gp import so3_log_map
from safetensors.torch import load_file

POSE_DIM = 12
POSE_ROT_DIM = 9
DEFAULT_STAGE_FOUR_SKIP_OBJECTS = {"force sensor"}


def to_torch(x: Any, device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    if torch.is_tensor(x):
        return x.to(device=device, dtype=dtype)
    return torch.as_tensor(x, device=device, dtype=dtype)


def axis_angle_to_matrix(rotvec: torch.Tensor) -> torch.Tensor:
    theta2 = (rotvec * rotvec).sum(dim=-1, keepdim=True)
    theta = torch.sqrt(theta2.clamp_min(1e-16))
    x, y, z = rotvec.unbind(dim=-1)
    zero = torch.zeros_like(x)
    K = torch.stack(
        [
            torch.stack([zero, -z, y], dim=-1),
            torch.stack([z, zero, -x], dim=-1),
            torch.stack([-y, x, zero], dim=-1),
        ],
        dim=-2,
    )
    I = torch.eye(3, device=rotvec.device, dtype=rotvec.dtype).expand(K.shape)
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
    return I + a[..., None] * K + b[..., None] * (K @ K)


def project_to_so3(R: torch.Tensor) -> torch.Tensor:
    U, _, Vh = torch.linalg.svd(R)
    R_proj = U @ Vh
    determinant = torch.det(R_proj)
    correction_diagonal = torch.ones((*R.shape[:-2], 3), device=R.device, dtype=R.dtype)
    correction_diagonal[..., -1] = torch.where(
        determinant < 0.0, -torch.ones_like(determinant), torch.ones_like(determinant)
    )
    correction = torch.diag_embed(correction_diagonal)
    return U @ correction @ Vh


@dataclass
class RigidPose:
    R: np.ndarray
    t: np.ndarray


@dataclass
class SDFGridMeta:
    grid_shape: Tuple[int, int, int]


class SDFGrid:

    def __init__(
        self, sdf_grid: torch.Tensor, origin_center: np.ndarray, pitch: float, meta: SDFGridMeta
    ):
        self.sdf_grid = sdf_grid
        self.origin_center = np.asarray(origin_center, dtype=np.float32)
        self.pitch = float(pitch)
        self.meta = meta

    def sample_with_details(
        self, points_local: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        output_shape = points_local.shape[:-1]
        points_flat = points_local.reshape(-1, 3)
        D, H, W = self.meta.grid_shape
        origin = to_torch(self.origin_center, device=points_flat.device)
        pitch = torch.tensor(self.pitch, device=points_flat.device, dtype=points_flat.dtype)
        idx = (points_flat - origin[None, :]) / pitch
        ix = 2.0 * (idx[:, 2] / max(W - 1, 1)) - 1.0
        iy = 2.0 * (idx[:, 1] / max(H - 1, 1)) - 1.0
        iz = 2.0 * (idx[:, 0] / max(D - 1, 1)) - 1.0
        grid = torch.stack([ix, iy, iz], dim=-1).view(1, -1, 1, 1, 3)
        val = F.grid_sample(
            self.sdf_grid.to(points_local.device),
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        val = val.view(-1)
        max_index = torch.tensor(
            [max(D - 1, 0), max(H - 1, 0), max(W - 1, 0)],
            device=points_local.device,
            dtype=points_local.dtype,
        )
        below = torch.relu(-idx)
        above = torch.relu(idx - max_index[None, :])
        outside_distance = torch.linalg.norm((below + above) * pitch, dim=1)
        inside = outside_distance <= 0.0
        signed = torch.where(inside, val, val.abs() + outside_distance)
        return (
            signed.reshape(output_shape),
            outside_distance.reshape(output_shape),
            inside.reshape(output_shape),
        )

    def sample(self, points_local: torch.Tensor) -> torch.Tensor:
        return self.sample_with_details(points_local)[0]


@dataclass
class ProjectionConfig:
    steps: int = 120
    lr_rot: float = 0.01
    lr_trans: float = 0.002
    w_sdf: float = 1.0
    w_penetration: float = 2.0
    w_t_reg: float = 0.005
    w_r_reg: float = 0.005
    penetration_tolerance: float = 0.001
    robust_delta: float = 0.01
    print_every: int = 20
    early_stop_patience: int = 12
    early_stop_min_delta: float = 1e-06
    min_sdf_loss: float = 0.0005
    min_sdf_mean_abs: float = 0.002
    min_improvement_abs: float = 1e-05
    min_improvement_ratio: float = 0.03
    max_translation_delta: float = 0.015
    max_rotation_delta_deg: float = 6.0
    blend_alpha_min: float = 0.15
    blend_alpha_max: float = 0.5
    blend_reference_ratio: float = 0.1


@dataclass
class BatchProjectionResult:
    rotations: np.ndarray
    translations: np.ndarray
    applied: np.ndarray
    decisions: List[str]
    blend_alpha: np.ndarray


def huber_loss(x: torch.Tensor, delta: float) -> torch.Tensor:
    abs_x = x.abs()
    delta_t = torch.as_tensor(delta, device=x.device, dtype=x.dtype)
    quad = torch.minimum(abs_x, delta_t)
    lin = abs_x - quad
    return 0.5 * quad**2 + delta_t * lin


def rotation_geodesic_distance_deg(R_a: torch.Tensor, R_b: torch.Tensor) -> torch.Tensor:
    rel = R_a.transpose(-1, -2) @ R_b
    trace = rel[..., 0, 0] + rel[..., 1, 1] + rel[..., 2, 2]
    cos_theta = ((trace - 1.0) * 0.5).clamp(-1.0 + 1e-07, 1.0 - 1e-07)
    return torch.rad2deg(torch.arccos(cos_theta))


def blend_rotation_matrices(
    R_start: torch.Tensor, R_target: torch.Tensor, alpha: torch.Tensor
) -> torch.Tensor:
    alpha = torch.as_tensor(alpha, device=R_start.device, dtype=R_start.dtype)
    return project_to_so3((1.0 - alpha) * R_start + alpha * R_target)


def compute_projection_terms_batch(
    contact_points_world: torch.Tensor,
    R: torch.Tensor,
    t: torch.Tensor,
    R_ref: torch.Tensor,
    t_ref: torch.Tensor,
    sdf_grid: SDFGrid,
    cfg: ProjectionConfig,
    contact_weights: torch.Tensor,
    collision_points_world: torch.Tensor,
    prior_strength: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    xl = torch.matmul(contact_points_world - t[:, None, :], R)
    sdf, outside_distance, _ = sdf_grid.sample_with_details(xl)
    weights = contact_weights.to(device=sdf.device, dtype=sdf.dtype).clamp_min(0.0)
    weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-08)
    loss_sdf = (huber_loss(sdf, cfg.robust_delta) * weights).sum(dim=1)
    collision_local = torch.matmul(collision_points_world - t[:, None, :], R)
    collision_sdf, collision_outside, _ = sdf_grid.sample_with_details(collision_local)
    penetration = torch.relu(-collision_sdf - float(cfg.penetration_tolerance))
    loss_penetration = huber_loss(penetration, cfg.robust_delta).mean(dim=1)
    loss_t_reg = (t - t_ref).square().mean(dim=1)
    loss_r_reg = (R - R_ref).square().mean(dim=(1, 2))
    prior = prior_strength.to(device=R.device, dtype=R.dtype).reshape(-1)
    loss = (
        cfg.w_sdf * loss_sdf
        + cfg.w_penetration * loss_penetration
        + prior * (cfg.w_t_reg * loss_t_reg + cfg.w_r_reg * loss_r_reg)
    )
    return {
        "loss": loss,
        "loss_sdf": loss_sdf,
        "loss_penetration": loss_penetration,
        "loss_t_reg": loss_t_reg,
        "loss_r_reg": loss_r_reg,
        "sdf_mean_abs": sdf.abs().mean(dim=1),
        "sdf_max_abs": sdf.abs().amax(dim=1),
        "penetration_mean": penetration.mean(dim=1),
        "outside_fraction": (outside_distance > 0).float().mean(dim=1),
        "collision_outside_fraction": (collision_outside > 0).float().mean(dim=1),
    }


class PoseProjectionRefiner:

    def __init__(self, config: ProjectionConfig, device: torch.device):
        self.config = config
        self.device = device

    def refine_batch(
        self,
        *,
        rotations: torch.Tensor,
        translations: torch.Tensor,
        contact_points_world: torch.Tensor,
        contact_weights: torch.Tensor,
        collision_points_world: torch.Tensor,
        sdf_grid: SDFGrid,
        prior_strength: torch.Tensor,
    ) -> BatchProjectionResult:
        cfg = self.config
        R0 = rotations.to(self.device, dtype=torch.float32)
        t0 = translations.to(self.device, dtype=torch.float32)
        xw = contact_points_world.to(self.device, dtype=torch.float32)
        weights = contact_weights.to(self.device, dtype=torch.float32)
        collision = collision_points_world.to(self.device, dtype=torch.float32)
        prior = prior_strength.to(self.device, dtype=torch.float32)
        batch_size = int(R0.shape[0])
        with torch.no_grad():
            init_terms = compute_projection_terms_batch(
                xw, R0, t0, R0, t0, sdf_grid, cfg, weights, collision, prior
            )
            skip_small = (init_terms["loss_sdf"] < float(cfg.min_sdf_loss)) | (
                init_terms["sdf_mean_abs"] < float(cfg.min_sdf_mean_abs)
            )
        rot_delta = nn.Parameter(torch.zeros(batch_size, 3, device=self.device))
        t_delta = nn.Parameter(torch.zeros(batch_size, 3, device=self.device))
        optimizer = torch.optim.Adam(
            [{"params": [rot_delta], "lr": cfg.lr_rot}, {"params": [t_delta], "lr": cfg.lr_trans}]
        )
        best_loss = init_terms["loss"].detach().clone()
        best_rot_delta = torch.zeros_like(rot_delta)
        best_t_delta = torch.zeros_like(t_delta)
        patience = torch.zeros(batch_size, device=self.device, dtype=torch.int64)
        active = ~skip_small
        for _ in range(int(cfg.steps)):
            optimizer.zero_grad(set_to_none=True)
            R = torch.matmul(axis_angle_to_matrix(rot_delta), R0)
            t = t0 + t_delta
            terms = compute_projection_terms_batch(
                xw, R, t, R0, t0, sdf_grid, cfg, weights, collision, prior
            )
            (terms["loss"] * active.to(terms["loss"].dtype)).sum().backward()
            with torch.no_grad():
                improved = active & (
                    terms["loss"].detach() + float(cfg.early_stop_min_delta) < best_loss
                )
                best_loss = torch.where(improved, terms["loss"].detach(), best_loss)
                best_rot_delta = torch.where(improved[:, None], rot_delta.detach(), best_rot_delta)
                best_t_delta = torch.where(improved[:, None], t_delta.detach(), best_t_delta)
                patience = torch.where(
                    improved, torch.zeros_like(patience), patience + active.long()
                )
                active = active & (patience < int(cfg.early_stop_patience))
            optimizer.step()
        with torch.no_grad():
            R_best = project_to_so3(torch.matmul(axis_angle_to_matrix(best_rot_delta), R0))
            t_best = t0 + best_t_delta
            best_terms = compute_projection_terms_batch(
                xw, R_best, t_best, R0, t0, sdf_grid, cfg, weights, collision, prior
            )
            improvement_abs = init_terms["loss"] - best_terms["loss"]
            improvement_ratio = improvement_abs / init_terms["loss"].clamp_min(1e-08)
            sufficient_gain = (improvement_abs >= float(cfg.min_improvement_abs)) & (
                improvement_ratio >= float(cfg.min_improvement_ratio)
            )
            rot_delta_deg = rotation_geodesic_distance_deg(R0, R_best)
            trans_delta_norm = torch.linalg.vector_norm(t_best - t0, dim=1)
            severity = torch.maximum(
                (best_terms["sdf_mean_abs"] - float(cfg.min_sdf_mean_abs)).clamp_min(0.0),
                (init_terms["sdf_mean_abs"] - float(cfg.min_sdf_mean_abs)).clamp_min(0.0),
            )
            alpha = float(cfg.blend_alpha_min) + max(
                float(cfg.blend_alpha_max) - float(cfg.blend_alpha_min), 0.0
            ) * (
                improvement_ratio
                / max(float(cfg.blend_reference_ratio), float(cfg.min_improvement_ratio), 1e-08)
            ).clamp(
                0.0, 1.0
            )
            alpha = torch.where(
                severity <= 0.0,
                torch.minimum(alpha, torch.full_like(alpha, float(cfg.blend_alpha_min))),
                alpha,
            )
            alpha = torch.minimum(
                alpha,
                torch.where(
                    trans_delta_norm > 0,
                    float(cfg.max_translation_delta) / trans_delta_norm.clamp_min(1e-12),
                    torch.ones_like(alpha),
                ),
            )
            alpha = torch.minimum(
                alpha,
                torch.where(
                    rot_delta_deg > 0,
                    float(cfg.max_rotation_delta_deg) / rot_delta_deg.clamp_min(1e-12),
                    torch.ones_like(alpha),
                ),
            ).clamp(0.0, 1.0)
            motion_ok = alpha > 0.0001
            R_blend = blend_rotation_matrices(R0, R_best, alpha[:, None, None])
            t_blend = t0 + alpha[:, None] * (t_best - t0)
            blended_terms = compute_projection_terms_batch(
                xw, R_blend, t_blend, R0, t0, sdf_grid, cfg, weights, collision, prior
            )
            physical_improvement = blended_terms["loss"] < init_terms["loss"]
            applied = ~skip_small & sufficient_gain & motion_ok & physical_improvement
            R_out = torch.where(applied[:, None, None], R_blend, R0)
            t_out = torch.where(applied[:, None], t_blend, t0)
        skip_cpu = skip_small.cpu().numpy()
        gain_cpu = sufficient_gain.cpu().numpy()
        motion_cpu = motion_ok.cpu().numpy()
        physical_cpu = physical_improvement.cpu().numpy()
        applied_cpu = applied.cpu().numpy()
        decisions: List[str] = []
        for index in range(batch_size):
            if bool(skip_cpu[index]):
                decisions.append("skip_small_residual")
            elif not bool(gain_cpu[index]):
                decisions.append("reject_small_gain")
            elif not bool(motion_cpu[index]):
                decisions.append("reject_large_motion")
            elif not bool(physical_cpu[index]):
                decisions.append("reject_negative_optimization")
            else:
                decisions.append("applied_soft_projection")
        return BatchProjectionResult(
            rotations=R_out.cpu().numpy().astype(np.float32),
            translations=t_out.cpu().numpy().astype(np.float32),
            applied=applied_cpu.astype(bool),
            decisions=decisions,
            blend_alpha=alpha.cpu().numpy().astype(np.float32),
        )


@dataclass
class ContactSet:
    points_world: np.ndarray
    weights: np.ndarray
    collision_points_world: np.ndarray
    source: str


def decode_if_bytes(value: Any) -> str:
    if isinstance(value, (bytes, np.bytes_)):
        return value.decode("utf-8", errors="ignore").strip()
    return str(value).strip()


def normalize_skip_objects(skip_objects: Iterable[str] | None = None) -> set[str]:
    values = DEFAULT_STAGE_FOUR_SKIP_OBJECTS if skip_objects is None else skip_objects
    return {decode_if_bytes(value) for value in values if decode_if_bytes(value)}


def _deterministic_fps_indices(points: np.ndarray, k: int, *, start_index: int = 0) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    n = int(points.shape[0])
    if n <= k:
        return np.arange(n, dtype=np.int64)
    chosen = np.zeros((k,), dtype=np.int64)
    distances = np.full((n,), np.inf, dtype=np.float32)
    chosen[0] = int(np.clip(start_index, 0, n - 1))
    current = points[chosen[0]]
    for index in range(1, k):
        distance = np.sum((points - current[None, :]) ** 2, axis=1)
        distances = np.minimum(distances, distance)
        chosen[index] = int(np.argmax(distances))
        current = points[chosen[index]]
    return chosen


def load_sdf_grids(path, device: torch.device) -> Dict[str, SDFGrid]:
    tensors = load_file(str(path))
    result = {}
    for key, grid in tensors.items():
        if key.endswith('/sdf_grid'):
            name = key.removesuffix('/sdf_grid')
            result[name] = SDFGrid(
                sdf_grid=grid.to(device=device, dtype=torch.float32),
                origin_center=tensors[f'{name}/origin_center'].numpy(),
                pitch=tensors[f'{name}/pitch'].item(),
                meta=SDFGridMeta(grid_shape=tuple(grid.shape[-3:])),
            )
    return result


def _nearest_surface_contact_set(
    points_world: np.ndarray,
    init_pose: RigidPose,
    sdf_grid: SDFGrid,
    *,
    device: torch.device,
    num_contact_points: int,
) -> Tuple[np.ndarray, np.ndarray]:
    points_t = torch.as_tensor(points_world, device=device, dtype=torch.float32)
    rotation = torch.as_tensor(init_pose.R, device=device, dtype=torch.float32)
    translation = torch.as_tensor(init_pose.t, device=device, dtype=torch.float32)
    points_local = (points_t - translation[None, :]) @ rotation
    distance = sdf_grid.sample(points_local).abs()
    candidate_count = min(
        max(int(num_contact_points) * 4, int(num_contact_points)), points_world.shape[0]
    )
    candidate_idx = (
        torch.topk(distance, k=candidate_count, largest=False).indices.detach().cpu().numpy()
    )
    candidate_points = points_world[candidate_idx]
    fps_local = _deterministic_fps_indices(
        candidate_points, min(int(num_contact_points), len(candidate_points))
    )
    selected_idx = candidate_idx[fps_local]
    selected_distance = distance[selected_idx].detach().cpu().numpy()
    weights = 1.0 / np.maximum(selected_distance, max(float(sdf_grid.pitch), 1e-06))
    weights = weights / max(float(weights.sum()), 1e-08)
    return (points_world[selected_idx], weights.astype(np.float32))


def extract_contact_from_hand(
    hand: np.ndarray,
    *,
    init_pose: RigidPose,
    sdf_grid: SDFGrid,
    device: torch.device,
    num_contact_points: int = 48,
    num_collision_points: int = 128,
    pressure_threshold: float = 0.0,
    pressure_quantile: float = 0.8,
    min_pressure_points: int = 8,
) -> ContactSet:
    hand = np.asarray(hand, dtype=np.float32)
    points = hand[:, :3]
    pressure = np.nan_to_num(hand[:, 3], nan=0.0, posinf=0.0, neginf=0.0).clip(min=0.0)
    positive = pressure[pressure > float(pressure_threshold)]
    selected_points: np.ndarray
    weights: np.ndarray
    source: str
    if positive.size >= int(min_pressure_points):
        adaptive_threshold = max(
            float(pressure_threshold), float(np.quantile(positive, pressure_quantile))
        )
        candidate_idx = np.flatnonzero(pressure >= adaptive_threshold)
        if candidate_idx.size < int(min_pressure_points):
            candidate_idx = np.argsort(pressure)[
                -max(int(min_pressure_points), int(num_contact_points) * 4) :
            ]
        candidate_idx = candidate_idx[np.argsort(pressure[candidate_idx])[::-1]]
        max_candidates = min(
            candidate_idx.size, max(int(num_contact_points) * 4, int(num_contact_points))
        )
        candidate_idx = candidate_idx[:max_candidates]
        candidate_points = points[candidate_idx]
        fps_idx = _deterministic_fps_indices(
            candidate_points, min(int(num_contact_points), len(candidate_points)), start_index=0
        )
        selected_idx = candidate_idx[fps_idx]
        selected_points = points[selected_idx]
        weights = pressure[selected_idx] + 1e-06
        weights = weights / max(float(weights.sum()), 1e-08)
        source = "pressure_weighted"
    else:
        selected_points, weights = _nearest_surface_contact_set(
            points, init_pose, sdf_grid, device=device, num_contact_points=int(num_contact_points)
        )
        source = "nearest_surface_fallback"
    collision_idx = _deterministic_fps_indices(
        points, min(int(num_collision_points), len(points)), start_index=0
    )
    return ContactSet(
        points_world=selected_points.astype(np.float32),
        weights=np.asarray(weights, dtype=np.float32),
        collision_points_world=points[collision_idx].astype(np.float32),
        source=source,
    )


def refine_poses_with_sdf(
    proposal_latent14: torch.Tensor,
    hand_data: torch.Tensor,
    object_names: List[str],
    sdf_cache: Dict[str, SDFGrid],
    *,
    scale: torch.Tensor,
    device: torch.device,
    config: ProjectionConfig,
    num_contact_points: int = 48,
    num_collision_points: int = 128,
    pressure_threshold: float = 0.0,
    pressure_quantile: float = 0.8,
    min_pressure_points: int = 8,
    pose_mask: torch.Tensor | None = None,
    gp_confidence: torch.Tensor | None = None,
    skip_objects: Iterable[str] | None = None,
    fallback_latent14: torch.Tensor,
    physics_batch_size: int = 256,
    predicted_angle_gain_mean: torch.Tensor | None = None,
    predicted_angle_gain_std: torch.Tensor | None = None,
    predicted_distance_gain_mean: torch.Tensor | None = None,
    predicted_distance_gain_std: torch.Tensor | None = None,
    gp_residual: torch.Tensor | None = None,
    gain_confidence_k: float = 1.0,
    gain_margin: float = 0.0,
    consistency_cosine_threshold: float = 0.0,
    consistency_min_norm: float = 1e-06,
    network_physics_cosine_threshold: float = 0.5,
    physical_output_alpha: float = 0.25,
) -> torch.Tensor:
    """Refine in metres, then return poses in the input's per-sample scaled space.

    SDF geometry and ProjectionConfig distance/translation settings are in metres.
    Only hand XYZ and pose translation cross this unit boundary; pressure,
    rotation, and network/GP acceptance checks retain their existing units.
    """
    fallback = fallback_latent14
    refined = proposal_latent14.clone()
    refiner = PoseProjectionRefiner(config, device=device)
    skip_set = normalize_skip_objects(skip_objects)
    total = int(refined.shape[0])
    pose_valid = None if pose_mask is None else pose_mask.reshape(-1) > 0.5
    scales = torch.as_tensor(scale).detach().to(device="cpu", dtype=torch.float32)
    if tuple(scales.shape) not in ((total,), (total, 1)):
        raise ValueError(f"scale must have shape ({total},) or ({total}, 1), got {tuple(scales.shape)}")
    scales = scales.reshape(-1)
    invalid_scale = ~torch.isfinite(scales) | (scales <= 0)
    if pose_valid is not None:
        invalid_scale &= pose_valid.cpu()
    if invalid_scale.any():
        rows = torch.nonzero(invalid_scale).reshape(-1)[:8].tolist()
        raise ValueError(f"Pose-valid samples require finite, positive scale; invalid rows: {rows}")
    grouped_indices: Dict[str, List[int]] = {}
    for index in range(total):
        name = object_names[index]
        if bool(pose_valid[index]) and name not in skip_set:
            grouped_indices.setdefault(name, []).append(index)

    def pad_points(points: np.ndarray, count: int) -> np.ndarray:
        points = np.asarray(points, dtype=np.float32)
        if len(points) >= count:
            return points[:count]
        if len(points) == 0:
            return np.zeros((count, 3), dtype=np.float32)
        return np.concatenate([points, np.repeat(points[-1:], count - len(points), axis=0)], axis=0)

    def pad_weights(weights: np.ndarray, count: int) -> np.ndarray:
        weights = np.asarray(weights, dtype=np.float32).reshape(-1)
        if len(weights) >= count:
            out = weights[:count]
        else:
            out = np.concatenate([weights, np.zeros(count - len(weights), dtype=np.float32)])
        return out / max(float(out.sum()), 1e-08)

    batch_size = max(1, int(physics_batch_size))
    for name, indices in grouped_indices.items():
        sdf = sdf_cache[name]
        for batch_start in range(0, len(indices), batch_size):
            batch_indices = indices[batch_start : batch_start + batch_size]
            rotations: List[np.ndarray] = []
            translations: List[np.ndarray] = []
            contact_points: List[np.ndarray] = []
            contact_weights_batch: List[np.ndarray] = []
            collision_points: List[np.ndarray] = []
            prior_strengths: List[float] = []
            for index in batch_indices:
                sample_scale = float(scales[index].item())
                rot9 = proposal_latent14[index, :POSE_ROT_DIM].numpy().astype(np.float32)
                trans3 = proposal_latent14[index, POSE_ROT_DIM:POSE_DIM].numpy().astype(np.float32)
                # Convert before contact selection as well as SDF optimization.
                trans3 /= sample_scale
                hand_metres = hand_data[index].numpy().copy()
                hand_metres[..., :3] /= sample_scale
                initial_pose = RigidPose(
                    R=project_to_so3(torch.from_numpy(rot9.reshape(3, 3)))
                    .numpy()
                    .astype(np.float32),
                    t=trans3,
                )
                contacts = extract_contact_from_hand(
                    hand_metres,
                    init_pose=initial_pose,
                    sdf_grid=sdf,
                    device=device,
                    num_contact_points=int(num_contact_points),
                    num_collision_points=int(num_collision_points),
                    pressure_threshold=float(pressure_threshold),
                    pressure_quantile=float(pressure_quantile),
                    min_pressure_points=int(min_pressure_points),
                )
                confidence_value = 0.0
                row = gp_confidence[index]
                confidence_value = float(row[: min(6, row.numel())].mean().clamp(0.0, 1.0).item())
                rotations.append(initial_pose.R)
                translations.append(initial_pose.t)
                contact_points.append(pad_points(contacts.points_world, int(num_contact_points)))
                contact_weights_batch.append(pad_weights(contacts.weights, int(num_contact_points)))
                collision_points.append(
                    pad_points(contacts.collision_points_world, int(num_collision_points))
                )
                prior_strengths.append(1.0 + 4.0 * confidence_value)
            result = refiner.refine_batch(
                rotations=torch.from_numpy(np.stack(rotations)),
                translations=torch.from_numpy(np.stack(translations)),
                contact_points_world=torch.from_numpy(np.stack(contact_points)),
                contact_weights=torch.from_numpy(np.stack(contact_weights_batch)),
                collision_points_world=torch.from_numpy(np.stack(collision_points)),
                sdf_grid=sdf,
                prior_strength=torch.tensor(prior_strengths, dtype=torch.float32),
            )
            for local_index, index in enumerate(batch_indices):
                if not bool(result.applied[local_index]):
                    continue
                physical_rotation = torch.from_numpy(result.rotations[local_index]).reshape(3, 3)
                # Restore scaled units before blending and network/GP comparisons.
                physical_translation = torch.from_numpy(result.translations[local_index]) * float(
                    scales[index].item()
                )
                network_rotation = project_to_so3(
                    proposal_latent14[index, :POSE_ROT_DIM].reshape(3, 3)
                )
                network_translation = proposal_latent14[index, POSE_ROT_DIM:POSE_DIM]
                output_alpha = float(np.clip(physical_output_alpha, 0.0, 1.0))
                projected_rotation = blend_rotation_matrices(
                    network_rotation,
                    physical_rotation,
                    torch.tensor(output_alpha, dtype=network_rotation.dtype),
                )
                projected_translation = network_translation + output_alpha * (
                    physical_translation - network_translation
                )
                gain_ok = False
                angle_lcb = float(predicted_angle_gain_mean.reshape(-1)[index].item()) - float(
                    gain_confidence_k
                ) * float(predicted_angle_gain_std.reshape(-1)[index].item())
                distance_lcb = float(
                    predicted_distance_gain_mean.reshape(-1)[index].item()
                ) - float(gain_confidence_k) * float(
                    predicted_distance_gain_std.reshape(-1)[index].item()
                )
                gain_ok = bool(
                    np.isfinite(angle_lcb)
                    and np.isfinite(distance_lcb)
                    and (angle_lcb > float(gain_margin))
                    and (distance_lcb > float(gain_margin))
                )
                gp_consistency_scores: List[float] = []
                network_consistency_scores: List[float] = []
                fallback_rotation = project_to_so3(fallback[index, :POSE_ROT_DIM].reshape(3, 3))
                stage_rotation_delta = so3_log_map(
                    (projected_rotation @ fallback_rotation.transpose(0, 1)).unsqueeze(0)
                )[0]
                stage_translation_delta = (
                    projected_translation - fallback[index, POSE_ROT_DIM:POSE_DIM]
                )
                network_rotation_delta = so3_log_map(
                    (network_rotation @ fallback_rotation.transpose(0, 1)).unsqueeze(0)
                )[0]
                network_translation_delta = (
                    network_translation - fallback[index, POSE_ROT_DIM:POSE_DIM]
                )
                gp_rotation_delta = gp_residual[index, :3].float()
                gp_translation_delta = gp_residual[index, 3:6].float()
                for stage_delta, gp_delta in (
                    (stage_rotation_delta, gp_rotation_delta),
                    (stage_translation_delta, gp_translation_delta),
                ):
                    stage_norm = float(torch.linalg.vector_norm(stage_delta).item())
                    gp_norm = float(torch.linalg.vector_norm(gp_delta).item())
                    if stage_norm > float(consistency_min_norm) and gp_norm > float(
                        consistency_min_norm
                    ):
                        cosine = float(
                            torch.dot(stage_delta, gp_delta).item()
                            / max(stage_norm * gp_norm, 1e-12)
                        )
                        gp_consistency_scores.append(cosine)
                for stage_delta, network_delta in (
                    (stage_rotation_delta, network_rotation_delta),
                    (stage_translation_delta, network_translation_delta),
                ):
                    stage_norm = float(torch.linalg.vector_norm(stage_delta).item())
                    network_norm = float(torch.linalg.vector_norm(network_delta).item())
                    if stage_norm > float(consistency_min_norm) and network_norm > float(
                        consistency_min_norm
                    ):
                        network_consistency_scores.append(
                            float(
                                torch.dot(stage_delta, network_delta).item()
                                / max(stage_norm * network_norm, 1e-12)
                            )
                        )
                consistency_score = (
                    float(sum(gp_consistency_scores) / len(gp_consistency_scores))
                    if gp_consistency_scores
                    else float("-inf")
                )
                network_consistency_score = (
                    float(sum(network_consistency_scores) / len(network_consistency_scores))
                    if network_consistency_scores
                    else 1.0
                )
                consistency_ok = consistency_score >= float(
                    consistency_cosine_threshold
                ) and network_consistency_score >= float(network_physics_cosine_threshold)
                accept = gain_ok and consistency_ok
                if not accept:
                    continue
                refined[index, :POSE_ROT_DIM] = projected_rotation.reshape(9)
                refined[index, POSE_ROT_DIM:POSE_DIM] = projected_translation
    return refined
