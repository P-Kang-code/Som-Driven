from dataclasses import dataclass

import gpytorch
import torch
import torch.nn.functional as functional


POSE_ROTATION_DIM = 9
POSE_DIM = 12
LATENT_DIM = 14
RESIDUAL_DIM = 8
GROUPS = {
    "rotation": (0, 3),
    "translation": (3, 6),
    "force_torque": (6, 8),
}


def project_rotation(rotation: torch.Tensor) -> torch.Tensor:
    matrix = torch.nan_to_num(rotation.float()).reshape(-1, 3, 3)
    left, _, right = torch.linalg.svd(matrix)
    sign = torch.where(torch.det(left @ right) < 0, -1.0, 1.0)
    correction = torch.eye(3, device=matrix.device, dtype=matrix.dtype).repeat(matrix.shape[0], 1, 1)
    correction[:, 2, 2] = sign
    return left @ correction @ right


def skew(vector: torch.Tensor) -> torch.Tensor:
    x, y, z = vector.unbind(dim=1)
    zero = torch.zeros_like(x)
    return torch.stack((zero, -z, y, z, zero, -x, -y, x, zero), dim=1).reshape(-1, 3, 3)


def so3_exp(vector: torch.Tensor) -> torch.Tensor:
    angle_square = vector.square().sum(dim=1, keepdim=True)
    angle = angle_square.sqrt()
    first = torch.where(
        angle < 0.0001,
        1 - angle_square / 6,
        torch.sin(angle) / angle.clamp_min(0.00000001),
    )
    second = torch.where(
        angle < 0.0001,
        0.5 - angle_square / 24,
        (1 - torch.cos(angle)) / angle_square.clamp_min(0.00000001),
    )
    matrix = skew(vector)
    identity = torch.eye(3, device=vector.device, dtype=vector.dtype).unsqueeze(0)
    return identity + first.unsqueeze(2) * matrix + second.unsqueeze(2) * (matrix @ matrix)


def so3_log(matrix: torch.Tensor) -> torch.Tensor:
    cosine = ((matrix.diagonal(dim1=1, dim2=2).sum(dim=1) - 1) * 0.5).clamp(-0.999999, 0.999999)
    angle = torch.acos(cosine)
    vee = torch.stack(
        (
            matrix[:, 2, 1] - matrix[:, 1, 2],
            matrix[:, 0, 2] - matrix[:, 2, 0],
            matrix[:, 1, 0] - matrix[:, 0, 1],
        ),
        dim=1,
    )
    scale = angle / (2 * torch.sin(angle).clamp_min(0.00000001))
    scale = torch.where(angle < 0.0001, torch.full_like(scale, 0.5), scale)
    return vee * scale.unsqueeze(1)


def project_latent(latent: torch.Tensor) -> torch.Tensor:
    if latent.ndim != 2 or latent.shape[1] != LATENT_DIM:
        raise ValueError(f"Expected [B,{LATENT_DIM}], received {tuple(latent.shape)}")
    return torch.cat(
        (
            project_rotation(latent[:, :POSE_ROTATION_DIM]).reshape(-1, POSE_ROTATION_DIM),
            latent[:, POSE_ROTATION_DIM:],
        ),
        dim=1,
    )


def residual_targets(
    target: torch.Tensor,
    prediction: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    target = project_latent(target)
    prediction = project_latent(prediction)
    target_rotation = target[:, :POSE_ROTATION_DIM].reshape(-1, 3, 3)
    prediction_rotation = prediction[:, :POSE_ROTATION_DIM].reshape(-1, 3, 3)
    rotation = so3_log(target_rotation @ prediction_rotation.transpose(1, 2))
    translation = target[:, POSE_ROTATION_DIM:POSE_DIM] - prediction[:, POSE_ROTATION_DIM:POSE_DIM]
    force_torque = target[:, POSE_DIM:LATENT_DIM] - prediction[:, POSE_DIM:LATENT_DIM]
    pose_valid = mask[:, :POSE_DIM].amax(dim=1, keepdim=True)
    force_valid = mask[:, POSE_DIM:LATENT_DIM].amax(dim=1, keepdim=True)
    residual_mask = torch.cat((pose_valid.expand(-1, 6), force_valid.expand(-1, 2)), dim=1)
    residual = torch.cat((rotation, translation, force_torque), dim=1)
    return residual * residual_mask, residual_mask


def apply_residual(latent: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
    latent = project_latent(latent)
    rotation = latent[:, :POSE_ROTATION_DIM].reshape(-1, 3, 3)
    rotation = so3_exp(residual[:, :3]) @ rotation
    translation = latent[:, POSE_ROTATION_DIM:POSE_DIM] + residual[:, 3:6]
    force_torque = latent[:, POSE_DIM:LATENT_DIM] + residual[:, 6:8]
    return torch.cat((rotation.reshape(-1, POSE_ROTATION_DIM), translation, force_torque), dim=1)


def sample_evenly(points: torch.Tensor, count: int) -> torch.Tensor:
    if points.shape[1] <= count:
        return points
    index = torch.linspace(0, points.shape[1] - 1, count, device=points.device).round().long()
    return points.index_select(1, index)


def geometry_features(
    hand: torch.Tensor,
    object_surface: torch.Tensor,
    prediction: torch.Tensor,
    hand_points: int = 64,
    object_points: int = 256,
) -> torch.Tensor:
    prediction = project_latent(prediction)
    hand_xyz = hand[:, :, :3].float()
    pressure = hand[:, :, 3].float().clamp_min(0)
    count = min(hand_points, hand_xyz.shape[1])
    top_pressure, top_index = torch.topk(pressure, count, dim=1, sorted=False)
    contact_world = torch.gather(hand_xyz, 1, top_index.unsqueeze(2).expand(-1, -1, 3))
    rotation = prediction[:, :POSE_ROTATION_DIM].reshape(-1, 3, 3)
    translation = prediction[:, POSE_ROTATION_DIM:POSE_DIM]
    contact_local = (contact_world - translation.unsqueeze(1)) @ rotation
    object_local = sample_evenly(object_surface[:, :, :3].float(), object_points)
    nearest = torch.cdist(contact_local, object_local).amin(dim=2)
    weight = top_pressure + 0.000001
    weight = weight / weight.sum(dim=1, keepdim=True).clamp_min(0.000001)
    centroid = (contact_local * weight.unsqueeze(2)).sum(dim=1)
    spread = torch.sqrt(
        ((contact_local - centroid.unsqueeze(1)).square() * weight.unsqueeze(2))
        .sum(dim=1)
        .clamp_min(0.000000000001)
    )
    scalars = torch.stack(
        (
            nearest.mean(dim=1),
            nearest.amin(dim=1),
            nearest.amax(dim=1),
            (nearest * weight).sum(dim=1),
            pressure.mean(dim=1),
            pressure.amax(dim=1),
            (pressure > 0).float().mean(dim=1),
            pressure.sum(dim=1) / max(pressure.shape[1], 1),
        ),
        dim=1,
    )
    return torch.cat(
        (
            contact_local.mean(dim=1),
            contact_local.std(dim=1),
            centroid,
            spread,
            scalars,
        ),
        dim=1,
    )


def compressed_code(code: torch.Tensor, size: int = 32) -> torch.Tensor:
    if code.shape[1] < size:
        return functional_pad(code.float(), size - code.shape[1])
    return functional.adaptive_avg_pool1d(code.float().unsqueeze(1), size).squeeze(1)


def functional_pad(value: torch.Tensor, amount: int) -> torch.Tensor:
    return torch.cat((value, torch.zeros(value.shape[0], amount, device=value.device, dtype=value.dtype)), dim=1)


def gp_features(
    hand_code: torch.Tensor,
    object_code: torch.Tensor,
    hand: torch.Tensor,
    object_surface: torch.Tensor,
    prediction: torch.Tensor,
) -> torch.Tensor:
    feature = torch.cat(
        (
            compressed_code(hand_code),
            compressed_code(object_code),
            project_latent(prediction),
            geometry_features(hand, object_surface, prediction),
        ),
        dim=1,
    )
    if feature.shape[1] != 98:
        raise RuntimeError(f"Expected 98 GP features, received {feature.shape[1]}")
    return feature


class GroupedExactGP(gpytorch.models.ExactGP):
    def __init__(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        likelihood: gpytorch.likelihoods.MultitaskGaussianLikelihood,
        rank: int,
    ):
        super().__init__(train_x, train_y, likelihood)
        tasks = train_y.shape[1]
        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ConstantMean(),
            num_tasks=tasks,
        )
        data_kernel = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(nu=1.5, ard_num_dims=train_x.shape[1])
        )
        self.covariance_module = gpytorch.kernels.MultitaskKernel(
            data_kernel,
            num_tasks=tasks,
            rank=max(1, min(rank, tasks)),
        )

    def forward(self, value: torch.Tensor) -> gpytorch.distributions.MultitaskMultivariateNormal:
        return gpytorch.distributions.MultitaskMultivariateNormal(
            self.mean_module(value),
            self.covariance_module(value),
        )


@dataclass(frozen=True)
class GPConfig:
    iterations: int = 300
    learning_rate: float = 0.03
    max_points: int = 3000
    coregionalization_rank: int = 1
    uncertainty_threshold: float = 0.75
    gate_temperature: float = 0.1
    max_correction_sigma: float = 2.5


@dataclass(frozen=True)
class GroupPolicy:
    correction_scale: float = 1.0
    uncertainty_threshold: float = 0.75
    gate_temperature: float = 0.1
    uncertainty_gated: bool = True


@dataclass
class GPBundle:
    model: GroupedExactGP
    likelihood: gpytorch.likelihoods.MultitaskGaussianLikelihood
    start: int
    end: int
    target_center: torch.Tensor
    target_scale: torch.Tensor


class GroupedResidualGP:
    def __init__(
        self,
        config: GPConfig = GPConfig(),
        policies: dict[str, GroupPolicy] | None = None,
    ):
        self.config = config
        self.feature_center: torch.Tensor | None = None
        self.feature_scale: torch.Tensor | None = None
        self.groups: dict[str, GPBundle] = {}
        default = GroupPolicy(
            uncertainty_threshold=config.uncertainty_threshold,
            gate_temperature=config.gate_temperature,
        )
        values = policies or {}
        self.policies = {name: values.get(name, default) for name in GROUPS}

    def fit(
        self,
        features: torch.Tensor,
        target: torch.Tensor,
        prediction: torch.Tensor,
        mask: torch.Tensor,
    ) -> None:
        residual, residual_mask = residual_targets(target, prediction, mask)
        self.feature_center = features.mean(dim=0)
        self.feature_scale = features.std(dim=0).clamp_min(0.0001)
        normalized = (features - self.feature_center) / self.feature_scale
        self.groups = {}
        for name, (start, end) in GROUPS.items():
            valid = residual_mask[:, start:end].amax(dim=1) > 0.5
            train_x = normalized[valid]
            train_y = residual[valid, start:end]
            if train_x.shape[0] > self.config.max_points:
                index = torch.linspace(
                    0,
                    train_x.shape[0] - 1,
                    self.config.max_points,
                    device=train_x.device,
                ).round().long()
                train_x = train_x.index_select(0, index)
                train_y = train_y.index_select(0, index)
            center = train_y.mean(dim=0)
            scale = train_y.std(dim=0).clamp_min(0.001)
            train_y = (train_y - center) / scale
            likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(
                num_tasks=end - start,
            ).to(train_x.device)
            model = GroupedExactGP(
                train_x,
                train_y,
                likelihood,
                self.config.coregionalization_rank,
            ).to(train_x.device)
            model.train()
            likelihood.train()
            parameters = {id(parameter): parameter for parameter in tuple(model.parameters()) + tuple(likelihood.parameters())}
            optimizer = torch.optim.Adam(parameters.values(), lr=self.config.learning_rate)
            marginal = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
            for _ in range(self.config.iterations):
                optimizer.zero_grad()
                loss = -marginal(model(train_x), train_y)
                loss.backward()
                optimizer.step()
            model.eval()
            likelihood.eval()
            self.groups[name] = GPBundle(model, likelihood, start, end, center, scale)

    @torch.no_grad()
    def correct(self, features: torch.Tensor, prediction: torch.Tensor) -> dict[str, torch.Tensor]:
        if self.feature_center is None or self.feature_scale is None or set(self.groups) != set(GROUPS):
            raise RuntimeError("The grouped GP must be fitted before use")
        normalized = (features - self.feature_center) / self.feature_scale
        residual = torch.zeros(features.shape[0], RESIDUAL_DIM, device=features.device)
        confidence = torch.zeros_like(residual)
        uncertainty = torch.zeros_like(residual)
        with gpytorch.settings.fast_pred_var():
            for name, (start, end) in GROUPS.items():
                bundle = self.groups[name]
                policy = self.policies[name]
                posterior = bundle.likelihood(bundle.model(normalized))
                mean = posterior.mean * bundle.target_scale + bundle.target_center
                standard = posterior.variance.clamp_min(0.000000001).sqrt() * bundle.target_scale
                if policy.uncertainty_gated:
                    gate = torch.sigmoid(
                        (policy.uncertainty_threshold - standard.mean(dim=1, keepdim=True))
                        / policy.gate_temperature
                    )
                else:
                    gate = torch.ones(standard.shape[0], 1, device=standard.device, dtype=standard.dtype)
                limit = self.config.max_correction_sigma * bundle.target_scale
                mean = mean.clamp(-limit, limit)
                residual[:, start:end] = policy.correction_scale * mean * gate
                confidence[:, start:end] = gate.expand(-1, end - start)
                uncertainty[:, start:end] = standard
        return {
            "latent": apply_residual(prediction, residual),
            "residual": residual,
            "confidence": confidence,
            "uncertainty": uncertainty,
        }
