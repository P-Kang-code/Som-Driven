import math
from dataclasses import dataclass

import torch
import torch.nn.functional as functional
from torch import nn

from stage_3 import LATENT_DIM, POSE_DIM, POSE_ROTATION_DIM, project_latent, project_rotation, so3_exp, so3_log


@dataclass(frozen=True)
class ContactProposalConfig:
    hidden_dim: int = 256
    depth: int = 4
    dropout: float = 0.05
    contact_points: int = 48
    object_points: int = 256
    max_rotation_degrees: float = 8.0
    max_translation: float = 0.02
    angle_weight: float = 3.0
    translation_weight: float = 5.0
    improvement_weight: float = 2.0
    improvement_margin: float = 0.0
    correction_weight: float = 0.001
    contact_weight: float = 2.0
    gain_weight: float = 1.0
    gain_std_floor: float = 0.001


class ResidualLayer(nn.Module):
    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + self.net(value)


class ContactProposalNetwork(nn.Module):
    def __init__(self, feature_dim: int, config: ContactProposalConfig = ContactProposalConfig()):
        super().__init__()
        self.config = config
        self.input = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, config.hidden_dim),
            nn.SiLU(),
        )
        self.layers = nn.ModuleList(
            ResidualLayer(config.hidden_dim, config.dropout) for _ in range(config.depth)
        )
        self.pose_output = nn.Linear(config.hidden_dim, 7)
        self.gain_output = nn.Linear(config.hidden_dim, 4)
        nn.init.zeros_(self.pose_output.weight)
        nn.init.zeros_(self.pose_output.bias)
        nn.init.constant_(self.pose_output.bias[6], -2.0)
        nn.init.zeros_(self.gain_output.weight)
        nn.init.zeros_(self.gain_output.bias)
        nn.init.constant_(self.gain_output.bias[1], -1.0)
        nn.init.constant_(self.gain_output.bias[3], -1.0)

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        value = self.input(features)
        for layer in self.layers:
            value = layer(value)
        pose = self.pose_output(value)
        gain = self.gain_output(value.detach())
        rotation_limit = math.radians(self.config.max_rotation_degrees)
        return {
            "rotation": torch.tanh(pose[:, :3]) * rotation_limit,
            "translation": torch.tanh(pose[:, 3:6]) * self.config.max_translation,
            "gate": torch.sigmoid(pose[:, 6:7]),
            "angle_gain_mean": gain[:, 0:1],
            "angle_gain_std": functional.softplus(gain[:, 1:2]) + self.config.gain_std_floor,
            "distance_gain_mean": gain[:, 2:3],
            "distance_gain_std": functional.softplus(gain[:, 3:4]) + self.config.gain_std_floor,
        }


def sample_evenly(points: torch.Tensor, count: int) -> torch.Tensor:
    if points.shape[1] <= count:
        return points
    index = torch.linspace(0, points.shape[1] - 1, count, device=points.device).round().long()
    return points.index_select(1, index)


def proposal_features(
    base: torch.Tensor,
    hand: torch.Tensor,
    object_surface: torch.Tensor,
    gp_confidence: torch.Tensor,
    config: ContactProposalConfig,
) -> torch.Tensor:
    base = project_latent(base)
    rotation = base[:, :POSE_ROTATION_DIM].reshape(-1, 3, 3)
    translation = base[:, POSE_ROTATION_DIM:POSE_DIM]
    hand_xyz = hand[:, :, :3].float()
    pressure = hand[:, :, 3].float().clamp_min(0)
    count = min(config.contact_points, hand_xyz.shape[1])
    top_pressure, index = torch.topk(pressure, count, dim=1, sorted=False)
    contact_world = torch.gather(hand_xyz, 1, index.unsqueeze(2).expand(-1, -1, 3))
    contact_local = (contact_world - translation.unsqueeze(1)) @ rotation
    object_local = sample_evenly(object_surface[:, :, :3].float(), config.object_points)
    distance = torch.cdist(contact_local, object_local)
    nearest_distance, nearest_index = distance.min(dim=2)
    nearest_point = torch.gather(
        object_local,
        1,
        nearest_index.unsqueeze(2).expand(-1, -1, 3),
    )
    displacement = nearest_point - contact_local
    weight = top_pressure + 0.000001
    weight = weight / weight.sum(dim=1, keepdim=True).clamp_min(0.000001)
    geometry = torch.cat(
        (
            contact_local.mean(dim=1),
            contact_local.std(dim=1),
            displacement.mean(dim=1),
            displacement.std(dim=1),
            torch.stack(
                (
                    nearest_distance.mean(dim=1),
                    nearest_distance.std(dim=1),
                    nearest_distance.amin(dim=1),
                    nearest_distance.amax(dim=1),
                    (nearest_distance * weight).sum(dim=1),
                ),
                dim=1,
            ),
            torch.stack(
                (
                    pressure.mean(dim=1),
                    pressure.amax(dim=1),
                    (pressure > 0).float().mean(dim=1),
                ),
                dim=1,
            ),
            torch.stack(
                (
                    gp_confidence.mean(dim=1),
                    gp_confidence.amin(dim=1),
                    gp_confidence.amax(dim=1),
                ),
                dim=1,
            ),
        ),
        dim=1,
    )
    return torch.cat((base, geometry), dim=1)


def apply_proposal(
    base: torch.Tensor,
    prediction: dict[str, torch.Tensor],
    alpha: float = 1.0,
) -> torch.Tensor:
    base = project_latent(base)
    gate = prediction["gate"] * alpha
    rotation = base[:, :POSE_ROTATION_DIM].reshape(-1, 3, 3)
    rotation = so3_exp(prediction["rotation"] * gate) @ rotation
    translation = base[:, POSE_ROTATION_DIM:POSE_DIM] + prediction["translation"] * gate
    return torch.cat((rotation.reshape(-1, POSE_ROTATION_DIM), translation, base[:, POSE_DIM:]), dim=1)


def pose_errors(target: torch.Tensor, prediction: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    target = project_latent(target)
    target_rotation = target[:, :POSE_ROTATION_DIM].reshape(-1, 3, 3)
    prediction_rotation = prediction[:, :POSE_ROTATION_DIM].reshape(-1, 3, 3)
    relative = target_rotation.transpose(1, 2) @ prediction_rotation
    cosine = ((relative.diagonal(dim1=1, dim2=2).sum(dim=1) - 1) * 0.5).clamp(-0.999999, 0.999999)
    angle = torch.acos(cosine)
    distance = torch.linalg.vector_norm(
        target[:, POSE_ROTATION_DIM:POSE_DIM] - prediction[:, POSE_ROTATION_DIM:POSE_DIM],
        dim=1,
    )
    return angle, distance


def contact_proxy(
    prediction: torch.Tensor,
    hand: torch.Tensor,
    object_surface: torch.Tensor,
    config: ContactProposalConfig,
) -> torch.Tensor:
    rotation = prediction[:, :POSE_ROTATION_DIM].reshape(-1, 3, 3)
    translation = prediction[:, POSE_ROTATION_DIM:POSE_DIM]
    pressure = hand[:, :, 3].float().clamp_min(0)
    count = min(config.contact_points, hand.shape[1])
    top_pressure, index = torch.topk(pressure, count, dim=1, sorted=False)
    contact_world = torch.gather(hand[:, :, :3].float(), 1, index.unsqueeze(2).expand(-1, -1, 3))
    contact_local = (contact_world - translation.unsqueeze(1)) @ rotation
    object_local = sample_evenly(object_surface[:, :, :3].float(), config.object_points)
    nearest = torch.cdist(contact_local, object_local).amin(dim=2)
    weight = top_pressure + 0.000001
    weight = weight / weight.sum(dim=1, keepdim=True).clamp_min(0.000001)
    robust = functional.smooth_l1_loss(nearest, torch.zeros_like(nearest), reduction="none", beta=0.01)
    return (robust * weight).sum(dim=1)


def gaussian_nll(target: torch.Tensor, mean: torch.Tensor, standard: torch.Tensor) -> torch.Tensor:
    standard = standard.clamp_min(0.000001)
    return 0.5 * ((target - mean) / standard).square() + standard.log()


def proposal_loss(
    model: ContactProposalNetwork,
    base: torch.Tensor,
    target: torch.Tensor,
    hand: torch.Tensor,
    object_surface: torch.Tensor,
    gp_confidence: torch.Tensor,
) -> torch.Tensor:
    config = model.config
    features = proposal_features(base, hand, object_surface, gp_confidence, config)
    prediction = model(features)
    proposed = apply_proposal(base, prediction)
    base_angle, base_distance = pose_errors(target, base)
    angle, distance = pose_errors(target, proposed)
    risk = config.angle_weight * angle + config.translation_weight * distance
    base_risk = config.angle_weight * base_angle + config.translation_weight * base_distance
    hinge = functional.relu(risk - base_risk + config.improvement_margin)
    angle_scale = base_angle.mean().clamp_min(0.001)
    distance_scale = base_distance.mean().clamp_min(0.00001)
    angle_gain = ((base_angle - angle) / angle_scale).clamp(-2, 2)
    distance_gain = ((base_distance - distance) / distance_scale).clamp(-2, 2)
    gain_loss = gaussian_nll(
        angle_gain.unsqueeze(1),
        prediction["angle_gain_mean"],
        prediction["angle_gain_std"],
    ).mean() + gaussian_nll(
        distance_gain.unsqueeze(1),
        prediction["distance_gain_mean"],
        prediction["distance_gain_std"],
    ).mean()
    correction = prediction["rotation"].square().mean() + prediction["translation"].square().mean()
    contact = contact_proxy(proposed, hand, object_surface, config).mean()
    return (
        risk.mean()
        + config.improvement_weight * hinge.mean()
        + config.correction_weight * correction
        + config.contact_weight * contact
        + config.gain_weight * gain_loss
    )


@dataclass(frozen=True)
class SDFProjectionConfig:
    steps: int = 80
    rotation_learning_rate: float = 0.002
    translation_learning_rate: float = 0.0005
    contact_points: int = 48
    collision_points: int = 128
    surface_points: int = 512
    sdf_weight: float = 1.0
    penetration_weight: float = 2.0
    translation_regularization: float = 0.05
    rotation_regularization: float = 0.05
    penetration_tolerance: float = 0.001
    robust_delta: float = 0.004
    minimum_improvement: float = 0.0000001
    minimum_improvement_ratio: float = 0.01
    maximum_translation: float = 0.02
    maximum_rotation_degrees: float = 8.0
    blend_minimum: float = 0.10
    blend_maximum: float = 0.35
    blend_reference_ratio: float = 0.10


class PointCloudSDF:
    def __init__(self, surface: torch.Tensor):
        surface = surface[:, :, :3].float().contiguous()
        self.surface = surface
        self.center = surface.mean(dim=1)
        normal = surface - self.center.unsqueeze(1)
        self.normal = normal / torch.linalg.vector_norm(normal, dim=2, keepdim=True).clamp_min(0.000001)

    def sample(self, local_points: torch.Tensor) -> torch.Tensor:
        distance = torch.cdist(local_points, self.surface)
        index = distance.argmin(dim=2)
        nearest = torch.gather(
            self.surface,
            1,
            index.unsqueeze(2).expand(-1, -1, 3),
        )
        normal = torch.gather(
            self.normal,
            1,
            index.unsqueeze(2).expand(-1, -1, 3),
        )
        return ((local_points - nearest) * normal).sum(dim=2)


def huber(value: torch.Tensor, delta: float) -> torch.Tensor:
    absolute = value.abs()
    quadratic = torch.minimum(absolute, torch.full_like(absolute, delta))
    linear = absolute - quadratic
    return 0.5 * quadratic.square() + delta * linear


def projection_terms(
    contact_world: torch.Tensor,
    contact_weight: torch.Tensor,
    collision_world: torch.Tensor,
    rotation: torch.Tensor,
    translation: torch.Tensor,
    reference_rotation: torch.Tensor,
    reference_translation: torch.Tensor,
    sdf: PointCloudSDF,
    config: SDFProjectionConfig,
) -> dict[str, torch.Tensor]:
    contact_local = torch.matmul(contact_world - translation.unsqueeze(1), rotation)
    signed = sdf.sample(contact_local)
    weight = contact_weight / contact_weight.sum(dim=1, keepdim=True).clamp_min(0.00000001)
    sdf_loss = (huber(signed, config.robust_delta) * weight).sum(dim=1)
    collision_local = torch.matmul(collision_world - translation.unsqueeze(1), rotation)
    collision_signed = sdf.sample(collision_local)
    penetration = functional.relu(-collision_signed - config.penetration_tolerance)
    penetration_loss = huber(penetration, config.robust_delta).mean(dim=1)
    translation_loss = (translation - reference_translation).square().mean(dim=1)
    rotation_loss = (rotation - reference_rotation).square().mean(dim=(1, 2))
    total = (
        config.sdf_weight * sdf_loss
        + config.penetration_weight * penetration_loss
        + config.translation_regularization * translation_loss
        + config.rotation_regularization * rotation_loss
    )
    return {
        "total": total,
        "sdf": sdf_loss,
        "sdf_mean": signed.abs().mean(dim=1),
        "penetration": penetration_loss,
    }


def blend_rotation(start: torch.Tensor, target: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    mixed = (1 - alpha.unsqueeze(2)) * start + alpha.unsqueeze(2) * target
    return project_rotation(mixed)


class SDFProjector:
    def __init__(self, config: SDFProjectionConfig = SDFProjectionConfig()):
        self.config = config

    def contact_sets(self, hand: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pressure = hand[:, :, 3].float().clamp_min(0)
        contact_count = min(self.config.contact_points, hand.shape[1])
        top_pressure, top_index = torch.topk(pressure, contact_count, dim=1, sorted=False)
        contact = torch.gather(
            hand[:, :, :3].float(),
            1,
            top_index.unsqueeze(2).expand(-1, -1, 3),
        )
        weight = top_pressure + 0.000001
        collision = sample_evenly(hand[:, :, :3].float(), self.config.collision_points)
        return contact, weight, collision

    def refine(
        self,
        proposal: torch.Tensor,
        hand: torch.Tensor,
        object_surface: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        config = self.config
        proposal = project_latent(proposal).detach()
        reference_rotation = proposal[:, :POSE_ROTATION_DIM].reshape(-1, 3, 3)
        reference_translation = proposal[:, POSE_ROTATION_DIM:POSE_DIM]
        contact, weight, collision = self.contact_sets(hand)
        sdf = PointCloudSDF(sample_evenly(object_surface, config.surface_points))
        with torch.no_grad():
            initial = projection_terms(
                contact,
                weight,
                collision,
                reference_rotation,
                reference_translation,
                reference_rotation,
                reference_translation,
                sdf,
                config,
            )
        rotation_delta = nn.Parameter(torch.zeros(proposal.shape[0], 3, device=proposal.device))
        translation_delta = nn.Parameter(torch.zeros(proposal.shape[0], 3, device=proposal.device))
        optimizer = torch.optim.Adam(
            (
                {"params": (rotation_delta,), "lr": config.rotation_learning_rate},
                {"params": (translation_delta,), "lr": config.translation_learning_rate},
            )
        )
        best_loss = initial["total"].clone()
        best_rotation = torch.zeros_like(rotation_delta)
        best_translation = torch.zeros_like(translation_delta)
        for _ in range(config.steps):
            optimizer.zero_grad()
            rotation = so3_exp(rotation_delta) @ reference_rotation
            translation = reference_translation + translation_delta
            terms = projection_terms(
                contact,
                weight,
                collision,
                rotation,
                translation,
                reference_rotation,
                reference_translation,
                sdf,
                config,
            )
            terms["total"].sum().backward()
            with torch.no_grad():
                improved = terms["total"] < best_loss
                best_loss = torch.where(improved, terms["total"], best_loss)
                best_rotation = torch.where(improved.unsqueeze(1), rotation_delta, best_rotation)
                best_translation = torch.where(improved.unsqueeze(1), translation_delta, best_translation)
            optimizer.step()
        with torch.no_grad():
            optimized_rotation = so3_exp(best_rotation) @ reference_rotation
            optimized_translation = reference_translation + best_translation
            optimized = projection_terms(
                contact,
                weight,
                collision,
                optimized_rotation,
                optimized_translation,
                reference_rotation,
                reference_translation,
                sdf,
                config,
            )
            improvement = initial["total"] - optimized["total"]
            ratio = improvement / initial["total"].clamp_min(0.00000001)
            alpha = config.blend_minimum + (
                config.blend_maximum - config.blend_minimum
            ) * (ratio / config.blend_reference_ratio).clamp(0, 1)
            rotation_degrees = torch.rad2deg(
                torch.linalg.vector_norm(so3_log(optimized_rotation @ reference_rotation.transpose(1, 2)), dim=1)
            )
            translation_norm = torch.linalg.vector_norm(optimized_translation - reference_translation, dim=1)
            alpha = torch.minimum(
                alpha,
                torch.where(
                    translation_norm > 0,
                    config.maximum_translation / translation_norm.clamp_min(0.000000000001),
                    torch.ones_like(alpha),
                ),
            )
            alpha = torch.minimum(
                alpha,
                torch.where(
                    rotation_degrees > 0,
                    config.maximum_rotation_degrees / rotation_degrees.clamp_min(0.000000000001),
                    torch.ones_like(alpha),
                ),
            ).clamp(0, 1)
            blended_rotation = blend_rotation(
                reference_rotation,
                optimized_rotation,
                alpha.unsqueeze(1),
            )
            blended_translation = reference_translation + alpha.unsqueeze(1) * (
                optimized_translation - reference_translation
            )
            blended = projection_terms(
                contact,
                weight,
                collision,
                blended_rotation,
                blended_translation,
                reference_rotation,
                reference_translation,
                sdf,
                config,
            )
            applied = (
                (improvement >= config.minimum_improvement)
                & (ratio >= config.minimum_improvement_ratio)
                & (alpha > 0.0001)
                & (blended["total"] < initial["total"])
            )
            latent = torch.cat(
                (
                    blended_rotation.reshape(-1, POSE_ROTATION_DIM),
                    blended_translation,
                    proposal[:, POSE_DIM:LATENT_DIM],
                ),
                dim=1,
            )
        return {
            "latent": latent,
            "applied": applied,
            "blend": alpha,
            "initial_sdf": initial["sdf_mean"],
            "final_sdf": blended["sdf_mean"],
        }


@dataclass(frozen=True)
class HybridStageFourConfig:
    gain_confidence: float = 0.25
    gain_margin: float = 0.0
    consistency_threshold: float = -0.25
    network_physics_threshold: float = 0.50
    consistency_minimum_norm: float = 0.000001
    physical_output_alpha: float = 0.25


def grouped_direction_consistency(
    first: torch.Tensor,
    second: torch.Tensor,
    minimum_norm: float,
    empty_value: float,
) -> torch.Tensor:
    values = []
    validities = []
    for start, end in ((0, 3), (3, 6)):
        first_group = first[:, start:end]
        second_group = second[:, start:end]
        first_norm = torch.linalg.vector_norm(first_group, dim=1)
        second_norm = torch.linalg.vector_norm(second_group, dim=1)
        valid = (first_norm > minimum_norm) & (second_norm > minimum_norm)
        cosine = functional.cosine_similarity(first_group, second_group, dim=1, eps=0.000000000001)
        values.append(torch.where(valid, cosine, torch.zeros_like(cosine)))
        validities.append(valid.float())
    value = torch.stack(values, dim=1).sum(dim=1)
    count = torch.stack(validities, dim=1).sum(dim=1)
    return torch.where(count > 0, value / count.clamp_min(1), torch.full_like(value, empty_value))


class HybridStageFour(nn.Module):
    def __init__(
        self,
        proposal_config: ContactProposalConfig = ContactProposalConfig(),
        projection_config: SDFProjectionConfig = SDFProjectionConfig(),
        hybrid_config: HybridStageFourConfig = HybridStageFourConfig(),
    ):
        super().__init__()
        self.proposal_config = proposal_config
        self.hybrid_config = hybrid_config
        self.proposal = ContactProposalNetwork(37, proposal_config)
        self.projector = SDFProjector(projection_config)

    def forward(
        self,
        base: torch.Tensor,
        hand: torch.Tensor,
        object_surface: torch.Tensor,
        gp_confidence: torch.Tensor,
        gp_residual: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        base = project_latent(base)
        features = proposal_features(base, hand, object_surface, gp_confidence, self.proposal_config)
        prediction = self.proposal(features)
        network = apply_proposal(base, prediction)
        physical = self.projector.refine(network, hand, object_surface)
        angle_bound = prediction["angle_gain_mean"].reshape(-1) - self.hybrid_config.gain_confidence * prediction[
            "angle_gain_std"
        ].reshape(-1)
        distance_bound = prediction["distance_gain_mean"].reshape(-1) - self.hybrid_config.gain_confidence * prediction[
            "distance_gain_std"
        ].reshape(-1)
        gain_ok = (angle_bound > self.hybrid_config.gain_margin) & (
            distance_bound > self.hybrid_config.gain_margin
        )
        base_rotation = base[:, :POSE_ROTATION_DIM].reshape(-1, 3, 3)
        network_rotation = network[:, :POSE_ROTATION_DIM].reshape(-1, 3, 3)
        physical_rotation = physical["latent"][:, :POSE_ROTATION_DIM].reshape(-1, 3, 3)
        final_rotation = blend_rotation(
            network_rotation,
            physical_rotation,
            torch.full(
                (network.shape[0], 1),
                self.hybrid_config.physical_output_alpha,
                device=network.device,
            ),
        )
        final_translation = network[:, POSE_ROTATION_DIM:POSE_DIM] + self.hybrid_config.physical_output_alpha * (
            physical["latent"][:, POSE_ROTATION_DIM:POSE_DIM] - network[:, POSE_ROTATION_DIM:POSE_DIM]
        )
        candidate = torch.cat(
            (final_rotation.reshape(-1, POSE_ROTATION_DIM), final_translation, base[:, POSE_DIM:]),
            dim=1,
        )
        candidate_rotation = candidate[:, :POSE_ROTATION_DIM].reshape(-1, 3, 3)
        candidate_delta = torch.cat(
            (
                so3_log(candidate_rotation @ base_rotation.transpose(1, 2)),
                candidate[:, POSE_ROTATION_DIM:POSE_DIM] - base[:, POSE_ROTATION_DIM:POSE_DIM],
            ),
            dim=1,
        )
        network_delta = torch.cat(
            (
                so3_log(network_rotation @ base_rotation.transpose(1, 2)),
                network[:, POSE_ROTATION_DIM:POSE_DIM] - base[:, POSE_ROTATION_DIM:POSE_DIM],
            ),
            dim=1,
        )
        gp_consistency = grouped_direction_consistency(
            candidate_delta,
            gp_residual[:, :6],
            self.hybrid_config.consistency_minimum_norm,
            float("-inf"),
        )
        network_physics_consistency = grouped_direction_consistency(
            candidate_delta,
            network_delta,
            self.hybrid_config.consistency_minimum_norm,
            1.0,
        )
        consistency_ok = (
            (gp_consistency >= self.hybrid_config.consistency_threshold)
            & (network_physics_consistency >= self.hybrid_config.network_physics_threshold)
        )
        accepted = physical["applied"] & gain_ok & consistency_ok
        output = torch.where(accepted.unsqueeze(1), candidate, network)
        return {
            "latent": output,
            "network": network,
            "physical": physical["latent"],
            "accepted": accepted,
            "gain_ok": gain_ok,
            "gp_consistency": gp_consistency,
            "network_physics_consistency": network_physics_consistency,
            "blend": physical["blend"],
        }
