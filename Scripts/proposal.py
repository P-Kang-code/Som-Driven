from __future__ import annotations
from typing import Any, Dict, Mapping, Tuple
from dataclasses import dataclass
from pathlib import Path
import math
import json
from safetensors.torch import load_file
import torch
from torch import nn
import torch.nn.functional as F
from gp import project_latent_rotation, so3_exp_map

POSE_DIM = 12
POSE_ROT_DIM = 9


@dataclass(frozen=True)
class ContactProposalConfig:
    hidden_dim: int = 256
    depth: int = 4
    dropout: float = 0.05
    contact_points: int = 48
    object_points: int = 256
    max_rotation_delta_deg: float = 8.0
    max_translation_delta: float = 0.02
    batch_size: int = 512
    gain_std_floor: float = 0.001


class ResidualBlock(nn.Module):

    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class ContactProposalNetwork(nn.Module):

    def __init__(self, feature_dim: int, config: ContactProposalConfig):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.config = config
        self.input = nn.Sequential(
            nn.LayerNorm(feature_dim), nn.Linear(feature_dim, config.hidden_dim), nn.SiLU()
        )
        self.blocks = nn.ModuleList(
            (ResidualBlock(config.hidden_dim, config.dropout) for _ in range(config.depth))
        )
        self.output = nn.Linear(config.hidden_dim, 7)
        self.gain_output = nn.Linear(config.hidden_dim, 4)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        nn.init.constant_(self.output.bias[6], -2.0)
        nn.init.zeros_(self.gain_output.weight)
        nn.init.zeros_(self.gain_output.bias)
        nn.init.constant_(self.gain_output.bias[1], -1.0)
        nn.init.constant_(self.gain_output.bias[3], -1.0)

    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self.input(features)
        for block in self.blocks:
            x = block(x)
        raw = self.output(x)
        gain_raw = self.gain_output(x.detach())
        rot_limit = math.radians(float(self.config.max_rotation_delta_deg))
        rotvec = torch.tanh(raw[:, :3]) * rot_limit
        translation = torch.tanh(raw[:, 3:6]) * float(self.config.max_translation_delta)
        gate = torch.sigmoid(raw[:, 6:7])
        angle_gain_mean = gain_raw[:, 0:1]
        angle_gain_std = F.softplus(gain_raw[:, 1:2]) + float(self.config.gain_std_floor)
        distance_gain_mean = gain_raw[:, 2:3]
        distance_gain_std = F.softplus(gain_raw[:, 3:4]) + float(self.config.gain_std_floor)
        return {
            "rotvec": rotvec,
            "translation": translation,
            "gate": gate,
            "angle_gain_mean": angle_gain_mean,
            "angle_gain_std": angle_gain_std,
            "distance_gain_mean": distance_gain_mean,
            "distance_gain_std": distance_gain_std,
        }


def _sample_evenly(points: torch.Tensor, count: int) -> torch.Tensor:
    if points.shape[1] <= count:
        return points
    index = torch.linspace(0, points.shape[1] - 1, steps=count, device=points.device).round().long()
    return points.index_select(1, index)


def build_proposal_features(
    base_latent14: torch.Tensor,
    hand_data: torch.Tensor,
    object_data: torch.Tensor,
    gp_confidence: torch.Tensor,
    config: ContactProposalConfig,
) -> torch.Tensor:
    base = project_latent_rotation(base_latent14.float())
    rotation = base[:, :POSE_ROT_DIM].reshape(-1, 3, 3)
    translation = base[:, POSE_ROT_DIM:POSE_DIM]
    hand_xyz = hand_data[..., :3].float()
    pressure = hand_data[..., 3].float().clamp_min(0.0)
    count = min(int(config.contact_points), hand_xyz.shape[1])
    pressure_top, indices = torch.topk(pressure, k=count, dim=1, largest=True, sorted=False)
    contact_world = torch.gather(hand_xyz, 1, indices.unsqueeze(-1).expand(-1, -1, 3))
    contact_local = (contact_world - translation[:, None, :]) @ rotation
    object_local = _sample_evenly(object_data[..., :3].float(), int(config.object_points))
    distances = torch.cdist(contact_local, object_local)
    nearest_distance, nearest_index = distances.min(dim=2)
    nearest_point = torch.gather(object_local, 1, nearest_index.unsqueeze(-1).expand(-1, -1, 3))
    displacement = nearest_point - contact_local
    weights = pressure_top + 1e-06
    weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-06)
    geometry = torch.cat(
        [
            contact_local.mean(dim=1),
            contact_local.std(dim=1),
            displacement.mean(dim=1),
            displacement.std(dim=1),
            torch.stack(
                [
                    nearest_distance.mean(dim=1),
                    nearest_distance.std(dim=1),
                    nearest_distance.amin(dim=1),
                    nearest_distance.amax(dim=1),
                    (nearest_distance * weights).sum(dim=1),
                ],
                dim=1,
            ),
            torch.stack(
                [pressure.mean(dim=1), pressure.amax(dim=1), (pressure > 0).float().mean(dim=1)],
                dim=1,
            ),
            torch.stack(
                [gp_confidence.mean(dim=1), gp_confidence.amin(dim=1), gp_confidence.amax(dim=1)],
                dim=1,
            ),
        ],
        dim=1,
    )
    return torch.cat([base, geometry], dim=1)


def apply_proposal(
    base_latent14: torch.Tensor, prediction: Mapping[str, torch.Tensor], *, global_alpha: float
) -> torch.Tensor:
    base = project_latent_rotation(base_latent14.float())
    alpha = prediction["gate"] * float(global_alpha)
    base_rotation = base[:, :POSE_ROT_DIM].reshape(-1, 3, 3)
    delta_rotation = so3_exp_map(prediction["rotvec"] * alpha)
    proposed_rotation = (delta_rotation @ base_rotation).reshape(-1, POSE_ROT_DIM)
    proposed_translation = base[:, POSE_ROT_DIM:POSE_DIM] + prediction["translation"] * alpha
    return torch.cat([proposed_rotation, proposed_translation, base[:, POSE_DIM:]], dim=1)


def load_contact_proposal_network(
    checkpoint_path: Path, device: torch.device
) -> Tuple[ContactProposalNetwork, ContactProposalConfig, Dict[str, Any]]:
    payload = json.loads(checkpoint_path.with_suffix(".json").read_text(encoding="utf-8"))
    config = ContactProposalConfig(
        **payload["config"]
    )
    model = ContactProposalNetwork(int(payload["feature_dim"]), config).to(device)
    model.load_state_dict(load_file(str(checkpoint_path), device=str(device)))
    model.eval()
    return (model, config, dict(payload["policy"]))


@torch.no_grad()
def generate_contact_proposal(
    model: ContactProposalNetwork,
    config: ContactProposalConfig,
    policy: Mapping[str, Any],
    data: Mapping[str, Any],
    *,
    device: torch.device,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    output = data["corrected"].clone()
    diagnostics = {
        "angle_gain_mean": torch.zeros(output.shape[0], dtype=torch.float32),
        "angle_gain_std": torch.full((output.shape[0],), float("inf"), dtype=torch.float32),
        "distance_gain_mean": torch.zeros(output.shape[0], dtype=torch.float32),
        "distance_gain_std": torch.full((output.shape[0],), float("inf"), dtype=torch.float32),
    }
    rows = torch.nonzero(data["saved"]["mask_pose"].reshape(-1) > 0.5).reshape(-1)
    alpha = float(policy.get("global_alpha", 0.0)) if policy.get("enabled", False) else 0.0
    if alpha <= 0.0:
        return (output, diagnostics)
    saved = data["saved"]
    for start in range(0, rows.numel(), config.batch_size):
        batch_rows = rows[start : start + config.batch_size]
        base = data["corrected"][batch_rows].to(device)
        hand = saved["hand_som"][batch_rows].to(device)
        obj = saved["object_data"][batch_rows].to(device)
        gp = saved["gp_gate"][batch_rows].to(device)
        features = build_proposal_features(base, hand, obj, gp, config)
        prediction = model(features)
        output[batch_rows] = apply_proposal(base, prediction, global_alpha=alpha).cpu()
        for name in (
            "angle_gain_mean",
            "angle_gain_std",
            "distance_gain_mean",
            "distance_gain_std",
        ):
            diagnostics[name][batch_rows] = prediction[name].reshape(-1).cpu()
    return (output, diagnostics)
