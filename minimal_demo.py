from pathlib import Path

import torch

from stage_1 import FullAfferentEncoder, SimpleRawAfferentEncoder
from stage_2 import StageTwoConfig, TokenDiffusion
from stage_3 import GPConfig, GroupedResidualGP, gp_features, project_rotation
from stage_4 import HybridStageFour, proposal_loss


def pose12(pose: torch.Tensor) -> torch.Tensor:
    if pose.shape[1] == 16:
        rotation = pose[:, (0, 1, 2, 4, 5, 6, 8, 9, 10)]
        translation = pose[:, (3, 7, 11)]
    elif pose.shape[1] == 12:
        rotation = pose[:, :9]
        translation = pose[:, 9:12]
    else:
        raise ValueError(f"Unsupported pose shape {tuple(pose.shape)}")
    return torch.cat((project_rotation(rotation).reshape(-1, 9), translation), dim=1)


def sensor_wrench(hand: torch.Tensor) -> torch.Tensor:
    xyz = hand[:, :, :3].float()
    pressure = hand[:, :, 3].float().clamp_min(0)
    force = pressure.mean(dim=1, keepdim=True)
    center = xyz.mean(dim=1, keepdim=True)
    load = torch.zeros_like(xyz)
    load[:, :, 2] = pressure
    torque = torch.linalg.vector_norm(torch.linalg.cross(xyz - center, load, dim=2), dim=2).mean(
        dim=1,
        keepdim=True,
    )
    return torch.cat((force, torque), dim=1)


def joint_targets(pose: torch.Tensor, hand: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    wrench = sensor_wrench(hand)
    pose_target = torch.cat((pose, torch.zeros(pose.shape[0], 2, device=pose.device)), dim=1)
    force_target = torch.cat((pose, wrench), dim=1)
    pose_mask = torch.cat(
        (
            torch.ones(pose.shape[0], 12, device=pose.device),
            torch.zeros(pose.shape[0], 2, device=pose.device),
        ),
        dim=1,
    )
    force_mask = torch.cat(
        (
            torch.zeros(pose.shape[0], 12, device=pose.device),
            torch.ones(pose.shape[0], 2, device=pose.device),
        ),
        dim=1,
    )
    target = torch.cat((pose_target, force_target), dim=0)
    mask = torch.cat((pose_mask, force_mask), dim=0)
    task = torch.cat(
        (
            torch.zeros(pose.shape[0], device=pose.device, dtype=torch.long),
            torch.ones(pose.shape[0], device=pose.device, dtype=torch.long),
        )
    )
    return target, mask, task


def main() -> None:
    root = Path(__file__).resolve().parent
    values = torch.load(root / "data" / "demo.pt", map_location="cpu", weights_only=True)
    if set(values) != {"hand", "object", "pose"}:
        raise ValueError("The demo file must contain only hand, object, and pose")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    count = min(2, len(values["pose"]))
    hand = values["hand"][:count].float().to(device)
    object_surface = values["object"][:count].float().to(device)
    pose = pose12(values["pose"][:count].float().to(device))
    full_encoder = FullAfferentEncoder()
    full_parameters = sum(parameter.numel() for parameter in full_encoder.parameters())
    del full_encoder
    raw_encoder = SimpleRawAfferentEncoder().to(device)
    with torch.no_grad():
        hand_code, object_code = raw_encoder(hand, object_surface)
    target, mask, task = joint_targets(pose, hand)
    hand_code = torch.cat((hand_code, hand_code), dim=0)
    object_code = torch.cat((object_code, object_code), dim=0)
    hand_joint = torch.cat((hand, hand), dim=0)
    object_joint = torch.cat((object_surface, object_surface), dim=0)
    diffusion = TokenDiffusion(StageTwoConfig()).to(device)
    optimizer = torch.optim.Adam(diffusion.parameters(), lr=0.00001)
    diffusion.train()
    optimizer.zero_grad()
    loss = diffusion.training_loss(target, mask, hand_code, object_code, task)
    loss.backward()
    optimizer.step()
    diffusion.eval()
    initial = diffusion.sample(hand_code, object_code, task)
    features = gp_features(hand_code, object_code, hand_joint, object_joint, initial)
    grouped_gp = GroupedResidualGP(GPConfig(iterations=1, max_points=32))
    grouped_gp.fit(features, target, initial, mask)
    corrected = grouped_gp.correct(features, initial)
    stage_four = HybridStageFour().to(device)
    pose_base = corrected["latent"][:count]
    pose_target = target[:count]
    pose_confidence = corrected["confidence"][:count]
    proposal_optimizer = torch.optim.Adam(stage_four.proposal.parameters(), lr=0.0003)
    stage_four.train()
    proposal_optimizer.zero_grad()
    loss_four = proposal_loss(
        stage_four.proposal,
        pose_base,
        pose_target,
        hand,
        object_surface,
        pose_confidence,
    )
    loss_four.backward()
    proposal_optimizer.step()
    stage_four.eval()
    refined = stage_four(
        pose_base,
        hand,
        object_surface,
        pose_confidence,
        corrected["residual"][:count],
    )
    print(
        {
            "samples": count,
            "stage_1_full": (full_parameters, 512),
            "stage_1_raw": (tuple(hand_code.shape), tuple(object_code.shape)),
            "stage_2": tuple(initial.shape),
            "stage_3": (tuple(features.shape), tuple(corrected["residual"].shape)),
            "stage_4": (tuple(refined["latent"].shape), int(refined["accepted"].sum().item())),
        }
    )


if __name__ == "__main__":
    main()
