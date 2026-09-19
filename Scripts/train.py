#!/usr/bin/env python3
"""Minimal sequential training of the three inference models, initialized with seed 42."""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import random

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

if __name__ == '__main__':
    print('Starting training script; loading PyTorch and model modules...', flush=True)

import numpy as np
import torch
import torch.nn.functional as F
import gpytorch
from safetensors import safe_open
from safetensors.torch import save_file

from diffusion import PoseTokenDiffusionConfig, PoseTokenDiffusionModel, extract
from gp import (GPFeatureConfig, GP_GROUPS, GroupedResidualGPModel,
                build_gp_features, deterministic_diffusion_noise, project_latent_rotation,
                so3_log_map, correct_batch)
from proposal import (ContactProposalConfig, ContactProposalNetwork,
                      build_proposal_features, apply_proposal)

ROOT = Path(__file__).resolve().parent
SEED = 42
TENSORS = {'index', 'participant_id', 'mask_pose', 'mask_force', 'scale',
           'hand_code', 'object_code', 'target', 'object_data', 'hand_som'}
LABELS = {'participant', 'source_type', 'trial_key', 'section', 'object_name'}
# Structural dimensions match the inference release; these are not fitted statistics.
DIFFUSION = PoseTokenDiffusionConfig(object_code_dim=512)
PROPOSAL = ContactProposalConfig()
GP_CONFIG = {
    'features': asdict(GPFeatureConfig(hand_projection_dim=32, object_projection_dim=32,
                                     projection_seed=SEED, geometry_hand_points=64,
                                     geometry_object_points=256)),
    'fixed_noise_seed': SEED, 'max_correction_sigma': 2.5,
}


def seed_everything():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def save_tensors(path, tensors):
    save_file({key: value.detach().cpu().contiguous().clone() for key, value in tensors.items()}, str(path))


def load_batch(path):
    """Read ten tensors and five JSON labels; accept a legacy duplicate sample_index."""
    with safe_open(str(path), framework='pt', device='cpu') as handle:
        if set(handle.keys()) != TENSORS:
            raise ValueError(f'{path}: expected exactly the ten inference tensors.')
        data = {key: handle.get_tensor(key) for key in handle.keys()}
    labels = json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
    n = len(data['index'])
    if set(labels) not in (LABELS, LABELS | {'sample_index'}) or any(not isinstance(v, list) or len(v) != n for v in labels.values()):
        raise ValueError(f'{path}: expected five aligned JSON label fields and optional legacy sample_index.')
    if 'sample_index' in labels and labels['sample_index'] != data['index'].reshape(-1).tolist():
        raise ValueError(f'{path}: sample_index does not match tensor index.')
    if any(v not in ('human', 'robot') for v in labels['source_type']):
        raise ValueError(f'{path}: invalid source_type.')
    if any(not isinstance(v, str) or not v.strip() for v in labels['participant']):
        raise ValueError(f'{path}: empty participant identity.')
    shapes = {'index': (n,), 'participant_id': (n,), 'mask_pose': (n, 1),
              'mask_force': (n, 1), 'scale': (n, 1), 'hand_code': (n, 512),
              'object_code': (n, 512), 'target': (n, 14),
              'object_data': (n, 2048, 3), 'hand_som': (n, 778, 4)}
    for key, shape in shapes.items():
        dtype = torch.int64 if key in ('index', 'participant_id') else torch.float32
        if tuple(data[key].shape) != shape or data[key].dtype != dtype:
            raise ValueError(f'{path}: {key} must have shape {shape} and dtype {dtype}.')
    for key in ('mask_pose', 'mask_force'):
        if not ((data[key] == 0) | (data[key] == 1)).all():
            raise ValueError(f'{path}: {key} must be binary.')
    mask = target_mask(data).bool()
    if not mask.any(dim=1).all() or not torch.isfinite(data['target'][mask]).all():
        raise ValueError(f'{path}: every sample needs a finite, valid target.')
    for key in ('hand_code', 'object_code', 'object_data', 'hand_som', 'scale'):
        if not torch.isfinite(data[key]).all():
            raise ValueError(f'{path}: non-finite {key}.')
    if not (data['scale'][data['mask_pose'] > 0.5] > 0).all():
        raise ValueError(f'{path}: pose-valid scale must be positive.')
    return data


def target_mask(data):
    return torch.cat([data['mask_pose'].expand(-1, 12), data['mask_force'].expand(-1, 2)], dim=1)


def batches(paths, batch_size, device, shuffle=False):
    order = torch.randperm(len(paths)).tolist() if shuffle else range(len(paths))
    for shard_number, i in enumerate(order, 1):
        print(f'Loading shard {shard_number}/{len(paths)}: {paths[i].name}', flush=True)
        data = load_batch(paths[i])
        rows = torch.randperm(len(data['index'])) if shuffle else torch.arange(len(data['index']))
        for selected in rows.split(batch_size):
            yield {key: value[selected].to(device) for key, value in data.items()}


def update(loss, optimizer, model):
    if not torch.isfinite(loss):
        raise ValueError('Non-finite training loss.')
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
    optimizer.step()


def train_diffusion(model, paths, args, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    model.train()
    diffusion = model.diffusion
    for epoch in range(args.epochs):
        print(f'diffusion epoch {epoch + 1}/{args.epochs}: starting', flush=True)
        total, count = 0.0, 0
        for data in batches(paths, args.batch_size, device, shuffle=True):
            mask = target_mask(data)
            # Missing task labels never enter either the noisy target or the loss.
            target = torch.where(mask.bool(), data['target'], 0.0)
            x0 = target * diffusion.data_scale + diffusion.data_shift
            time = torch.randint(0, diffusion.num_timesteps, (len(target),), device=device)
            noisy = (extract(diffusion.sqrt_alphas_cumprod, time, x0.shape) * x0
                     + extract(diffusion.sqrt_one_minus_alphas_cumprod, time, x0.shape) * torch.randn_like(x0))
            tokens = model.encode_condition_tokens(data['hand_code'], data['object_code'],
                                                  (data['mask_force'][:, 0] > 0.5).long())
            prediction = diffusion.model(noisy, time, tokens)
            loss = ((prediction - x0).square() * mask).sum() / mask.sum()
            update(loss, optimizer, model)
            total += loss.item()
            count += 1
            if count == 1 or count % 10 == 0:
                print(f'diffusion epoch {epoch + 1}/{args.epochs}, batch {count}: loss={loss.item():.6f}', flush=True)
        print(f'diffusion epoch {epoch + 1}: loss={total / count:.6f}', flush=True)
    model.eval()


@torch.no_grad()
def gp_training_data(diffusion, paths, args, device):
    features, residuals, masks = [], [], []
    print('GP: generating diffusion predictions and residual training data...', flush=True)
    for batch_number, data in enumerate(batches(paths, args.batch_size, device), 1):
        if batch_number == 1 or batch_number % 10 == 0:
            print(f'GP feature extraction: batch {batch_number}', flush=True)
        noise = deterministic_diffusion_noise(data['index'], 14, seed=SEED, device=device)
        pred = diffusion.generate_latent(data['hand_code'], data['object_code'], noise=noise,
                                        task_code=(data['mask_force'][:, 0] > 0.5).long())
        base = project_latent_rotation(pred)
        residual = torch.zeros(len(pred), 8, device=device)
        pose = data['mask_pose'][:, 0] > 0.5
        force = data['mask_force'][:, 0] > 0.5
        truth = project_latent_rotation(data['target'][pose])
        residual[pose, :3] = so3_log_map(truth[:, :9].reshape(-1, 3, 3)
                                        @ base[pose, :9].reshape(-1, 3, 3).transpose(-1, -2))
        residual[pose, 3:6] = truth[:, 9:12] - base[pose, 9:12]
        residual[force, 6:8] = data['target'][force, 12:14] - base[force, 12:14]
        features.append(build_gp_features(data['hand_code'], data['object_code'],
                                          data['hand_som'], data['object_data'], pred,
                                          GP_CONFIG['features']).cpu())
        residuals.append(residual.cpu())
        masks.append(torch.cat([data['mask_pose'].expand(-1, 6), data['mask_force'].expand(-1, 2)], 1).cpu())
    return torch.cat(features), torch.cat(residuals), torch.cat(masks)


def train_gp(diffusion, paths, args, device):
    x, y, mask = gp_training_data(diffusion, paths, args, device)
    stats = {'x_center': x.mean(0, keepdim=True), 'x_scale': x.std(0, unbiased=False, keepdim=True).clamp_min(1e-6),
             'y_center': torch.zeros(1, 8), 'y_scale': torch.ones(1, 8)}
    x = (x - stats['x_center']) / stats['x_scale']
    models = []
    for name, (start, end) in GP_GROUPS.items():
        rows = torch.where(mask[:, start] > 0.5)[0]
        values = y[rows, start:end]
        stats['y_center'][:, start:end] = values.mean(0)
        stats['y_scale'][:, start:end] = values.std(0, unbiased=False).clamp_min(1e-6)
        # ExactGP fits a fixed training reference set; its size does not change its architecture.
        selected = rows[torch.randperm(len(rows))[:args.gp_max_samples]]
        train_x = x[selected].to(device)
        train_y = ((y[selected, start:end] - stats['y_center'][:, start:end])
                   / stats['y_scale'][:, start:end]).to(device)
        likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(
            num_tasks=end - start, rank=2, noise_constraint=gpytorch.constraints.GreaterThan(0.0001)).to(device)
        model = GroupedResidualGPModel(train_x, train_y, likelihood, coregionalization_rank=2).to(device)
        model.train()
        likelihood.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.gp_lr)
        objective = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
        print(f'gp {name}: optimizing {len(selected)} references for {args.gp_steps} steps', flush=True)
        for step in range(args.gp_steps):
            with gpytorch.settings.cholesky_jitter(1e-4):
                loss = -objective(model(train_x), train_y)
                update(loss, optimizer, model)
            if step == 0 or (step + 1) % 10 == 0 or step + 1 == args.gp_steps:
                print(f'gp {name}, step {step + 1}/{args.gp_steps}: loss={loss.item():.6f}', flush=True)
        print(f'gp {name}: {len(selected)} references, loss={loss.item():.6f}', flush=True)
        save_tensors(args.output_dir / f'gp_{name}.safetensors', {
            **{f'state.{key}': value for key, value in model.state_dict().items()},
            'reference_inputs': train_x, 'reference_residuals': train_y})
        model.eval()
        likelihood.eval()
        models.append(model)
    save_tensors(args.output_dir / 'gp_normalization.safetensors', stats)
    # Fixed demonstration policy: no thresholds learned from validation data.
    policies = {name: {'enabled': True, 'alpha': 1.0, 'gate_mode': 'none',
                       'std_threshold': 1.0, 'gate_temperature': 0.1} for name in GP_GROUPS}
    write_json(args.output_dir / 'gp.json', {'group_policies': policies})
    stats['group_policies'] = policies
    return models, stats


def pose_errors(pred, truth):
    relative = truth[:, :9].reshape(-1, 3, 3).transpose(-1, -2) @ pred[:, :9].reshape(-1, 3, 3)
    cosine = ((relative.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2).clamp(-1 + 1e-7, 1 - 1e-7)
    return torch.rad2deg(torch.acos(cosine)), (pred[:, 9:12] - truth[:, 9:12]).norm(dim=1)


def train_proposal(diffusion, models, stats, paths, args, device):
    feature_parts, base_parts, truth_parts = [], [], []
    print('Proposal: preparing features using the diffusion and GP models...', flush=True)
    # Upstream models are frozen. Cache just small proposal features and pose labels.
    for data in batches(paths, args.batch_size, device):
        _, corrected, saved = correct_batch(diffusion, models, stats, data, GP_CONFIG, device)
        rows = (data['mask_pose'][:, 0] > 0.5).cpu()
        if not rows.any():
            continue
        with torch.no_grad():
            base = corrected[rows].to(device)
            features = build_proposal_features(base, saved['hand_som'][rows.to(device)],
                                               saved['object_data'][rows.to(device)],
                                               saved['gp_gate'][rows].to(device), PROPOSAL)
            feature_parts.append(features.cpu())
            base_parts.append(base.cpu())
            truth_parts.append(project_latent_rotation(data['target'][rows.to(device)]).cpu())
    features, base, truth = map(torch.cat, (feature_parts, base_parts, truth_parts))
    model = ContactProposalNetwork(feature_dim=37, config=PROPOSAL).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    model.train()
    for epoch in range(args.proposal_epochs):
        print(f'proposal epoch {epoch + 1}/{args.proposal_epochs}: starting', flush=True)
        total, count = 0.0, 0
        for rows in torch.randperm(len(features)).split(args.batch_size):
            x, b, y = (value[rows].to(device) for value in (features, base, truth))
            prediction = model(x)
            proposed = apply_proposal(b, prediction, global_alpha=1.0)
            pose_loss = (F.mse_loss(proposed[:, :9], y[:, :9])
                         + F.mse_loss(proposed[:, 9:12], y[:, 9:12]) / PROPOSAL.max_translation_delta ** 2)
            with torch.no_grad():
                before, after = pose_errors(b, y), pose_errors(proposed, y)
                gains = [old - new for old, new in zip(before, after)]
            gain_loss = 0.0
            for name, gain in zip(('angle', 'distance'), gains):
                mean = prediction[f'{name}_gain_mean'][:, 0]
                std = prediction[f'{name}_gain_std'][:, 0]
                gain_loss = gain_loss + (std.log() + 0.5 * ((gain - mean) / std).square()).mean()
            loss = pose_loss + gain_loss
            update(loss, optimizer, model)
            total += loss.item()
            count += 1
            if count == 1 or count % 10 == 0:
                print(f'proposal epoch {epoch + 1}/{args.proposal_epochs}, batch {count}: loss={loss.item():.6f}', flush=True)
        print(f'proposal epoch {epoch + 1}: loss={total / count:.6f}', flush=True)
    save_tensors(args.output_dir / 'proposal.safetensors', model.state_dict())
    write_json(args.output_dir / 'proposal.json', {
        'feature_dim': 37, 'config': asdict(PROPOSAL),
        'policy': {'enabled': True, 'global_alpha': 1.0, 'gain_confidence_k': 0.25,
                   'gain_margin': 0.0, 'consistency_cosine_threshold': -0.25, 'consistency_min_norm': 1e-6}})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'data',
                        help='Training shards directory (default: data beside train.py).')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'models',
                        help='Model output directory (default: Scripts/models).')
    parser.add_argument('--device', default='cuda:1', help='Training device (default: cuda:1).')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--proposal-epochs', type=int, default=1)
    parser.add_argument('--gp-steps', type=int, default=20)
    parser.add_argument('--gp-max-samples', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--gp-lr', type=float, default=0.05)
    args = parser.parse_args(argv)
    print(f'Requested device: {args.device}; data: {args.data_dir}; output: {args.output_dir}', flush=True)
    device = torch.device(args.device)
    if device.type == 'cuda':
        visible_count = torch.cuda.device_count()
        selected_device = device.index if device.index is not None else 0
        if not torch.cuda.is_available() or selected_device >= visible_count:
            parser.error(f'{args.device} is unavailable; {visible_count} CUDA devices are visible. '
                         'Check CUDA_VISIBLE_DEVICES or set --device explicitly.')
        torch.cuda.set_device(device)
        print(f'Using {device}: {torch.cuda.get_device_name(device)}', flush=True)
    for key in ('batch_size', 'epochs', 'proposal_epochs', 'gp_steps', 'gp_max_samples', 'lr', 'gp_lr'):
        if not np.isfinite(getattr(args, key)) or getattr(args, key) <= 0:
            parser.error(f'--{key.replace("_", "-")} must be positive and finite.')
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error('Use an empty output directory.')
    manifest = args.data_dir / 'manifest.json'
    if manifest.exists():
        names = json.loads(manifest.read_text(encoding='utf-8'))['shards']
        if any(Path(name).name != name for name in names):
            parser.error('Shard names must be plain filenames.')
        paths = [args.data_dir / name for name in names]
    else:
        paths = sorted(args.data_dir.glob('*.safetensors'))
        paths = [path for path in paths if path.name != 'sdf.safetensors']
    if not paths or len(set(paths)) != len(paths):
        parser.error('Provide unique safetensors shards with matching JSON files.')
    seen, pose_count, force_count = set(), 0, 0
    print(f'Checking {len(paths)} data shards before training...', flush=True)
    for shard_number, path in enumerate(paths, 1):
        print(f'Checking shard {shard_number}/{len(paths)}: {path.name}', flush=True)
        data = load_batch(path)
        indices = data['index'].tolist()
        if len(set(indices)) != len(indices) or seen.intersection(indices):
            parser.error('Sample indices must be unique within and across shards.')
        seen.update(indices)
        pose_count += int(data['mask_pose'].sum())
        force_count += int(data['mask_force'].sum())
        print(f'  Loaded {len(indices)} samples', flush=True)
    if not pose_count or not force_count:
        parser.error('Training all models requires both pose-valid and force-valid samples.')
    del data
    print(f'Data ready: {len(seen)} samples; pose-valid={pose_count}, force-valid={force_count}', flush=True)
    seed_everything()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f'Initializing diffusion model on {device}...', flush=True)
    diffusion = PoseTokenDiffusionModel(DIFFUSION).to(device)
    train_diffusion(diffusion, paths, args, device)
    save_tensors(args.output_dir / 'diffusion.safetensors', diffusion.state_dict())
    write_json(args.output_dir / 'diffusion.json', asdict(DIFFUSION))
    models, stats = train_gp(diffusion, paths, args, device)
    train_proposal(diffusion, models, stats, paths, args, device)
    write_json(args.output_dir / 'training.json', {
        'seed': SEED, 'samples': len(seen), 'gp': GP_CONFIG,
        'epochs': args.epochs, 'proposal_epochs': args.proposal_epochs,
        'gp_steps': args.gp_steps, 'gp_max_samples': args.gp_max_samples,
        'batch_size': args.batch_size, 'lr': args.lr, 'gp_lr': args.gp_lr})
    print(f'Trained all three models from initialization on {len(seen)} samples: {args.output_dir}', flush=True)


if __name__ == '__main__':
    main()
