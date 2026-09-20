# Som-Driven

Minimal downstream model code for somatosensory-driven state estimation in hand–object interaction. The implementation combines conditional diffusion, grouped Gaussian-process residual correction, a learned contact-based pose proposal, and signed-distance-field (SDF) pose refinement.

# Code layout

| File | Role |
| --- | --- |
| [diffusion.py](diffusion.py) | Conditional transformer diffusion model and DDIM sampling. |
| [gp.py](gp.py) | GP feature construction, grouped residual models, rotation operations, and correction utilities. |
| [proposal.py](proposal.py) | Contact-based pose proposal network and predicted gain distributions. |
| [physics.py](physics.py) | SDF loading, contact selection, pose optimization, blending, and physical-update acceptance checks. |
| [train.py](train.py) | Sequential training of the diffusion model, three grouped GPs, and proposal network. |
| [requirements.txt](requirements.txt) | Python package dependencies. |


# Installation and training

Run these commands from the directory containing `train.py`, after preparing the training data:

```bash
python -m pip install -r requirements.txt
python train.py --data-dir data --output-dir models --device cpu
```

For the first visible CUDA device:

```bash
python train.py --data-dir data --output-dir models --device cuda:0
```

Set the device explicitly: the script otherwise defaults to `cuda:1`. The output directory must be absent or empty. Without path overrides, data and model directories default to `data/` and `models/` beside `train.py`.

# Saved outputs

A completed training run writes:

```text
models/
    diffusion.safetensors
    diffusion.json
    gp_rotation.safetensors
    gp_translation.safetensors
    gp_force_torque.safetensors
    gp_normalization.safetensors
    gp.json
    proposal.safetensors
    proposal.json
    training.json
```

