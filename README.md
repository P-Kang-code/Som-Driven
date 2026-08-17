# Som-Driven

This repository provides a minimal four-stage model implementation.

1. `stage_1.py` includes complete 512-dimensional variational point autoencoders with three Conv1D encoding stages and residual decoders, plus a compact 512-dimensional Conv1D encoder that accepts the bundled raw point tensors directly.
2. `stage_2.py` implements a 14-dimensional pose, force, and torque diffusion state with hand, object, and task tokens, six transformer layers, eight 64-dimensional attention heads, 1000 diffusion steps, 50 DDIM steps, and Gaussian noise.
3. `stage_3.py` builds 98-dimensional inputs and an eight-dimensional SO(3)-safe residual for three exact multitask Matérn-3/2 Gaussian processes covering rotation, translation, and force-torque.
4. `stage_4.py` combines a supervised contact proposal and heteroscedastic gain prediction with an 80-step point-surface signed-distance projection, contact and collision terms, bounded blending, GP and network consistency checks, and acceptance criteria.


```bash
pip install -r requirements.txt
python minimal_demo.py
```
