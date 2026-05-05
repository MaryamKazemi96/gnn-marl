# FCAI — Multi-Robot Task Allocation with PPO + GNN (SB3)

This repository implements a multi-robot task allocation environment (warehouse-style pickup & delivery) and trains a PPO agent with a GNN-based policy using Stable-Baselines3.

## What’s inside

- **Environment (base)**: `src/environment/environment.py`
  - Discrete-time simulation with robots, tasks, pickups/deliveries, deadlines/obsolescence, and reward shaping.
  - Uses an **assignment interval** (e.g., every 5 steps) to decide when new task assignments can be made.

- **SB3 Wrapper**: `src/environment/sb3_env_wrapper.py`
  - Converts the base environment into an SB3-compatible env.
  - Action space is **MultiDiscrete**: one action per robot.
  - Adds `info["action_mask"]`, `info["cand_task_ids"]`, reward component logs under `rew/*`, and decision-step flags.

- **Policy / Model**: `src/models/sb3_gnn_policy.py`
  - GNN-based policy (e.g., GraphSAGE/GCN) for learning from robot/task graph observations.

- **Training Callback**: `src/utils/callbacks.py`
  - Logs reward components (`rew/*`) and episode outcomes (completed/obsolete).
  - Optionally logs meaningful decision-step policy behavior:
    - `policy/noop_fraction_meaningful`
    - `policy/assigned_fraction_meaningful`
    - `policy/collision_drop_fraction_meaningful`

- **Evaluation & Plotting**
  - `eval_ppo.py`: runs evaluation (deterministic + stochastic) and saves JSON results.
  - `plot_evaluation.py`: generates per-seed and aggregate plots from eval JSON + TensorBoard logs.

## Setup

Create a Python environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Make sure the data files exist:
- `data/agents.npy`
- `data/tasks_batch_0.npy`, `data/tasks_batch_1.npy`, ...

## Configuration

Training is controlled by `configs/training_config.yaml`. Key parameters:
- `environment.assignment_interval`: how often assignment decisions are made
- `ppo.n_steps`: rollout length per PPO update
- `ppo.ent_coef`: entropy coefficient (exploration strength)

## Train

Run your training script (the repo’s training entrypoint) using the YAML config.

TensorBoard logs and checkpoints are written under:
- `checkpoints_ppo/seed_<seed>/tensorboard/`
- `checkpoints_ppo/seed_<seed>/ppo_final.zip` (and periodic saves)

## Evaluate (Deterministic vs Stochastic)

After training, generate evaluation JSONs:

```bash
python3 eval_ppo.py --checkpoint-dir checkpoints_ppo --episodes 100
```

This writes per-seed:
- `checkpoints_ppo/seed_<seed>/eval_results_deterministic.json`
- `checkpoints_ppo/seed_<seed>/eval_results_stochastic.json`

## Plot results

Generate plots (evaluation + training curves):

```bash
python3 plot_evaluation.py --checkpoint-dir checkpoints_ppo
```

Outputs:
- Per-seed plots: `checkpoints_ppo/seed_<seed>/eval_plots/`
- Aggregate plots: `checkpoints_ppo/eval_plots/`

## Notes / Common gotchas

- **Decision steps vs non-decision steps**: actions only matter every `assignment_interval` steps; many metrics should be computed only on *meaningful* decision steps (tasks + available robots).
- **NOOP action**: the last index in each robot’s discrete action head represents “no assignment”.
- If `policy/*meaningful` TensorBoard tags are missing, ensure the wrapper inserts `action_mask` into `info` on every step.

## License

Add a license if you plan to distribute this code publicly.
