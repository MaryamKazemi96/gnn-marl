print("GNNPPOEnvWrapper imported successfully!")

"""
Minimal wrapper for GNN-PPO training with the new MultiAgentTaskEnv.

The new MultiAgentTaskEnv already:
- Generates graph observations natively
- Handles action decoding
- Provides action masking
- Manages rewards

This wrapper just provides SB3 compatibility.
"""
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, Any, Tuple, Optional

from src.environment.environment import MultiAgentTaskEnv
class GNNPPOEnvWrapper(gym.Env):
    """
    Minimal wrapper for SB3 MaskablePPO compatibility.
    """

    def __init__(
        self,
        agents: np.ndarray,
        tasks_batches: list,
        K_max: int = 5,
        N_max: int = 15,
        E_max: int = 50,
        use_xy_pickup: bool = False,
        normalize_features: bool = True,
        use_node_type: bool = True,
        use_ego_robot: bool = True,
        use_edge_rt: bool = False,
        edge_features: Optional[list] = None,
        two_hop: bool = False,
        two_hop_directed: bool = False,
        vicinity_m: float = 50.0,
        max_steps: int = 1000,
        max_robot_capacity: int = 2,
        max_wait_delay_s: float = 600.0,
        max_travel_delay_s: float = 3600.0,
    ):
        super().__init__()

        self.base_env = MultiAgentTaskEnv(
            agents=agents,
            tasks_batches=tasks_batches,
            K_max=K_max,
            N_max=N_max,
            E_max=E_max,
            use_xy_pickup=use_xy_pickup,
            normalize_features=normalize_features,
            use_node_type=use_node_type,
            use_ego_robot=use_ego_robot,
            use_edge_rt=use_edge_rt,
            edge_features=edge_features or [],
            two_hop=two_hop,
            two_hop_directed=two_hop_directed,
            vicinity_m=vicinity_m,
            max_steps=max_steps,
            max_robot_capacity=max_robot_capacity,
            max_wait_delay_s=max_wait_delay_s,
            max_travel_delay_s=max_travel_delay_s,
        )

        self.observation_space = self.base_env.observation_space
        self.action_space      = self.base_env.action_space
        self._noop_index       = K_max
        self._episode_return   = 0.0

    # ------------------------------------------------------------------ #
    #  Helpers                                                             #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _fix_info(info: Dict) -> Dict:
        """Rename action_mask → action_masks for MaskablePPO."""
        if "action_mask" in info:
            info["action_masks"] = info.pop("action_mask")
        return info

    # ------------------------------------------------------------------ #
    #  Gym API                                                             #
    # ------------------------------------------------------------------ #

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None):
        super().reset(seed=seed)
        obs, info = self.base_env.reset(seed=seed)
        self._episode_return = 0.0
        return obs, self._fix_info(info)

    def step(self, action: np.ndarray) -> Tuple[Dict, float, bool, bool, Dict]:
        obs, reward, terminated, truncated, info = self.base_env.step(action)

        self._episode_return += reward

        if terminated or truncated:
            info["episode_return"]   = self._episode_return
            info["episode"] = {"r": self._episode_return}  # SB3 Monitor convention
            self._episode_return = 0.0

        return obs, reward, terminated, truncated, self._fix_info(info)

    def action_mask(self) -> np.ndarray:
        """Convenience method — calls base env directly."""
        return self.base_env.action_mask()

    def close(self):
        if hasattr(self.base_env, "close"):
            self.base_env.close()


# ============================================================================
# Testing
# ============================================================================

if __name__ == "__main__":
    import yaml
    from pathlib import Path

    config_path = Path("configs/training_config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    from train_ppo import load_generated_data
    agents, tasks_batches = load_generated_data("../data")

    env = GNNPPOEnvWrapper(
        agents=agents,
        tasks_batches=tasks_batches,
        K_max=config.get("K_max", 5),
        N_max=config.get("N_max", 15),
        E_max=config.get("E_max", 50),
        use_xy_pickup=config.get("use_xy_pickup", False),
        normalize_features=config.get("normalize_features", True),
        use_node_type=config.get("use_node_type", True),
        use_ego_robot=config.get("use_ego_robot", True),
        use_edge_rt=config.get("use_edge_rt", False),
        two_hop=config.get("two_hop", False),
        vicinity_m=config.get("vicinity_m", 40.0),
        max_steps=config.get("max_steps", 1000),
    )

    # ── Reset ──────────────────────────────────────────────────────────
    obs, info = env.reset()
    print("✓ Reset successful")
    print(f"  Observation keys : {sorted(obs.keys())}")
    print(f"  Action space     : {env.action_space}")
    assert "action_masks" in info, "action_masks missing from reset info"
    print(f"  action_masks shape: {info['action_masks'].shape}")

    # ── Observation invariants ─────────────────────────────────────────
    for k in ["x", "node_mask", "edge_index", "edge_mask", "cand_idx", "cand_mask"]:
        v = obs[k]
        print(f"  {k:12s} shape={v.shape} dtype={v.dtype}")

    R = obs["x"].shape[0]
    K = obs["cand_idx"].shape[1]

    for r in range(R):
        n = int(obs["node_mask"][r].sum())
        assert obs["node_mask"][r, 0] == 1, f"Robot {r} missing at node 0"
        for kk in range(K):
            if obs["cand_mask"][r, kk]:
                idx = int(obs["cand_idx"][r, kk])
                assert 0 < idx < N_max, f"Robot {r} cand_idx {idx} out of range"
                assert obs["node_mask"][r, idx] == 1, \
                    f"Robot {r} cand_idx {idx} points to masked node"
    print("✓ Ego-graph + candidate invariants OK")

    # ── Action mask invariants ─────────────────────────────────────────
    masks = info["action_masks"]  # (R, K_max+1)
    assert masks.shape == (R, K + 1), f"Mask shape mismatch: {masks.shape}"
    for r in range(R):
        assert masks[r, -1] == 1, f"Robot {r} NO-OP not always valid"
        for kk in range(K):
            if masks[r, kk]:
                assert obs["cand_mask"][r, kk] == 1, \
                    f"Robot {r}: action_mask valid but cand_mask=0 at slot {kk}"
    print("✓ Action mask invariants OK")

    # ── Step ───────────────────────────────────────────────────────────
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    assert "action_masks" in info, "action_masks missing from step info"
    print(f"✓ Step successful — reward={reward:.3f} terminated={terminated}")

    # ── Full random episode ────────────────────────────────────────────
    obs, info = env.reset()
    ep_steps, ep_return = 0, 0.0
    while True:
        masks  = info["action_masks"]           # (R, K+1)
        action = np.array([
            np.random.choice(np.where(masks[r])[0])
            for r in range(R)
        ])
        obs, reward, terminated, truncated, info = env.step(action)
        ep_return += reward
        ep_steps  += 1
        if terminated or truncated:
            break

    print(f"✓ Full episode done — steps={ep_steps} return={ep_return:.2f}")
    assert "episode_return" in info, "episode_return missing from terminal info"

    env.close()
    print("✓ All wrapper tests passed!")