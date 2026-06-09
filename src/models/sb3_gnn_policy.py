from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, List, cast, Literal

import torch as th
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

# IMPORTANT: import your actor-critic module here
# It must implement: logits_rk, value = self.gnn_ac(obs_one_dict)
# logits_rk shape: [R, K] (candidate-only)
from src.models.actor_critic import EgoActorCritic  # <-- adjust path to your project


class DictPassthroughExtractor(BaseFeaturesExtractor):
    """
    Passthrough extractor: keep raw dict observations for the custom policy.
    SB3 requires a tensor output, so we return a dummy tensor.
    """
    def __init__(self, observation_space: spaces.Dict):
        super().__init__(observation_space, features_dim=1)
        self.last_obs: Optional[Dict[str, th.Tensor]] = None

    def forward(self, obs):
        if isinstance(obs, dict):
            obs = {
                k: (v.detach() if th.is_tensor(v) else v)
                for k, v in obs.items()
            }

        self.last_obs = obs

        any_tensor = next(iter(obs.values()))
        B = any_tensor.shape[0]
        return th.ones((B, 1), device=any_tensor.device, dtype=any_tensor.dtype)

def to_numpy(x):
        if isinstance(x, th.Tensor):
            return x.detach().cpu().numpy()
        return x 
class RTGNNPolicy(ActorCriticPolicy):
    """
    SB3 PPO policy for her-style ego-graph observations.

    Observation keys expected (batched by SB3 VecEnv): each has leading dim B
      x:         [B, R, N_max, F]
      node_mask: [B, R, N_max]
      edge_index:[B, R, 2, E_max]
      edge_mask: [B, R, E_max]
      cand_idx:  [B, R, K]
      cand_mask: [B, R, K]

    Action space:
      MultiDiscrete([K+1] * R), where action==K means NOOP for that robot.

    The model produces candidate-only logits [B,R,K], we append a shared learnable NOOP logit.
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule,
        *,
        in_dim: int,
        hidden: int,
        k_max: int,
        logit_temperature: float = 5.0,
        noop_init: float = -1.0,
        freeze_noop_logit: bool = False,
        edge_dim: int = 0,
        use_competitor_fusion: bool = False,
        use_two_hop_actor: bool = False,
        use_two_hop_critic: bool = False,
        eta_index: int = -1,
        lambda_init: float = 0.0,
        backbone: str = "sage",
        critic_aggregation: str = "joint_mean",
        gnn_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        assert isinstance(action_space, spaces.MultiDiscrete), "RTGNNPolicy requires MultiDiscrete action space"
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            features_extractor_class=DictPassthroughExtractor,
            features_extractor_kwargs={},
            **kwargs,
        )

        # --- infer R from obs space ---
        assert isinstance(self.observation_space, spaces.Dict), "Expected Dict observation space"
        x_shape = self.observation_space.spaces["x"].shape  # (R, N, F)
        self.R = int(x_shape[0])

        # --- infer K from action space ---
        self.Kp1 = int(action_space.nvec[0])
        for n in action_space.nvec:
            assert int(n) == self.Kp1, "All robots must share same (K+1) action size"
        self.K = self.Kp1 - 1
        if int(k_max) != self.K:
            raise ValueError(f"k_max mismatch: action space implies K={self.K}, got k_max={k_max}")
        self.noop_index = self.K

        self.logit_temperature = float(logit_temperature)

        # --- build GNN actor-critic ---
        gnn_kwargs = dict(gnn_kwargs or {})
        bb_allowed = ("dummy", "sage")
        agg_allowed = ("per_robot", "joint_mean", "joint_attn")
        if backbone not in bb_allowed:
            raise ValueError(f"Invalid backbone='{backbone}'. Allowed: {bb_allowed}")
        if critic_aggregation not in agg_allowed:
            raise ValueError(f"Invalid critic_aggregation='{critic_aggregation}'. Allowed: {agg_allowed}")

        self.gnn_ac = EgoActorCritic(
            in_dim=int(in_dim),
            hidden=int(hidden),
            k_max=int(self.K),
            backbone=cast(Literal["dummy", "sage"], backbone),
            critic_aggregation=cast(Literal["per_robot", "joint_mean", "joint_attn"], critic_aggregation),
            edge_dim=int(edge_dim),
            use_competitor_fusion=bool(use_competitor_fusion),
            use_two_hop_actor=bool(use_two_hop_actor),
            use_two_hop_critic=bool(use_two_hop_critic),
            eta_index=int(eta_index),
            lambda_init=float(lambda_init),
            **gnn_kwargs,
        )

        self.noop_logit = nn.Parameter(
            th.tensor(float(noop_init), dtype=th.float32),
            requires_grad=(not bool(freeze_noop_logit)),
        )

        # SB3 expects these heads; we bypass with custom forward/evaluate_actions anyway.
        self.action_net = nn.Identity()
        self.value_net = nn.Identity()

        # Rebuild optimizer AFTER adding gnn_ac + noop_logit
        gnn_params = list(self.gnn_ac.parameters())
        assert len(gnn_params) > 0, "GNN has no parameters before _build"
        self._build(lr_schedule)

    # ---------------- helpers ----------------

    def _build_batch_outputs(self, obs_b: Dict[str, th.Tensor]) -> Tuple[th.Tensor, th.Tensor]:
        """
        Run EgoActorCritic over each batch element.

        Returns:
          logits_k: [B, R, K] candidate-only logits
          values:   [B, 1]
        """
        any_tensor = next(iter(obs_b.values()))
        B = int(any_tensor.shape[0])

        logits_list: List[th.Tensor] = []
        values_list: List[th.Tensor] = []

        for b in range(B):
            obs_one = {k: v[b] for k, v in obs_b.items()}  # drop batch dim => [R,...]
            logits_rk, value = self.gnn_ac(obs_one)         # logits: [R,K]

            if logits_rk.dim() != 2 or logits_rk.shape[0] != self.R or logits_rk.shape[1] != self.K:
                raise RuntimeError(f"EgoActorCritic must return logits [R,K]={self.R,self.K}, got {tuple(logits_rk.shape)}")

            logits_list.append(logits_rk)

            # reduce value to scalar per batch element
            if not isinstance(value, th.Tensor):
                value_t = th.tensor(float(value), device=logits_rk.device, dtype=th.float32)
            else:
                if value.dim() == 0:
                    value_t = value
                elif value.dim() == 1:
                    value_t = value.mean()
                else:
                    value_t = value.squeeze().mean()
            values_list.append(value_t)

        logits_k = th.stack(logits_list, dim=0)              # [B,R,K]
        if self.logit_temperature and self.logit_temperature != 1.0:
            logits_k = logits_k / float(self.logit_temperature)
        values = th.stack(values_list, dim=0).unsqueeze(-1)  # [B,1]
        return logits_k, values

    def _append_noop_and_mask(
        self,
        logits_k: th.Tensor,          # [B,R,K]
        cand_mask: th.Tensor,         # [B,R,K] bool/int
    ) -> Tuple[th.Tensor, th.Tensor]:
        """
        Returns:
          logits_full: [B,R,K+1]
          mask_full:   [B,R,K+1] bool (NOOP always valid)
        """
        B, R, K = logits_k.shape
        assert R == self.R and K == self.K

        noop_col = self.noop_logit.expand(B, R, 1)          # [B,R,1]
        logits_full = th.cat([logits_k, noop_col], dim=-1)  # [B,R,K+1]

        if cand_mask.dtype != th.bool:
            cand_mask = cand_mask.bool()

        noop_mask = th.ones((B, R, 1), dtype=th.bool, device=cand_mask.device)
        mask_full = th.cat([cand_mask, noop_mask], dim=-1)  # [B,R,K+1]
        return logits_full, mask_full

    @staticmethod
    def masked_logprob_entropy(
        logits_full: th.Tensor,   # [B,R,K+1], already masked with -1e9 for invalid
        actions: th.Tensor,       # [B,R]
        active: th.Tensor,        # [B,R] bool
    ) -> Tuple[th.Tensor, th.Tensor]:
        """
        Returns:
          log_prob_sum: [B]
          entropy_sum:  [B]
        """
        logp = th.log_softmax(logits_full, dim=-1)                   # [B,R,K+1]
        a = actions.long().unsqueeze(-1)                             # [B,R,1]
        chosen_logp = logp.gather(-1, a).squeeze(-1)                 # [B,R]

        p = th.softmax(logits_full, dim=-1)
        ent = -th.sum(p * logp, dim=-1)                              # [B,R]

        active_f = active.to(dtype=chosen_logp.dtype)
        chosen_logp = chosen_logp * active_f
        ent = ent * active_f

        return chosen_logp.sum(dim=1), ent.sum(dim=1)

    def _dist_from_logits_flat(self, logits_flat: th.Tensor):
        # logits_flat must be [B, sum(nvec)] = [B, R*(K+1)]
        return self.action_dist.proba_distribution(action_logits=logits_flat)

    # ---------------- SB3 API ----------------

    # def forward(self, obs: Any, deterministic: bool = False) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
    #     obs_tensor, _ = self.obs_to_tensor(obs)
    #     _ = self.extract_features(obs_tensor, features_extractor=self.features_extractor)
    #     obs_b = cast(Dict[str, th.Tensor], self.features_extractor.last_obs)
    #     assert obs_b is not None

    #     logits_k, values = self._build_batch_outputs(obs_b)             # [B,R,K], [B,1]
    #     cand_mask = obs_b["cand_mask"]                                  # [B,R,K]
    #     logits_full, mask_full = self._append_noop_and_mask(logits_k, cand_mask)

    #     # apply mask to logits (invalid actions -> -inf)
    #     logits_full = logits_full.masked_fill(~mask_full.bool(), -1e9)

    #     B = logits_full.shape[0]
    #     logits_flat = logits_full.reshape(B, -1)                        # [B,R*(K+1)]
    #     dist = self._dist_from_logits_flat(logits_flat)

    #     actions = dist.get_actions(deterministic=deterministic)         # [B,R]  <-- IMPORTANT
    #     active = cand_mask.bool().any(dim=-1)                           # [B,R]  (robots with >=1 candidate)

    #     log_prob, _ = self.masked_logprob_entropy(logits_full, actions, active)
    #     return actions, values, log_prob

    # def evaluate_actions(self, obs: Any, actions: th.Tensor) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
    #     obs_tensor, _ = self.obs_to_tensor(obs)
    #     _ = self.extract_features(obs_tensor, features_extractor=self.features_extractor)
    #     obs_b = cast(Dict[str, th.Tensor], self.features_extractor.last_obs)
    #     assert obs_b is not None

    #     logits_k, values = self._build_batch_outputs(obs_b)
    #     cand_mask = obs_b["cand_mask"]

    #     logits_full, mask_full = self._append_noop_and_mask(logits_k, cand_mask)
    #     logits_full = logits_full.masked_fill(~mask_full.bool(), -1e9)

    #     B = logits_full.shape[0]
    #     actions = actions.reshape(B, self.R)  # SB3 usually passes [B,R]; reshape is safe

    #     active = cand_mask.bool().any(dim=-1)
    #     log_prob, entropy = self.masked_logprob_entropy(logits_full, actions, active)
    #     return values, log_prob, entropy
    def forward(self, obs, deterministic=False):
        obs = {
            k: to_numpy(v)
            for k, v in obs.items()
        }
        obs_tensor, _ = self.obs_to_tensor(obs)
        _ = self.extract_features(obs_tensor, features_extractor=self.features_extractor)
        obs_b = self.features_extractor.last_obs
        assert obs_b is not None

        logits_k, values = self._build_batch_outputs(obs_b)      # [B,R,K], [B,1]
        cand_mask = obs_b["cand_mask"]                            # [B,R,K]
        logits_full, mask_full = self._append_noop_and_mask(logits_k, cand_mask)
        logits_full = logits_full.masked_fill(~mask_full, -1e9)

        B = logits_full.shape[0]
        logits_flat = logits_full.reshape(B, -1)                  # [B, R*(K+1)]
        dist = self._dist_from_logits_flat(logits_flat)
        actions_flat = dist.get_actions(deterministic=deterministic)  # [B, R]

        # Reshape for per-robot log_prob computation
        actions = actions_flat.reshape(B, self.R)
        active  = mask_full[..., :self.K].any(dim=-1)             # [B,R] — has real candidates
        log_prob, _ = self.masked_logprob_entropy(logits_full, actions, active)

        return actions_flat, values, log_prob                      # SB3 expects flat actions

    def evaluate_actions(self, obs, actions):
        obs = {
            k: to_numpy(v)
            for k, v in obs.items()
        }
        obs_tensor, _ = self.obs_to_tensor(obs)
        _ = self.extract_features(obs_tensor, features_extractor=self.features_extractor)
        obs_b = self.features_extractor.last_obs
        assert obs_b is not None

        logits_k, values = self._build_batch_outputs(obs_b)
        cand_mask = obs_b["cand_mask"]
        logits_full, mask_full = self._append_noop_and_mask(logits_k, cand_mask)
        logits_full = logits_full.masked_fill(~mask_full, -1e9)

        B = logits_full.shape[0]
        actions = actions.reshape(B, self.R)
        active  = mask_full[..., :self.K].any(dim=-1)
        log_prob, entropy = self.masked_logprob_entropy(logits_full, actions, active)

        return values, log_prob, entropy
    
    def predict_values(self, obs: Any) -> th.Tensor:
        obs = {
            k: to_numpy(v)
            for k, v in obs.items()
        }
        obs_tensor, _ = self.obs_to_tensor(obs)
        _ = self.extract_features(obs_tensor, features_extractor=self.features_extractor)
        obs_b = cast(Dict[str, th.Tensor], self.features_extractor.last_obs)
        assert obs_b is not None

        _logits_k, values = self._build_batch_outputs(obs_b)
        return values

    def _predict(self, observation: th.Tensor, deterministic: bool = False) -> th.Tensor:
        actions, _values, _log_prob = self.forward(observation, deterministic=deterministic)
        return actions