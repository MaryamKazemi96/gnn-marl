# """
# Stable-Baselines3 policy wrapper for GNN-based actor-critic.
# Integrates with PPO trainer and handles masked action sampling.
# """
# import torch as th
# import torch.nn as nn
# from typing import Any, Dict, Optional, List, Literal, cast, Tuple
# from gymnasium import spaces
# from stable_baselines3.common.policies import ActorCriticPolicy
# from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

# from .actor_critic import EgoActorCritic


# class DictPassthroughExtractor(BaseFeaturesExtractor):
#     """
#     Passthrough extractor that preserves raw dictionary observations.
#     Returns dummy feature tensor to satisfy SB3 interface.
#     """
#     def __init__(self, observation_space: spaces.Dict):
#         super().__init__(observation_space, features_dim=1)
#         self.last_obs: Optional[Dict[str, th.Tensor]] = None
    
#     def forward(self, obs: Dict[str, th.Tensor]) -> th.Tensor:
#         """Capture raw observation and return dummy features."""
#         self.last_obs = obs
#         any_tensor = next(iter(obs.values()))
#         B = any_tensor.shape[0]
#         return th.ones((B, 1), device=any_tensor.device, dtype=any_tensor.dtype)


# class RTGNNPolicy(ActorCriticPolicy):
#     """
#     PPO-compatible policy using GNN-based actor-critic.
    
#     Key features:
#     - Action space: MultiDiscrete [K_max+1] * R (K_max task choices + 1 NO-OP per robot)
#     - Learnable NO-OP logit: Single shared parameter across all robots/batch
#     - Masked action sampling: Respects cand_mask to allow only valid actions
#     - Optional 2-hop architecture: Competitor-aware task selection
#     """
#     def __init__(
#         self,
#         *args,
#         in_dim: int,
#         hidden: int,
#         k_max: int,
#         logit_temperature: float = 5.0,
#         noop_init: float = -1.0,
#         freeze_noop_logit: bool = False,
#         edge_dim: int = 0,
#         use_competitor_fusion: bool = False,
#         use_two_hop_actor: bool = False,
#         use_two_hop_critic: bool = False,
#         eta_index: int = -1,
#         lambda_init: float = 0.0,
#         backbone: str = "sage",
#         critic_aggregation: str = "joint_mean",
#         **kwargs,
#     ):
#         """Initialize policy."""
#         gnn_kwargs: Dict[str, Any] = kwargs.pop("gnn_kwargs", {})
        
#         # Use passthrough extractor to preserve dict observations
#         super().__init__(*args, features_extractor_class=DictPassthroughExtractor, **kwargs)
        
#         # Get n_robots from observation space
#         if isinstance(self.observation_space, spaces.Dict):
#             x_shape = self.observation_space.spaces["x"].shape
#             self.n_robots = x_shape[0]  # First dim is R
#         else:
#             self.n_robots = 1
        
#         self.k_max = k_max
#         self.k_out = k_max + 1  # +1 for explicit NO-OP
        
#         # Validate backbone and aggregation
#         _bb_allowed = ("dummy", "sage")
#         _agg_allowed = ("per_robot", "joint_mean", "joint_attn")
#         if backbone not in _bb_allowed:
#             raise ValueError(f"Invalid backbone='{backbone}'. Allowed: {_bb_allowed}")
#         if critic_aggregation not in _agg_allowed:
#             raise ValueError(f"Invalid critic_aggregation='{critic_aggregation}'. Allowed: {_agg_allowed}")
        
#         bb_lit = cast(Literal["dummy", "sage"], backbone)
#         agg_lit = cast(Literal["per_robot", "joint_mean", "joint_attn"], critic_aggregation)
        
#         # Create GNN actor-critic
#         self.gnn_ac = EgoActorCritic(
#             in_dim=in_dim,
#             hidden=hidden,
#             k_max=k_max,
#             backbone=bb_lit,
#             critic_aggregation=agg_lit,
#             edge_dim=int(edge_dim),
#             use_competitor_fusion=bool(use_competitor_fusion),
#             use_two_hop_actor=bool(use_two_hop_actor),
#             use_two_hop_critic=bool(use_two_hop_critic),
#             eta_index=int(eta_index),
#             lambda_init=float(lambda_init),
#             **gnn_kwargs,
#         )
        
#         # Learnable NO-OP logit
#         self.noop_logit = nn.Parameter(th.tensor(noop_init))
#         self.logit_temperature = float(logit_temperature)
        
#         # Add GNN parameters to optimizer
#         extra_params = list(self.gnn_ac.parameters())
#         if not freeze_noop_logit:
#             extra_params.append(self.noop_logit)
#         if len(extra_params) > 0:
#             self.optimizer.add_param_group({"params": extra_params})
    
#     def _build_batch_outputs(self, obs_b: Dict[str, th.Tensor]) -> Tuple[th.Tensor, th.Tensor]:
#         """
#         Process batch of observations through GNN actor-critic.
        
#         Returns:
#             logits: [B, R, K_max] action logits
#             values: [B, 1] value estimates
#         """
#         B = next(iter(obs_b.values())).shape[0]
#         logits_list: List[th.Tensor] = []
#         values_list: List[th.Tensor] = []
        
#         # Process each environment sample independently
#         for b in range(B):
#             obs_one = {k: v[b] for k, v in obs_b.items()}
#             logits_b, value_b = self.gnn_ac(obs_one)  # logits: [R, K_max]
#             logits_list.append(logits_b)
            
#             # Normalize value to scalar
#             if value_b.dim() == 0:
#                 v_b = value_b
#             elif value_b.dim() == 1:
#                 v_b = value_b.mean()
#             else:
#                 v_b = value_b.squeeze()
#             values_list.append(v_b)
        
#         logits = th.stack(logits_list, dim=0)  # [B, R, K_max]
#         logits = logits / self.logit_temperature  # Apply temperature
#         values = th.stack(values_list, dim=0).unsqueeze(-1)  # [B, 1]
#         return logits, values
    
#     def _append_noop(self, logits: th.Tensor, mask_k: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
#         """
#         Append NO-OP column and flatten for MultiDiscrete action space.
        
#         Args:
#             logits: [B, R, K_max]
#             mask_k: [B, R, K_max]
        
#         Returns:
#             logits_flat: [B, R*(K_max+1)]
#             mask_flat: [B, R*(K_max+1)]
#         """
#         B, R, _K = logits.shape
#         noop_col = self.noop_logit.expand(B, R, 1)
#         logits_full = th.cat([logits, noop_col], dim=-1)  # [B, R, K_max+1]
        
#         if mask_k.dtype != th.bool:
#             mask_k = mask_k.bool()
#         ones = th.ones((B, R, 1), dtype=th.bool, device=mask_k.device)
#         mask_full = th.cat([mask_k, ones], dim=-1)  # [B, R, K_max+1]
        
#         # Reshape to flat format for MultiDiscrete
#         logits_flat = logits_full.reshape(B, R * (self.k_max + 1))
#         mask_flat = mask_full.reshape(B, R * (self.k_max + 1))
        
#         return logits_flat, mask_flat
    
#     @staticmethod
#     def masked_logprob_entropy(
#         logits: th.Tensor, 
#         actions: th.Tensor, 
#         active: th.Tensor
#     ) -> Tuple[th.Tensor, th.Tensor]:
#         """
#         Compute log-probability and entropy of actions, masked by active robots.
        
#         Args:
#             logits: [B, R, K] (unflattened)
#             actions: [B, R]
#             active: [B, R] bool
        
#         Returns:
#             log_prob: [B]
#             entropy: [B]
#         """
#         B, R, K = logits.shape
        
#         # Compute log probabilities
#         logp = th.log_softmax(logits, dim=-1)
#         a = actions.long().unsqueeze(-1)  # [B, R, 1]
#         chosen_logp = logp.gather(-1, a).squeeze(-1)  # [B, R]
        
#         # Compute entropy
#         p = th.softmax(logits, dim=-1)
#         ent = -th.sum(p * logp, dim=-1)  # [B, R]
        
#         # Mask by active robots
#         chosen_logp = chosen_logp * active.float()
#         ent = ent * active.float()
        
#         return chosen_logp.sum(dim=1), ent.sum(dim=1)
    
#     def _dist_from_logits(self, logits: th.Tensor, mask: th.Tensor):
#         """Create SB3 action distribution from logits."""
#         # logits already flattened [B, R*K_out]
#         return self.action_dist.proba_distribution(action_logits=logits)
    
#     def forward(self, obs: Any, deterministic: bool = False) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
#         """
#         Policy forward pass: sample actions and compute log-probabilities.
        
#         Returns:
#             actions: [B, R] sampled actions
#             values: [B, 1] value estimates
#             log_prob: [B] log-probabilities
#         """
#         _ = self.extract_features(obs, features_extractor=self.features_extractor)
#         obs_dict_b = cast(Dict[str, th.Tensor], self.features_extractor.last_obs)
#         assert obs_dict_b is not None
        
#         B = next(iter(obs_dict_b.values())).shape[0]
#         R = self.n_robots
        
#         logits_k, values = self._build_batch_outputs(obs_dict_b)  # [B, R, K_max], [B, 1]
#         mask_k = obs_dict_b["cand_mask"]  # [B, R, K_max]
#         logits_flat, mask_flat = self._append_noop(logits_k, mask_k)  # [B, R*(K_max+1)]
        
#         # Unflatten for masking
#         logits_shaped = logits_flat.reshape(B, R, self.k_max + 1)
#         mask_shaped = mask_flat.reshape(B, R, self.k_max + 1)
#         logits_shaped = logits_shaped.masked_fill(~mask_shaped, -1e9)
#         logits_flat = logits_shaped.reshape(B, -1)
        
#         dist = self._dist_from_logits(logits_flat, mask_flat)
#         actions_flat = dist.get_actions(deterministic=deterministic)  # [B, R*(K_max+1)]
#         actions = actions_flat.reshape(B, R)  # [B, R]
        
#         active = mask_k.any(dim=-1)
#         log_prob, _ = self.masked_logprob_entropy(logits_shaped, actions, active)
        
#         return actions, values, log_prob
    
#     def evaluate_actions(self, obs: Any, actions: th.Tensor) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
#         """Evaluate log-prob and entropy for given actions (used during training)."""
#         _ = self.extract_features(obs, features_extractor=self.features_extractor)
#         obs_dict_b = cast(Dict[str, th.Tensor], self.features_extractor.last_obs)
#         assert obs_dict_b is not None
        
#         B = next(iter(obs_dict_b.values())).shape[0]
#         R = self.n_robots
        
#         logits_k, values = self._build_batch_outputs(obs_dict_b)
#         mask_k = obs_dict_b["cand_mask"]
#         logits_flat, mask_flat = self._append_noop(logits_k, mask_k)
        
#         # Unflatten for masking
#         logits_shaped = logits_flat.reshape(B, R, self.k_max + 1)
#         mask_shaped = mask_flat.reshape(B, R, self.k_max + 1)
#         logits_shaped = logits_shaped.masked_fill(~mask_shaped, -1e9)
        
#         # Flatten actions back to [B, R]
#         actions = actions.reshape(B, R)
#         active = mask_k.any(dim=-1)
        
#         log_prob, entropy = self.masked_logprob_entropy(logits_shaped, actions, active)
#         return values, log_prob, entropy
    
#     def predict_values(self, obs: Any) -> th.Tensor:
#         """Predict values without sampling actions."""
#         _ = self.extract_features(obs, features_extractor=self.features_extractor)
#         obs_dict_b = cast(Dict[str, th.Tensor], self.features_extractor.last_obs)
#         assert obs_dict_b is not None
#         _, values = self._build_batch_outputs(obs_dict_b)
#         return values
    
#     def _predict(self, observation: th.Tensor, deterministic: bool = False) -> th.Tensor:
#         """Override for model.predict() calls."""
#         actions, _, _ = self.forward(observation, deterministic=deterministic)
#         return actions

# sb3_gnn_policy.py
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

    def forward(self, obs: Dict[str, th.Tensor]) -> th.Tensor:
        self.last_obs = obs
        any_tensor = next(iter(obs.values()))
        B = any_tensor.shape[0]
        return th.ones((B, 1), device=any_tensor.device, dtype=any_tensor.dtype)


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
        logit_temperature: float = 1.0,
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

    def forward(self, obs: Any, deterministic: bool = False) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        obs_tensor, _ = self.obs_to_tensor(obs)
        _ = self.extract_features(obs_tensor, features_extractor=self.features_extractor)
        obs_b = cast(Dict[str, th.Tensor], self.features_extractor.last_obs)
        assert obs_b is not None

        logits_k, values = self._build_batch_outputs(obs_b)             # [B,R,K], [B,1]
        cand_mask = obs_b["cand_mask"]                                  # [B,R,K]
        logits_full, mask_full = self._append_noop_and_mask(logits_k, cand_mask)

        # apply mask to logits (invalid actions -> -inf)
        logits_full = logits_full.masked_fill(~mask_full.bool(), -1e9)

        B = logits_full.shape[0]
        logits_flat = logits_full.reshape(B, -1)                        # [B,R*(K+1)]
        dist = self._dist_from_logits_flat(logits_flat)

        actions = dist.get_actions(deterministic=deterministic)         # [B,R]  <-- IMPORTANT
        active = cand_mask.bool().any(dim=-1)                           # [B,R]  (robots with >=1 candidate)

        log_prob, _ = self.masked_logprob_entropy(logits_full, actions, active)
        return actions, values, log_prob

    def evaluate_actions(self, obs: Any, actions: th.Tensor) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        obs_tensor, _ = self.obs_to_tensor(obs)
        _ = self.extract_features(obs_tensor, features_extractor=self.features_extractor)
        obs_b = cast(Dict[str, th.Tensor], self.features_extractor.last_obs)
        assert obs_b is not None

        logits_k, values = self._build_batch_outputs(obs_b)
        cand_mask = obs_b["cand_mask"]

        logits_full, mask_full = self._append_noop_and_mask(logits_k, cand_mask)
        logits_full = logits_full.masked_fill(~mask_full.bool(), -1e9)

        B = logits_full.shape[0]
        actions = actions.reshape(B, self.R)  # SB3 usually passes [B,R]; reshape is safe

        active = cand_mask.bool().any(dim=-1)
        log_prob, entropy = self.masked_logprob_entropy(logits_full, actions, active)
        return values, log_prob, entropy

    def predict_values(self, obs: Any) -> th.Tensor:
        obs_tensor, _ = self.obs_to_tensor(obs)
        _ = self.extract_features(obs_tensor, features_extractor=self.features_extractor)
        obs_b = cast(Dict[str, th.Tensor], self.features_extractor.last_obs)
        assert obs_b is not None

        _logits_k, values = self._build_batch_outputs(obs_b)
        return values

    def _predict(self, observation: th.Tensor, deterministic: bool = False) -> th.Tensor:
        actions, _values, _log_prob = self.forward(observation, deterministic=deterministic)
        return actions