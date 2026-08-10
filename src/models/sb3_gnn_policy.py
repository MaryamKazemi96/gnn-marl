# from __future__ import annotations

# from typing import Any, Dict, Optional, Tuple, List, cast, Literal

# import torch as th
# import torch.nn as nn
# import numpy as np
# from gymnasium import spaces
# from stable_baselines3.common.policies import ActorCriticPolicy
# from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

# # IMPORTANT: import your actor-critic module here
# # It must implement: logits_rk, value = self.gnn_ac(obs_one_dict)
# # logits_rk shape: [R, K] (candidate-only)
# from src.models.actor_critic import EgoActorCritic  # <-- adjust path to your project


# class DictPassthroughExtractor(BaseFeaturesExtractor):
#     """
#     Passthrough extractor: keep raw dict observations for the custom policy.
#     SB3 requires a tensor output, so we return a dummy tensor.
#     """
#     def __init__(self, observation_space: spaces.Dict):
#         super().__init__(observation_space, features_dim=1)
#         self.last_obs: Optional[Dict[str, th.Tensor]] = None

#     def forward(self, obs):
#         if isinstance(obs, dict):
#             obs = {
#                 k: (v.detach() if th.is_tensor(v) else v)
#                 for k, v in obs.items()
#             }

#         self.last_obs = obs

#         any_tensor = next(iter(obs.values()))
#         B = any_tensor.shape[0]
#         return th.ones((B, 1), device=any_tensor.device, dtype=any_tensor.dtype)

# def to_numpy(x):
#         if isinstance(x, th.Tensor):
#             return x.detach().cpu().numpy()
#         return x 

# class RTGNNPolicy(ActorCriticPolicy):
#     """
#     SB3 PPO policy for her-style ego-graph observations.

#     Observation keys expected (batched by SB3 VecEnv): each has leading dim B
#       x:         [B, R, N_max, F]
#       node_mask: [B, R, N_max]
#       edge_index:[B, R, 2, E_max]
#       edge_mask: [B, R, E_max]
#       cand_idx:  [B, R, K]
#       cand_mask: [B, R, K]

#     Action space:
#       MultiDiscrete([K+1] * R), where action==K means NOOP for that robot.

#     The model produces candidate-only logits [B,R,K], we append a shared learnable NOOP logit.
#     """

#     def __init__(
#         self,
#         observation_space: spaces.Space,
#         action_space: spaces.Space,
#         lr_schedule,
#         *,
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
#         gnn_kwargs: Optional[Dict[str, Any]] = None,
#         **kwargs,
#     ):
#         assert isinstance(action_space, spaces.MultiDiscrete), "RTGNNPolicy requires MultiDiscrete action space"
#         super().__init__(
#             observation_space,
#             action_space,
#             lr_schedule,
#             features_extractor_class=DictPassthroughExtractor,
#             features_extractor_kwargs={},
#             **kwargs,
#         )

#         # --- infer R from obs space ---
#         assert isinstance(self.observation_space, spaces.Dict), "Expected Dict observation space"
#         x_shape = self.observation_space.spaces["x"].shape  # (R, N, F)
#         self.R = int(x_shape[0])

#         # --- infer K from action space ---
#         self.Kp1 = int(action_space.nvec[0])
#         for n in action_space.nvec:
#             assert int(n) == self.Kp1, "All robots must share same (K+1) action size"
#         self.K = self.Kp1 - 1
#         if int(k_max) != self.K:
#             raise ValueError(f"k_max mismatch: action space implies K={self.K}, got k_max={k_max}")
#         self.noop_index = self.K

#         self.logit_temperature = float(logit_temperature)

#         # --- build GNN actor-critic ---
#         gnn_kwargs = dict(gnn_kwargs or {})
#         bb_allowed = ("dummy", "sage")
#         agg_allowed = ("per_robot", "joint_mean", "joint_attn")
#         if backbone not in bb_allowed:
#             raise ValueError(f"Invalid backbone='{backbone}'. Allowed: {bb_allowed}")
#         if critic_aggregation not in agg_allowed:
#             raise ValueError(f"Invalid critic_aggregation='{critic_aggregation}'. Allowed: {agg_allowed}")

#         self.gnn_ac = EgoActorCritic(
#             in_dim=int(in_dim),
#             hidden=int(hidden),
#             k_max=int(self.K),
#             backbone=cast(Literal["dummy", "sage"], backbone),
#             critic_aggregation=cast(Literal["per_robot", "joint_mean", "joint_attn"], critic_aggregation),
#             edge_dim=int(edge_dim),
#             use_competitor_fusion=bool(use_competitor_fusion),
#             use_two_hop_actor=bool(use_two_hop_actor),
#             use_two_hop_critic=bool(use_two_hop_critic),
#             eta_index=int(eta_index),
#             lambda_init=float(lambda_init),
#             **gnn_kwargs,
#         )

#         self.noop_logit = nn.Parameter(
#             th.tensor(float(noop_init), dtype=th.float32),
#             requires_grad=(not bool(freeze_noop_logit)),
#         )

#         # SB3 expects these heads; we bypass with custom forward/evaluate_actions anyway.
#         self.action_net = nn.Identity()
#         self.value_net = nn.Identity()

#         # Rebuild optimizer AFTER adding gnn_ac + noop_logit
#         gnn_params = list(self.gnn_ac.parameters())
#         assert len(gnn_params) > 0, "GNN has no parameters before _build"
#         self._build(lr_schedule)

#     # ---------------- helpers ----------------

#     def get_distribution(self, obs):
#         obs = {k: to_numpy(v) for k, v in obs.items()}
#         obs_tensor, _ = self.obs_to_tensor(obs)

#         _ = self.extract_features(
#             obs_tensor,
#             features_extractor=self.features_extractor,
#         )
#         obs_b = self.features_extractor.last_obs

#         logits_k, _ = self._build_batch_outputs(obs_b)
#         cand_mask = obs_b["cand_mask"]

#         logits_full, mask_full = self._append_noop_and_mask(
#             logits_k,
#             cand_mask,
#         )
#         logits_full = logits_full.masked_fill(~mask_full, -1e9)

#         B = logits_full.shape[0]
#         logits_flat = logits_full.reshape(B, -1)

#         return self._dist_from_logits_flat(logits_flat)
#     def _build_batch_outputs(self, obs_b: Dict[str, th.Tensor]) -> Tuple[th.Tensor, th.Tensor]:
#         """
#         Run EgoActorCritic over each batch element.

#         Returns:
#           logits_k: [B, R, K] candidate-only logits
#           values:   [B, 1]
#         """
#         any_tensor = next(iter(obs_b.values()))
#         B = int(any_tensor.shape[0])

#         logits_list: List[th.Tensor] = []
#         values_list: List[th.Tensor] = []

#         for b in range(B):
#             obs_one = {k: v[b] for k, v in obs_b.items()}  # drop batch dim => [R,...]
#             logits_rk, value = self.gnn_ac(obs_one)         # logits: [R,K]
#             # print debug
#             # if not hasattr(self, "_actor_debug"):
#             #     self._actor_debug = 0
#             # self._actor_debug += 1

#             # if self._actor_debug % 200 == 0:
#             #     print("\n========================")
#             #     print("Actor output")
#             #     print("mean :", logits_rk.mean().item())
#             #     print("std  :", logits_rk.std().item())
#             #     print("min  :", logits_rk.min().item())
#             #     print("max  :", logits_rk.max().item())
#             #     print(logits_rk)
#             # ---------
#             if logits_rk.dim() != 2 or logits_rk.shape[0] != self.R or logits_rk.shape[1] != self.K:
#                 raise RuntimeError(f"EgoActorCritic must return logits [R,K]={self.R,self.K}, got {tuple(logits_rk.shape)}")

#             logits_list.append(logits_rk)

#             # reduce value to scalar per batch element
#             if not isinstance(value, th.Tensor):
#                 value_t = th.tensor(float(value), device=logits_rk.device, dtype=th.float32)
#             else:
#                 if value.dim() == 0:
#                     value_t = value
#                 elif value.dim() == 1:
#                     value_t = value.mean()
#                 else:
#                     value_t = value.squeeze().mean()
#             values_list.append(value_t)

#         logits_k = th.stack(logits_list, dim=0)              # [B,R,K]
#         if self.logit_temperature and self.logit_temperature != 1.0:
#             logits_k = logits_k / float(self.logit_temperature)
#         values = th.stack(values_list, dim=0).unsqueeze(-1)  # [B,1]
#         return logits_k, values

#     def _append_noop_and_mask(
#         self,
#         logits_k: th.Tensor,          # [B,R,K]
#         cand_mask: th.Tensor,         # [B,R,K] bool/int
#     ) -> Tuple[th.Tensor, th.Tensor]:
#         """
#         Returns:
#           logits_full: [B,R,K+1]
#           mask_full:   [B,R,K+1] bool (NOOP always valid)
#         """
#         B, R, K = logits_k.shape
#         assert R == self.R and K == self.K

#         noop_col = self.noop_logit.expand(B, R, 1)          # [B,R,1]
#         logits_full = th.cat([logits_k, noop_col], dim=-1)  # [B,R,K+1]

#         if cand_mask.dtype != th.bool:
#             cand_mask = cand_mask.bool()

#         noop_mask = th.ones((B, R, 1), dtype=th.bool, device=cand_mask.device)
#         mask_full = th.cat([cand_mask, noop_mask], dim=-1)  # [B,R,K+1]
#         return logits_full, mask_full

#     @staticmethod
#     def masked_logprob_entropy(
#         logits_full: th.Tensor,   # [B,R,K+1], already masked with -1e9 for invalid
#         actions: th.Tensor,       # [B,R]
#         active: th.Tensor,        # [B,R] bool
#     ) -> Tuple[th.Tensor, th.Tensor]:
#         """
#         Returns:
#           log_prob_sum: [B]
#           entropy_sum:  [B]
#         """
#         logp = th.log_softmax(logits_full, dim=-1)                   # [B,R,K+1]
#         a = actions.long().unsqueeze(-1)                             # [B,R,1]
#         chosen_logp = logp.gather(-1, a).squeeze(-1)                 # [B,R]

#         p = th.softmax(logits_full, dim=-1)
#         ent = -th.sum(p * logp, dim=-1)                              # [B,R]

#         active_f = active.to(dtype=chosen_logp.dtype)
#         chosen_logp = chosen_logp * active_f
#         ent = ent * active_f

#         return chosen_logp.sum(dim=1), ent.sum(dim=1)

#     def _dist_from_logits_flat(self, logits_flat: th.Tensor):
#         # logits_flat must be [B, sum(nvec)] = [B, R*(K+1)]
#         return self.action_dist.proba_distribution(action_logits=logits_flat)

#     # ---------------- SB3 API ----------------

#     # def forward(self, obs: Any, deterministic: bool = False) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
#     #     obs_tensor, _ = self.obs_to_tensor(obs)
#     #     _ = self.extract_features(obs_tensor, features_extractor=self.features_extractor)
#     #     obs_b = cast(Dict[str, th.Tensor], self.features_extractor.last_obs)
#     #     assert obs_b is not None

#     #     logits_k, values = self._build_batch_outputs(obs_b)             # [B,R,K], [B,1]
#     #     cand_mask = obs_b["cand_mask"]                                  # [B,R,K]
#     #     logits_full, mask_full = self._append_noop_and_mask(logits_k, cand_mask)

#     #     # apply mask to logits (invalid actions -> -inf)
#     #     logits_full = logits_full.masked_fill(~mask_full.bool(), -1e9)

#     #     B = logits_full.shape[0]
#     #     logits_flat = logits_full.reshape(B, -1)                        # [B,R*(K+1)]
#     #     dist = self._dist_from_logits_flat(logits_flat)

#     #     actions = dist.get_actions(deterministic=deterministic)         # [B,R]  <-- IMPORTANT
#     #     active = cand_mask.bool().any(dim=-1)                           # [B,R]  (robots with >=1 candidate)

#     #     log_prob, _ = self.masked_logprob_entropy(logits_full, actions, active)
#     #     return actions, values, log_prob

#     # def evaluate_actions(self, obs: Any, actions: th.Tensor) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
#     #     obs_tensor, _ = self.obs_to_tensor(obs)
#     #     _ = self.extract_features(obs_tensor, features_extractor=self.features_extractor)
#     #     obs_b = cast(Dict[str, th.Tensor], self.features_extractor.last_obs)
#     #     assert obs_b is not None

#     #     logits_k, values = self._build_batch_outputs(obs_b)
#     #     cand_mask = obs_b["cand_mask"]

#     #     logits_full, mask_full = self._append_noop_and_mask(logits_k, cand_mask)
#     #     logits_full = logits_full.masked_fill(~mask_full.bool(), -1e9)

#     #     B = logits_full.shape[0]
#     #     actions = actions.reshape(B, self.R)  # SB3 usually passes [B,R]; reshape is safe

#     #     active = cand_mask.bool().any(dim=-1)
#     #     log_prob, entropy = self.masked_logprob_entropy(logits_full, actions, active)
#     #     return values, log_prob, entropy
#     def forward(self, obs, deterministic=False):
#         obs = {
#             k: to_numpy(v)
#             for k, v in obs.items()
#         }
#         obs_tensor, _ = self.obs_to_tensor(obs)
#         _ = self.extract_features(obs_tensor, features_extractor=self.features_extractor)
#         obs_b = self.features_extractor.last_obs
#         assert obs_b is not None

#         logits_k, values = self._build_batch_outputs(obs_b)      # [B,R,K], [B,1]
#         cand_mask = obs_b["cand_mask"]                            # [B,R,K]

#         #------------debug for candidate mask
#         # print("cand_mask:", cand_mask[0])
#         # if not hasattr(self, "_mask_debug"):
#         #     self._mask_debug = 0

#         # self._mask_debug += 1

#         # if self._mask_debug % 200 == 0:

#         #     print("\nCandidate mask")
#         # print(cand_mask[0])


#         # print(
#         #     "valid candidates per robot:",
#         #     cand_mask[0].sum(dim=-1)
#         # )
#         #------------debug for candidate mask---------------
#         logits_full, mask_full = self._append_noop_and_mask(logits_k, cand_mask)
#         # pribt debug
#         # probs = th.softmax(logits_full, dim=-1)
#         # print(logits_full[0], '====================')
#         # print(probs[0], '====================')
#         # -------
#         logits_full = logits_full.masked_fill(~mask_full, -1e9)

#         B = logits_full.shape[0]
#         logits_flat = logits_full.reshape(B, -1)                  # [B, R*(K+1)]
#         # print(logits_flat[0].reshape(self.R, self.K+1), 'logit before logit_flat')
#         dist = self._dist_from_logits_flat(logits_flat)
#         actions_flat = dist.get_actions(deterministic=deterministic)  # [B, R]
#         # print('action flat', actions_flat)
#         # Reshape for per-robot log_prob computation
#         actions = actions_flat.reshape(B, self.R)
#         active  = mask_full[..., :self.K].any(dim=-1)             # [B,R] — has real candidates
#         log_prob, _ = self.masked_logprob_entropy(logits_full, actions, active)
#         # print(type(dist))
#         # print(type(dist.distribution))
#         # print(len(dist.distribution))
#         # print(type(dist.distribution[0]))
#         # print(dist.distribution[0])
#         # print(print(dist.distribution[0].logits))
#         # print(dist.distribution.probs[0], 'dist.distribution.probs[0]')
#         return actions_flat, values, log_prob                      # SB3 expects flat actions

#     def evaluate_actions(self, obs, actions):
#         obs = {
#             k: to_numpy(v)
#             for k, v in obs.items()
#         }
#         obs_tensor, _ = self.obs_to_tensor(obs)
#         _ = self.extract_features(obs_tensor, features_extractor=self.features_extractor)
#         obs_b = self.features_extractor.last_obs
#         assert obs_b is not None

#         logits_k, values = self._build_batch_outputs(obs_b)
#         #----debug for actor weights--------------
#         # if not hasattr(self, "_debug_counter"):
#         #     self._debug_counter = 0

#         # self._debug_counter += 1

#         # if self._debug_counter % 200 == 0:
#         #     print("\n===== ACTOR DEBUG =====")
#         #     print("actor_head weight norm:",
#         #         self.gnn_ac.actor_head.weight.norm().item())
#         #     print("actor_head bias:",
#         #         self.gnn_ac.actor_head.bias.data.cpu().numpy())
#         #----------------------
#         cand_mask = obs_b["cand_mask"]
#         logits_full, mask_full = self._append_noop_and_mask(logits_k, cand_mask)
#         logits_full = logits_full.masked_fill(~mask_full, -1e9)
#         #-------------debug for actor weights--------------
#         # print(
#         #     "logits:",
#         #     logits_full[0, 0].detach().cpu().numpy()
#         # )
#         #-------------debug for actor weights--------------
#         B = logits_full.shape[0]
#         actions = actions.reshape(B, self.R)
#         active  = mask_full[..., :self.K].any(dim=-1)
#         log_prob, entropy = self.masked_logprob_entropy(logits_full, actions, active)

#         return values, log_prob, entropy
    
#     def predict_values(self, obs: Any) -> th.Tensor:
#         obs = {
#             k: to_numpy(v)
#             for k, v in obs.items()
#         }
#         obs_tensor, _ = self.obs_to_tensor(obs)
#         _ = self.extract_features(obs_tensor, features_extractor=self.features_extractor)
#         obs_b = cast(Dict[str, th.Tensor], self.features_extractor.last_obs)
#         assert obs_b is not None

#         _logits_k, values = self._build_batch_outputs(obs_b)
#         return values

#     @th.no_grad()
#     @th.no_grad()
#     def _get_masked_logits(self, obs: Any) -> th.Tensor:
#         """Shared internal computation for get_action_probs/get_action_logits —
#         identical masked-logit path used by forward()/evaluate_actions().
#         Returns logits_full [B, R, K_max+1], with invalid (masked-out)
#         candidate slots set to -1e9 (same convention as training)."""
#         obs = {
#             k: to_numpy(v)
#             for k, v in obs.items()
#         }
#         obs_tensor, _ = self.obs_to_tensor(obs)
#         _ = self.extract_features(obs_tensor, features_extractor=self.features_extractor)
#         obs_b = self.features_extractor.last_obs
#         assert obs_b is not None

#         logits_k, _values = self._build_batch_outputs(obs_b)      # [B,R,K]
#         cand_mask = obs_b["cand_mask"]                            # [B,R,K]
#         logits_full, mask_full = self._append_noop_and_mask(logits_k, cand_mask)  # [B,R,K+1]
#         logits_full = logits_full.masked_fill(~mask_full, -1e9)
#         return logits_full

#     @th.no_grad()
#     def get_action_logits(self, obs: Any) -> np.ndarray:
#         """Diagnostic helper — raw (pre-softmax) logits, shape [B, R, K_max+1].
#         Masked-out (invalid) candidate slots are set to -1e9, matching the
#         exact masking used in forward()/evaluate_actions() — filter those
#         out using cand_mask before computing statistics (e.g. mean/spread),
#         since -1e9 entries would otherwise dominate any naive average.
#         """
#         logits_full = self._get_masked_logits(obs)
#         return logits_full.cpu().numpy()

#     @th.no_grad()
#     def get_action_probs(self, obs: Any) -> np.ndarray:
#         """Diagnostic helper — NOT used by training/predict(), only for
#         inspecting the actual per-robot categorical distribution (softmax
#         over K_max real candidates + 1 noop slot) that deterministic/
#         stochastic action selection is drawn from.

#         Returns probs with shape [B, R, K_max+1], softmax-normalized over
#         the last axis, with masked-out (invalid) candidate slots at ~0
#         probability (since their logits were set to -1e9 before softmax,
#         same as in forward()/evaluate_actions() — this method reuses that
#         exact same masked logit computation so the numbers reported here
#         are guaranteed to match what deterministic()/sampling actually see).
#         """
#         logits_full = self._get_masked_logits(obs)
#         probs = th.softmax(logits_full, dim=-1)  # [B, R, K+1]
#         return probs.cpu().numpy()

#     def _predict(self, observation: th.Tensor, deterministic: bool = False) -> th.Tensor:
#         actions, _values, _log_prob = self.forward(observation, deterministic=deterministic)
#         return actions


# def compute_noop_logit_stats(policy: "RTGNNPolicy", obs: Any):
#     """Shared helper for training-time and eval-time logit/probability
#     logging (noop vs best-real-candidate summary). Given a batched obs
#     dict (any batch size), returns aggregate stats across every
#     (batch, robot) decision that had >=1 real candidate available (i.e.
#     noop was a genuine choice, not forced), or None if no such decisions
#     exist in this batch."""
#     probs = policy.get_action_probs(obs)    # [B,R,K+1]
#     logits = policy.get_action_logits(obs)  # [B,R,K+1]
#     cand_mask = np.asarray(obs["cand_mask"]).astype(bool)  # [B,R,K]
#     K = cand_mask.shape[-1]

#     has_real = cand_mask.any(axis=-1)  # [B,R]
#     if not has_real.any():
#         return None

#     p_noop_all = probs[..., K]
#     l_noop_all = logits[..., K]

#     masked_p_real = np.where(cand_mask, probs[..., :K], -np.inf)
#     masked_l_real = np.where(cand_mask, logits[..., :K], -np.inf)
#     p_real_sum_all = np.where(cand_mask, probs[..., :K], 0.0).sum(axis=-1)
#     p_best_real_all = masked_p_real.max(axis=-1)
#     l_best_real_all = masked_l_real.max(axis=-1)

#     p_noop = p_noop_all[has_real]
#     l_noop = l_noop_all[has_real]
#     p_real_sum = p_real_sum_all[has_real]
#     p_best_real = p_best_real_all[has_real]
#     l_best_real = l_best_real_all[has_real]

#     overall_max_prob = probs[has_real].max(axis=-1)
#     is_plurality = np.isclose(p_noop, overall_max_prob)
#     is_majority = p_noop > 0.5

#     valid_mask_full = np.concatenate([cand_mask, np.ones_like(cand_mask[..., :1])], axis=-1)
#     logits_masked_full = np.where(valid_mask_full, logits, np.nan)[has_real]
#     overall_max_logit = np.nanmax(logits_masked_full, axis=-1)
#     overall_mean_logit = np.nanmean(logits_masked_full, axis=-1)

#     return {
#         "n": int(has_real.sum()),
#         "p_noop_mean": float(p_noop.mean()),
#         "p_best_real_mean": float(p_best_real.mean()),
#         "p_real_sum_mean": float(p_real_sum.mean()),
#         "logit_noop_mean": float(l_noop.mean()),
#         "logit_best_real_mean": float(l_best_real.mean()),
#         "logit_gap_mean": float((l_noop - l_best_real).mean()),
#         "noop_plurality_rate": float(is_plurality.mean()),
#         "noop_majority_rate": float(is_majority.mean()),
#         "overall_max_logit_mean": float(overall_max_logit.mean()),
#         "overall_mean_logit_mean": float(overall_mean_logit.mean()),
#     }


# def compute_all_action_logit_stats(policy: "RTGNNPolicy", obs: Any, K_max: int):
#     """'Logits of all actions' — per-CANDIDATE-RANK mean logit, plus noop.
#     Unlike compute_noop_logit_stats (which only tracks the single best real
#     candidate vs noop), this ranks each decision's real-candidate logits
#     descending and tracks rank 0 (best), rank 1 (2nd best), ... rank K-1
#     separately, so you can see the whole shape of the action distribution
#     over training, not just the noop-vs-winner margin. A decision only
#     contributes to rank i if it actually had >=i+1 valid candidates that
#     step (fewer candidates just means fewer ranks get a contribution from
#     that particular decision, tracked via a per-rank count for correct
#     weighted averaging).

#     Returns dict with:
#       "noop_logit_mean": float
#       "rank_logit_means": list[float] of length K_max (NaN-safe: rank i is
#           None if no decision in this batch ever had that many candidates)
#       "rank_counts": list[int] of length K_max
#     """
#     logits = policy.get_action_logits(obs)  # [B,R,K+1]
#     cand_mask = np.asarray(obs["cand_mask"]).astype(bool)  # [B,R,K]
#     K = cand_mask.shape[-1]
#     assert K == K_max, f"K_max mismatch: obs has {K}, expected {K_max}"

#     l_noop = logits[..., K]
#     has_real = cand_mask.any(axis=-1)
#     noop_logit_mean = float(l_noop[has_real].mean()) if has_real.any() else None

#     cand_logits = logits[..., :K]  # [B,R,K]
#     masked = np.where(cand_mask, cand_logits, -np.inf)
#     # sort each decision's valid candidate logits descending, pad invalid
#     # slots to -inf so they sort to the end and don't pollute real ranks
#     sorted_desc = -np.sort(-masked, axis=-1)  # [B,R,K], descending

#     rank_means = []
#     rank_counts = []
#     for rank in range(K_max):
#         col = sorted_desc[..., rank]
#         valid = np.isfinite(col)
#         count = int(valid.sum())
#         rank_counts.append(count)
#         rank_means.append(float(col[valid].mean()) if count > 0 else None)

#     return {
#         "noop_logit_mean": noop_logit_mean,
#         "rank_logit_means": rank_means,
#         "rank_counts": rank_counts,
#     }

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, List, cast, Literal

import torch as th
import torch.nn as nn
import numpy as np
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

    def get_distribution(self, obs):
        obs = {k: to_numpy(v) for k, v in obs.items()}
        obs_tensor, _ = self.obs_to_tensor(obs)

        _ = self.extract_features(
            obs_tensor,
            features_extractor=self.features_extractor,
        )
        obs_b = self.features_extractor.last_obs

        logits_k, _ = self._build_batch_outputs(obs_b)
        cand_mask = obs_b["cand_mask"]

        logits_full, mask_full = self._append_noop_and_mask(
            logits_k,
            cand_mask,
        )
        logits_full = logits_full.masked_fill(~mask_full, -1e9)

        B = logits_full.shape[0]
        logits_flat = logits_full.reshape(B, -1)

        return self._dist_from_logits_flat(logits_flat)
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
            # print debug
            # if not hasattr(self, "_actor_debug"):
            #     self._actor_debug = 0
            # self._actor_debug += 1

            # if self._actor_debug % 200 == 0:
            #     print("\n========================")
            #     print("Actor output")
            #     print("mean :", logits_rk.mean().item())
            #     print("std  :", logits_rk.std().item())
            #     print("min  :", logits_rk.min().item())
            #     print("max  :", logits_rk.max().item())
            #     print(logits_rk)
            # ---------
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

        # For conflict_resolution='hungarian_bids': push this step's raw
        # candidate logits (pre-noop, pre-mask — bid strength is the
        # policy's actual preference, not a masked sentinel) to the env
        # so its conflict resolver can use them as bid values instead of
        # distance. self._bid_env is set externally (see train_ppo.py)
        # to the VecEnv this policy is collecting rollouts in; None by
        # default, so this is a no-op unless explicitly wired up. Only
        # meaningful during rollout collection (this method), never during
        # evaluate_actions() (the training-update replay path), so it
        # cannot leak into gradient computation or affect what actually
        # gets trained on — it only affects the live environment step
        # that's about to happen.
        if getattr(self, "_bid_env", None) is not None:
            logits_k_np = logits_k.detach().cpu().numpy()
            for b in range(logits_k_np.shape[0]):
                self._bid_env.env_method("set_pending_logits", logits_k_np[b], indices=b)

        #------------debug for candidate mask
        # print("cand_mask:", cand_mask[0])
        # if not hasattr(self, "_mask_debug"):
        #     self._mask_debug = 0

        # self._mask_debug += 1

        # if self._mask_debug % 200 == 0:

        #     print("\nCandidate mask")
        # print(cand_mask[0])


        # print(
        #     "valid candidates per robot:",
        #     cand_mask[0].sum(dim=-1)
        # )
        #------------debug for candidate mask---------------
        logits_full, mask_full = self._append_noop_and_mask(logits_k, cand_mask)
        # pribt debug
        # probs = th.softmax(logits_full, dim=-1)
        # print(logits_full[0], '====================')
        # print(probs[0], '====================')
        # -------
        logits_full = logits_full.masked_fill(~mask_full, -1e9)

        B = logits_full.shape[0]
        logits_flat = logits_full.reshape(B, -1)                  # [B, R*(K+1)]
        # print(logits_flat[0].reshape(self.R, self.K+1), 'logit before logit_flat')
        dist = self._dist_from_logits_flat(logits_flat)
        actions_flat = dist.get_actions(deterministic=deterministic)  # [B, R]
        # print('action flat', actions_flat)
        # Reshape for per-robot log_prob computation
        actions = actions_flat.reshape(B, self.R)
        active  = mask_full[..., :self.K].any(dim=-1)             # [B,R] — has real candidates
        log_prob, _ = self.masked_logprob_entropy(logits_full, actions, active)
        # print(type(dist))
        # print(type(dist.distribution))
        # print(len(dist.distribution))
        # print(type(dist.distribution[0]))
        # print(dist.distribution[0])
        # print(print(dist.distribution[0].logits))
        # print(dist.distribution.probs[0], 'dist.distribution.probs[0]')
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
        #----debug for actor weights--------------
        # if not hasattr(self, "_debug_counter"):
        #     self._debug_counter = 0

        # self._debug_counter += 1

        # if self._debug_counter % 200 == 0:
        #     print("\n===== ACTOR DEBUG =====")
        #     print("actor_head weight norm:",
        #         self.gnn_ac.actor_head.weight.norm().item())
        #     print("actor_head bias:",
        #         self.gnn_ac.actor_head.bias.data.cpu().numpy())
        #----------------------
        cand_mask = obs_b["cand_mask"]
        logits_full, mask_full = self._append_noop_and_mask(logits_k, cand_mask)
        logits_full = logits_full.masked_fill(~mask_full, -1e9)
        #-------------debug for actor weights--------------
        # print(
        #     "logits:",
        #     logits_full[0, 0].detach().cpu().numpy()
        # )
        #-------------debug for actor weights--------------
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

    @th.no_grad()
    @th.no_grad()
    def _get_masked_logits(self, obs: Any) -> th.Tensor:
        """Shared internal computation for get_action_probs/get_action_logits —
        identical masked-logit path used by forward()/evaluate_actions().
        Returns logits_full [B, R, K_max+1], with invalid (masked-out)
        candidate slots set to -1e9 (same convention as training)."""
        obs = {
            k: to_numpy(v)
            for k, v in obs.items()
        }
        obs_tensor, _ = self.obs_to_tensor(obs)
        _ = self.extract_features(obs_tensor, features_extractor=self.features_extractor)
        obs_b = self.features_extractor.last_obs
        assert obs_b is not None

        logits_k, _values = self._build_batch_outputs(obs_b)      # [B,R,K]
        cand_mask = obs_b["cand_mask"]                            # [B,R,K]
        logits_full, mask_full = self._append_noop_and_mask(logits_k, cand_mask)  # [B,R,K+1]
        logits_full = logits_full.masked_fill(~mask_full, -1e9)
        return logits_full

    @th.no_grad()
    def get_action_logits(self, obs: Any) -> np.ndarray:
        """Diagnostic helper — raw (pre-softmax) logits, shape [B, R, K_max+1].
        Masked-out (invalid) candidate slots are set to -1e9, matching the
        exact masking used in forward()/evaluate_actions() — filter those
        out using cand_mask before computing statistics (e.g. mean/spread),
        since -1e9 entries would otherwise dominate any naive average.
        """
        logits_full = self._get_masked_logits(obs)
        return logits_full.cpu().numpy()

    @th.no_grad()
    def get_action_probs(self, obs: Any) -> np.ndarray:
        """Diagnostic helper — NOT used by training/predict(), only for
        inspecting the actual per-robot categorical distribution (softmax
        over K_max real candidates + 1 noop slot) that deterministic/
        stochastic action selection is drawn from.

        Returns probs with shape [B, R, K_max+1], softmax-normalized over
        the last axis, with masked-out (invalid) candidate slots at ~0
        probability (since their logits were set to -1e9 before softmax,
        same as in forward()/evaluate_actions() — this method reuses that
        exact same masked logit computation so the numbers reported here
        are guaranteed to match what deterministic()/sampling actually see).
        """
        logits_full = self._get_masked_logits(obs)
        probs = th.softmax(logits_full, dim=-1)  # [B, R, K+1]
        return probs.cpu().numpy()

    def _predict(self, observation: th.Tensor, deterministic: bool = False) -> th.Tensor:
        actions, _values, _log_prob = self.forward(observation, deterministic=deterministic)
        return actions


def compute_noop_logit_stats(policy: "RTGNNPolicy", obs: Any):
    """Shared helper for training-time and eval-time logit/probability
    logging (noop vs best-real-candidate summary). Given a batched obs
    dict (any batch size), returns aggregate stats across every
    (batch, robot) decision that had >=1 real candidate available (i.e.
    noop was a genuine choice, not forced), or None if no such decisions
    exist in this batch."""
    probs = policy.get_action_probs(obs)    # [B,R,K+1]
    logits = policy.get_action_logits(obs)  # [B,R,K+1]
    cand_mask = np.asarray(obs["cand_mask"]).astype(bool)  # [B,R,K]
    K = cand_mask.shape[-1]

    has_real = cand_mask.any(axis=-1)  # [B,R]
    if not has_real.any():
        return None

    p_noop_all = probs[..., K]
    l_noop_all = logits[..., K]

    masked_p_real = np.where(cand_mask, probs[..., :K], -np.inf)
    masked_l_real = np.where(cand_mask, logits[..., :K], -np.inf)
    p_real_sum_all = np.where(cand_mask, probs[..., :K], 0.0).sum(axis=-1)
    p_best_real_all = masked_p_real.max(axis=-1)
    l_best_real_all = masked_l_real.max(axis=-1)

    p_noop = p_noop_all[has_real]
    l_noop = l_noop_all[has_real]
    p_real_sum = p_real_sum_all[has_real]
    p_best_real = p_best_real_all[has_real]
    l_best_real = l_best_real_all[has_real]

    overall_max_prob = probs[has_real].max(axis=-1)
    is_plurality = np.isclose(p_noop, overall_max_prob)
    is_majority = p_noop > 0.5

    valid_mask_full = np.concatenate([cand_mask, np.ones_like(cand_mask[..., :1])], axis=-1)
    logits_masked_full = np.where(valid_mask_full, logits, np.nan)[has_real]
    overall_max_logit = np.nanmax(logits_masked_full, axis=-1)
    overall_mean_logit = np.nanmean(logits_masked_full, axis=-1)

    return {
        "n": int(has_real.sum()),
        "p_noop_mean": float(p_noop.mean()),
        "p_best_real_mean": float(p_best_real.mean()),
        "p_real_sum_mean": float(p_real_sum.mean()),
        "logit_noop_mean": float(l_noop.mean()),
        "logit_best_real_mean": float(l_best_real.mean()),
        "logit_gap_mean": float((l_noop - l_best_real).mean()),
        "noop_plurality_rate": float(is_plurality.mean()),
        "noop_majority_rate": float(is_majority.mean()),
        "overall_max_logit_mean": float(overall_max_logit.mean()),
        "overall_mean_logit_mean": float(overall_mean_logit.mean()),
    }


def compute_all_action_logit_stats(policy: "RTGNNPolicy", obs: Any, K_max: int):
    """'Logits of all actions' — per-CANDIDATE-RANK mean logit, plus noop.
    Unlike compute_noop_logit_stats (which only tracks the single best real
    candidate vs noop), this ranks each decision's real-candidate logits
    descending and tracks rank 0 (best), rank 1 (2nd best), ... rank K-1
    separately, so you can see the whole shape of the action distribution
    over training, not just the noop-vs-winner margin. A decision only
    contributes to rank i if it actually had >=i+1 valid candidates that
    step (fewer candidates just means fewer ranks get a contribution from
    that particular decision, tracked via a per-rank count for correct
    weighted averaging).

    Returns dict with:
      "noop_logit_mean": float
      "rank_logit_means": list[float] of length K_max (NaN-safe: rank i is
          None if no decision in this batch ever had that many candidates)
      "rank_counts": list[int] of length K_max
    """
    logits = policy.get_action_logits(obs)  # [B,R,K+1]
    cand_mask = np.asarray(obs["cand_mask"]).astype(bool)  # [B,R,K]
    K = cand_mask.shape[-1]
    assert K == K_max, f"K_max mismatch: obs has {K}, expected {K_max}"

    l_noop = logits[..., K]
    has_real = cand_mask.any(axis=-1)
    noop_logit_mean = float(l_noop[has_real].mean()) if has_real.any() else None

    cand_logits = logits[..., :K]  # [B,R,K]
    masked = np.where(cand_mask, cand_logits, -np.inf)
    # sort each decision's valid candidate logits descending, pad invalid
    # slots to -inf so they sort to the end and don't pollute real ranks
    sorted_desc = -np.sort(-masked, axis=-1)  # [B,R,K], descending

    rank_means = []
    rank_counts = []
    for rank in range(K_max):
        col = sorted_desc[..., rank]
        valid = np.isfinite(col)
        count = int(valid.sum())
        rank_counts.append(count)
        rank_means.append(float(col[valid].mean()) if count > 0 else None)

    return {
        "noop_logit_mean": noop_logit_mean,
        "rank_logit_means": rank_means,
        "rank_counts": rank_counts,
    }