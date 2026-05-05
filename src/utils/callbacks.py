

"""
Callback that logs episode stats + reward components + policy behavior metrics:
- NOOP fraction on meaningful decision steps
- Assigned fraction on meaningful decision steps
- Collision-drop fraction on meaningful decision steps
"""
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from pathlib import Path
import json
from typing import Any, Dict
import torch as th
from typing import Any, Dict, Tuple, Optional

class FinalTaskAllocationCallback(BaseCallback):
    def __init__(self, save_freq=1000, save_path="./logs", verbose=1):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = Path(save_path)
        self.save_path.mkdir(parents=True, exist_ok=True)

        self.episode_completions = []
        self.episode_obsolete = []
        self.episode_count = 0

        # per-episode reward sums
        self._ep_rew_sums = {}

        # --- policy metrics accumulators (meaningful decision steps only) ---
        self._meaningful_decision_steps = 0
        self._noop_count = 0
        self._action_count = 0
        self._assigned_count = 0
        self._drop_count = 0

    #add a helper to extract reward component for new reward structure

    @staticmethod
    def _extract_reward_info(info: Dict[str, Any]) -> Tuple[Optional[str], Dict[str, float]]:
        """
        Returns (reward_mode, rew_dict) where rew_dict keys are like 'rew/...'
        and values are floats for logging.
        Supports both:
          - flattened rew/* already in info
          - env-provided info_reward dict (new or old)
        """
        rew: Dict[str, float] = {}

        # Case A) already flattened into info as rew/*
        for k, v in info.items():
            if isinstance(k, str) and k.startswith("rew/") and isinstance(v, (int, float, np.number)):
                rew[k] = float(v)

        # If we already have rew/* keys, we can still also try to add missing ones from info_reward.
        info_reward = info.get("info_reward", None)
        if not isinstance(info_reward, dict):
            # Some envs might put reward info directly in "reward_info" or similar
            info_reward = info.get("reward_info", None)

        if not isinstance(info_reward, dict):
            return (info.get("reward_mode", None), rew)

        reward_mode = info_reward.get("reward_mode", None)

        # Always log total reward if present
        if "sum_rewards" in info_reward and isinstance(info_reward["sum_rewards"], (int, float, np.number)):
            rew.setdefault("rew/sum_rewards", float(info_reward["sum_rewards"]))

        # OLD reward components
        if "pickups_this_step" in info_reward:
            if isinstance(info_reward.get("pickups_this_step"), (int, float, np.number)):
                rew["rew/pickups_this_step"] = float(info_reward["pickups_this_step"])
            if isinstance(info_reward.get("deliveries_this_step"), (int, float, np.number)):
                rew["rew/deliveries_this_step"] = float(info_reward["deliveries_this_step"])
            if isinstance(info_reward.get("obsolete_this_step"), (int, float, np.number)):
                rew["rew/obsolete_this_step"] = float(info_reward["obsolete_this_step"])

            # infer mode if not set
            if reward_mode is None:
                reward_mode = "old"

        # NEW reward components
        terms = info_reward.get("terms", None)
        if isinstance(terms, dict):
            for key in ["completion_events", "abandoned_events", "missed_dropoff_events", "backlog_tasks"]:
                v = terms.get(key, None)
                if isinstance(v, (int, float, np.number)):
                    rew[f"rew/{key}"] = float(v)

            if reward_mode is None:
                reward_mode = "new"

        return (reward_mode, rew)

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        actions = self.locals.get("actions", None)  # VecEnv actions

        # ---------- policy behavior metrics (use env0 info + env0 actions) ----------
        if actions is not None and isinstance(infos, (list, tuple)) and len(infos) > 0:
            info0 = infos[0] if isinstance(infos[0], dict) else {}
            is_meaningful = bool(info0.get("meaningful_decision_step", False))

            # Only compute these stats when decisions actually matter
            if is_meaningful:
                # --- log per-action mean logits (from policy) ---
                pol = getattr(self.model, "policy", None)
                means = getattr(pol, "last_action_logit_means", None)# --- NEW: logit separation metrics (need current obs + policy forward) ---
                try:
                    pol = getattr(self.model, "policy", None)

                    # Try to get latest obs dict from SB3 locals
                    obs_any = self.locals.get("new_obs", None)
                    if obs_any is None:
                        obs_any = self.locals.get("obs", None)

                    # VecEnv => obs is typically a dict of arrays with leading dim n_envs
                    if pol is not None and isinstance(obs_any, dict):
                        # take env0
                        # obs0 = {k: v[0] for k, v in obs_any.items()}

                        # # convert to torch tensors on correct device
                        # obs_t = {
                        #     k: th.as_tensor(v, device=pol.device)
                        #     for k, v in obs0.items()
                        # }

                        # # get masked logits directly
                        # logits_t, _values_t, _active_t = pol._masked_logits_and_value(obs_t)  # [1,R,K+1]
                        
                        obs0 = {k: v[0] for k, v in obs_any.items()}  # env0

                        obs_t = {}
                        for k, v in obs0.items():
                            t = th.as_tensor(v, device=pol.device)

                            # Add batch dim (B=1) to match RTGNNPolicy expectations
                            # edge_index should be [B,2,E]
                            if k == "edge_index":
                                if t.ndim == 2:
                                    t = t.unsqueeze(0)
                            else:
                                if t.ndim >= 1:
                                    t = t.unsqueeze(0)

                            obs_t[k] = t

                        logits_t, _values_t, _active_t = pol._masked_logits_and_value(obs_t)  # [1,R,K+1]
                        logits_np = logits_t.detach().cpu().numpy()[0]  # [R,K+1]
                        logits_np = logits_t.detach().cpu().numpy()[0]  # [R,K+1]

                        mask = info0.get("action_mask", None)
                        if isinstance(mask, np.ndarray) and mask.ndim == 2:
                            R, Kp1 = mask.shape
                            noop_index = Kp1 - 1
                            stats = self._logit_separation_stats(
                                logits=logits_np,
                                mask=mask,
                                noop_index=noop_index,
                            )
                            for k, v in stats.items():
                                if isinstance(v, (int, float, np.number)) and np.isfinite(v):
                                    self.logger.record(f"logits/{k}", float(v))
                except Exception:
                    pass
                if isinstance(means, dict) and means:
                    for k, v in means.items():
                        if isinstance(v, (int, float, np.number)):
                            self.logger.record(f"logits/{k}", float(v))
                mask = info0.get("action_mask", None)
                # mask should be [R, K+1] and NOOP is last index
                if isinstance(mask, np.ndarray) and mask.ndim == 2:
                    R, Kp1 = mask.shape
                    noop_index = int(Kp1 - 1)

                    # actions usually shape (n_envs, R); take env0 and flatten
                    a = np.asarray(actions)
                    a0 = a[0] if a.ndim >= 2 else a
                    a0 = np.asarray(a0).astype(int).reshape(-1)

                    if a0.size == R:
                        resolved = info0.get("resolved_assignments", {}) or {}
                        if not isinstance(resolved, dict):
                            resolved = {}

                        assigned = int(len(resolved))  # after conflict resolution

                        # chosen non-NOOP actions
                        non_noop = int(np.sum(a0 != noop_index))
                        noop = int(np.sum(a0 == noop_index))

                        # robots that tried to assign but got nothing after conflict resolution
                        dropped = max(0, non_noop - assigned)

                        # accumulate
                        self._meaningful_decision_steps += 1
                        self._noop_count += noop
                        self._action_count += int(R)
                        self._assigned_count += assigned
                        self._drop_count += dropped

                        # log running fractions (over meaningful decision steps)
                        denom = float(max(1, self._action_count))
                        self.logger.record("policy/noop_fraction_meaningful", float(self._noop_count) / denom)
                        self.logger.record("policy/assigned_fraction_meaningful", float(self._assigned_count) / denom)
                        self.logger.record("policy/collision_drop_fraction_meaningful", float(self._drop_count) / denom)

        # ---------- reward component logging + episode stats ----------
        for info in infos:
            if not isinstance(info, dict):
                continue

            # log rew/*
            # for k, v in info.items():
            #     if not (isinstance(k, str) and k.startswith("rew/")):
            #         continue
            #     if isinstance(v, (int, float, np.number)):
            #         v = float(v)
            #         self.logger.record(f"{k}_step", v)
            #         self._ep_rew_sums[k] = self._ep_rew_sums.get(k, 0.0) + v
            reward_mode, rew = self._extract_reward_info(info)
            if reward_mode is not None:
                self.logger.record("rew/reward_mode", 0.0 if reward_mode == "old" else 1.0)

            for k, v in rew.items():
                self.logger.record(f"{k}_step", float(v))
                self._ep_rew_sums[k] = self._ep_rew_sums.get(k, 0.0) + float(v)
            # episode end
            # Support both the new key names (completed_count / obsolete_count)
            # produced by MultiAgentTaskEnv and the legacy names.
            _has_episode_end = (
                "episode_completed" in info
                or "completed_count" in info
            )
            if _has_episode_end:
                completed = int(
                    info.get("episode_completed",
                             info.get("completed_count", 0))
                )
                obsolete = int(
                    info.get("episode_obsolete",
                             info.get("obsolete_count", 0))
                )

                self.episode_completions.append(completed)
                self.episode_obsolete.append(obsolete)
                self.episode_count += 1

                self.logger.record("task/completed", completed)
                self.logger.record("task/obsolete", obsolete)
                self.logger.record("task/completion_rate", 100 * completed / 40)

                for k, s in self._ep_rew_sums.items():
                    self.logger.record(f"{k}_episode_sum", float(s))
                self._ep_rew_sums = {}

                if self.verbose > 0 and self.episode_count % 10 == 0:
                    recent = min(10, len(self.episode_completions))
                    print(f"\n[Episode {self.episode_count}] Completed: {completed}/40, Obsolete: {obsolete}")
                    print(f"  Last {recent} avg: {np.mean(self.episode_completions[-recent:]):.1f} completed")

        # Save periodically
        if self.n_calls % self.save_freq == 0 and len(self.episode_completions) > 0:
            self._save_metrics()

        return True
    @staticmethod
    def _logit_separation_stats(
        logits: np.ndarray,  # [R, K+1]
        mask: np.ndarray,    # [R, K+1] bool or 0/1
        noop_index: int,
    ) -> Dict[str, float]:
        """
        Compute per-robot best/2nd-best task logits and gaps, then average across robots
        that have at least one valid task (excluding NOOP).
        """
        m = mask.astype(bool)
        R, Kp1 = logits.shape

        # valid tasks exclude noop
        valid_tasks = m.copy()
        valid_tasks[:, noop_index] = False

        # active robots: at least one valid task
        active = valid_tasks.any(axis=1)
        if not np.any(active):
            return {}

        # Mask invalid task logits to -inf so max works
        task_logits = logits[:, :].copy()
        task_logits[~valid_tasks] = -np.inf

        # best task per robot
        best = np.max(task_logits, axis=1)

        # second best: set best index to -inf and max again
        best_idx = np.argmax(task_logits, axis=1)
        task_logits2 = task_logits.copy()
        for r in range(R):
            task_logits2[r, best_idx[r]] = -np.inf
        second = np.max(task_logits2, axis=1)

        noop = logits[:, noop_index]

        # filter active and finite
        a = active & np.isfinite(best)
        if not np.any(a):
            return {}

        best_a = best[a]
        second_a = second[a]
        noop_a = noop[a]

        # if second is -inf (only 1 valid task), ignore those in best-second gap
        finite_second = np.isfinite(second_a)

        out: Dict[str, float] = {}
        out["max_task_logit_mean"] = float(np.mean(best_a))
        out["noop_logit_mean"] = float(np.mean(noop_a))
        out["gap_best_minus_noop_mean"] = float(np.mean(best_a - noop_a))

        if np.any(finite_second):
            out["second_best_task_logit_mean"] = float(np.mean(second_a[finite_second]))
            out["gap_best_minus_second_mean"] = float(np.mean(best_a[finite_second] - second_a[finite_second]))
        else:
            out["second_best_task_logit_mean"] = float("nan")
            out["gap_best_minus_second_mean"] = float("nan")

        return out
    def _save_metrics(self):
        metrics = {
            "episode_completions": self.episode_completions,
            "episode_obsolete": self.episode_obsolete,
            "num_episodes": self.episode_count,
        }
        with open(self.save_path / f"metrics_step_{self.n_calls}.json", "w") as f:
            json.dump(metrics, f, indent=2)