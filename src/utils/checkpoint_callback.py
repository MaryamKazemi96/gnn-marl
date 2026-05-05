from __future__ import annotations

import re
import glob
from pathlib import Path
from typing import Optional, Tuple

from stable_baselines3.common.callbacks import BaseCallback


def latest_model_path(model_dir: str) -> Tuple[str, int, int]:
    """
    Find latest model_episode<ep>_ts<ts>.zip in model_dir and return (path, ep, ts).
    """
    pattern = str(Path(model_dir) / "model_episode*_ts*.zip")
    candidates = []
    for path in glob.glob(pattern):
        m = re.search(r"model_episode(\d+)_ts(\d+)\.zip$", Path(path).name)
        if m:
            ep = int(m.group(1))
            ts = int(m.group(2))
            candidates.append((ts, ep, path))
    if not candidates:
        raise FileNotFoundError(f"No saved models found in {model_dir}")
    candidates.sort(key=lambda x: x[0])
    ts, ep, path = candidates[-1]
    return path, ep, ts


class EpisodeTimestepCheckpointCallback(BaseCallback):
    """
    Save checkpoints named like: model_episode{episode}_ts{timesteps}.zip

    Notes:
    - SB3 callback has access to self.num_timesteps.
    - We infer episode count by counting 'episode_completed' occurrences in infos.
      (Works with your env which provides episode stats in info at episode end.)
    """

    def __init__(self, save_dir: str | Path, save_every_steps: int = 5000, verbose: int = 0):
        super().__init__(verbose)
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.save_every_steps = int(save_every_steps)

        self.episode_idx = 0
        self._last_saved_ts = 0

    def _on_step(self) -> bool:
        # count episode ends
        infos = self.locals.get("infos", [])
        for info in infos:
            if isinstance(info, dict) and ("episode_completed" in info):
                self.episode_idx += 1

        # periodic save
        ts = int(self.num_timesteps)
        if ts - self._last_saved_ts >= self.save_every_steps:
            self._last_saved_ts = ts
            path = self.save_dir / f"model_episode{self.episode_idx}_ts{ts}.zip"
            self.model.save(path)
            if self.verbose:
                print(f"[CKPT] Saved {path}")

        return True