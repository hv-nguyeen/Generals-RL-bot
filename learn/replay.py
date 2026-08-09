"""Small game-balanced replay for critic refreshes.

Policy samples remain strictly on-policy. Only the value model reads this ring:
games are sampled uniformly first and positions uniformly second, so a 900-turn
draw cannot outweigh a 150-turn decisive game merely by contributing six times
as many rows.
"""

from __future__ import annotations

from collections import deque

import numpy as np


class GameBalancedReplay:
    """Bounded episode ring storing float16 observations and scalar outcomes."""

    def __init__(self, capacity_episodes: int):
        if capacity_episodes < 0:
            raise ValueError("capacity_episodes must be non-negative")
        self.capacity = int(capacity_episodes)
        self._episodes: deque[tuple[np.ndarray, float, str]] = deque(
            maxlen=self.capacity or None)

    def __len__(self) -> int:
        return len(self._episodes)

    @property
    def rows(self) -> int:
        return sum(len(x) for x, _, _ in self._episodes)

    def add(self, x: np.ndarray, outcome: float, source: str = "current") -> None:
        if self.capacity == 0:
            return
        x = np.asarray(x)
        if x.ndim != 4 or len(x) == 0:
            raise ValueError(f"episode observations must be non-empty (T,C,H,W), got {x.shape}")
        if outcome not in (-1.0, 0.0, 1.0):
            raise ValueError(f"episode outcome must be -1/0/+1, got {outcome}")
        self._episodes.append((x.astype(np.float16, copy=True), float(outcome), str(source)))

    def sample(self, rng: np.random.Generator, rows: int,
               source_floor: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
        """Uniform episode then uniform position, optionally diversifying sources.

        ``source_floor`` mixes the empirical source distribution with uniform
        over sources. It is useful for league critics whose latest opponent can
        otherwise occupy the entire ring before older styles are revisited.
        """
        if rows < 1 or not self._episodes:
            raise ValueError("sample needs positive rows and a non-empty replay")
        if not 0.0 <= source_floor <= 1.0:
            raise ValueError("source_floor must be in [0, 1]")
        episodes = list(self._episodes)
        if source_floor and len({e[2] for e in episodes}) > 1:
            sources = sorted({e[2] for e in episodes})
            by_source = {s: np.asarray([i for i, e in enumerate(episodes) if e[2] == s])
                         for s in sources}
            empirical = np.full(len(episodes), (1.0 - source_floor) / len(episodes))
            for ids in by_source.values():
                empirical[ids] += source_floor / len(sources) / len(ids)
            episode_ids = rng.choice(len(episodes), size=rows, p=empirical)
        else:
            episode_ids = rng.integers(0, len(episodes), size=rows)
        xs, z = [], np.empty(rows, np.float32)
        for j, i in enumerate(episode_ids):
            episode, outcome, _ = episodes[int(i)]
            xs.append(episode[int(rng.integers(len(episode)))])
            z[j] = outcome
        return np.stack(xs), z
