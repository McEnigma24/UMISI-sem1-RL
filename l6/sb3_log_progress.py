"""
Postęp treningu SB3: interaktywny pasek (TTY) vs linie w pliku logów (Slurm, przekierowania).

Stable-Baselines3 używa ``tqdm.rich`` — w logach batchowych (brak TTY) pasek zwykle
znika lub jest bezużyteczny. Ten moduł dodaje okresowe ``print`` z procentem.
"""
from __future__ import annotations

import os
import sys

from stable_baselines3.common.callbacks import BaseCallback


def use_sb3_interactive_progress_bar() -> bool:
    """True: zostaw ``progress_bar=True`` (Rich). False: wyłącz pasek, użyj :class:`LogFileProgressCallback`."""
    return sys.stdout.isatty()


def log_progress_interval_steps() -> int:
    raw = os.environ.get("SB3_LOG_PROGRESS_EVERY", "25000").strip().replace("_", "")
    try:
        n = int(raw, 10)
    except ValueError:
        return 25_000
    return max(1_000, n)


class LogFileProgressCallback(BaseCallback):
    """
    Co ``log_every_steps`` kroków środowiska wypisuje jedną linię ``[progress] ...`` (flush),
    czytelna w ``GRID_log_out_*.txt`` na Cyfronecie.
    """

    def __init__(self, log_every_steps: int | None = None) -> None:
        super().__init__(verbose=0)
        self.log_every_steps = (
            log_every_steps if log_every_steps is not None else log_progress_interval_steps()
        )
        self._goal: int = 0
        self._next_milestone: int = 0

    def _on_training_start(self) -> None:
        self._goal = int(self.locals["total_timesteps"])
        every = self.log_every_steps
        ts = self.num_timesteps
        self._next_milestone = min(self._goal, ((ts // every) + 1) * every)
        print(
            f"[progress] start timesteps={ts:_} / goal={self._goal:_} "
            f"(log co {every:_} kroków — tryb pliku logów / Slurm)",
            flush=True,
        )

    def _on_step(self) -> bool:
        while self.num_timesteps >= self._next_milestone and self._next_milestone <= self._goal:
            ts = min(self.num_timesteps, self._goal)
            pct = 100.0 * ts / max(self._goal, 1)
            print(
                f"[progress] timesteps {ts:_}/{self._goal:_} ({pct:.1f}%)",
                flush=True,
            )
            self._next_milestone += self.log_every_steps
        return True

    def _on_training_end(self) -> None:
        ts = min(self.num_timesteps, self._goal)
        pct = 100.0 * ts / max(self._goal, 1)
        print(
            f"[progress] finished timesteps {ts:_}/{self._goal:_} ({pct:.1f}%)",
            flush=True,
        )
