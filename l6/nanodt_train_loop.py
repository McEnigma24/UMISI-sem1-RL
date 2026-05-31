"""
Patched Decision Transformer training loop for nanoDT.

Upstream ``nanodt.trainer.DecisionTransformerTrainer.train`` uses a finite
``WeightedRandomSampler`` and a single-pass ``iter(DataLoader)``, but calls
``next(dataloader_iter)`` more often than ``num_samples`` (train + eval),
which raises ``StopIteration`` on short runs and on Windows.

Changes vs nanoDT 0.1.0 (MIT):
- ``dataloader_iter = itertools.cycle(iter(dataloader))``
- ``num_workers=0`` on Windows, else ``2``
- Optional early stopping (plateau, consecutive high loss, optional online success)
- Optional periodic checkpoints compatible with ``NanoDTAgent.load``

Source reference: https://github.com/lubiluk/nanoDT
"""
from __future__ import annotations

import itertools
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from nanodt.trainer import DecisionTransformerDataCollator, calculate_dataset_stats

if TYPE_CHECKING:
    from nanodt.trainer import DecisionTransformerTrainer


@dataclass
class L6TrainHooks:
    """Set via ``set_pending_l6_train_hooks`` before ``NanoDTAgent.learn``; read at train start."""

    # Early stop: consecutive evals with loss >= max (disabled if loss_max is None)
    early_stop_patience: int | None = None
    early_stop_loss_max: float | None = None
    early_stop_on: str = "val"  # "val" | "train" | "both"

    # Plateau: stop if (max - min) of val loss over last window evals < epsilon (disabled if epsilon None)
    early_stop_plateau_epsilon: float | None = None
    early_stop_plateau_window: int | None = None

    # Optional online rollout (same iter modulo as eval_interval alignment optional)
    online_eval_every_iters: int | None = None
    online_eval_episodes: int = 10
    online_eval_fn: Callable[[int], dict[str, float]] | None = None
    # Stop if online success_rate_final >= this (good); disabled if None
    early_stop_success_min: float | None = None
    # Stop after this many consecutive online checks with success_rate_final <= max (disabled if None)
    early_stop_success_max: float | None = None
    early_stop_success_patience: int | None = None

    save_every_iters: int | None = None
    checkpoint_dir: Path | None = None
    # Required for checkpoints: same object as ``NanoDTAgent`` used in learn()
    checkpoint_agent: Any | None = None


_PENDING_L6_TRAIN_HOOKS: L6TrainHooks | None = None

# Filled when training ends (early or max iters); consumed by train_dt for manifest / provenance
LAST_TRAIN_REPORT: dict[str, Any] = {
    "finished_iter": 0,
    "stop_reason": "unknown",
    "eval_history": [],
}


def set_pending_l6_train_hooks(hooks: L6TrainHooks | None) -> None:
    global _PENDING_L6_TRAIN_HOOKS
    _PENDING_L6_TRAIN_HOOKS = hooks


def clear_pending_l6_train_hooks() -> None:
    global _PENDING_L6_TRAIN_HOOKS
    _PENDING_L6_TRAIN_HOOKS = None


def _loss_metric(losses: dict[str, float], on: str) -> float:
    if on == "val":
        return float(losses["val"])
    if on == "train":
        return float(losses["train"])
    if on == "both":
        return max(float(losses["train"]), float(losses["val"]))
    raise ValueError(f"early_stop_on must be val|train|both, got {on!r}")


def _save_mid_checkpoint(
    trainer: Any,
    agent: Any,
    path: Path,
    extras: dict[str, Any] | None = None,
) -> None:
    """Match ``NanoDTAgent.save`` tensor dict so ``NanoDTAgent.load`` works."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "model_state_dict": trainer.model.to("cpu").state_dict(),
        "model_config": agent.model_config,
        "trainer_config": agent.trainer_config,
        "state_mean": trainer.dataset_stats_.state_mean,
        "state_std": trainer.dataset_stats_.state_std,
        "reward_scale": trainer.config.reward_scale,
    }
    if extras:
        payload.update(extras)
    torch.save(payload, str(path))
    trainer.model.to(trainer.config.device)


def decision_transformer_train_cyclic(self: DecisionTransformerTrainer) -> None:
    global LAST_TRAIN_REPORT
    hooks = _PENDING_L6_TRAIN_HOOKS or L6TrainHooks()
    LAST_TRAIN_REPORT = {
        "finished_iter": 0,
        "stop_reason": "max_iters",
        "eval_history": [],
    }

    self.dataset_stats_ = calculate_dataset_stats(self.dataset)

    num_timesteps = sum(self.dataset_stats_.traj_lens)
    num_timesteps = max(int(self.config.pct_traj * num_timesteps), 1)
    sorted_inds = np.argsort(self.dataset_stats_.returns)
    num_trajectories = 1
    timesteps = self.dataset_stats_.traj_lens[sorted_inds[-1]]
    ind = len(self.dataset) - 2
    while (
        ind >= 0
        and timesteps + self.dataset_stats_.traj_lens[sorted_inds[ind]] <= num_timesteps
    ):
        timesteps += self.dataset_stats_.traj_lens[sorted_inds[ind]]
        num_trajectories += 1
        ind -= 1
    sorted_inds = sorted_inds[-num_trajectories:]

    p_sample = self.dataset_stats_.traj_lens[sorted_inds] / sum(
        self.dataset_stats_.traj_lens[sorted_inds]
    )

    optimizer = self.configure_optimizers(
        self.config.weight_decay,
        self.config.learning_rate,
        (self.config.beta1, self.config.beta2),
        self.config.device,
    )

    collator = DecisionTransformerDataCollator(
        state_mean=self.dataset_stats_.state_mean,
        state_std=self.dataset_stats_.state_std,
        K=self.model.config.K,
        max_ep_len=self.model.config.max_ep_len,
        reward_scale=self.config.reward_scale,
        act_discrete=self.model.config.act_discrete,
    )
    n_samples = (
        self.config.max_iters
        * self.config.batch_size
        * self.config.gradient_accumulation_steps
    )
    n_samples *= 2
    sampler = WeightedRandomSampler(
        weights=p_sample,
        num_samples=n_samples,
        replacement=True,
    )
    num_workers = 0 if sys.platform == "win32" else 2
    dataloader = DataLoader(
        self.dataset,
        batch_size=self.config.batch_size,
        collate_fn=collator,
        sampler=sampler,
        num_workers=num_workers,
    )
    dataloader_iter = itertools.cycle(iter(dataloader))

    iter_num = 0
    self.model.to(self.config.device)

    states, actions, rewards, rtgs, tsteps, mask = next(dataloader_iter)
    t0 = time.time()
    local_iter_num = 0
    running_mfu = -1.0

    bad_loss_streak = 0
    bad_success_streak = 0
    val_history: list[float] = []

    def record_eval(entry: dict[str, Any]) -> None:
        LAST_TRAIN_REPORT["eval_history"].append(entry)

    def maybe_early_stop_after_eval() -> str | None:
        nonlocal bad_loss_streak, bad_success_streak, val_history
        hist = LAST_TRAIN_REPORT["eval_history"]
        if not hist:
            return None
        last = hist[-1]
        losses = {"train": last["train_loss"], "val": last["val_loss"]}
        m = _loss_metric(losses, hooks.early_stop_on)
        val_history.append(float(losses["val"]))
        if len(val_history) > 512:
            val_history = val_history[-512:]

        if hooks.early_stop_loss_max is not None and hooks.early_stop_patience is not None:
            if m >= hooks.early_stop_loss_max:
                bad_loss_streak += 1
            else:
                bad_loss_streak = 0
            if bad_loss_streak >= hooks.early_stop_patience:
                return "consecutive_high_loss"

        w = hooks.early_stop_plateau_window
        eps = hooks.early_stop_plateau_epsilon
        if w is not None and eps is not None and w >= 2 and len(val_history) >= w:
            window = val_history[-w:]
            if max(window) - min(window) < eps:
                return "loss_plateau"

        sr = last.get("online_success_rate_final")
        if sr is not None:
            if hooks.early_stop_success_min is not None and sr >= hooks.early_stop_success_min:
                return "online_success_target"
            if (
                hooks.early_stop_success_max is not None
                and hooks.early_stop_success_patience is not None
            ):
                if sr <= hooks.early_stop_success_max:
                    bad_success_streak += 1
                else:
                    bad_success_streak = 0
                if bad_success_streak >= hooks.early_stop_success_patience:
                    return "consecutive_low_online_success"

        return None

    stop_reason: str | None = None

    while True:
        lr = (
            self.get_lr(iter_num)
            if self.config.decay_lr
            else self.config.learning_rate
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        if iter_num % self.config.eval_interval == 0:
            losses = self.estimate_loss(dataloader_iter)
            print(
                f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
            )
            entry: dict[str, Any] = {
                "iter": iter_num,
                "train_loss": float(losses["train"]),
                "val_loss": float(losses["val"]),
            }

            if (
                hooks.online_eval_fn is not None
                and hooks.online_eval_every_iters is not None
                and hooks.online_eval_every_iters > 0
                and iter_num % hooks.online_eval_every_iters == 0
            ):
                try:
                    online = hooks.online_eval_fn(iter_num)
                    entry.update(online)
                    print(
                        f"  online eval: success_final={online.get('online_success_rate_final', float('nan')):.3f}, "
                        f"mean_return={online.get('online_mean_return', float('nan')):.2f}"
                    )
                except Exception as e:
                    entry["online_eval_error"] = str(e)
                    print(f"  online eval failed: {e}")

            record_eval(entry)
            stop_reason = maybe_early_stop_after_eval()
            if stop_reason is not None:
                LAST_TRAIN_REPORT["finished_iter"] = iter_num
                LAST_TRAIN_REPORT["stop_reason"] = stop_reason
                print(f"Early stop: {stop_reason} at iter {iter_num}")
                break

        if iter_num == 0 and self.config.eval_only:
            LAST_TRAIN_REPORT["finished_iter"] = 0
            LAST_TRAIN_REPORT["stop_reason"] = "eval_only"
            break

        for micro_step in range(self.config.gradient_accumulation_steps):
            states, actions, rewards, rtgs, tsteps, mask = (
                states.to(self.config.device),
                actions.to(self.config.device),
                rewards.to(self.config.device),
                rtgs.to(self.config.device),
                tsteps.to(self.config.device),
                mask.to(self.config.device),
            )

            logits, loss = self.model(
                states, actions, rtgs, tsteps, mask, targets=actions
            )
            loss = loss / self.config.gradient_accumulation_steps
            states, actions, rewards, rtgs, tsteps, mask = next(dataloader_iter)
            loss.backward()
        if self.config.grad_clip != 0.0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.grad_clip
            )
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        if iter_num % self.config.log_interval == 0:
            lossf = loss.item() * self.config.gradient_accumulation_steps
            if local_iter_num >= 5:
                mfu = self.estimate_mfu(
                    self.config.batch_size * self.config.gradient_accumulation_steps,
                    dt,
                )
                running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
            print(
                f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%"
            )
        iter_num += 1
        local_iter_num += 1

        if (
            hooks.save_every_iters
            and hooks.save_every_iters > 0
            and hooks.checkpoint_dir is not None
            and hooks.checkpoint_agent is not None
            and iter_num > 0
            and iter_num % hooks.save_every_iters == 0
        ):
            ckpt = hooks.checkpoint_dir / f"ckpt_iter_{iter_num:08d}.pth"
            _save_mid_checkpoint(self, hooks.checkpoint_agent, ckpt)

        if iter_num > self.config.max_iters:
            LAST_TRAIN_REPORT["finished_iter"] = iter_num - 1
            LAST_TRAIN_REPORT["stop_reason"] = "max_iters"
            break

    if stop_reason is None and LAST_TRAIN_REPORT.get("stop_reason") == "max_iters":
        LAST_TRAIN_REPORT["finished_iter"] = min(iter_num - 1, self.config.max_iters)


def apply_nanodt_cyclic_train_patch() -> None:
    """Replace ``DecisionTransformerTrainer.train`` for the current process."""
    import nanodt.trainer as nt

    nt.DecisionTransformerTrainer.train = decision_transformer_train_cyclic  # type: ignore[method-assign]
