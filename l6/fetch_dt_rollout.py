"""
Shared Fetch Dict-obs flattening and online rollout for Decision Transformer (nanoDT).

Used by ``train_dt_minari_fetch`` (optional online eval during training) and
``eval_dt_minari_fetch`` (evaluation vs SB3 baseline).
"""
from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np

from train_expert_ppo import RolloutEvalResult, _is_success_scalar


def flatten_fetch_obs_dict(obs: dict[str, np.ndarray], key_order: tuple[str, ...]) -> np.ndarray:
    parts = [np.asarray(obs[k], dtype=np.float32) for k in key_order]
    return np.concatenate(parts, axis=-1)


def rollout_dt_agent(
    agent: Any,
    env_id: str,
    flat_observation_keys: tuple[str, ...],
    *,
    n_episodes: int,
    seed: int,
    target_return: float,
) -> RolloutEvalResult:
    """
    Rollout ``NanoDTAgent`` in ``gym.make(env_id)`` with Dict observations flattened
    to match training. RTG: ``reset(target_return)`` then ``act(obs)`` first step,
    then ``act(obs, rew=...)``.
    """
    env = gym.make(env_id)
    success_final = 0
    success_any = 0
    returns: list[float] = []
    lengths: list[float] = []
    final_dists: list[float] = []

    for ep in range(n_episodes):
        agent.reset(target_return)
        obs, _ = env.reset(seed=seed + ep)
        done = False
        total = 0.0
        steps = 0
        any_success = False
        first = True
        prev_rew: float | None = None

        while not done:
            flat = flatten_fetch_obs_dict(obs, flat_observation_keys)
            if first:
                action = agent.act(flat, rew=None)
                first = False
            else:
                assert prev_rew is not None
                action = agent.act(flat, rew=prev_rew)

            obs, prev_rew, terminated, truncated, info = env.step(
                np.asarray(action, dtype=np.float32).reshape(-1)
            )
            total += float(prev_rew)
            steps += 1
            if _is_success_scalar(info):
                any_success = True
            done = terminated or truncated
            if done:
                if _is_success_scalar(info):
                    success_final += 1
                if any_success:
                    success_any += 1
                returns.append(total)
                lengths.append(float(steps))
                ag = np.asarray(obs["achieved_goal"], dtype=np.float64).reshape(-1)
                dg = np.asarray(obs["desired_goal"], dtype=np.float64).reshape(-1)
                final_dists.append(float(np.linalg.norm(ag - dg)))

    env.close()
    n = float(n_episodes) if n_episodes else 1.0
    return RolloutEvalResult(
        mean_return=float(np.mean(returns)) if returns else 0.0,
        mean_length=float(np.mean(lengths)) if lengths else 0.0,
        success_rate_final=success_final / n,
        success_rate_any=success_any / n,
        mean_final_goal_dist=float(np.mean(final_dists)) if final_dists else 0.0,
    )


def rollout_result_to_jsonable(r: RolloutEvalResult) -> dict[str, float]:
    return {
        "mean_return": r.mean_return,
        "mean_length": r.mean_length,
        "success_rate_final": r.success_rate_final,
        "success_rate_any": r.success_rate_any,
        "mean_final_goal_dist": r.mean_final_goal_dist,
    }
