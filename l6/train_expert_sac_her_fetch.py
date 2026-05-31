"""
Ekspert SAC + HER na środowiskach Fetch (Gymnasium Robotics v4).

SAC z ``HerReplayBuffer`` to standard z pracy Plappert et al. / benchmarków Fetch
(lepszy niż samo PPO na rzadkiej nagrodzie).

Domyślnie trenuje **po kolei** na całym zestawie sparse Fetch-v4:
  FetchReach, FetchPush, FetchSlide, FetchPickAndPlace.

Przykłady:
  python train_expert_sac_her_fetch.py --timesteps 500_000
  python train_expert_sac_her_fetch.py --env-id FetchPickAndPlace-v4 --timesteps 1_000_000
  python train_expert_sac_her_fetch.py --check-device
"""
from __future__ import annotations

import argparse
import json
from collections import deque
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import gymnasium as gym
import gymnasium_robotics
import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.her import HerReplayBuffer
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from train_expert_ppo import (
    RolloutEvalResult,
    allocate_run_dir,
    rollout_eval,
)

gym.register_envs(gymnasium_robotics)

L6_ROOT = Path(__file__).resolve().parent
WEIGHTS_ROOT = L6_ROOT / "weights"

# Klasyczny zestaw „Fetch” z literatury (sparse, v4 — zgodny z Gymnasium).
FETCH_SPARSE_V4: tuple[str, ...] = (
    "FetchReach-v4",
    "FetchPush-v4",
    "FetchSlide-v4",
    "FetchPickAndPlace-v4",
)


@dataclass
class SacHerFetchConfig:
    timesteps: int
    n_envs: int
    seed: int
    learning_rate: float
    buffer_size: int
    learning_starts: int
    batch_size: int
    tau: float
    gamma: float
    train_freq: int
    gradient_steps: int
    n_sampled_goal: int
    goal_selection_strategy: str
    eval_freq: int
    n_eval_episodes: int
    policy_net_arch: tuple[int, ...]


def default_config() -> SacHerFetchConfig:
    return SacHerFetchConfig(
        timesteps=500_000,
        n_envs=4,
        seed=0,
        learning_rate=3e-4,
        buffer_size=1_000_000,
        learning_starts=10_000,
        batch_size=256,
        tau=0.005,
        gamma=0.98,
        train_freq=1,
        gradient_steps=1,
        n_sampled_goal=4,
        goal_selection_strategy="future",
        eval_freq=10_000,
        n_eval_episodes=20,
        policy_net_arch=(256, 256, 256),
    )


def resolve_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def print_device_check() -> None:
    print(f"torch.__version__ = {torch.__version__}")
    print(f"torch.cuda.is_available() = {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"  [{i}] {torch.cuda.get_device_name(i)}")
    print(f"resolve_device() -> {resolve_device()}")


def _make_monitored_env(env_id: str):
    def _init():
        return Monitor(gym.make(env_id))

    return _init


def make_vec_env(env_id: str, n_envs: int) -> DummyVecEnv:
    return DummyVecEnv([_make_monitored_env(env_id) for _ in range(n_envs)])


class FetchEpisodeTensorboardCallback(BaseCallback):
    """
    Loguje do TensorBoard metryki z końcówki epizodu (Monitor: ``episode``)
    oraz ``is_success`` z Fetch GoalEnv — uzupełnia wbudowane logi SAC.
    """

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self._rew = deque(maxlen=200)
        self._len = deque(maxlen=200)
        self._succ = deque(maxlen=200)

    def _on_step(self) -> bool:
        infos = self.locals.get("infos")
        if not infos:
            return True
        for info in infos:
            if not isinstance(info, dict) or "episode" not in info:
                continue
            ep = info["episode"]
            r = float(ep["r"])
            ln = float(ep["l"])
            self._rew.append(r)
            self._len.append(ln)
            succ = 0.0
            if "is_success" in info:
                succ = float(np.asarray(info["is_success"]).reshape(-1)[0])
            self._succ.append(succ)
            self.logger.record("fetch/episode_reward", r)
            self.logger.record("fetch/episode_length", ln)
            self.logger.record("fetch/episode_success", succ)
            self.logger.record("fetch/ep_rew_mean_200", float(np.mean(self._rew)))
            self.logger.record("fetch/ep_len_mean_200", float(np.mean(self._len)))
            self.logger.record("fetch/success_mean_200", float(np.mean(self._succ)))
        return True


def env_slug(env_id: str) -> str:
    return env_id.replace("/", "_").replace(":", "_")


def train_sac_her_one_env(
    env_id: str,
    parent_dir: Path,
    cfg: SacHerFetchConfig,
    *,
    no_eval_callback: bool,
    use_subdir: bool,
) -> dict[str, Any]:
    device = resolve_device()
    run_dir = (parent_dir / env_slug(env_id)) if use_subdir else parent_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    tb_dir = run_dir / "tensorboard"
    tb_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== SAC+HER | {env_id} | device={device} | n_envs={cfg.n_envs} ===")

    train_env = make_vec_env(env_id, cfg.n_envs)
    eval_env = make_vec_env(env_id, 1)

    callbacks: list[BaseCallback] = [FetchEpisodeTensorboardCallback()]
    if not no_eval_callback:
        callbacks.append(
            EvalCallback(
                eval_env,
                best_model_save_path=str(run_dir / "best"),
                log_path=str(run_dir / "eval_logs"),
                eval_freq=max(cfg.eval_freq // cfg.n_envs, 1),
                n_eval_episodes=cfg.n_eval_episodes,
                deterministic=True,
                render=False,
            )
        )

    model = SAC(
        "MultiInputPolicy",
        train_env,
        learning_rate=cfg.learning_rate,
        buffer_size=cfg.buffer_size,
        learning_starts=cfg.learning_starts,
        batch_size=cfg.batch_size,
        tau=cfg.tau,
        gamma=cfg.gamma,
        train_freq=cfg.train_freq,
        gradient_steps=cfg.gradient_steps,
        ent_coef="auto",
        replay_buffer_class=HerReplayBuffer,
        replay_buffer_kwargs=dict(
            n_sampled_goal=cfg.n_sampled_goal,
            goal_selection_strategy=cfg.goal_selection_strategy,
        ),
        policy_kwargs=dict(net_arch=list(cfg.policy_net_arch)),
        verbose=1,
        seed=cfg.seed,
        device=device,
        tensorboard_log=str(tb_dir),
    )

    model.learn(
        total_timesteps=cfg.timesteps,
        callback=callbacks if callbacks else None,
        progress_bar=True,
        log_interval=10,
    )

    model_path = run_dir / "sac_her_model.zip"
    model.save(str(model_path))

    metrics = rollout_eval(
        model,
        env_id,
        n_episodes=30,
        seed=cfg.seed + 1,
        deterministic=True,
    )

    manifest: dict[str, Any] = {
        "format_version": 1,
        "algo": "SAC+HER",
        "policy_class": "MultiInputPolicy",
        "env_id": env_id,
        "device": device,
        "train_config": asdict(cfg),
        "post_train_eval": asdict(metrics),
        "artifacts": {
            "model": model_path.name,
            "tensorboard": str(tb_dir.relative_to(run_dir)),
            "best_model_dir": "best" if not no_eval_callback else None,
        },
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    train_env.close()
    eval_env.close()

    print(
        f"Zapisano: {model_path}\n"
        f"  TensorBoard: tensorboard --logdir {tb_dir}\n"
        f"  Eval: success_final={metrics.success_rate_final:.3f}, "
        f"success_any={metrics.success_rate_any:.3f}, "
        f"mean_final_dist_m={metrics.mean_final_goal_dist:.4f}"
    )

    return {
        "env_id": env_id,
        "run_dir": str(run_dir),
        "model": str(model_path),
        "metrics": asdict(metrics),
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="SAC+HER expert on Fetch-v4 suite (or single env)")
    p.add_argument(
        "--env-id",
        type=str,
        default=None,
        help="Jedno środowisko (np. FetchPickAndPlace-v4). Domyślnie: cały zestaw FETCH_SPARSE_V4.",
    )
    p.add_argument("--timesteps", type=int, default=500_000, help="Budżet kroków na *każde* środowisko w trybie suite")
    p.add_argument("--n-envs", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--buffer-size", type=int, default=1_000_000)
    p.add_argument("--learning-starts", type=int, default=10_000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--gamma", type=float, default=0.98)
    p.add_argument("--tau", type=float, default=0.005)
    p.add_argument("--n-sampled-goal", type=int, default=4)
    p.add_argument("--goal-strategy", type=str, default="future", choices=["future", "episode", "final"])
    p.add_argument("--eval-freq", type=int, default=10_000)
    p.add_argument("--n-eval-episodes", type=int, default=20)
    p.add_argument("--no-eval-callback", action="store_true")
    p.add_argument("--check-device", action="store_true")
    p.add_argument(
        "--suite-dir",
        type=str,
        default=None,
        help="Opcjonalnie: istniejący katalog suite (dopisz kolejne envy ręcznie — zaawansowane)",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()
    if args.check_device:
        print_device_check()
        return

    cfg = default_config()
    cfg.timesteps = args.timesteps
    cfg.n_envs = args.n_envs
    cfg.seed = args.seed
    cfg.learning_rate = args.lr
    cfg.buffer_size = args.buffer_size
    cfg.learning_starts = args.learning_starts
    cfg.batch_size = args.batch_size
    cfg.gamma = args.gamma
    cfg.tau = args.tau
    cfg.n_sampled_goal = args.n_sampled_goal
    cfg.goal_selection_strategy = args.goal_strategy
    cfg.eval_freq = args.eval_freq
    cfg.n_eval_episodes = args.n_eval_episodes

    envs: tuple[str, ...]
    if args.env_id:
        envs = (args.env_id,)
    else:
        envs = FETCH_SPARSE_V4

    WEIGHTS_ROOT.mkdir(parents=True, exist_ok=True)

    if args.suite_dir:
        parent = Path(args.suite_dir).resolve()
        parent.mkdir(parents=True, exist_ok=True)
    elif len(envs) == 1:
        parent = allocate_run_dir(WEIGHTS_ROOT)
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M")
        parent = WEIGHTS_ROOT / f"{stamp}_sac_her_fetch_suite"
        parent.mkdir(parents=False, exist_ok=False)

    use_subdir = len(envs) > 1
    results: list[dict[str, Any]] = []
    for i, eid in enumerate(envs):
        cfg_run = replace(cfg, seed=cfg.seed + i * 10_000)
        results.append(
            train_sac_her_one_env(
                eid,
                parent,
                cfg_run,
                no_eval_callback=args.no_eval_callback,
                use_subdir=use_subdir,
            )
        )

    suite_manifest = {
        "format_version": 1,
        "algo": "SAC+HER",
        "fetch_envs_order": list(envs),
        "parent_dir": str(parent),
        "runs": results,
    }
    (parent / "manifest_suite.json").write_text(
        json.dumps(suite_manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"\nSuite manifest: {parent / 'manifest_suite.json'}")
    if len(envs) > 1:
        print(f"TensorBoard (wszystkie envy): tensorboard --logdir {parent}")


if __name__ == "__main__":
    main()
