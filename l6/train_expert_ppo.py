"""
Trening eksperta PPO na FetchPickAndPlace-v4 (Gymnasium Robotics, obserwacja Dict).

Wymaga: Python 3.12.x, zależności z ``requirements.txt`` (SB3, gymnasium-robotics, torch).

Przykład:
  python train_expert_ppo.py --check-device
  python train_expert_ppo.py --timesteps 1_000_000
  python train_expert_ppo.py --eval-only weights/2026-05-30_12-00/ppo_model.zip
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import gymnasium as gym
import gymnasium_robotics
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

gym.register_envs(gymnasium_robotics)

L6_ROOT = Path(__file__).resolve().parent
WEIGHTS_ROOT = L6_ROOT / "weights"
DEFAULT_ENV_ID = "FetchPickAndPlace-v4"


@dataclass
class TrainConfig:
    env_id: str
    seed: int
    total_timesteps: int
    n_envs: int
    learning_rate: float
    n_steps: int
    batch_size: int
    n_epochs: int
    gamma: float
    gae_lambda: float
    clip_range: float
    ent_coef: float
    max_grad_norm: float
    eval_freq: int
    n_eval_episodes: int


def default_train_config() -> TrainConfig:
    return TrainConfig(
        env_id=DEFAULT_ENV_ID,
        seed=0,
        total_timesteps=1_000_000,
        n_envs=4,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        max_grad_norm=0.5,
        eval_freq=20_000,
        n_eval_episodes=15,
    )


def resolve_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def print_device_check() -> None:
    """Szybki podgląd: czy PyTorch widzi CUDA / MPS (bez treningu)."""
    print(f"torch.__version__ = {torch.__version__}")
    print(f"torch.cuda.is_available() = {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        n = torch.cuda.device_count()
        print(f"torch.cuda.device_count() = {n}")
        for i in range(n):
            print(f"  [{i}] {torch.cuda.get_device_name(i)}")
        print(f"torch.cuda.get_device_capability(0) = {torch.cuda.get_device_capability(0)}")
        v = getattr(torch.version, "cuda", None)
        if v is not None:
            print(f"torch.version.cuda = {v}")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None:
        print(f"torch.backends.mps.is_available() = {mps.is_available()}")
    print(f"resolve_device() -> {resolve_device()}  (tak wybierze PPO)")
    if not torch.cuda.is_available() and resolve_device() == "cpu":
        print(
            "Uwaga: brak CUDA — trening na CPU będzie wolny. "
            "Sprawdź sterownik NVIDIA i instalację torch z requirements.txt (indeks cu124)."
        )


def _make_monitored_env(env_id: str):
    def _init():
        env = gym.make(env_id)
        return Monitor(env)

    return _init


def make_training_vec_env(env_id: str, n_envs: int) -> DummyVecEnv:
    """Środowisko wektorowe; Fetch ma Dict obs — PPO używa MultiInputPolicy."""
    return DummyVecEnv([_make_monitored_env(env_id) for _ in range(n_envs)])


def allocate_run_dir(root: Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M")
    run = root / stamp
    run.mkdir(parents=True, exist_ok=False)
    return run


def success_rate(
    model: PPO,
    env_id: str,
    *,
    n_episodes: int,
    seed: int,
    deterministic: bool = True,
) -> tuple[float, float, float]:
    """
    Średni zwrot, średnia długość epizodu, ułamek sukcesów (``info['is_success']``).
    """
    env = gym.make(env_id)
    successes = 0
    returns: list[float] = []
    lengths: list[float] = []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        total = 0.0
        steps = 0
        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, rew, terminated, truncated, info = env.step(action)
            total += float(rew)
            steps += 1
            done = terminated or truncated
            if done:
                s = info.get("is_success")
                if s is not None and float(np.asarray(s).reshape(-1)[0]) >= 0.5:
                    successes += 1
                returns.append(total)
                lengths.append(float(steps))

    env.close()
    mean_ret = float(np.mean(returns)) if returns else 0.0
    mean_len = float(np.mean(lengths)) if lengths else 0.0
    rate = successes / n_episodes if n_episodes else 0.0
    return mean_ret, mean_len, rate


def train(
    cfg: TrainConfig,
    *,
    tensorboard: bool,
    no_eval_callback: bool,
) -> Path:
    device = resolve_device()
    print(f"Urządzenie: {device} | env={cfg.env_id} | n_envs={cfg.n_envs}")

    WEIGHTS_ROOT.mkdir(parents=True, exist_ok=True)
    run_dir = allocate_run_dir(WEIGHTS_ROOT)
    tb_dir = run_dir / "tensorboard" if tensorboard else None
    if tb_dir is not None:
        tb_dir.mkdir(parents=True, exist_ok=True)

    train_env = make_training_vec_env(cfg.env_id, cfg.n_envs)
    eval_env = make_training_vec_env(cfg.env_id, 1)

    callbacks = []
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

    model = PPO(
        "MultiInputPolicy",
        train_env,
        learning_rate=cfg.learning_rate,
        n_steps=cfg.n_steps,
        batch_size=cfg.batch_size,
        n_epochs=cfg.n_epochs,
        gamma=cfg.gamma,
        gae_lambda=cfg.gae_lambda,
        clip_range=cfg.clip_range,
        ent_coef=cfg.ent_coef,
        max_grad_norm=cfg.max_grad_norm,
        verbose=0,
        seed=cfg.seed,
        device=device,
        tensorboard_log=str(tb_dir) if tb_dir else None,
    )

    model.learn(
        total_timesteps=cfg.total_timesteps,
        callback=callbacks if callbacks else None,
        progress_bar=True,
    )

    model_path = run_dir / "ppo_model.zip"
    model.save(str(model_path))

    mean_ret, mean_len, succ = success_rate(
        model,
        cfg.env_id,
        n_episodes=30,
        seed=cfg.seed + 1,
        deterministic=True,
    )

    manifest: dict[str, Any] = {
        "format_version": 1,
        "algo": "PPO",
        "policy_class": "MultiInputPolicy",
        "env_id": cfg.env_id,
        "device": device,
        "train_config": asdict(cfg),
        "post_train_eval": {
            "n_episodes": 30,
            "mean_episode_return": mean_ret,
            "mean_episode_length": mean_len,
            "success_rate": succ,
        },
        "artifacts": {
            "model": model_path.name,
            "best_model_dir": "best" if not no_eval_callback else None,
        },
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    train_env.close()
    eval_env.close()

    print(f"Zapisano: {model_path}")
    print(f"manifest: {run_dir / 'manifest.json'}")
    print(
        f"Krótka ewaluacja (30 ep.): return={mean_ret:.3f}, len={mean_len:.1f}, success_rate={succ:.3f}"
    )
    return run_dir


def eval_only(model_path: Path, env_id: str, n_episodes: int, seed: int) -> None:
    model = PPO.load(str(model_path), device=resolve_device())
    mean_ret, mean_len, succ = success_rate(
        model,
        env_id,
        n_episodes=n_episodes,
        seed=seed,
        deterministic=True,
    )
    print(
        f"{env_id} | {model_path}\n"
        f"  mean_return={mean_ret:.3f}, mean_len={mean_len:.1f}, success_rate={succ:.3f} ({n_episodes} ep.)"
    )


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="PPO expert on FetchPickAndPlace-v4")
    p.add_argument("--env-id", type=str, default=DEFAULT_ENV_ID)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--timesteps", type=int, default=1_000_000)
    p.add_argument("--n-envs", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--n-steps", type=int, default=2048)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--n-epochs", type=int, default=10)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--eval-freq", type=int, default=20_000)
    p.add_argument("--n-eval-episodes", type=int, default=15)
    p.add_argument("--tensorboard", action="store_true", help="Logi TB w katalogu runu")
    p.add_argument(
        "--no-eval-callback",
        action="store_true",
        help="Wyłącz EvalCallback (szybszy smoke test)",
    )
    p.add_argument(
        "--eval-only",
        type=str,
        default=None,
        metavar="PATH",
        help="Tylko ewaluacja zapisanego modelu (.zip)",
    )
    p.add_argument("--eval-episodes", type=int, default=50)
    p.add_argument(
        "--check-device",
        action="store_true",
        help="Wypisz torch / CUDA / MPS i urządzenie używane przez skrypt; bez treningu",
    )
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.check_device:
        print_device_check()
        return
    if args.eval_only:
        eval_only(Path(args.eval_only), args.env_id, args.eval_episodes, args.seed)
        return

    cfg = default_train_config()
    cfg.env_id = args.env_id
    cfg.seed = args.seed
    cfg.total_timesteps = args.timesteps
    cfg.n_envs = args.n_envs
    cfg.learning_rate = args.lr
    cfg.n_steps = args.n_steps
    cfg.batch_size = args.batch_size
    cfg.n_epochs = args.n_epochs
    cfg.ent_coef = args.ent_coef
    cfg.eval_freq = args.eval_freq
    cfg.n_eval_episodes = args.n_eval_episodes

    train(
        cfg,
        tensorboard=args.tensorboard,
        no_eval_callback=args.no_eval_callback,
    )


if __name__ == "__main__":
    main()
