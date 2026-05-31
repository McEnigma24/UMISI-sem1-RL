"""
Ekspert SAC + HER na środowiskach Fetch (Gymnasium Robotics v4).

SAC z ``HerReplayBuffer`` to standard z pracy Plappert et al. / benchmarków Fetch
(lepszy niż samo PPO na rzadkiej nagrodzie).

Domyślnie: **wczesne zatrzymanie**, gdy ``success_rate_final >= próg`` (własny ``rollout_eval``);
pierwsze sprawdzenie od ``min_steps_before_early_stop``, potem co ``early_stop_check_freq``
(startowe wartości w ``default_config()``; to samo ustawiają domyślne flagi CLI).
Próg jest **osobny dla każdego Fetch** (łatwiejsze zadania = wyższy próg), z fallbackiem dla
innych ``--env-id``. Opcjonalnie ``--success-threshold`` nadpisuje **wszystkie** envy
jedną wartością. Pełny budżet: ``--timesteps`` (w skrypcie: ``MY_TIMESTEPS_LIMIT``).

Domyślnie katalog akumulacji **``weights/besties``** (per env: checkpointy, ``quality_metadata.json``, skip jeśli już certyfikacja). Flagi: ``--fresh-run-dir``, ``--accumulate-dir``, ``--resume``, ``--early-stop-streak``, ``--no-checkpoints``.

Przykłady:
  python train_expert_sac_her_fetch.py
  python train_expert_sac_her_fetch.py --env-id FetchPickAndPlace-v4
  python train_expert_sac_her_fetch.py --check-device
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
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
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.her import HerReplayBuffer
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from sb3_log_progress import LogFileProgressCallback, use_sb3_interactive_progress_bar
from train_expert_ppo import (
    RolloutEvalResult,
    allocate_run_dir,
    rollout_eval,
)

gym.register_envs(gymnasium_robotics)

L6_ROOT = Path(__file__).resolve().parent
WEIGHTS_ROOT = L6_ROOT / "weights"
DEFAULT_ACCUMULATE_DIR = WEIGHTS_ROOT / "besties"
QUALITY_METADATA_NAME = "quality_metadata.json"
CRITERIA_SPEC_VERSION = 1
_CHECKPOINT_STEP_RE = re.compile(r"_(\d+)_steps\.zip$")

# Klasyczny zestaw „Fetch” z literatury (sparse, v4 — zgodny z Gymnasium).
FETCH_SPARSE_V4: tuple[str, ...] = (
    "FetchReach-v4",
    "FetchPush-v4",
    "FetchSlide-v4",
    "FetchPickAndPlace-v4",
)

# # Progi early-stop (success_rate_final z rollout_eval) — per zadanie, trochę poniżej
# # typowych mocnych runów (~0.9+), sensownie pod amatorski trening.
# DEFAULT_EARLY_STOP_SUCCESS_THRESHOLD_BY_ENV: dict[str, float] = {
#     "FetchReach-v4": 0.93,
#     "FetchPush-v4": 0.78,
#     "FetchSlide-v4": 0.66,
#     "FetchPickAndPlace-v4": 0.62,
# }

# moje #
DEFAULT_EARLY_STOP_SUCCESS_THRESHOLD_BY_ENV: dict[str, float] = {
    "FetchReach-v4": 0.95,
    "FetchPush-v4": 0.85,
    "FetchSlide-v4": 0.70,
    "FetchPickAndPlace-v4": 0.50,
}

MY_TIMESTEPS_LIMIT = 1_500_000

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
    # Wczesne zatrzymanie po progu sukcesu (rollout_eval): per-env z tabeli albo
    # ``early_stop_success_threshold_uniform`` (np. z --success-threshold).
    early_stop_enabled: bool
    early_stop_success_threshold_uniform: float | None
    early_stop_success_threshold_fallback: float
    early_stop_check_freq: int
    early_stop_eval_episodes: int
    min_steps_before_early_stop: int
    early_stop_streak_required: int
    checkpoint_save_freq: int
    checkpoint_save_replay_buffer: bool
    checkpoints_enabled: bool


def default_config() -> SacHerFetchConfig:
    return SacHerFetchConfig(
        timesteps=MY_TIMESTEPS_LIMIT,
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
        early_stop_enabled=True,
        early_stop_success_threshold_uniform=None,
        early_stop_success_threshold_fallback=0.75,
        # Częstszy rollout_eval + niski próg startu: sensowne przy łatwych env (np. Reach),
        # żeby nie mielić do pełnego budżetu; trudniejsze Fetch — ewentualnie podnieś min_steps / check_freq z CLI.
        early_stop_check_freq=5_000,
        early_stop_eval_episodes=24,
        min_steps_before_early_stop=6_000,
        early_stop_streak_required=1,
        checkpoint_save_freq=100_000,
        checkpoint_save_replay_buffer=False,
        checkpoints_enabled=True,
    )


def resolved_early_stop_success_threshold(env_id: str, cfg: SacHerFetchConfig) -> float:
    """Jednolity override z CLI albo tabela + fallback dla nieznanego env_id."""
    if cfg.early_stop_success_threshold_uniform is not None:
        return float(cfg.early_stop_success_threshold_uniform)
    return float(
        DEFAULT_EARLY_STOP_SUCCESS_THRESHOLD_BY_ENV.get(
            env_id,
            cfg.early_stop_success_threshold_fallback,
        )
    )


def criteria_dict_for_env(env_id: str, cfg: SacHerFetchConfig, success_threshold: float) -> dict[str, Any]:
    """Kryteria zapisane w quality_metadata — poza streakiem muszą się zgadzać z cfg przy skipie."""
    return {
        "spec_version": CRITERIA_SPEC_VERSION,
        "env_id": env_id,
        "early_stop_enabled": bool(cfg.early_stop_enabled),
        "success_threshold": float(success_threshold),
        "early_stop_streak_required": int(cfg.early_stop_streak_required),
        "early_stop_check_freq": int(cfg.early_stop_check_freq),
        "min_steps_before_early_stop": int(cfg.min_steps_before_early_stop),
        "early_stop_eval_episodes": int(cfg.early_stop_eval_episodes),
    }


def _criteria_equal(a: dict[str, Any], b: dict[str, Any]) -> bool:
    keys = (
        "spec_version",
        "env_id",
        "early_stop_enabled",
        "early_stop_streak_required",
        "early_stop_check_freq",
        "min_steps_before_early_stop",
        "early_stop_eval_episodes",
    )
    for k in keys:
        if a.get(k) != b.get(k):
            return False
    ta, tb = float(a.get("success_threshold", 0.0)), float(b.get("success_threshold", 0.0))
    if abs(ta - tb) > 1e-5:
        return False
    return True


def _criteria_equal_excluding_streak(a: dict[str, Any], b: dict[str, Any]) -> bool:
    """Jak ``_criteria_equal``, ale ignoruje ``early_stop_streak_required`` (streak porównujemy osobno)."""
    keys = (
        "spec_version",
        "env_id",
        "early_stop_enabled",
        "early_stop_check_freq",
        "min_steps_before_early_stop",
        "early_stop_eval_episodes",
    )
    for k in keys:
        if a.get(k) != b.get(k):
            return False
    ta, tb = float(a.get("success_threshold", 0.0)), float(b.get("success_threshold", 0.0))
    if abs(ta - tb) > 1e-5:
        return False
    return True


def _streak_certification_satisfied(
    crit_file: dict[str, Any],
    criteria_now: dict[str, Any],
    cert: dict[str, Any],
) -> bool:
    """
    Osiągnięty streak H jest OK względem wymagania zapisane w pliku (K_file) i bieżącego (K_now):
    ``H >= max(K_file, K_now)`` — certyfikat przy K_file=10 i H=10 zadowala K_now=5.
    """
    K_now = int(criteria_now["early_stop_streak_required"])
    K_file = int(crit_file.get("early_stop_streak_required") or K_now)
    H = int(cert.get("consecutive_hits_achieved", 0))
    return H >= max(K_file, K_now)


def read_quality_metadata(run_dir: Path) -> dict[str, Any] | None:
    path = run_dir / QUALITY_METADATA_NAME
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def write_quality_metadata_atomic(run_dir: Path, payload: dict[str, Any]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / QUALITY_METADATA_NAME
    tmp = run_dir / f".{QUALITY_METADATA_NAME}.tmp"
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def latest_checkpoint_path(checkpoints_dir: Path) -> Path | None:
    if not checkpoints_dir.is_dir():
        return None
    best: tuple[int, Path] | None = None
    for p in checkpoints_dir.glob("*.zip"):
        m = _CHECKPOINT_STEP_RE.search(p.name)
        if not m:
            continue
        step = int(m.group(1))
        if best is None or step > best[0]:
            best = (step, p)
    return best[1] if best else None


def default_resume_zip(run_dir: Path, resume_cli: Path | None) -> Path | None:
    if resume_cli is not None and resume_cli.is_file():
        return resume_cli
    latest = run_dir / "latest" / "sac_her_resume.zip"
    if latest.is_file():
        return latest
    ck = latest_checkpoint_path(run_dir / "checkpoints")
    if ck is not None:
        return ck
    final = run_dir / "sac_her_model.zip"
    if final.is_file():
        return final
    return None


def copy_to_latest_resume(run_dir: Path, zip_path: Path) -> None:
    dest_dir = run_dir / "latest"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / "sac_her_resume.zip"
    shutil.copy2(zip_path, dest)


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


class EarlySuccessStopCallback(BaseCallback):
    """
    Co ``check_freq`` kroków (po ``min_steps``) uruchamia ``rollout_eval``.
    Sukces = ``success_rate_final >= threshold``. Wymaga ``streak_required`` kolejnych
    udanych sprawdzeń z rzędu; jeden fail zeruje licznik. Dopiero wtedy ``learn()`` się kończy.
    """

    def __init__(
        self,
        env_id: str,
        *,
        success_threshold: float,
        check_freq: int,
        eval_episodes: int,
        min_steps: int,
        eval_seed: int,
        streak_required: int,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.env_id = env_id
        self.success_threshold = success_threshold
        self.check_freq = max(int(check_freq), 1)
        self.eval_episodes = max(int(eval_episodes), 5)
        self.min_steps = max(int(min_steps), 0)
        self.eval_seed = eval_seed
        self.streak_required = max(int(streak_required), 1)
        self._last_bucket = -1
        self.stopped_early = False
        self.stop_at_timesteps: int | None = None
        self.consecutive_hits = 0
        self.last_rollout_success_rate_final: float | None = None
        self.last_rollout_eval: RolloutEvalResult | None = None

    def _on_step(self) -> bool:
        t = int(self.num_timesteps)
        if t < self.min_steps:
            return True
        bucket = t // self.check_freq
        if bucket <= self._last_bucket:
            return True
        self._last_bucket = bucket

        m = rollout_eval(
            self.model,  # type: ignore[arg-type]
            self.env_id,
            n_episodes=self.eval_episodes,
            seed=self.eval_seed + t,
            deterministic=True,
        )
        self.last_rollout_success_rate_final = float(m.success_rate_final)
        self.last_rollout_eval = m
        self.logger.record("early_stop/success_rate_final", m.success_rate_final)
        self.logger.record("early_stop/success_rate_any", m.success_rate_any)
        self.logger.record("early_stop/mean_final_goal_dist_m", m.mean_final_goal_dist)
        self.logger.record("early_stop/success_threshold", float(self.success_threshold))
        self.logger.record("early_stop/consecutive_hits", float(self.consecutive_hits))

        if m.success_rate_final >= self.success_threshold:
            self.consecutive_hits += 1
        else:
            self.consecutive_hits = 0

        self.logger.record("early_stop/consecutive_hits_after", float(self.consecutive_hits))
        print(
            f"[early stop check] t={t} success_rate_final={m.success_rate_final:.3f} "
            f"(próg {self.success_threshold:.3f}) streak={self.consecutive_hits}/{self.streak_required}",
            flush=True,
        )

        if self.consecutive_hits >= self.streak_required:
            self.stopped_early = True
            self.stop_at_timesteps = t
            print(
                f"\n[early stop] streak {self.consecutive_hits} >= {self.streak_required}, "
                f"ostatni success_rate_final={m.success_rate_final:.3f}, num_timesteps={t}\n",
                flush=True,
            )
            return False
        return True


def persist_streak_certification(
    run_dir: Path,
    criteria_now: dict[str, Any],
    early_cb: EarlySuccessStopCallback,
    model_path: Path,
) -> None:
    """Zapis certyfikatu zaraz po zapisie wag — przed rollout_eval (uniknięcie utraty metadanych)."""
    lev = early_cb.last_rollout_eval
    if lev is not None:
        cert_body: dict[str, Any] = {
            "certified": True,
            "consecutive_hits_achieved": early_cb.consecutive_hits,
            "last_eval_success_rate_final": early_cb.last_rollout_success_rate_final,
            "last_eval_success_rate_any": lev.success_rate_any,
            "mean_return": lev.mean_return,
            "mean_length": lev.mean_length,
            "mean_final_goal_dist_m": lev.mean_final_goal_dist,
            "certified_at_num_timesteps": early_cb.stop_at_timesteps,
            "artifact_primary": model_path.name,
        }
    else:
        cert_body = {
            "certified": True,
            "consecutive_hits_achieved": early_cb.consecutive_hits,
            "last_eval_success_rate_final": early_cb.last_rollout_success_rate_final,
            "last_eval_success_rate_any": None,
            "mean_return": None,
            "mean_length": None,
            "mean_final_goal_dist_m": None,
            "certified_at_num_timesteps": early_cb.stop_at_timesteps,
            "artifact_primary": model_path.name,
        }
    write_quality_metadata_atomic(
        run_dir,
        {"criteria": criteria_now, "certification": cert_body},
    )
    print(
        f"  Zapisano {run_dir / QUALITY_METADATA_NAME} (certyfikacja streak, artefakt={model_path.name}).",
        flush=True,
    )


def env_slug(env_id: str) -> str:
    return env_id.replace("/", "_").replace(":", "_")


def train_sac_her_one_env(
    env_id: str,
    parent_dir: Path,
    cfg: SacHerFetchConfig,
    *,
    no_eval_callback: bool,
    use_subdir: bool,
    resume_path: str | None = None,
    additional_timesteps: int | None = None,
) -> dict[str, Any]:
    device = resolve_device()
    run_dir = (parent_dir / env_slug(env_id)) if use_subdir else parent_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run_dir / "latest").mkdir(parents=True, exist_ok=True)
    tb_dir = run_dir / "tensorboard"
    tb_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== SAC+HER | {env_id} | device={device} | n_envs={cfg.n_envs} ===")

    stop_thr = resolved_early_stop_success_threshold(env_id, cfg)
    thr_src = (
        "uniform_override"
        if cfg.early_stop_success_threshold_uniform is not None
        else (
            "builtin_table"
            if env_id in DEFAULT_EARLY_STOP_SUCCESS_THRESHOLD_BY_ENV
            else "fallback"
        )
    )
    criteria_now = criteria_dict_for_env(env_id, cfg, stop_thr)
    qm_path = run_dir / QUALITY_METADATA_NAME
    print(f"  {QUALITY_METADATA_NAME}: {qm_path} (istnieje: {qm_path.is_file()})", flush=True)
    meta = read_quality_metadata(run_dir)
    if cfg.early_stop_enabled and meta:
        cert = meta.get("certification") or {}
        crit_file = meta.get("criteria") or {}
        art = cert.get("artifact_primary", "sac_her_model.zip")
        art_path = Path(art) if Path(art).is_absolute() else run_dir / str(art)
        K_file = int(crit_file.get("early_stop_streak_required") or criteria_now["early_stop_streak_required"])
        K_now = int(criteria_now["early_stop_streak_required"])
        H = int(cert.get("consecutive_hits_achieved", 0))
        streak_ok = _streak_certification_satisfied(crit_file, criteria_now, cert)
        criteria_ok = _criteria_equal_excluding_streak(crit_file, criteria_now)
        cert_ok = cert.get("certified") is True
        if cert_ok and criteria_ok and streak_ok and art_path.is_file():
            need = max(K_file, K_now)
            print(
                f"[skip] {env_id}: już certyfikowane (criteria OK; streak plik K_file={K_file}, "
                f"bieżące K_now={K_now}, hits H={H} >= max(K_file,K_now)={need}). Model: {art_path}",
                flush=True,
            )
            skip_metrics = {
                "mean_return": float(cert.get("mean_return", 0.0)),
                "mean_length": float(cert.get("mean_length", 0.0)),
                "success_rate_final": float(cert.get("last_eval_success_rate_final", 0.0)),
                "success_rate_any": float(cert.get("last_eval_success_rate_any", 0.0)),
                "mean_final_goal_dist": float(cert.get("mean_final_goal_dist_m", 0.0)),
            }
            return {
                "env_id": env_id,
                "run_dir": str(run_dir),
                "model": str(art_path),
                "metrics": skip_metrics,
                "actual_timesteps": int(cert.get("certified_at_num_timesteps", 0)),
                "stopped_early": False,
                "stop_at_timesteps": None,
                "skipped": True,
                "skip_reason": "already_certified",
            }
        if cert_ok and criteria_ok and streak_ok and not art_path.is_file():
            print(
                f"[reconcile] W metadata jest certyfikat, ale brakuje pliku wag {art_path} — "
                "kontynuuję trening (resume jeśli możliwy).",
                flush=True,
            )

    if cfg.early_stop_enabled:
        print(
            f"  Early stop: success_rate_final >= {stop_thr:.3f}, "
            f"{cfg.early_stop_streak_required} kolejnych passów rollout_eval "
            f"({thr_src}; co {cfg.early_stop_check_freq} kroków od num_timesteps >= {cfg.min_steps_before_early_stop:_})."
        )
        print(
            f"  Log SB3 „Eval … Success rate” = EvalCallback co {cfg.eval_freq:_} kroków "
            "(osobna ewaluacja od early-stop).",
            flush=True,
        )
    resume_cli = Path(resume_path).resolve() if resume_path else None
    resume_zip = default_resume_zip(run_dir, resume_cli)
    if resume_zip is not None:
        print(f"  Resume z: {resume_zip}", flush=True)
    elif meta and not _criteria_equal_excluding_streak(meta.get("criteria") or {}, criteria_now):
        print("  [reconcile] Kryteria w quality_metadata różnią się od bieżących — trening od checkpointu.", flush=True)

    train_env = make_vec_env(env_id, cfg.n_envs)
    eval_env = make_vec_env(env_id, 1)

    use_pbar = use_sb3_interactive_progress_bar()
    callbacks: list[BaseCallback] = [FetchEpisodeTensorboardCallback()]
    if not use_pbar:
        callbacks.append(LogFileProgressCallback())
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
    if cfg.checkpoints_enabled:
        save_every = max(cfg.checkpoint_save_freq // cfg.n_envs, 1)
        callbacks.append(
            CheckpointCallback(
                save_freq=save_every,
                save_path=str(run_dir / "checkpoints"),
                name_prefix="sac_her",
                save_replay_buffer=cfg.checkpoint_save_replay_buffer,
            )
        )

    early_cb: EarlySuccessStopCallback | None = None
    if cfg.early_stop_enabled:
        early_cb = EarlySuccessStopCallback(
            env_id,
            success_threshold=stop_thr,
            check_freq=cfg.early_stop_check_freq,
            eval_episodes=cfg.early_stop_eval_episodes,
            min_steps=cfg.min_steps_before_early_stop,
            eval_seed=cfg.seed,
            streak_required=cfg.early_stop_streak_required,
        )
        callbacks.append(early_cb)

    if resume_zip is not None:
        model = SAC.load(
            str(resume_zip),
            env=train_env,
            device=device,
            tensorboard_log=str(tb_dir),
        )
        if additional_timesteps is not None:
            learn_budget = int(model.num_timesteps) + int(additional_timesteps)
        else:
            learn_budget = int(cfg.timesteps)
    else:
        learn_budget = int(cfg.timesteps)
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
        total_timesteps=learn_budget,
        callback=callbacks if callbacks else None,
        progress_bar=use_pbar,
        log_interval=10,
    )

    actual_ts = int(model.num_timesteps)
    stopped_early = bool(early_cb and early_cb.stopped_early)
    stop_at = early_cb.stop_at_timesteps if early_cb and early_cb.stopped_early else None

    model_path = run_dir / "sac_her_model.zip"
    try:
        model.save(str(model_path), save_replay_buffer=cfg.checkpoint_save_replay_buffer)
    except TypeError:
        model.save(str(model_path))
    copy_to_latest_resume(run_dir, model_path)

    if cfg.early_stop_enabled and early_cb and early_cb.stopped_early:
        persist_streak_certification(run_dir, criteria_now, early_cb, model_path)

    try:
        metrics = rollout_eval(
            model,
            env_id,
            n_episodes=30,
            seed=cfg.seed + 1,
            deterministic=True,
        )
    except Exception as exc:  # noqa: BLE001 — manifest i tak chcemy domknąć
        print(f"  [uwaga] rollout_eval po treningu nie powiódł się: {exc}", flush=True)
        metrics = RolloutEvalResult(
            mean_return=0.0,
            mean_length=0.0,
            success_rate_final=0.0,
            success_rate_any=0.0,
            mean_final_goal_dist=0.0,
        )

    manifest: dict[str, Any] = {
        "format_version": 1,
        "algo": "SAC+HER",
        "policy_class": "MultiInputPolicy",
        "env_id": env_id,
        "device": device,
        "train_config": asdict(cfg),
        "training": {
            "requested_timesteps": cfg.timesteps,
            "learn_budget_timesteps": learn_budget,
            "actual_timesteps": actual_ts,
            "early_stop_enabled": cfg.early_stop_enabled,
            "early_stop_streak_required": cfg.early_stop_streak_required
            if cfg.early_stop_enabled
            else None,
            "early_stop_streak_at_stop": early_cb.consecutive_hits
            if (early_cb and early_cb.stopped_early)
            else None,
            "stopped_early": stopped_early,
            "stop_at_timesteps": stop_at,
            "early_stop_success_threshold_resolved": stop_thr
            if cfg.early_stop_enabled
            else None,
            "early_stop_success_threshold_source": thr_src if cfg.early_stop_enabled else None,
            "resume_from": str(resume_zip) if resume_zip else None,
            "additional_timesteps": additional_timesteps,
        },
        "post_train_eval": asdict(metrics),
        "artifacts": {
            "model": model_path.name,
            "tensorboard": str(tb_dir.relative_to(run_dir)),
            "best_model_dir": "best" if not no_eval_callback else None,
            "checkpoints_dir": "checkpoints" if cfg.checkpoints_enabled else None,
            "latest_resume": str(Path("latest") / "sac_her_resume.zip"),
        },
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    train_env.close()
    eval_env.close()

    es_note = ""
    if stopped_early and stop_at is not None:
        es_note = (
            f"\n  Wczesny stop przy {stop_at} kroków "
            f"(streak {early_cb.consecutive_hits if early_cb else '?'}/{cfg.early_stop_streak_required})."
        )
    elif cfg.early_stop_enabled:
        es_note = f"\n  Pełny budżet learn_budget={learn_budget} — streak nie domknięty."

    print(
        f"Zapisano: {model_path}\n"
        f"  TensorBoard: tensorboard --logdir {tb_dir}\n"
        f"  Kroki: {actual_ts} / learn_budget={learn_budget}{es_note}\n"
        f"  Eval: success_final={metrics.success_rate_final:.3f}, "
        f"success_any={metrics.success_rate_any:.3f}, "
        f"mean_final_dist_m={metrics.mean_final_goal_dist:.4f}",
        flush=True,
    )

    return {
        "env_id": env_id,
        "run_dir": str(run_dir),
        "model": str(model_path),
        "metrics": asdict(metrics),
        "actual_timesteps": actual_ts,
        "stopped_early": stopped_early,
        "stop_at_timesteps": stop_at,
        "skipped": False,
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="SAC+HER expert on Fetch-v4 suite (or single env)")
    p.add_argument(
        "--env-id",
        type=str,
        default=None,
        help="Jedno środowisko (np. FetchPickAndPlace-v4). Domyślnie: cały zestaw FETCH_SPARSE_V4.",
    )
    p.add_argument(
        "--timesteps",
        type=int,
        default=MY_TIMESTEPS_LIMIT,
        help="Budżet kroków środowiska na jeden run (suite: ten sam budżet *na każde* env z FETCH_SPARSE_V4)",
    )
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
    p.add_argument(
        "--no-early-stop",
        action="store_true",
        help="Wyłącza wczesne zatrzymanie — zawsze pełny budżet --timesteps.",
    )
    p.add_argument(
        "--success-threshold",
        type=float,
        default=None,
        metavar="P",
        help=(
            "Jednolity próg success_rate_final dla **wszystkich** envów (nadpisuje tabelę per-zadanie). "
            "Bez tego argumentu: domyślna tabela progów w skrypcie + --early-stop-threshold-fallback "
            "dla nieznanego --env-id."
        ),
    )
    p.add_argument(
        "--early-stop-threshold-fallback",
        type=float,
        default=0.75,
        metavar="P",
        help=(
            "Próg early-stop, gdy --env-id nie ma wpisu w domyślnej tabeli i nie podano --success-threshold."
        ),
    )
    p.add_argument(
        "--early-stop-check-freq",
        type=int,
        default=default_config().early_stop_check_freq,
        help="Co ile kroków sprawdzać próg (po --min-steps-before-early-stop).",
    )
    p.add_argument(
        "--early-stop-eval-episodes",
        type=int,
        default=default_config().early_stop_eval_episodes,
        help="Liczba epizodów w rollout_eval przy sprawdzaniu wczesnego stopu.",
    )
    p.add_argument(
        "--min-steps-before-early-stop",
        type=int,
        default=default_config().min_steps_before_early_stop,
        help="Minimalna liczba kroków zanim zaczniemy sprawdzać wczesny stop.",
    )
    p.add_argument(
        "--early-stop-streak",
        type=int,
        default=default_config().early_stop_streak_required,
        help="Liczba kolejnych udanych rollout_eval (success_rate_final >= próg) wymagana do wczesnego stopu.",
    )
    p.add_argument(
        "--checkpoint-save-freq",
        type=int,
        default=default_config().checkpoint_save_freq,
        help="Co ile kroków środowiska zapisywać checkpoint (dzielone wewnętrznie przez n_envs dla SB3).",
    )
    p.add_argument(
        "--no-checkpoints",
        action="store_true",
        help="Wyłącza CheckpointCallback (brak okresowych .zip w checkpoints/).",
    )
    p.add_argument(
        "--checkpoint-save-replay-buffer",
        action="store_true",
        help="Zapisuj replay buffer w checkpointach i końcowym save (duże pliki).",
    )
    p.add_argument(
        "--accumulate-dir",
        type=str,
        default=str(DEFAULT_ACCUMULATE_DIR.relative_to(L6_ROOT)),
        help="Katalog root akumulacji (względem katalogu l6/), np. weights/besties — bez nowego timestampu.",
    )
    p.add_argument(
        "--fresh-run-dir",
        action="store_true",
        help="Stare zachowanie: nowy katalog z timestampem (suite) lub allocate_run_dir (pojedynczy env).",
    )
    p.add_argument(
        "--resume",
        type=str,
        default=None,
        metavar="PATH",
        help="Wymuś ładowanie wag z tego pliku .zip (inaczej: latest/ lub checkpoints/ w katalogu env).",
    )
    p.add_argument(
        "--additional-timesteps",
        type=int,
        default=None,
        metavar="N",
        help="Po wczytaniu checkpointu (--resume lub latest/checkpoints): learn do num_timesteps + N.",
    )
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
    cfg.early_stop_enabled = not args.no_early_stop
    cfg.early_stop_success_threshold_uniform = args.success_threshold
    cfg.early_stop_success_threshold_fallback = args.early_stop_threshold_fallback
    cfg.early_stop_check_freq = args.early_stop_check_freq
    cfg.early_stop_eval_episodes = args.early_stop_eval_episodes
    cfg.min_steps_before_early_stop = args.min_steps_before_early_stop
    cfg.early_stop_streak_required = args.early_stop_streak
    cfg.checkpoint_save_freq = args.checkpoint_save_freq
    cfg.checkpoint_save_replay_buffer = bool(args.checkpoint_save_replay_buffer)
    cfg.checkpoints_enabled = not args.no_checkpoints

    envs: tuple[str, ...]
    if args.env_id:
        envs = (args.env_id,)
    else:
        envs = FETCH_SPARSE_V4

    WEIGHTS_ROOT.mkdir(parents=True, exist_ok=True)

    if args.suite_dir:
        parent = Path(args.suite_dir).resolve()
        parent.mkdir(parents=True, exist_ok=True)
    elif args.fresh_run_dir:
        if len(envs) == 1:
            parent = allocate_run_dir(WEIGHTS_ROOT)
        else:
            stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M")
            parent = WEIGHTS_ROOT / f"{stamp}_sac_her_fetch_suite"
            parent.mkdir(parents=False, exist_ok=False)
    else:
        parent = (L6_ROOT / args.accumulate_dir).resolve()
        parent.mkdir(parents=True, exist_ok=True)

    resume_arg = args.resume
    add_ts = args.additional_timesteps
    if len(envs) > 1 and resume_arg:
        print(
            "[uwaga] --resume jest ignorowane przy suite (>1 env); każde env ładuje własne latest/checkpoints.",
            flush=True,
        )
        resume_arg = None
        add_ts = None

    if args.suite_dir is not None:
        use_subdir = len(envs) > 1
    elif args.fresh_run_dir:
        use_subdir = len(envs) > 1
    else:
        # accumulate-dir: zawsze podkatalog per env (także dla pojedynczego --env-id)
        use_subdir = True

    results: list[dict[str, Any]] = []
    print(
        f"\n[suite] Root: {parent}\n"
        f"[suite] Kolejka: {len(envs)} envów — dla każdego: [skip] gdy certyfikat + wagi OK; "
        "w przeciwnym razie resume (latest/checkpoints/sac_her_model.zip) albo start od zera.",
        flush=True,
    )
    for i, eid in enumerate(envs):
        print(f"[suite] --- ({i + 1}/{len(envs)}) {eid} ---", flush=True)
        cfg_run = replace(cfg, seed=cfg.seed + i * 10_000)
        results.append(
            train_sac_her_one_env(
                eid,
                parent,
                cfg_run,
                no_eval_callback=args.no_eval_callback,
                use_subdir=use_subdir,
                resume_path=resume_arg,
                additional_timesteps=add_ts,
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
