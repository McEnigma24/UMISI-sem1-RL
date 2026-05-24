# A script to train a Soft Actor-Critic (SAC) agent with Hindsight Experience Replay (HER) on a specified gym environment.
# By default it uses the PandaReach-v3 environment from the panda_gym package.
# But it can be easily modified to use any other gym environment, including those from the gymnasium_robotics package.
# Just uncomment the import statement for gymnasium_robotics and register the environments and use for example the FetchReach-v3 environment.

# Also give it a try on push and pick and place tasks: PandaPush-v3, PandaPickAndPlace-v3, FetchPush-v3, FetchPickAndPlace-v3

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Tuple

import gymnasium as gym
# Uncomment the following line to use gymnasium_robotics environments
# import gymnasium_robotics
import panda_gym
import torch

# Uncomment the following lines to register gymnasium_robotics environments
# gym.register_envs(gymnasium_robotics)

from asdf.algos import SAC
from asdf.buffers import HerReplayBuffer
from asdf.extractors import DictExtractor
from asdf.loggers import SilentLogger, TensorboardLogger
from asdf.policies import MlpPolicy


# --- Konfiguracja treningu (musi być zgodna z zapisanym ``signature`` w metadata.json) ---

DEFAULT_ENV_ID = "PandaReach-v3"
POLICY_HIDDEN = [64, 64]
POLICY_EXTRACTOR = "DictExtractor"

HER_N_SAMPLED_GOAL = 3
HER_STRATEGY = "future"
REPLAY_BUFFER_SIZE_TRAIN = 1_000_000
REPLAY_BUFFER_SIZE_LOAD = 50_000

# Argumenty przekazywane do ``SAC(...)`` (łącznie z domyślnymi z algos, które tu utrwalamy w sygnaturze)
SAC_KWARGS: dict[str, Any] = dict(
    seed=0,
    gamma=0.9,
    polyak=0.995,
    lr=1e-4,
    alpha="auto",
    target_entropy="auto",
    batch_size=64,
    start_steps=1_000,
    update_after=1_000,
    update_every=1,
    n_test_episodes=25,
    max_episode_len=100,
    n_updates=None,
)

TRAIN_N_STEPS = 100_000
TRAIN_LOG_INTERVAL = 1_000

EVAL_N_EPISODES = 50
EVAL_RENDER_SLEEP = 1.0 / 30.0

WEIGHTS_ROOT = Path(__file__).resolve().parent / "weights"
METADATA_FILENAME = "metadata.json"
WEIGHTS_FILENAME = "policy.pt"


def canonical_signature() -> dict[str, Any]:
    """
    Pełna sygnatura eksperymentu — zapisywana przy treningu i porównywana przy ``--load``.
    Zmiana któregokolwiek pola wymaga nowego treningu (stare katalogi ``weights/`` przestaną pasować).
    """
    return {
        "format_version": 1,
        "env_id": DEFAULT_ENV_ID,
        "policy": {
            "hidden_sizes": list(POLICY_HIDDEN),
            "extractor": POLICY_EXTRACTOR,
        },
        "her": {
            "n_sampled_goal": HER_N_SAMPLED_GOAL,
            "goal_selection_strategy": HER_STRATEGY,
            "replay_buffer_size_train": REPLAY_BUFFER_SIZE_TRAIN,
            "replay_buffer_size_load": REPLAY_BUFFER_SIZE_LOAD,
        },
        "sac": dict(SAC_KWARGS),
        "train_loop": {
            "n_steps": TRAIN_N_STEPS,
            "log_interval": TRAIN_LOG_INTERVAL,
        },
        "eval_render": {
            "n_episodes": EVAL_N_EPISODES,
            "sleep": EVAL_RENDER_SLEEP,
        },
    }


def _normalize_signature(obj: Any) -> Any:
    """Ujednolicenie typów po wczytaniu z JSON (np. int vs float tam gdzie trzeba)."""
    if isinstance(obj, dict):
        return {k: _normalize_signature(v) for k, v in sorted(obj.items())}
    if isinstance(obj, list):
        return [_normalize_signature(x) for x in obj]
    if isinstance(obj, float) and obj.is_integer():
        return int(obj)
    return obj


def resolve_device() -> str:
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        return "cuda"
    if torch.backends.mps.is_available():
        print("Using Apple Silicon GPU")
        return "mps"
    print("Using CPU")
    return "cpu"


def allocate_run_directory(weights_root: Path) -> Path:
    weights_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    candidate = weights_root / stamp
    if not candidate.exists():
        candidate.mkdir(parents=False)
        return candidate
    for suffix in range(1, 10_000):
        c = weights_root / f"{stamp}-{suffix}"
        if not c.exists():
            c.mkdir(parents=False)
            return c
    raise RuntimeError("Nie udało się utworzyć unikalnego katalogu pod weights/")


def save_training_run(algo: SAC, run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    sig = canonical_signature()
    doc = {
        "signature": sig,
        "weights_file": WEIGHTS_FILENAME,
        "saved_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    meta_path = run_dir / METADATA_FILENAME
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2, sort_keys=True)
        f.write("\n")
    weights_path = run_dir / WEIGHTS_FILENAME
    algo.save(str(weights_path))
    print(f"Zapisano run: {run_dir}")
    print(f"  - {meta_path.name}")
    print(f"  - {weights_path.name}")


def _assert_signature_matches(loaded: dict[str, Any]) -> None:
    expected = canonical_signature()
    if _normalize_signature(loaded) == _normalize_signature(expected):
        return
    import pprint

    raise SystemExit(
        "Metadane (signature) nie zgadzają się z aktualnym train.py — przerywam.\n"
        "--- Oczekiwane (kod) ---\n"
        f"{pprint.pformat(expected, width=100, sort_dicts=True)}\n"
        "--- W pliku ---\n"
        f"{pprint.pformat(loaded, width=100, sort_dicts=True)}"
    )


def instantiate_sac_from_signature(
    signature: dict[str, Any], *, device: str
) -> Tuple[gym.Env, SAC]:
    if signature["policy"]["extractor"] != POLICY_EXTRACTOR:
        raise ValueError(f"Nieobsługiwany extractor: {signature['policy']['extractor']}")
    env_id = signature["env_id"]
    env = gym.make(env_id)
    policy = MlpPolicy(
        env.observation_space,
        env.action_space,
        hidden_sizes=signature["policy"]["hidden_sizes"],
        extractor_type=DictExtractor,
    )
    policy.to(device)
    her = signature["her"]
    buffer = HerReplayBuffer(
        env=env,
        size=her["replay_buffer_size_load"],
        n_sampled_goal=her["n_sampled_goal"],
        goal_selection_strategy=her["goal_selection_strategy"],
        device=device,
    )
    algo = SAC(
        env,
        policy=policy,
        buffer=buffer,
        logger=SilentLogger(),
        **signature["sac"],
    )
    return env, algo


def load_sac_from_run_dir(run_dir: str | Path) -> Tuple[gym.Env, SAC, dict[str, Any]]:
    """
    Wczytuje ``metadata.json`` z katalogu runu, weryfikuje ``signature`` względem ``canonical_signature()``,
    potem ładuje wagi z pliku wskazanego w metadanych.
    Zwraca także ``doc`` metadanych (bez ponownego czytania pliku).
    """
    run_path = Path(run_dir).expanduser().resolve()
    meta_path = run_path / METADATA_FILENAME
    if not meta_path.is_file():
        raise SystemExit(f"Brak pliku metadanych: {meta_path}")
    with meta_path.open(encoding="utf-8") as f:
        doc = json.load(f)
    if "signature" not in doc:
        raise SystemExit("metadata.json nie zawiera pola 'signature'.")
    _assert_signature_matches(doc["signature"])
    weights_name = doc.get("weights_file", WEIGHTS_FILENAME)
    weights_path = run_path / weights_name
    if not weights_path.is_file():
        raise SystemExit(f"Brak pliku wag: {weights_path}")

    device = resolve_device()
    env, algo = instantiate_sac_from_signature(doc["signature"], device=device)
    algo.load(str(weights_path))
    return env, algo, doc


def run_eval_render_from_run_dir(run_dir: str | Path) -> dict[str, Any]:
    run_path = Path(run_dir).expanduser().resolve()
    env, algo, doc = load_sac_from_run_dir(run_path)
    eval_cfg = doc["signature"]["eval_render"]

    env.close()
    algo.policy.cpu()
    env_vis = gym.make(doc["signature"]["env_id"], render_mode="human")
    try:
        results = algo.test(
            env_vis,
            n_episodes=eval_cfg["n_episodes"],
            sleep=eval_cfg["sleep"],
        )
    finally:
        env_vis.close()
    print(
        f"Test reward {results['mean_ep_ret']}, "
        f"Test episode length: {results['mean_ep_len']}"
    )
    if "success_rate" in results:
        print(f"Success rate: {results['success_rate']}")
    return results


def main_train() -> None:
    device = resolve_device()
    env = gym.make(DEFAULT_ENV_ID)

    policy = MlpPolicy(
        env.observation_space,
        env.action_space,
        hidden_sizes=POLICY_HIDDEN,
        extractor_type=DictExtractor,
    )
    policy.to(device)

    buffer = HerReplayBuffer(
        env=env,
        size=REPLAY_BUFFER_SIZE_TRAIN,
        n_sampled_goal=HER_N_SAMPLED_GOAL,
        goal_selection_strategy=HER_STRATEGY,
        device=device,
    )
    logger = TensorboardLogger()
    logger.open()

    algo = SAC(
        env,
        policy=policy,
        buffer=buffer,
        logger=logger,
        **SAC_KWARGS,
    )
    algo.train(n_steps=TRAIN_N_STEPS, log_interval=TRAIN_LOG_INTERVAL)
    env.close()
    logger.close()

    run_dir = allocate_run_directory(WEIGHTS_ROOT)
    save_training_run(algo, run_dir)

    policy.cpu()
    run_eval_render_from_run_dir(run_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SAC + HER: domyślnie trening + zapis do weights/<data>_<godz-min>/; "
        "``--load RUN_DIR`` — tylko test (render), po weryfikacji metadanych."
    )
    parser.add_argument(
        "--load",
        type=Path,
        metavar="RUN_DIR",
        help="Katalog runu, np. weights/2026-05-24_14-30 (metadata.json + policy.pt)",
    )

    args = parser.parse_args()

    if args.load is None:
        main_train()
    else:
        run_dir = args.load.expanduser().resolve()
        if not run_dir.is_dir():
            raise SystemExit(f"Brak katalogu runu: {run_dir}")
        run_eval_render_from_run_dir(run_dir)
