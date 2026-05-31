"""
Trening Decision Transformera (nanoDT) na lokalnym datassecie **Minari** z Fetch (Dict obs).

nanoDT zakłada wektor stanu (``Box``); Minari zapisuje ``Dict`` — skrypt spłaszcza klucze
w stałej kolejności i zapisuje ją w manifeście (spójnie z późniejszą ewaluacją).

Wymaga: ``nanodt``, ``minari``, dataset nagrany np. ``record_expert_minari.py``.

Przykład:

  python train_dt_minari_fetch.py --dataset-id l6/fetchreach-v4/expert-sac-v0 \\
      --max-iters 50000
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import multiprocessing
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import gymnasium as gym
import gymnasium_robotics
import numpy as np
import torch
from minari import MinariDataset

from nanodt.agent import NanoDTAgent
from nanodt.utils import seed_libraries

from fetch_dt_rollout import rollout_dt_agent, rollout_result_to_jsonable
from nanodt_train_loop import (
    L6TrainHooks,
    LAST_TRAIN_REPORT,
    apply_nanodt_cyclic_train_patch,
    clear_pending_l6_train_hooks,
    set_pending_l6_train_hooks,
)
from record_expert_minari import DEFAULT_MINARI_DATASETS_ROOT, configure_minari_local_root

gym.register_envs(gymnasium_robotics)

L6_ROOT = Path(__file__).resolve().parent
DT_WEIGHTS_ROOT = L6_ROOT / "dt_weights"


@dataclass
class _FlatTraj:
    """Jedna trajektoria z wektorowym ``observations`` (T, state_dim) — format oczekiwany przez nanoDT."""

    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray


def _flatten_fetch_obs_dict(
    obs: dict[str, np.ndarray], key_order: tuple[str, ...]
) -> np.ndarray:
    parts = [np.asarray(obs[k], dtype=np.float32) for k in key_order]
    return np.concatenate(parts, axis=-1)


class FlatDictMinariView:
    """
    Widok na ``MinariDataset`` ze ``spaces.Dict`` obserwacjami: każdy epizod ma ``observations`` (T, D).
    ``observation_space`` nadpisane na ``Box(D,)`` — wymagane przez ``NanoDTAgent.learn``.
    """

    def __init__(self, dataset: MinariDataset, key_order: tuple[str, ...]):
        self._ds = dataset
        self._keys = key_order
        ep0 = dataset[0]
        if not isinstance(ep0.observations, dict):
            raise TypeError(
                "Ten widok jest tylko dla Dict obs. Dla Box użyj MinariDataset bezpośrednio z nanoDT."
            )
        missing = [k for k in key_order if k not in ep0.observations]
        if missing:
            raise KeyError(f"Brak kluczy obserwacji w epizodzie: {missing}")
        d = int(sum(np.asarray(ep0.observations[k]).shape[-1] for k in key_order))
        low = np.full(d, -np.inf, dtype=np.float32)
        high = np.full(d, np.inf, dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)
        self.action_space = dataset.action_space

    def __len__(self) -> int:
        return len(self._ds)

    def __getitem__(self, idx: int) -> _FlatTraj:
        ep = self._ds[idx]
        assert isinstance(ep.observations, dict)
        flat = _flatten_fetch_obs_dict(ep.observations, self._keys)
        actions = np.asarray(ep.actions, dtype=np.float32)
        rewards = np.asarray(ep.rewards, dtype=np.float32)
        return _FlatTraj(observations=flat, actions=actions, rewards=rewards)

    def __iter__(self) -> Iterator[_FlatTraj]:
        for i in range(len(self)):
            yield self[i]


def default_flat_key_order(obs_space: gym.Space) -> tuple[str, ...]:
    """Kolejność konkatenacji dla ``Dict`` — domyślnie stabilna (alfabetyczna)."""
    if isinstance(obs_space, gym.spaces.Dict):
        return tuple(sorted(obs_space.spaces.keys()))
    raise TypeError(f"Oczekiwano Dict obs, jest {type(obs_space)}")


def allocate_dt_run_dir() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M")
    run = DT_WEIGHTS_ROOT / stamp
    run.mkdir(parents=True, exist_ok=False)
    return run


def default_rtg_from_minari_episode_returns(raw_ds: MinariDataset, *, max_episodes: int = 512) -> float:
    """
    Heurystyka ``target_return`` / RTG dla online eval i DT rollouts.

    Przy nagrodzie sparse średnia bywa silnie ujemna — ``max(średnia, 1)`` dawałoby absurdalne ``1.0``.
    Używamy percentyla 90. (górny rejestr demonstracji), co lepiej odpowiada „optymistycznemu” RTG.
    """
    n = min(max_episodes, len(raw_ds))
    if n <= 0:
        return 50.0
    rets = [float(np.sum(np.asarray(raw_ds[i].rewards, dtype=np.float64))) for i in range(n)]
    p90 = float(np.percentile(rets, 90))
    mx = float(np.max(rets))
    # Sukces na Fetch sparse często daje zwrot 0 — „optymistyczny” RTG nie powinien być ujemny.
    if mx <= 0.0:
        return float(max(0.0, p90))
    return float(max(p90, mx))


def merge_extra_into_torch_checkpoint(path: Path, extras: dict[str, Any]) -> None:
    """Dopisuje klucze do ``.pth`` (np. proweniencja Minari) bez zmiany formatu nanoDT."""
    blob = torch.load(str(path), map_location="cpu", weights_only=False)
    if not isinstance(blob, dict):
        raise TypeError(f"Oczekiwano dict w checkpointcie, jest {type(blob)}")
    blob.update(extras)
    torch.save(blob, str(path))


def resolve_device(explicit: str | None) -> str:
    if explicit and explicit != "auto":
        return explicit
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def maybe_wrap_dataset_for_dt(dataset: MinariDataset) -> Any:
    obs_sp = dataset.observation_space
    if isinstance(obs_sp, gym.spaces.Dict):
        keys = default_flat_key_order(obs_sp)
        return FlatDictMinariView(dataset, keys)
    return dataset


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Trening Decision Transformera (nanoDT) na Minari Fetch")
    p.add_argument(
        "--dataset-id",
        type=str,
        default="l6/fetchreach-v4/expert-sac-v0",
        help="ID datasetu Minari (np. z record_expert_minari)",
    )
    p.add_argument(
        "--minari-datasets-root",
        type=Path,
        default=None,
        help="Root Minari (MINARI_DATASETS_PATH). Domyślnie jak record_expert_minari: l6/minari_datasets",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="auto", help="auto | cpu | cuda | mps")
    p.add_argument("--max-iters", type=int, default=50_000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--eval-interval", type=int, default=2_000)
    p.add_argument("--eval-iters", type=int, default=50)
    p.add_argument("--log-interval", type=int, default=200)
    p.add_argument("--reward-scale", type=float, default=1_000.0)
    p.add_argument("--K", type=int, default=20, help="Długość kontekstu DT (seq. w collatorze)")
    p.add_argument(
        "--max-ep-len",
        type=int,
        default=128,
        help="Górne ograniczenie timestepów w modelu (>= długość epizodu w danych, Fetch ~50)",
    )
    p.add_argument("--n-layer", type=int, default=3)
    p.add_argument("--n-head", type=int, default=1)
    p.add_argument("--n-embd", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--out-dir", type=Path, default=None, help="Domyślnie: dt_weights/<timestamp UTC>/")
    p.add_argument(
        "--env-id",
        type=str,
        default=None,
        help="ID środowiska Gym (np. FetchReach-v4). Domyślnie z Minari env_spec.",
    )
    # Early stopping (loss on offline batches)
    p.add_argument(
        "--early-stop-patience",
        type=int,
        default=None,
        help="Liczba kolejnych evalów z loss >= --early-stop-loss-max (None = wyłączone)",
    )
    p.add_argument(
        "--early-stop-loss-max",
        type=float,
        default=None,
        help="Próg lossu (train/val/both wg --early-stop-on); patience kolejnych evalów",
    )
    p.add_argument(
        "--early-stop-on",
        type=str,
        choices=("val", "train", "both"),
        default="val",
        help="Który loss liczyć do progu consecutive",
    )
    p.add_argument(
        "--early-stop-plateau-epsilon",
        type=float,
        default=None,
        help="Plateau val loss: max-min w oknie < epsilon => stop (None = wyłączone)",
    )
    p.add_argument(
        "--early-stop-plateau-window",
        type=int,
        default=None,
        help="Liczba ostatnich punktów eval (>=2) do detekcji plateau",
    )
    # Optional online eval during training
    p.add_argument(
        "--online-eval-every-iters",
        type=int,
        default=None,
        help="Co ile iteracji rollout w env (wielokrotność --eval-interval zalecana)",
    )
    p.add_argument("--online-eval-episodes", type=int, default=10)
    p.add_argument(
        "--online-eval-target-return",
        type=float,
        default=None,
        help="RTG dla agent.reset() w rolloutach online (None = heurystyka ze średniego zwrotu w datassecie)",
    )
    p.add_argument(
        "--early-stop-success-min",
        type=float,
        default=None,
        help="Zatrzymaj gdy online success_rate_final >= ta wartość (None = wyłączone)",
    )
    p.add_argument(
        "--early-stop-success-max",
        type=float,
        default=None,
        help="Próg dolny success (np. 0.0); z --early-stop-success-patience",
    )
    p.add_argument(
        "--early-stop-success-patience",
        type=int,
        default=None,
        help="Liczba kolejnych online evalów z success <= --early-stop-success-max",
    )
    p.add_argument(
        "--save-every-iters",
        type=int,
        default=None,
        help="Zapis pośredni co N iteracji do out-dir (None = tylko końcowy zapis)",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()
    if sys.platform == "win32":
        multiprocessing.freeze_support()

    if (args.early_stop_patience is not None) ^ (args.early_stop_loss_max is not None):
        raise SystemExit("Uzyj obu: --early-stop-patience i --early-stop-loss-max albo zadnego.")
    if args.early_stop_plateau_epsilon is not None or args.early_stop_plateau_window is not None:
        if (
            args.early_stop_plateau_epsilon is None
            or args.early_stop_plateau_window is None
            or args.early_stop_plateau_window < 2
        ):
            raise SystemExit(
                "Plateau: podaj --early-stop-plateau-epsilon oraz --early-stop-plateau-window (>=2) oba."
            )
    if (args.early_stop_success_max is not None) ^ (args.early_stop_success_patience is not None):
        raise SystemExit(
            "Online success stop: uzyj obu --early-stop-success-max i --early-stop-success-patience albo zadnego."
        )
    if args.online_eval_every_iters is not None and args.online_eval_every_iters > 0:
        if args.eval_interval % args.online_eval_every_iters != 0 and args.online_eval_every_iters % args.eval_interval != 0:
            print(
                "Ostrzezenie: --online-eval-every-iters nie jest wielokrotnoscia "
                "ani dzielnikiem --eval-interval; online eval odpali tylko gdy oba warunki modulo spelnione."
            )

    root = configure_minari_local_root(
        args.minari_datasets_root.expanduser().resolve()
        if args.minari_datasets_root is not None
        else DEFAULT_MINARI_DATASETS_ROOT
    )
    print(f"MINARI_DATASETS_PATH={root}")

    import minari

    seed_libraries(args.seed)
    device = resolve_device(args.device)
    print(f"Device: {device}")
    apply_nanodt_cyclic_train_patch()

    raw_ds = minari.load_dataset(args.dataset_id)
    train_ds = maybe_wrap_dataset_for_dt(raw_ds)
    flat_keys: tuple[str, ...] | None = None
    if isinstance(train_ds, FlatDictMinariView):
        flat_keys = train_ds._keys

    env_id = args.env_id or raw_ds.env_spec.id
    mean_ret = default_rtg_from_minari_episode_returns(raw_ds)
    online_target = (
        float(args.online_eval_target_return)
        if args.online_eval_target_return is not None
        else float(mean_ret)
    )

    run_dir = args.out_dir.expanduser().resolve() if args.out_dir is not None else allocate_dt_run_dir()
    run_dir.mkdir(parents=True, exist_ok=True)
    model_path = run_dir / "dt_model.pth"

    agent = NanoDTAgent(
        device=device,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
        K=args.K,
        max_ep_len=args.max_ep_len,
    )

    online_fn = None
    if args.online_eval_every_iters and flat_keys:
        def online_fn(_iter: int) -> dict[str, float]:
            r = rollout_dt_agent(
                agent,
                env_id,
                flat_keys,
                n_episodes=int(args.online_eval_episodes),
                seed=int(args.seed) + int(_iter),
                target_return=online_target,
            )
            d = rollout_result_to_jsonable(r)
            return {
                "online_success_rate_final": d["success_rate_final"],
                "online_success_rate_any": d["success_rate_any"],
                "online_mean_return": d["mean_return"],
                "online_mean_length": d["mean_length"],
                "online_mean_final_goal_dist": d["mean_final_goal_dist"],
            }

    ckpt_extras_base = None
    if args.save_every_iters:
        ckpt_extras_base = {
            "minari_dataset_id": args.dataset_id,
            "flat_observation_keys": list(flat_keys) if flat_keys else None,
            "env_id": env_id,
        }

    hooks = L6TrainHooks(
        early_stop_patience=args.early_stop_patience,
        early_stop_loss_max=args.early_stop_loss_max,
        early_stop_on=args.early_stop_on,
        early_stop_plateau_epsilon=args.early_stop_plateau_epsilon,
        early_stop_plateau_window=args.early_stop_plateau_window,
        online_eval_every_iters=args.online_eval_every_iters,
        online_eval_episodes=args.online_eval_episodes,
        online_eval_fn=online_fn,
        early_stop_success_min=args.early_stop_success_min,
        early_stop_success_max=args.early_stop_success_max,
        early_stop_success_patience=args.early_stop_success_patience,
        save_every_iters=args.save_every_iters,
        checkpoint_dir=run_dir if args.save_every_iters else None,
        checkpoint_agent=agent if args.save_every_iters else None,
        checkpoint_extras_base=ckpt_extras_base,
    )

    set_pending_l6_train_hooks(hooks)
    try:
        agent.learn(
            train_ds,
            reward_scale=args.reward_scale,
            max_iters=args.max_iters,
            batch_size=args.batch_size,
            eval_interval=args.eval_interval,
            eval_iters=args.eval_iters,
            log_interval=args.log_interval,
        )
    finally:
        clear_pending_l6_train_hooks()

    agent.save(str(model_path))
    print(f"Saved model: {model_path}")

    early_info = {
        "stop_reason": LAST_TRAIN_REPORT.get("stop_reason"),
        "finished_iter": LAST_TRAIN_REPORT.get("finished_iter"),
        "eval_history_tail": LAST_TRAIN_REPORT.get("eval_history", [])[-16:],
    }
    merge_extra_into_torch_checkpoint(
        model_path,
        {
            "minari_dataset_id": args.dataset_id,
            "flat_observation_keys": list(flat_keys) if flat_keys else None,
            "env_id": env_id,
            "early_stop_reason": early_info.get("stop_reason"),
            "early_stop": early_info,
        },
    )

    nanodt_ver = importlib.metadata.version("nanodt")
    manifest = {
        "format_version": 1,
        "nanodt_version": nanodt_ver,
        "dataset_id": args.dataset_id,
        "minari_datasets_root": str(root),
        "flat_observation_keys": list(flat_keys) if flat_keys else None,
        "env_id": env_id,
        "device": device,
        "seed": args.seed,
        "early_stop": early_info,
        "hyperparams": {
            "max_iters": args.max_iters,
            "batch_size": args.batch_size,
            "eval_interval": args.eval_interval,
            "eval_iters": args.eval_iters,
            "log_interval": args.log_interval,
            "reward_scale": args.reward_scale,
            "K": args.K,
            "max_ep_len": args.max_ep_len,
            "n_layer": args.n_layer,
            "n_head": args.n_head,
            "n_embd": args.n_embd,
            "dropout": args.dropout,
            "early_stop_patience": args.early_stop_patience,
            "early_stop_loss_max": args.early_stop_loss_max,
            "early_stop_on": args.early_stop_on,
            "early_stop_plateau_epsilon": args.early_stop_plateau_epsilon,
            "early_stop_plateau_window": args.early_stop_plateau_window,
            "online_eval_every_iters": args.online_eval_every_iters,
            "online_eval_episodes": args.online_eval_episodes,
            "online_eval_target_return": online_target,
            "early_stop_success_min": args.early_stop_success_min,
            "early_stop_success_max": args.early_stop_success_max,
            "early_stop_success_patience": args.early_stop_success_patience,
            "save_every_iters": args.save_every_iters,
        },
        "model_path": str(model_path),
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Manifest: {run_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
