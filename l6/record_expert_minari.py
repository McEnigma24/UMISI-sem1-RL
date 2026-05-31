"""
Nagrywa trajektorie wytrenowanego eksperta (Stable-Baselines3: SAC lub PPO,
``MultiInputPolicy``) do lokalnego datasetu **Minari** — pod offline RL / Decision Transformera.
Minari 0.5 przy zapisie z ``DataCollector`` wymaga **JAX** (extra ``[create]`` w ``requirements.txt``).

Domyślnie HDF5 trafia do **``l6/minari_datasets/``** (``MINARI_DATASETS_PATH``); manifest: **``minari_recordings/<timestamp>/manifest.json``**.

Przykłady:

  python record_expert_minari.py --model weights/.../FetchReach-v4/sac_her_model.zip \\
      --env-id FetchReach-v4 --n-episodes 800

  python record_expert_minari.py --model weights/.../ppo_model.zip --env-id FetchPush-v4 \\
      --algo ppo --dataset-id l6/fetchpush-v4/expert-ppo-v0 --overwrite
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

import gymnasium as gym
import gymnasium_robotics
import numpy as np
import torch
from minari import DataCollector

from train_expert_ppo import RolloutEvalResult, allocate_run_dir, rollout_eval

gym.register_envs(gymnasium_robotics)

L6_ROOT = Path(__file__).resolve().parent
MANIFEST_ROOT = L6_ROOT / "minari_recordings"
# Katalog główny lokalnych datasetów Minari (``MINARI_DATASETS_PATH``) — domyślnie w repo zamiast ``~/.minari/datasets``.
DEFAULT_MINARI_DATASETS_ROOT = L6_ROOT / "minari_datasets"


def configure_minari_local_root(root: Path) -> Path:
    """Ustawia ``MINARI_DATASETS_PATH`` (API Minari: ``get_dataset_path``, ``list_local_datasets``, zapis HDF5)."""
    root = root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    os.environ["MINARI_DATASETS_PATH"] = str(root)
    return root


class _MinariGoalInfoPadWrapper(gym.Wrapper):
    """Minari ``EpisodeBuffer`` łączy ``info`` przez ``jax.tree_map`` — wszystkie kroki muszą mieć te same klucze.

    Gymnasium Robotics ``Fetch*`` zwraca przy ``reset()`` puste ``info``, a przy ``step()`` — ``is_success``,
    co psuje bufor po ``DataCollector.reset()`` (pierwszy krok nowego epizodu vs. puste ``infos`` z resetu).
    Uzupełniamy brakujące ``is_success`` tak jak w kroku środowiska (``np.float32``).
    """

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Any, dict[str, Any]]:
        obs, info = self.env.reset(seed=seed, options=options)
        return obs, self._pad_info(info)

    def step(self, action: Any) -> tuple[Any, Any, bool, bool, dict[str, Any]]:
        obs, rew, terminated, truncated, info = self.env.step(action)
        return obs, rew, terminated, truncated, self._pad_info(info)

    @staticmethod
    def _pad_info(info: Any) -> dict[str, Any]:
        d = dict(info) if isinstance(info, dict) else {}
        if "is_success" not in d:
            d["is_success"] = np.float32(0.0)
        return d


def _check_jax_for_minari() -> None:
    try:
        import jax  # noqa: F401
    except ImportError:
        print(
            "Błąd: brak pakietu JAX — Minari 0.5 wymaga go przy zapisie datasetu z DataCollector.\n"
            "Zainstaluj zależności (w tym extra [create]), np.:\n"
            '  uv pip install "minari[hdf5,create]>=0.5.3,<0.6"\n',
            file=sys.stderr,
        )
        raise SystemExit(1)


def resolve_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def slugify_dataset_token(s: str) -> str:
    s = s.strip().lower().replace("/", "-").replace(":", "-")
    s = re.sub(r"[^a-z0-9._-]+", "-", s)
    return re.sub(r"-+", "-", s).strip("-")


def default_dataset_id(env_id: str, algo_tag: str) -> str:
    """Format zgodny z konwencją Minari: ``namespace/envslug/name-v0``."""
    env_slug = slugify_dataset_token(env_id)
    algo_slug = slugify_dataset_token(algo_tag)
    return f"l6/{env_slug}/expert-{algo_slug}-v0"


def _fix_sac_policy_class_and_kwargs(
    policy_class: Any,
    policy_kwargs: Any,
    use_sde_alg: bool,
) -> tuple[Any, dict[str, Any]]:
    """
    Naprawia ``policy_class`` / ``policy_kwargs`` tak, by ``policy_class(obs, act, lr, **kwargs)``
    nie przekazywało ``use_sde`` dwukrotnie (typowe: ``functools.partial`` + ``policy_kwargs``).
    """
    import functools

    from stable_baselines3.sac.policies import MultiInputPolicy

    if policy_kwargs is None:
        pk = {}
    elif isinstance(policy_kwargs, dict):
        pk = dict(policy_kwargs)
    elif isinstance(policy_kwargs, Mapping):
        pk = dict(policy_kwargs)
    else:
        pk = {}
    pc = policy_class

    fek = pk.get("features_extractor_kwargs")
    if isinstance(fek, dict) and "use_sde" in fek:
        fek = dict(fek)
        fek.pop("use_sde", None)
        pk["features_extractor_kwargs"] = fek

    okw = pk.get("optimizer_kwargs")
    if isinstance(okw, dict) and "use_sde" in okw:
        okw = dict(okw)
        okw.pop("use_sde", None)
        pk["optimizer_kwargs"] = okw

    na = pk.get("net_arch")
    if isinstance(na, list):
        cleaned: list[Any] = []
        for item in na:
            if isinstance(item, dict) and "use_sde" in item:
                d = dict(item)
                d.pop("use_sde", None)
                cleaned.append(d)
            else:
                cleaned.append(item)
        pk["net_arch"] = cleaned
    elif isinstance(na, dict) and "use_sde" in na:
        d = dict(na)
        d.pop("use_sde", None)
        pk["net_arch"] = d

    if isinstance(pc, functools.partial):
        merged: dict[str, Any] = dict(pc.keywords or {})
        merged.update(pk)
        merged.pop("use_sde", None)
        if use_sde_alg:
            merged["use_sde"] = True
        func = pc.func
        if func is MultiInputPolicy:
            return MultiInputPolicy, merged
        if isinstance(func, type) and issubclass(func, MultiInputPolicy):
            return MultiInputPolicy, merged
        return func, merged

    merged = dict(pk)
    merged.pop("use_sde", None)
    if use_sde_alg:
        merged["use_sde"] = True
    return pc, merged


def _sac_her_requires_env(err: Exception) -> bool:
    """``HerReplayBuffer`` przy ``SAC.load`` wymaga przekazanego ``env`` (inaczej asercja SB3)."""
    err_s = str(err).lower()
    return "you must pass an environment" in err_s or (
        "herreplaybuffer" in err_s and "environment" in err_s
    )


def _sac_load_kwargs(device: str, env: gym.Env | None) -> dict[str, Any]:
    kw: dict[str, Any] = dict(device=device, print_system_info=False)
    if env is not None:
        kw["env"] = env
    return kw


def _sac_load_with_use_sde_monkeypatch(model_path: Path, device: str, env: gym.Env | None = None) -> Any:
    """
    Patch ``SAC._setup_model`` na czas ``SAC.load`` — naprawa **przed** ``super()._setup_model()``,
    które w ``OffPolicyAlgorithm`` tworzy ``self.policy`` (inaczej błąd powstaje zanim
    dotkniemy ``OffPolicyAlgorithm._setup_model`` w sensowny sposób).
    """
    import stable_baselines3.sac.sac as sac_mod
    from stable_baselines3 import SAC

    orig_sac_setup = sac_mod.SAC._setup_model

    def _wrapped_sac_setup(self: Any) -> None:
        pc, pk = _fix_sac_policy_class_and_kwargs(
            getattr(self, "policy_class", None),
            getattr(self, "policy_kwargs", None),
            bool(getattr(self, "use_sde", False)),
        )
        self.policy_class = pc
        self.policy_kwargs = pk
        return orig_sac_setup(self)

    sac_mod.SAC._setup_model = _wrapped_sac_setup  # type: ignore[method-assign]
    try:
        return SAC.load(str(model_path.resolve()), **_sac_load_kwargs(device, env))
    finally:
        sac_mod.SAC._setup_model = orig_sac_setup  # type: ignore[method-assign]


def _sac_load_with_use_sde_policy_kwargs_workaround(
    model_path: Path, device: str, env: gym.Env | None = None
) -> Any:
    """
    Zapasowa ścieżka: zapis tymczasowego zipa po nadpisaniu ``data`` (gdyby monkeypatch
    był niewystarczający w jakiejś wersji SB3).
    """
    import os
    import tempfile

    from stable_baselines3 import SAC
    from stable_baselines3.common.save_util import load_from_zip_file, save_to_zip_file

    p = str(model_path.resolve())
    data, params, pytorch_variables = load_from_zip_file(p, device=device)
    if data is None:
        return SAC.load(p, **_sac_load_kwargs(device, env))

    pc, pk = _fix_sac_policy_class_and_kwargs(
        data.get("policy_class"),
        data.get("policy_kwargs"),
        bool(data.get("use_sde")),
    )
    data["policy_class"] = pc
    data["policy_kwargs"] = pk

    fd, tmp_name = tempfile.mkstemp(suffix=".zip")
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        save_to_zip_file(tmp_path, data=data, params=params, pytorch_variables=pytorch_variables)
        return SAC.load(str(tmp_path), **_sac_load_kwargs(device, env))
    finally:
        tmp_path.unlink(missing_ok=True)


def load_sb3_model(model_path: Path, algo: str, *, env: gym.Env | None = None) -> tuple[Any, str]:
    """Zwraca (model, etykieta algorytmu do manifestu).

    Dla SAC z ``HerReplayBuffer`` **trzeba** podać ``env`` (np. ``gym.make(env_id)``) —
    inaczej ``SAC.load`` się wywali; ``record_expert_minari`` przekazuje je z ``--env-id``.
    """
    p = str(model_path.resolve())
    device = resolve_device()
    last_err: Exception | None = None

    if algo in ("auto", "sac"):
        try:
            from stable_baselines3 import SAC

            m = SAC.load(p, **_sac_load_kwargs(device, env))
            return m, "SAC"
        except Exception as e:
            last_err = e
            err_s = str(e).lower()
            if env is None and _sac_her_requires_env(e):
                raise RuntimeError(
                    "Nie wczytano SAC: checkpoint uzywa HerReplayBuffer i wymaga przekazania "
                    "`env` do `SAC.load` (np. gym.make('FetchReach-v4')). "
                    "Uruchom `record_expert_minari.py` z `--env-id` zgodnym z treningiem modelu."
                ) from e
            use_sde_dup = "use_sde" in err_s and (
                "multiple values" in err_s or "keyword argument" in err_s
            )
            if use_sde_dup:
                try:
                    m = _sac_load_with_use_sde_monkeypatch(model_path, device, env=env)
                    return m, "SAC"
                except Exception as e2:
                    last_err = e2
                try:
                    m = _sac_load_with_use_sde_policy_kwargs_workaround(model_path, device, env=env)
                    return m, "SAC"
                except Exception as e3:
                    last_err = e3
            if algo == "sac":
                raise RuntimeError(f"Nie wczytano SAC z {model_path}: {last_err}") from last_err

    if algo in ("auto", "ppo"):
        try:
            from stable_baselines3 import PPO

            m = PPO.load(p, **_sac_load_kwargs(device, env))
            return m, "PPO"
        except Exception as e:
            last_err = e
            if algo == "ppo":
                raise RuntimeError(f"Nie wczytano PPO z {model_path}: {e}") from e

    raise RuntimeError(
        f"Nie rozpoznano formatu checkpointu SB3 (algo={algo}): {model_path}. "
        f"Ostatni blad: {last_err}"
    )


def record_episodes(
    model: Any,
    env_id: str,
    *,
    n_episodes: int,
    seed: int,
    deterministic: bool,
    record_infos: bool,
) -> DataCollector:
    """Tworzy ``DataCollector`` + rollout; zwraca owinięte środowisko (przed ``create_dataset``)."""
    base = gym.make(env_id)
    if record_infos:
        base = _MinariGoalInfoPadWrapper(base)
    # Monitor nie jest wymagany do Minari; zostawiamy czyste env pod zgodność z DT.
    env = DataCollector(base, record_infos=record_infos)

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        terminated = truncated = False
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, _rew, terminated, truncated, _info = env.step(action)

    return env


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Nagrywanie trajektorii eksperta SB3 do datasetu Minari (Fetch / Dict obs)"
    )
    p.add_argument("--model", type=Path, required=True, help="Ścieżka do .zip (SAC lub PPO SB3)")
    p.add_argument("--env-id", type=str, required=True, help="np. FetchReach-v4")
    p.add_argument(
        "--algo",
        type=str,
        choices=("auto", "sac", "ppo"),
        default="auto",
        help="Który loader SB3 (domyślnie: auto — najpierw SAC, potem PPO)",
    )
    p.add_argument("--n-episodes", type=int, default=500, help="Liczba epizodów do nagrania")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--stochastic",
        action="store_true",
        help="``model.predict(..., deterministic=False)`` (domyślnie deterministycznie)",
    )
    p.add_argument(
        "--no-record-infos",
        action="store_true",
        help="Nie zapisuj pól ``info`` do HDF5 (mniejszy plik; Fetch i tak ma sukces w obs/reward)",
    )
    p.add_argument(
        "--dataset-id",
        type=str,
        default=None,
        help="ID datasetu Minari (np. l6/fetchreach-v4/expert-sac-v0). Domyślnie: z env + algorytmu.",
    )
    p.add_argument(
        "--minari-datasets-root",
        type=Path,
        default=None,
        help="Katalog główny Minari (zmienna MINARI_DATASETS_PATH). Domyślnie: <katalog l6>/minari_datasets — HDF5 w repo.",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Usuń istniejący lokalny dataset o tym samym ``dataset_id`` przed zapisem",
    )
    p.add_argument(
        "--skip-post-eval",
        action="store_true",
        help="Pomiń krótką ewaluację ``rollout_eval`` po nagraniu (szybciej)",
    )
    p.add_argument("--post-eval-episodes", type=int, default=30)
    return p


def main() -> None:
    _check_jax_for_minari()
    args = build_parser().parse_args()
    minari_datasets_root = configure_minari_local_root(
        args.minari_datasets_root
        if args.minari_datasets_root is not None
        else DEFAULT_MINARI_DATASETS_ROOT
    )
    print(f"Minari (MINARI_DATASETS_PATH): {minari_datasets_root}")
    import minari

    model_path = args.model.expanduser().resolve()
    if not model_path.is_file():
        raise SystemExit(f"Brak pliku modelu: {model_path}")

    load_env = gym.make(args.env_id)
    try:
        model, algo_name = load_sb3_model(model_path, args.algo, env=load_env)
    finally:
        load_env.close()
    dataset_id = args.dataset_id or default_dataset_id(args.env_id, algo_name)

    local = minari.list_local_datasets()
    if dataset_id in local:
        if not args.overwrite:
            raise SystemExit(
                f"Dataset '{dataset_id}' już istnieje lokalnie. Użyj --overwrite albo innego --dataset-id."
            )
        minari.delete_dataset(dataset_id)

    print(
        f"Nagrywanie: env={args.env_id} | model={model_path.name} | algo={algo_name} | "
        f"episodes={args.n_episodes} | dataset_id={dataset_id}"
    )

    collector_env = record_episodes(
        model,
        args.env_id,
        n_episodes=args.n_episodes,
        seed=args.seed,
        deterministic=not args.stochastic,
        record_infos=not args.no_record_infos,
    )

    description = (
        f"Ekspert SB3 {algo_name} z {model_path.name}; {args.n_episodes} epizodów; "
        f"env={args.env_id}; deterministic={not args.stochastic}."
    )
    dataset = collector_env.create_dataset(
        dataset_id=dataset_id,
        algorithm_name=f"SB3-{algo_name}",
        author="l6-local",
        author_email="l6-local@local",
        description=description,
    )

    collector_env.close()

    post: RolloutEvalResult | None = None
    if not args.skip_post_eval:
        post = rollout_eval(
            model,
            args.env_id,
            n_episodes=args.post_eval_episodes,
            seed=args.seed + 12345,
            deterministic=True,
        )
        print(
            "Ewaluacja po nagraniu (deterministyczna): "
            f"success_final={post.success_rate_final:.3f}, success_any={post.success_rate_any:.3f}, "
            f"mean_return={post.mean_return:.2f}"
        )

    MANIFEST_ROOT.mkdir(parents=True, exist_ok=True)
    run_dir = allocate_run_dir(MANIFEST_ROOT)
    minari_path: str | None = None
    try:
        minari_path = str(dataset.spec.data_path)
    except Exception:
        minari_path = None

    manifest = {
        "format_version": 1,
        "minari_datasets_root": str(minari_datasets_root),
        "minari_dataset_id": dataset_id,
        "minari_dataset_path": minari_path,
        "env_id": args.env_id,
        "model_path": str(model_path),
        "sb3_algo": algo_name,
        "n_episodes": args.n_episodes,
        "seed": args.seed,
        "deterministic_rollout": not args.stochastic,
        "record_infos": not args.no_record_infos,
        "post_eval": asdict(post) if post is not None else None,
    }

    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Minari: dataset_id={dataset_id}")
    print(f"Manifest run: {run_dir / 'manifest.json'}")
    print("Podgląd lokalnych datasetów:  python -m minari list local")


if __name__ == "__main__":
    main()
