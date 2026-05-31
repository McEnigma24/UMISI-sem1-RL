"""
Online evaluation of a trained nanoDT on Fetch (Dict obs): flatten obs like training,
correct RTG updates via ``act(obs, rew=...)``, optional SB3 baseline for the same rollouts.

Writes ``eval_metrics.json`` for ``plot_dt_vs_baseline.py``.

Example:

  python eval_dt_minari_fetch.py --model dt_weights/.../dt_model.pth \\
      --manifest dt_weights/.../manifest.json
  # baseline: auto z minari_recordings lub L6_EVAL_BASELINE_MODEL / --baseline-model
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import gymnasium as gym
import gymnasium_robotics
import numpy as np
import torch

from fetch_dt_rollout import rollout_dt_agent, rollout_result_to_jsonable
from nanodt.agent import NanoDTAgent
from record_expert_minari import load_sb3_model
from train_expert_ppo import RolloutEvalResult, rollout_eval

gym.register_envs(gymnasium_robotics)

L6_ROOT = Path(__file__).resolve().parent
EVAL_ROOT = L6_ROOT / "dt_eval_runs"


def allocate_eval_run_dir() -> Path:
    EVAL_ROOT.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M")
    run = EVAL_ROOT / stamp
    run.mkdir(parents=True, exist_ok=False)
    return run


def load_manifest(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TypeError("manifest.json must be a JSON object")
    return data


def resolve_flat_keys(
    manifest: dict[str, Any],
    model_blob: dict[str, Any],
    cli_keys: list[str] | None,
) -> tuple[str, ...]:
    if cli_keys:
        return tuple(cli_keys)
    fk = model_blob.get("flat_observation_keys") or manifest.get("flat_observation_keys")
    if fk is None:
        raise SystemExit(
            "Brak flat_observation_keys: podaj --flat-keys albo --manifest / checkpoint z tym polem."
        )
    return tuple(str(k) for k in fk)


def resolve_env_id(manifest: dict[str, Any], model_blob: dict[str, Any], cli: str | None) -> str:
    if cli:
        return cli
    e = model_blob.get("env_id") or manifest.get("env_id")
    if not e:
        raise SystemExit("Brak env_id: podaj --env-id albo manifest / .pth z env_id.")
    return str(e)


def find_expert_zip_from_minari_recordings(dataset_id: str) -> Path | None:
    """
    Szuka ``minari_recordings/<run>/manifest.json`` z tym samym ``minari_dataset_id``
    co dataset treningowy DT i zwraca ``model_path`` do zipa SB3, jeśli plik istnieje.
    Wybiera najnowszy run po nazwie katalogu (timestamp w sortowaniu).
    """
    root = L6_ROOT / "minari_recordings"
    if not root.is_dir():
        return None
    candidates: list[tuple[str, Path]] = []
    for sub in root.iterdir():
        if not sub.is_dir():
            continue
        mp = sub / "manifest.json"
        if not mp.is_file():
            continue
        try:
            rec = json.loads(mp.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if rec.get("minari_dataset_id") != dataset_id:
            continue
        zp = rec.get("model_path")
        if not zp:
            continue
        p = Path(str(zp)).expanduser().resolve()
        if p.is_file():
            candidates.append((sub.name, p))
    if not candidates:
        return None
    candidates.sort(key=lambda t: t[0], reverse=True)
    return candidates[0][1]


def resolve_baseline_model_path(
    *,
    train_manifest: dict[str, Any],
    cli_path: Path | None,
    recording_manifest: Path | None,
    no_baseline: bool,
) -> tuple[Path | None, str]:
    """Zwraca (ścieżka_zip, opis_źródła) lub (None, powód)."""
    if no_baseline:
        return None, "no_baseline_flag"
    if cli_path is not None:
        p = cli_path.expanduser().resolve()
        if not p.is_file():
            raise SystemExit(f"Brak pliku --baseline-model: {p}")
        return p, "cli"
    envp = os.environ.get("L6_EVAL_BASELINE_MODEL", "").strip()
    if envp:
        p = Path(envp).expanduser().resolve()
        if p.is_file():
            return p, "env_L6_EVAL_BASELINE_MODEL"
        return None, f"env_path_missing:{p}"
    if recording_manifest is not None:
        rm = load_manifest(recording_manifest.expanduser().resolve())
        zp = rm.get("model_path")
        if zp:
            p = Path(str(zp)).expanduser().resolve()
            if p.is_file():
                return p, "recording_manifest_cli"
        return None, "recording_manifest_no_zip"
    did = train_manifest.get("dataset_id")
    if isinstance(did, str) and did:
        p = find_expert_zip_from_minari_recordings(did)
        if p is not None:
            return p, "minari_recordings_auto"
        return None, f"minari_recordings_no_file_for_dataset:{did}"
    return None, "no_dataset_id_in_train_manifest"


def infer_target_return_from_minari_manifest(manifest: dict[str, Any]) -> float | None:
    """Percentyl 90 / max zwrotu epizodów z datasetu Minari (jak train_dt) — sensowny RTG przy sparse reward."""
    did = manifest.get("dataset_id")
    root = manifest.get("minari_datasets_root")
    if not did or not root:
        return None
    import os

    import minari

    os.environ["MINARI_DATASETS_PATH"] = str(Path(str(root)).resolve())
    ds = minari.load_dataset(str(did))
    n = min(512, len(ds))
    if n <= 0:
        return None
    rets = [float(np.sum(np.asarray(ds[i].rewards, dtype=np.float64))) for i in range(n)]
    p90 = float(np.percentile(rets, 90))
    mx = float(np.max(rets))
    if mx <= 0.0:
        return float(max(0.0, p90))
    return float(max(p90, mx))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Online eval nanoDT (Fetch Dict) + optional SB3 baseline")
    p.add_argument("--model", type=Path, required=True, help="Ścieżka do dt_model.pth")
    p.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="manifest.json z treningu (flat keys, env_id)",
    )
    p.add_argument("--device", type=str, default="cpu", help="cpu | cuda | mps")
    p.add_argument("--env-id", type=str, default=None)
    p.add_argument(
        "--flat-keys",
        type=str,
        nargs="+",
        default=None,
        help="Kolejność kluczy Dict (np. achieved_goal desired_goal observation); domyślnie z manifestu/.pth",
    )
    p.add_argument("--n-episodes", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--target-return",
        type=float,
        default=None,
        help="RTG dla agent.reset(); None = z Minari (p90/max zwrotu) jeśli manifest ma dataset_id, inaczej hyperparams / 50",
    )
    p.add_argument(
        "--baseline-model",
        type=Path,
        default=None,
        help="Zip SB3 eksperta; jeśli pominięte — szukamy zipa w minari_recordings (ten sam dataset_id) lub L6_EVAL_BASELINE_MODEL",
    )
    p.add_argument(
        "--recording-manifest",
        type=Path,
        default=None,
        help="Bezpośrednio manifest.json z nagrania Minari (model_path do zipa SB3)",
    )
    p.add_argument(
        "--no-baseline",
        action="store_true",
        help="Wyłącz automatyczne szukanie baseline (tylko metryki DT)",
    )
    p.add_argument(
        "--baseline-algo",
        type=str,
        choices=("auto", "sac", "ppo"),
        default="auto",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Katalog na eval_metrics.json (domyślnie dt_eval_runs/<timestamp>/)",
    )
    p.add_argument("--out-json", type=Path, default=None, help="Bezpośrednia ścieżka do JSON (nadpisuje out-dir)")
    return p


def main() -> None:
    args = build_parser().parse_args()
    model_path = args.model.expanduser().resolve()
    manifest = load_manifest(args.manifest.expanduser().resolve() if args.manifest else None)
    blob = torch.load(str(model_path), map_location="cpu", weights_only=False)
    if not isinstance(blob, dict):
        raise SystemExit("Checkpoint must be a dict")

    flat_keys = resolve_flat_keys(manifest, blob, args.flat_keys)
    env_id = resolve_env_id(manifest, blob, args.env_id)

    target = args.target_return
    if target is None:
        inferred = infer_target_return_from_minari_manifest(manifest) if manifest else None
        if inferred is not None:
            target = inferred
            print(f"target_return (Minari p90/max): {target:.4g}", file=sys.stderr)
        else:
            hp = manifest.get("hyperparams") or {}
            if isinstance(hp, dict) and hp.get("online_eval_target_return") is not None:
                target = float(hp["online_eval_target_return"])
            else:
                target = 50.0

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA niedostepne, uzywam CPU", file=sys.stderr)
        device = "cpu"

    agent = NanoDTAgent.load(str(model_path), device=device)

    dt_result = rollout_dt_agent(
        agent,
        env_id,
        flat_keys,
        n_episodes=args.n_episodes,
        seed=args.seed,
        target_return=float(target),
    )

    baseline_result: RolloutEvalResult | None = None
    baseline_label: str | None = None
    baseline_zip: Path | None = None
    baseline_how: str = "none"
    bpath, baseline_how = resolve_baseline_model_path(
        train_manifest=manifest,
        cli_path=args.baseline_model,
        recording_manifest=args.recording_manifest,
        no_baseline=args.no_baseline,
    )
    if bpath is not None:
        baseline_zip = bpath
        print(f"Baseline zip ({baseline_how}): {baseline_zip}", file=sys.stderr)
        load_env = gym.make(env_id)
        try:
            sb3, baseline_label = load_sb3_model(
                baseline_zip,
                args.baseline_algo,
                env=load_env,
            )
        finally:
            load_env.close()
        baseline_result = rollout_eval(
            sb3,
            env_id,
            n_episodes=args.n_episodes,
            seed=args.seed,
            deterministic=True,
        )
    else:
        print(
            f"Uwaga: brak baseline ({baseline_how}) — w wykresie tylko DT. "
            "Ustaw L6_EVAL_BASELINE_MODEL, --baseline-model lub dopasuj minari_recordings.",
            file=sys.stderr,
        )

    payload: dict[str, Any] = {
        "format_version": 1,
        "env_id": env_id,
        "flat_observation_keys": list(flat_keys),
        "n_episodes": args.n_episodes,
        "seed": args.seed,
        "target_return": float(target),
        "dt": rollout_result_to_jsonable(dt_result),
        "baseline": rollout_result_to_jsonable(baseline_result) if baseline_result else None,
        "baseline_algo": baseline_label,
        "baseline_model_path": str(baseline_zip) if baseline_zip else None,
        "baseline_resolution": baseline_how,
        "model_path": str(model_path),
    }

    if args.out_json is not None:
        out_path = args.out_json.expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        run_dir = args.out_dir.expanduser().resolve() if args.out_dir is not None else allocate_eval_run_dir()
        run_dir.mkdir(parents=True, exist_ok=True)
        out_path = run_dir / "eval_metrics.json"

    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")
    print("DT:", json.dumps(payload["dt"], indent=2))
    if payload["baseline"] is not None:
        print("Baseline:", json.dumps(payload["baseline"], indent=2))


if __name__ == "__main__":
    main()
