"""
Kolejny trening DT (``train_dt_minari_fetch.py``) dla wielu datasetów Minari.

Przykład (wszystkie lokalne datasety l6/*):

  uv run python train_dt_minari_multi.py --all-local -- --max-iters 30000

Przykład (wybrane ID):

  uv run python train_dt_minari_multi.py \\
      --dataset-ids l6/fetchreach-v4/expert-sac-v0 l6/fetchpush-v4/expert-sac-v0 \\
      -- --max-iters 50000 --device cuda

Argumenty po ``--`` trafiają do każdego wywołania ``train_dt_minari_fetch.py``.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from record_expert_minari import DEFAULT_MINARI_DATASETS_ROOT, configure_minari_local_root

L6_ROOT = Path(__file__).resolve().parent
TRAIN_SCRIPT = L6_ROOT / "train_dt_minari_fetch.py"


def _local_dataset_ids(root: Path) -> list[str]:
    import minari

    os.environ["MINARI_DATASETS_PATH"] = str(root)
    data = minari.list_local_datasets()
    if isinstance(data, dict):
        return sorted(data.keys())
    return sorted(data) if data else []


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Kolejno: trening DT dla wielu datasetów Minari")
    p.add_argument(
        "--minari-datasets-root",
        type=Path,
        default=None,
        help="Root Minari (jak w train_dt_minari_fetch)",
    )
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument(
        "--dataset-ids",
        nargs="+",
        metavar="ID",
        help="Lista ``--dataset-id`` Minari (kolejność = kolejność treningów)",
    )
    g.add_argument(
        "--all-local",
        action="store_true",
        help="Wszystkie datasety zwrócone przez ``minari.list_local_datasets()``",
    )
    p.add_argument(
        "--skip-missing",
        action="store_true",
        help="Pomiń ID, których nie ma lokalnie (domyślnie: błąd jeśli brak)",
    )
    return p


def main() -> None:
    argv = sys.argv[1:]
    if "--" in argv:
        idx = argv.index("--")
        our_argv = argv[:idx]
        passthrough = argv[idx + 1 :]
    else:
        our_argv = argv
        passthrough = []

    args = build_parser().parse_args(our_argv)

    root = configure_minari_local_root(
        args.minari_datasets_root.expanduser().resolve()
        if args.minari_datasets_root is not None
        else DEFAULT_MINARI_DATASETS_ROOT
    )
    print(f"MINARI_DATASETS_PATH={root}")

    if args.all_local:
        ids = _local_dataset_ids(root)
        if not ids:
            raise SystemExit("Brak lokalnych datasetów Minari — najpierw record_expert_minari.py.")
    else:
        ids = list(args.dataset_ids)

    available = set(_local_dataset_ids(root))
    to_run: list[str] = []
    for did in ids:
        if did not in available:
            msg = f"Brak datasetu lokalnie: {did}"
            if args.skip_missing:
                print(f"Pomijam: {msg}", file=sys.stderr)
                continue
            raise SystemExit(msg + "\n  (nagraj: record_expert_minari.py --dataset-id ... / --list-local w train_dt)")
        to_run.append(did)

    if not to_run:
        raise SystemExit("Nic do treningu (pusta lista po --skip-missing?).")

    for i, did in enumerate(to_run):
        print(f"\n=== [{i + 1}/{len(to_run)}] DT train: {did} ===\n", flush=True)
        cmd = [
            sys.executable,
            str(TRAIN_SCRIPT),
            "--dataset-id",
            did,
            *passthrough,
        ]
        if args.minari_datasets_root is not None:
            cmd.extend(["--minari-datasets-root", str(args.minari_datasets_root)])
        r = subprocess.run(cmd, cwd=str(L6_ROOT))
        if r.returncode != 0:
            raise SystemExit(f"Trening zakończony kodem {r.returncode} dla {did}")


if __name__ == "__main__":
    main()
