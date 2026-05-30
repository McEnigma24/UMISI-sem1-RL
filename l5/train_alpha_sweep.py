#!/usr/bin/env python3
"""
Kolejne treningi SAC+HER dla wielu wartości α (stała temperatura entropii), a na końcu **zawsze**
dodatkowo jeden run z ``alpha="auto"`` (baseline, chyba że ``--skip-auto``).

Każdy run:
  - osobny podkatalog TensorBoard: ``runs/<sweep_id>/alpha_<...>/``
  - osobny katalog wag: ``weights/<sweep_id>/<data>_<czas>/`` (jak ``allocate_run_directory``)
  - ``metadata.json`` z poprawnym ``signature.sac.alpha`` i polem ``tensorboard_log_dir``

Uruchom z katalogu ``l5/`` (np. ``python train_alpha_sweep.py``).
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from train import L5_ROOT, run_training_session

DEFAULT_ALPHAS: tuple[float, ...] = (0.05, 0.5, 0.95)


def alpha_tb_slug(alpha: float | str) -> str:
    """Bezpieczna nazwa podkatalogu dla TensorBoard."""
    if isinstance(alpha, str):
        return alpha.replace(".", "p").replace(os.sep, "_")
    return f"{float(alpha):g}".replace(".", "p").replace("-", "m")


def _write_manifest(path: Path, entries: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "runs": entries,
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2, sort_keys=True)
        f.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep α: wiele treningów z osobnym TB i wagami.")
    parser.add_argument(
        "--skip-auto",
        action="store_true",
        help="Nie dodawaj na końcu runu z alpha='auto' (domyślnie auto jest zawsze treningowany).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Wypisz plan (ścieżki TB / weights root) bez treningu.",
    )
    args = parser.parse_args()

    sweep_stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    sweep_id = f"alpha_sweep_{sweep_stamp}"
    tb_root = L5_ROOT / "runs" / sweep_id
    weights_sweep_root = L5_ROOT / "weights" / sweep_id
    manifest_path = weights_sweep_root / "manifest.json"

    alphas: list[float | str] = [float(x) for x in DEFAULT_ALPHAS]
    if not args.skip_auto:
        alphas.append("auto")

    print(f"Sweep id: {sweep_id}")
    print(f"TensorBoard (wszystkie runy): {tb_root}")
    print(f"Wagi (wszystkie runy):        {weights_sweep_root}")
    print("Kolejność α:", alphas)

    planned: list[dict[str, Any]] = []
    for a in alphas:
        slug = alpha_tb_slug(a)
        tb_dir = tb_root / f"alpha_{slug}"
        planned.append({"alpha": a, "tensorboard_log_dir": str(tb_dir.relative_to(L5_ROOT))})

    if args.dry_run:
        for row in planned:
            print("[dry-run]", row)
        return

    weights_sweep_root.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, Any]] = []

    for a in alphas:
        slug = alpha_tb_slug(a)
        tb_dir = tb_root / f"alpha_{slug}"
        override: dict[str, Any] = {"alpha": a}
        print(f"\n=== Trening alpha={a!r}  TB={tb_dir.name} ===\n")
        run_dir = run_training_session(
            sac_kwargs=override,
            tensorboard_log_dir=tb_dir,
            weights_root=weights_sweep_root,
        )
        rel_weights = str(run_dir.resolve().relative_to(L5_ROOT))
        rel_tb = str(tb_dir.resolve().relative_to(L5_ROOT))
        entries.append(
            {
                "alpha": a,
                "weights_dir": rel_weights,
                "tensorboard_log_dir": rel_tb,
            }
        )
        _write_manifest(manifest_path, entries)
        print(f"Zapisano manifest: {manifest_path}")

    print("\nZakończono sweep. Otwórz TensorBoard na katalogu:")
    print(f"  tensorboard --logdir {tb_root}")


if __name__ == "__main__":
    main()
