#!/usr/bin/env python3
"""
Pipeline „checkpoint -> DT -> porównanie” DLA JEDNEJ aktywności Fetch.

Cel eksperymentu: wziąć **niedorobionego** eksperta (checkpoint ~3/4 drogi, ~75%
maks. performansu), nagrać jego ślady do Minari, wytrenować na nich DT i sprawdzić,
czy DT (warunkowany na wysoki RTG) potrafi **przebić politykę, która zebrała dane**
— czyli klasyczny test „stitching / improvement over behavior policy”.

Jedna aktywność = jeden niezależny proces (idealne pod osobne joby SLURM na Cyfronecie):

  python run_ckpt_pipeline.py --env FetchSlide-v4        --checkpoint-steps 800000 --device cuda
  python run_ckpt_pipeline.py --env FetchPush-v4         --checkpoint-steps 300000 --device cuda
  python run_ckpt_pipeline.py --env FetchPickAndPlace-v4 --checkpoint-steps 400000 --device cuda

Etapy (kolejno): record -> train -> eval -> plot. Wybór: --stages.

NIE nadpisujemy danych eksperckich: dataset-id oraz katalogi DT/eval mają sufiks z
liczbą kroków checkpointu. Etap ``record`` jest POMIJANY, jeśli dataset już istnieje
(``--force-record`` wymusza ponowne nagranie; wtedy dokładamy ``--overwrite`` do
record_expert_minari, ale TYLKO dla tego ckpt-datasetu, nie dla expert-v0).
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

L6_ROOT = Path(__file__).resolve().parent
PY = sys.executable

ALL_STAGES = ("record", "train", "eval", "plot")
_STEPS_RE = re.compile(r"_(\d+)_steps\.zip$")


def env_slug(env_id: str) -> str:
    """FetchPickAndPlace-v4 -> fetchpickandplace-v4 (jak w dotychczasowych dataset-id)."""
    return env_id.lower()


def resolve_checkpoint(args: argparse.Namespace) -> tuple[Path, str]:
    """Zwraca (ścieżka_do_zip, tag_kroków) dla checkpointu."""
    if args.checkpoint is not None:
        ckpt = args.checkpoint.expanduser().resolve()
        m = _STEPS_RE.search(ckpt.name)
        steps = m.group(1) if m else (str(args.checkpoint_steps) if args.checkpoint_steps else "ckpt")
        return ckpt, steps
    if args.checkpoint_steps is None:
        raise SystemExit("Podaj --checkpoint <ścieżka.zip> albo --checkpoint-steps <int>.")
    steps = str(args.checkpoint_steps)
    ckpt = (
        args.besties_root.expanduser().resolve()
        / args.env
        / "checkpoints"
        / f"sac_her_{steps}_steps.zip"
    )
    return ckpt, steps


def minari_dataset_dir(minari_root: Path, dataset_id: str) -> Path:
    """Katalog datasetu Minari dla danego ID (root/<namespace.../name>)."""
    return minari_root.joinpath(*dataset_id.split("/"))


def run(cmd: list[str], *, title: str) -> None:
    print(f"\n========== [{title}] ==========", flush=True)
    print(" ".join(str(c) for c in cmd), flush=True)
    # UTF-8 w podprocesach: child-scripty drukują polskie znaki, a Windows cp1252
    # rzuca UnicodeEncodeError nawet po udanej pracy. PYTHONUTF8=1 to naprawia
    # (na Linux/Cyfronecie i tak UTF-8 — bez efektów ubocznych).
    env = dict(os.environ)
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    res = subprocess.run(cmd, cwd=str(L6_ROOT), env=env)
    if res.returncode != 0:
        raise SystemExit(f"[{title}] zakończone błędem (exit {res.returncode}).")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Pipeline checkpoint->DT->porównanie dla jednej aktywności Fetch",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--env", required=True, help="np. FetchSlide-v4 / FetchPush-v4 / FetchPickAndPlace-v4")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--checkpoint", type=Path, default=None, help="Pełna ścieżka do .zip checkpointu SB3")
    g.add_argument("--checkpoint-steps", type=int, default=None, help="Kroki checkpointu w weights/besties/<env>/checkpoints/")
    p.add_argument("--besties-root", type=Path, default=L6_ROOT / "weights" / "besties")
    p.add_argument(
        "--minari-datasets-root",
        type=Path,
        default=L6_ROOT / "minari_datasets",
        help="Root Minari (MINARI_DATASETS_PATH) dla record i train",
    )
    p.add_argument("--tag", default="ckpt", help="Prefiks ID datasetu/katalogów (np. ckpt -> ckpt800000)")
    # rozmiary etapów
    p.add_argument("--n-episodes", type=int, default=1800, help="Epizody do nagrania (ostatnio: 1800)")
    p.add_argument("--max-iters", type=int, default=40000, help="Iteracje treningu DT (jak modele eksperckie)")
    p.add_argument("--eval-episodes", type=int, default=50, help="Epizody ewaluacji DT vs checkpoint")
    p.add_argument("--target-return", type=float, default=0.0, help="RTG dla DT (Fetch sparse: 0.0 = maks.)")
    # seedy (spójnie z dotychczasowym pipeline)
    p.add_argument("--record-seed", type=int, default=0)
    p.add_argument("--train-seed", type=int, default=42)
    p.add_argument("--eval-seed", type=int, default=0)
    # urządzenie
    p.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda", "mps"))
    # sterowanie etapami (domyślnie: WZNAWIALNE — każdy etap pomijany, gdy jego wynik już istnieje)
    p.add_argument("--stages", default="record,train,eval,plot", help="Lista etapów po przecinku")
    p.add_argument("--force", action="store_true", help="Policz wszystkie etapy od nowa (ignoruj istniejące wyniki)")
    p.add_argument("--force-record", action="store_true", help="Nagraj ponownie nawet gdy dataset istnieje (--overwrite)")
    p.add_argument("--force-train", action="store_true", help="Trenuj DT od nowa nawet gdy dt_model.pth istnieje")
    p.add_argument("--force-eval", action="store_true", help="Ewaluuj od nowa nawet gdy eval_metrics.json istnieje")
    p.add_argument("--force-plot", action="store_true", help="Rysuj od nowa nawet gdy wykres istnieje")
    # przekazanie surowych flag do treningu DT (np. early stopping)
    p.add_argument("--train-extra", default="", help="Dodatkowe flagi do train_dt_minari_fetch.py (w cudzysłowie)")
    return p


def resolve_eval_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def main() -> None:
    args = build_parser().parse_args()
    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    for s in stages:
        if s not in ALL_STAGES:
            raise SystemExit(f"Nieznany etap: {s}. Dozwolone: {', '.join(ALL_STAGES)}")

    slug = env_slug(args.env)
    ckpt, steps = resolve_checkpoint(args)
    if "record" in stages or "eval" in stages:
        if not ckpt.is_file():
            raise SystemExit(f"Brak pliku checkpointu: {ckpt}")

    dataset_id = f"l6/{slug}/{args.tag}{steps}-sac-v0"
    minari_root = args.minari_datasets_root.expanduser().resolve()
    dt_out = L6_ROOT / "dt_weights_ckpt" / f"{args.env}-{args.tag}{steps}"
    eval_out = L6_ROOT / "dt_eval_runs" / f"{args.tag}_{slug}_{steps}"

    print("=" * 64, flush=True)
    print(f"AKTYWNOŚĆ        : {args.env}", flush=True)
    print(f"CHECKPOINT       : {ckpt}", flush=True)
    print(f"DATASET-ID       : {dataset_id}", flush=True)
    print(f"MINARI ROOT      : {minari_root}", flush=True)
    print(f"DT OUT           : {dt_out}", flush=True)
    print(f"EVAL OUT         : {eval_out}", flush=True)
    print(f"ETAPY            : {', '.join(stages)}", flush=True)
    print("=" * 64, flush=True)

    force_record = args.force or args.force_record
    force_train = args.force or args.force_train
    force_eval = args.force or args.force_eval
    force_plot = args.force or args.force_plot

    # 1) RECORD
    if "record" in stages:
        ds_dir = minari_dataset_dir(minari_root, dataset_id)
        if ds_dir.is_dir() and not force_record:
            print(f"[record] Dataset już istnieje ({ds_dir}) — pomijam (użyj --force-record/--force).", flush=True)
        else:
            cmd = [
                PY, "record_expert_minari.py",
                "--model", str(ckpt),
                "--env-id", args.env,
                "--algo", "sac",
                "--dataset-id", dataset_id,
                "--minari-datasets-root", str(minari_root),
                "--n-episodes", str(args.n_episodes),
                "--seed", str(args.record_seed),
            ]
            if force_record:
                cmd.append("--overwrite")
            run(cmd, title="record")

    dt_model = dt_out / "dt_model.pth"
    dt_manifest = dt_out / "manifest.json"
    eval_json = eval_out / "eval_metrics.json"
    plot_png = eval_out / "figures" / "dt_vs_baseline.png"

    # 2) TRAIN DT
    if "train" in stages:
        if dt_model.is_file() and dt_manifest.is_file() and not force_train:
            print(f"[train] DT już istnieje ({dt_model}) — pomijam (użyj --force-train/--force).", flush=True)
        else:
            cmd = [
                PY, "train_dt_minari_fetch.py",
                "--dataset-id", dataset_id,
                "--minari-datasets-root", str(minari_root),
                "--env-id", args.env,
                "--device", args.device,
                "--max-iters", str(args.max_iters),
                "--seed", str(args.train_seed),
                "--out-dir", str(dt_out),
            ]
            if args.train_extra.strip():
                cmd += args.train_extra.split()
            run(cmd, title="train")

    # 3) EVAL DT vs TEN checkpoint (te same seedy)
    if "eval" in stages:
        if eval_json.is_file() and not force_eval:
            print(f"[eval] Wynik już istnieje ({eval_json}) — pomijam (użyj --force-eval/--force).", flush=True)
        else:
            if not dt_model.is_file():
                raise SystemExit(f"[eval] Brak wytrenowanego DT: {dt_model} (uruchom etap train).")
            cmd = [
                PY, "eval_dt_minari_fetch.py",
                "--model", str(dt_model),
                "--manifest", str(dt_manifest),
                "--baseline-model", str(ckpt),
                "--baseline-algo", "sac",
                "--env-id", args.env,
                "--n-episodes", str(args.eval_episodes),
                "--target-return", str(args.target_return),
                "--seed", str(args.eval_seed),
                "--device", resolve_eval_device(args.device),
                "--out-dir", str(eval_out),
            ]
            run(cmd, title="eval")

    # 4) PLOT per aktywność (DT vs checkpoint)
    if "plot" in stages:
        if plot_png.is_file() and not force_plot:
            print(f"[plot] Wykres już istnieje ({plot_png}) — pomijam (użyj --force-plot/--force).", flush=True)
        else:
            if not eval_json.is_file():
                raise SystemExit(f"[plot] Brak {eval_json} (uruchom etap eval).")
            cmd = [
                PY, "plot_dt_vs_baseline.py",
                "--input", str(eval_json),
                "--title", f"DT (z checkpointu {steps}) vs checkpoint — {args.env}",
            ]
            run(cmd, title="plot")

    print("\n========== GOTOWE ==========", flush=True)
    print(f"Eval JSON : {eval_out / 'eval_metrics.json'}", flush=True)
    print(f"Wykres    : {eval_out / 'figures' / 'dt_vs_baseline.png'}", flush=True)
    print(
        "Zbiorczy wykres 3 aktywności (po wszystkich procesach):\n"
        f"  {PY} plot_dt_vs_expert_suite.py --inputs "
        "dt_eval_runs\\ckpt_*\\eval_metrics.json",
        flush=True,
    )


if __name__ == "__main__":
    main()
