"""
Zbiorcze porównanie DT vs ekspert (SB3) **po zadaniach** z wielu ``eval_metrics.json``.

Ekspert SAC+HER, na którego nagraniach uczył się DT, jest tu górną granicą
(ceiling) — dlatego oprócz surowych słupków liczymy też *normalized score*
(``DT / ekspert``), czyli „ile % eksperta DT odzyskuje” per zadanie i średnio.

Wejście: albo lista plików ``--inputs a.json b.json ...``, albo katalog
``--runs-root dt_eval_runs`` (wtedy bierzemy **najnowszy** ``eval_metrics.json``
dla każdego ``env_id``).

Przykład:

  python plot_dt_vs_expert_suite.py --runs-root dt_eval_runs
  python plot_dt_vs_expert_suite.py --inputs dt_eval_runs/reach/eval_metrics.json dt_eval_runs/push/eval_metrics.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

L6_ROOT = Path(__file__).resolve().parent

# (klucz w eval_metrics, etykieta panelu, czy "wyżej = lepiej")
# mean_return jako metryka GŁÓWNA (średnia z całego epizodu, nie jednego kroku).
# success_rate_final świadomie pominięty — to tylko ostatni krok = jedna próbka.
PANELS: tuple[tuple[str, str, bool], ...] = (
    ("mean_return", "Śr. zwrot epizodu (GŁÓWNA)", True),
    ("mean_final_goal_dist", "Śr. dystans do celu (koniec)", False),
    ("success_rate_any", "Sukces (kiedykolwiek w ep.)", True),
)

# Krótkie etykiety zadań na osi X
ENV_SHORT = {
    "FetchReach-v4": "Reach",
    "FetchPush-v4": "Push",
    "FetchSlide-v4": "Slide",
    "FetchPickAndPlace-v4": "Pick&Place",
}
# Kolejność prezentacji (od najłatwiejszego)
ENV_ORDER = ["FetchReach-v4", "FetchPush-v4", "FetchSlide-v4", "FetchPickAndPlace-v4"]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Zbiorczy wykres DT vs ekspert po zadaniach (+ normalized score)")
    src = p.add_mutually_exclusive_group(required=False)
    src.add_argument("--inputs", type=Path, nargs="+", default=None, help="Lista eval_metrics.json")
    src.add_argument(
        "--runs-root",
        type=Path,
        default=L6_ROOT / "dt_eval_runs",
        help="Katalog z runami eval (domyślnie dt_eval_runs); bierzemy najnowszy per env_id",
    )
    p.add_argument("--output", type=Path, default=None, help="PNG (domyślnie figures/dt_vs_expert_suite.png)")
    p.add_argument("--title", type=str, default="DT vs ekspert (SAC+HER) — zadania Fetch")
    return p


def discover_inputs(runs_root: Path) -> list[Path]:
    """Najnowszy eval_metrics.json per env_id (po nazwie katalogu runu = timestamp)."""
    runs_root = runs_root.expanduser().resolve()
    by_env: dict[str, tuple[str, Path]] = {}
    if not runs_root.is_dir():
        return []
    for jp in runs_root.rglob("eval_metrics.json"):
        try:
            data = json.loads(jp.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        env = str(data.get("env_id", "?"))
        # sortowalny klucz: nazwa katalogu runu (timestamp) + pełna ścieżka
        stamp = jp.parent.name
        prev = by_env.get(env)
        if prev is None or stamp > prev[0]:
            by_env[env] = (stamp, jp)
    return [v[1] for v in by_env.values()]


def load_records(paths: list[Path]) -> list[dict[str, Any]]:
    recs: list[dict[str, Any]] = []
    for p in paths:
        p = p.expanduser().resolve()
        data = json.loads(p.read_text(encoding="utf-8"))
        recs.append({"path": p, "data": data, "env_id": str(data.get("env_id", "?"))})
    # sortuj wg ENV_ORDER, nieznane na koniec
    recs.sort(key=lambda r: (ENV_ORDER.index(r["env_id"]) if r["env_id"] in ENV_ORDER else 999, r["env_id"]))
    return recs


def _fmt(key: str, v: float) -> str:
    if "success" in key:
        return f"{v * 100:.0f}%"
    if key == "mean_return":
        return f"{v:.1f}"
    return f"{v:.3f}"


def main() -> None:
    args = build_parser().parse_args()

    paths = list(args.inputs) if args.inputs else discover_inputs(args.runs_root)
    if not paths:
        raise SystemExit(
            "Brak eval_metrics.json. Najpierw odpal eval_dt_minari_fetch.py per zadanie "
            "(z --baseline-model), albo wskaż pliki przez --inputs."
        )

    recs = load_records(paths)
    envs = [r["env_id"] for r in recs]
    short = [ENV_SHORT.get(e, e) for e in envs]
    x = np.arange(len(envs))
    width = 0.38

    n_panels = len(PANELS)
    fig, axes = plt.subplots(1, n_panels, figsize=(min(4.0 * n_panels, 18.0), 4.4), squeeze=False)
    axes = list(axes[0])

    any_baseline = False
    for ax, (key, panel_title, _higher_better) in zip(axes, PANELS):
        dt_vals = [float((r["data"].get("dt") or {}).get(key, np.nan)) for r in recs]
        base_present = [isinstance(r["data"].get("baseline"), dict) for r in recs]
        base_vals = [
            float(r["data"]["baseline"].get(key, np.nan)) if bp else np.nan
            for r, bp in zip(recs, base_present)
        ]
        any_baseline = any_baseline or any(base_present)

        b_dt = ax.bar(x - width / 2, dt_vals, width, label="DT", color="#264653", edgecolor="white", zorder=2)
        b_ex = ax.bar(x + width / 2, base_vals, width, label="Ekspert", color="#e76f51", edgecolor="white", zorder=2)

        ax.set_title(panel_title, fontsize=10, pad=6)
        ax.set_xticks(x, short, fontsize=8, rotation=15)
        ax.grid(axis="y", linestyle=":", alpha=0.55, zorder=0)
        ax.tick_params(axis="y", labelsize=8)
        ax.axhline(0.0, color="#cccccc", linewidth=0.85, zorder=1)

        ax.bar_label(b_dt, labels=[_fmt(key, v) for v in dt_vals], fontsize=7, padding=2)
        ax.bar_label(b_ex, labels=["" if np.isnan(v) else _fmt(key, v) for v in base_vals], fontsize=7, padding=2)

        if "success" in key:
            ax.set_ylim(0.0, 1.05)

    axes[0].legend(loc="upper left", fontsize=8, framealpha=0.9)

    # Normalized score (DT / ekspert) dla success_rate_final
    # Normalized score liczony na ŚREDNIM ZWROCIE, nie na ostatnim kroku.
    # Zwrot jest ujemny i = -(kroki poza celem); zamieniamy na pozytywny "udział
    # czasu w celu" = (zwrot + L)/L, L = długość epizodu, i dzielimy DT/ekspert.
    lines: list[str] = []
    norm_vals: list[float] = []
    for r in recs:
        dt_blob = r["data"].get("dt") or {}
        base = r["data"].get("baseline")
        env_short = ENV_SHORT.get(r["env_id"], r["env_id"])
        dt_r = dt_blob.get("mean_return")
        be_r = base.get("mean_return") if isinstance(base, dict) else None
        if dt_r is None or be_r is None:
            lines.append(f"{env_short}: brak baseline")
            continue
        L = float(dt_blob.get("mean_length") or 50.0)
        dt_dwell = max(0.0, L + float(dt_r))   # kroki w celu (DT)
        be_dwell = max(0.0, L + float(be_r))   # kroki w celu (ekspert)
        if be_dwell <= 1e-9:
            lines.append(f"{env_short}: ekspert≈0 czasu w celu (DT {dt_dwell:.1f})")
        else:
            ratio = dt_dwell / be_dwell
            norm_vals.append(ratio)
            lines.append(
                f"{env_short}: {ratio * 100:.0f}% eksperta "
                f"(zwrot {dt_r:.1f} vs {be_r:.1f})"
            )
    mean_txt = f"  |  średnio: {np.mean(norm_vals) * 100:.0f}% eksperta" if norm_vals else ""

    footer = "Normalized score (śr. zwrot → udział czasu w celu, DT/ekspert)  —  " + "   ".join(lines) + mean_txt
    n_ep = recs[0]["data"].get("n_episodes", "?")
    fig.suptitle(f"{args.title}   (n_episodes={n_ep}, te same seedy)", fontsize=12, y=0.99)
    fig.text(0.5, 0.015, footer, ha="center", va="bottom", fontsize=8, color="#333333")
    if not any_baseline:
        fig.text(
            0.5, 0.055,
            "UWAGA: w żadnym eval_metrics.json nie ma baseline — odpal eval_dt_minari_fetch.py z --baseline-model.",
            ha="center", va="bottom", fontsize=8, color="#b00020",
        )

    fig.subplots_adjust(top=0.86, bottom=0.2, wspace=0.28)

    out = args.output
    if out is None:
        out = L6_ROOT / "figures" / "dt_vs_expert_suite.png"
    out = out.expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)

    def _ascii(s: str) -> str:
        return s.encode("ascii", "replace").decode("ascii")

    print("Zadania:", ", ".join(envs))
    print(_ascii("\n".join(lines)))
    if norm_vals:
        print(f"Srednio: {np.mean(norm_vals) * 100:.0f}% eksperta")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
