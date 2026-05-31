"""
Bar charts comparing DT vs optional baseline from ``eval_metrics.json`` (output of
``eval_dt_minari_fetch.py``).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


METRIC_KEYS = (
    "success_rate_final",
    "success_rate_any",
    "mean_return",
    "mean_length",
    "mean_final_goal_dist",
)

METRIC_LABELS = (
    "success_rate_final",
    "success_rate_any",
    "mean_return",
    "mean_length",
    "mean_final_goal_dist",
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Bar plot DT vs baseline from eval_metrics.json")
    p.add_argument("--input", type=Path, required=True, help="eval_metrics.json")
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="PNG path (default: figures/dt_vs_baseline.png next to input)",
    )
    p.add_argument("--title", type=str, default=None)
    return p


def main() -> None:
    args = build_parser().parse_args()
    path = args.input.expanduser().resolve()
    data = json.loads(path.read_text(encoding="utf-8"))
    dt = data.get("dt") or {}
    baseline = data.get("baseline")

    if not isinstance(dt, dict):
        raise SystemExit("eval_metrics.json: missing 'dt' object")

    has_base = baseline is not None and isinstance(baseline, dict)
    labels = list(METRIC_LABELS)
    x = np.arange(len(labels))
    width = 0.35
    dt_vals = [float(dt.get(k, 0.0)) for k in METRIC_KEYS]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, dt_vals, width, label="DT")
    if has_base:
        b_vals = [float(baseline.get(k, 0.0)) for k in METRIC_KEYS]
        ax.bar(x + width / 2, b_vals, width, label=f"Baseline ({data.get('baseline_algo', '?')})")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("value")
    ax.legend()
    title = args.title or f"DT vs baseline — {data.get('env_id', '')} (n={data.get('n_episodes', '')})"
    ax.set_title(title)
    fig.tight_layout()

    out = args.output
    if out is None:
        out = path.parent / "figures" / "dt_vs_baseline.png"
    out = out.expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
