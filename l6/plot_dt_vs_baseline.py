"""
Porównanie DT vs baseline z ``eval_metrics.json`` — osobna oś Y na metrykę
(bez mieszania skali success ~0–1 z mean_return ~ −50).

Gdy brak baseline w JSON, jedna dyskretna stopka pod figurą (nie tekst na każdym panelu).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PANELS: tuple[tuple[str, str], ...] = (
    ("success_rate_final", "Sukces (ostatni krok)"),
    ("success_rate_any", "Sukces (kiedykolwiek w ep.)"),
    ("mean_return", "Śr. zwrot epizodu"),
    ("mean_length", "Śr. długość epizodu"),
    ("mean_final_goal_dist", "Śr. dystans do celu (koniec)"),
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Wykres DT vs baseline z eval_metrics.json (panele per metryka)")
    p.add_argument("--input", type=Path, required=True, help="eval_metrics.json")
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="PNG (domyślnie figures/dt_vs_baseline.png obok input)",
    )
    p.add_argument("--title", type=str, default=None)
    return p


def _value_labels_for_panel(key: str, vals: list[float]) -> list[str]:
    labels: list[str] = []
    for v in vals:
        if "success" in key:
            labels.append(f"{v * 100:.0f}%")
        elif key == "mean_return":
            labels.append(f"{v:.2f}")
        elif key == "mean_length":
            labels.append(f"{v:.1f}")
        else:
            labels.append(f"{v:.4f}")
    return labels


def main() -> None:
    args = build_parser().parse_args()
    path = args.input.expanduser().resolve()
    data = json.loads(path.read_text(encoding="utf-8"))
    dt = data.get("dt") or {}
    baseline = data.get("baseline")

    if not isinstance(dt, dict):
        raise SystemExit("eval_metrics.json: brak obiektu 'dt'")

    has_base = baseline is not None and isinstance(baseline, dict)
    algo = data.get("baseline_algo") or "?"
    base_tick = f"Baseline\n({algo})"

    n = len(PANELS)
    # Jeden rząd: brak pustej „szóstej” komórki; więcej miejsca na podpisy osi X
    fig_w = min(2.85 * n, 16.0)
    fig, axes = plt.subplots(1, n, figsize=(fig_w, 3.9), squeeze=True)
    if n == 1:
        axes = [axes]
    else:
        axes = list(axes)

    for ax, (key, panel_title) in zip(axes, PANELS):
        dt_v = float(dt.get(key, 0.0))
        if has_base:
            b_v = float(baseline.get(key, 0.0))
            vals = [dt_v, b_v]
            x = np.arange(2)
            bars = ax.bar(
                x,
                vals,
                width=0.52,
                color=["#264653", "#e76f51"],
                edgecolor="white",
                linewidth=0.9,
                zorder=2,
            )
            ax.set_xticks(x, ["DT", base_tick], fontsize=8)
        else:
            vals = [dt_v]
            x = np.arange(1)
            bars = ax.bar(
                x,
                vals,
                width=0.42,
                color=["#264653"],
                edgecolor="white",
                linewidth=0.9,
                zorder=2,
            )
            ax.set_xticks(x, ["DT"], fontsize=9)

        ax.set_title(panel_title, fontsize=9.5, pad=6)
        ax.axhline(0.0, color="#cccccc", linewidth=0.85, zorder=1)
        ax.grid(axis="y", linestyle=":", alpha=0.55, zorder=0)
        ax.tick_params(axis="y", labelsize=8)
        if ax is not axes[0]:
            ax.set_ylabel("")
        ylow, yhigh = ax.get_ylim()
        if abs(yhigh - ylow) < 1e-9:
            ax.set_ylim(ylow - 0.05, yhigh + 0.05)
        elif ylow < 0 < yhigh:
            ax.set_ylim(bottom=min(ylow, 0), top=max(yhigh, 0))

        lbls = _value_labels_for_panel(key, vals)
        ax.bar_label(bars, labels=lbls, fontsize=8, padding=3, zorder=3)

        if "success" in key:
            hi = max(vals) if vals else 0.0
            ax.set_ylim(0.0, max(1.0, hi * 1.25, 0.12))

    env = data.get("env_id", "?")
    n_ep = data.get("n_episodes", "?")
    tr = data.get("target_return")
    sub = f"{env}  |  n_episodes={n_ep}"
    if tr is not None:
        sub += f"  |  target_return={tr}"
    if has_base:
        bsrc = data.get("baseline_model_path")
        if bsrc:
            sub += f"\nBaseline: {Path(bsrc).name}"
        br = data.get("baseline_resolution")
        if br:
            sub += f"  ({br})"
    title = args.title or (
        "DT vs baseline (SB3)" if has_base else "Metryki DT — brak baseline w JSON"
    )
    fig.suptitle(f"{title}\n{sub}", fontsize=11, y=0.98)

    if not has_base:
        fig.text(
            0.5,
            0.02,
            "Baseline: eval_dt_minari_fetch.py wybiera zip z minari_recordings (ten sam dataset_id), "
            "albo ustaw L6_EVAL_BASELINE_MODEL / --baseline-model. Wyłączenie: --no-baseline.",
            ha="center",
            va="bottom",
            fontsize=8,
            color="#333333",
            transform=fig.transFigure,
        )

    fig.subplots_adjust(top=0.78, bottom=0.2 if not has_base else 0.14, wspace=0.38)

    out = args.output
    if out is None:
        out = path.parent / "figures" / "dt_vs_baseline.png"
    out = out.expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
