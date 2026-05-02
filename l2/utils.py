from __future__ import annotations

import csv
from collections import defaultdict

from matplotlib import pyplot as plt
import numpy as np

import problem


def draw_arrow(axes: plt.Axes, begin: tuple[int, int], end: tuple[int, int]) -> None:
    (begin_y, begin_x), (end_y, end_x) = begin, end
    delta_x, delta_y = end_x - begin_x, end_y - begin_y
    axes.arrow(begin_x + 0.5, begin_y + 0.5, delta_x, delta_y,
               length_includes_head=True,
               head_width=0.8, head_length=0.8,
               fc='r', ec='r')


def draw_episode(track: np.ndarray, positions: list[problem.Position], episode: int) -> None:
    ax = plt.axes()
    ax.imshow(track)
    for i in range(len(positions) - 1):
        begin, end = positions[i], positions[i+1]
        draw_arrow(ax, begin, end)
    plt.savefig(f'plots/track_{episode}.png', dpi=300)
    plt.clf()


def draw_penalties_plot(penalties: list[int], window_size: int, episode: int) -> None:
    means = [np.mean(penalties[i:i+window_size]) for i in range(len(penalties) - window_size)]
    ax = plt.axes()
    ax.plot(means)
    plt.savefig(f'plots/penalties_{episode}.png', dpi=300)
    plt.clf()


# Kolory zbliżone do typowego wykresu z PDF (n-krokowy SARSA vs α).
_PARAM_STUDY_COLORS: dict[int, str] = {
    1: "#d62728",
    2: "#2ca02c",
    4: "#1f77b4",
    8: "#000000",
    16: "#e377c2",
    32: "#17becf",
    64: "#9467bd",
    128: "#c989c9",
    256: "#ff7f0e",
    512: "#8c564b",
}


def _color_for_n(n: int, series_index: int) -> str:
    if n in _PARAM_STUDY_COLORS:
        return _PARAM_STUDY_COLORS[n]
    return plt.cm.tab20.colors[series_index % 20]


def plot_param_study_n_alpha(
    csv_path: str,
    out_path: str,
    *,
    y_column: str = "normalized_cost",
    ylim: tuple[float, float] | None = (0.25, 0.55),
    ylabel: str | None = None,
) -> None:
    """Wykres α (oś X) vs koszt (oś Y), osobna krzywa dla każdego n — styl jak w instrukcji.

    Etykiety ``n=…`` przy krzywych (zamiast legendy), tylko lewa/dolna oś, opcjonalnie stałe ylim.
    """
    by_n: dict[int, dict[float, float]] = defaultdict(dict)
    with open(csv_path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            if not row.get("n") or row.get("n", "").strip() == "":
                continue
            n = int(row["n"])
            alpha = float(row["alpha"])
            by_n[n][alpha] = float(row[y_column])

    if not by_n:
        raise ValueError(f"Brak danych w {csv_path} (pusty lub sam nagłówek).")

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    fig.patch.set_facecolor("white")

    for si, n in enumerate(sorted(by_n.keys())):
        pts = sorted(by_n[n].items())
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        color = _color_for_n(n, si)
        ax.plot(
            xs,
            ys,
            color=color,
            linewidth=1.9,
            marker="o",
            markersize=4,
            markerfacecolor=color,
            markeredgecolor="white",
            markeredgewidth=0.6,
            clip_on=False,
        )

        # Etykiety jak w PDF: niższe n — przy dole (minimum), wyższe n — przy lewej stronie (stromizna)
        arr_y = np.array(ys, dtype=float)
        if n >= 64:
            i_mark = int(np.argmin(xs))
            ha, va = "left", "bottom"
            off = (6, 4)
        else:
            i_mark = int(np.argmin(arr_y))
            ha, va = "center", "top"
            off = (0, 5)
        lx, ly = xs[i_mark], ys[i_mark]
        ax.annotate(
            f"n={n}",
            xy=(lx, ly),
            xytext=off,
            textcoords="offset points",
            color=color,
            fontsize=9,
            fontweight="medium",
            ha=ha,
            va=va,
            zorder=5 + n,
        )

    ax.set_xlabel(r"$\alpha$", fontsize=12)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=11)

    ax.set_xlim(0.0, 1.0)
    ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

    all_y_flat = [y for d in by_n.values() for y in d.values()]
    if ylim is not None:
        lo, hi = ylim
        pad = 0.015 * (max(all_y_flat) - min(all_y_flat) or 1.0)
        if min(all_y_flat) < lo or max(all_y_flat) > hi:
            ax.set_ylim(min(lo, min(all_y_flat) - pad), max(hi, max(all_y_flat) + pad))
        else:
            ax.set_ylim(lo, hi)
    else:
        pad = 0.02 * (max(all_y_flat) - min(all_y_flat) or 1.0)
        ax.set_ylim(min(all_y_flat) - pad, max(all_y_flat) + pad)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", which="major", labelsize=10)
    ax.grid(True, axis="y", alpha=0.28, linestyle="-", linewidth=0.8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_compare_behavior_is(
    series_csv_path: str,
    out_path: str,
    *,
    roll_window: int = 50,
) -> None:
    """Krzywe uczenia: średnia krocząca kary po epizodach dla compare_push_is."""
    episodes: list[int] = []
    series: dict[str, list[float]] = {}
    with open(series_csv_path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        fieldnames = [fn for fn in (r.fieldnames or []) if fn != "episode"]
        for row in r:
            episodes.append(int(row["episode"]))
            for name in fieldnames:
                if name not in series:
                    series[name] = []
                series[name].append(float(row[name]))

    if not episodes:
        raise ValueError(f"Brak danych w {series_csv_path}")

    labels = {
        "epsilon_is": r"$\varepsilon$-greedy + IS",
        "push_is": r"push + IS",
        "push_no_is": r"push, bez IS",
    }
    colors = {
        "epsilon_is": "#1f77b4",
        "push_is": "#d62728",
        "push_no_is": "#9467bd",
    }

    fig, ax = plt.subplots(figsize=(9.5, 5.0))
    fig.patch.set_facecolor("white")

    w = max(1, min(roll_window, len(episodes)))

    def rolling_mean(vals: list[float]) -> list[float]:
        out: list[float] = []
        for i in range(len(vals)):
            lo = max(0, i - w + 1)
            chunk = vals[lo : i + 1]
            out.append(float(np.mean(chunk)))
        return out

    for key in ("epsilon_is", "push_is", "push_no_is"):
        if key not in series:
            continue
        ys = rolling_mean(series[key])
        ax.plot(
            episodes,
            ys,
            color=colors.get(key, "#333333"),
            linewidth=1.8,
            label=labels.get(key, key),
        )

    ax.set_xlabel("epizod", fontsize=11)
    ax.set_ylabel(f"średnia kara (okno {w})", fontsize=11)
    ax.legend(loc="best", fontsize=10, framealpha=0.92)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="y", alpha=0.28, linestyle="-", linewidth=0.8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _ordered_series_columns(fieldnames: list[str]) -> list[str]:
    cols = [c for c in fieldnames if c != "episode"]
    eps_cols = [c for c in cols if c == "epsilon_is"]
    push_cols = sorted(
        [c for c in cols if c.startswith("push_p")],
        key=lambda x: int(x.replace("push_p", "") or "0"),
    )
    rest = [c for c in cols if c not in eps_cols and c not in push_cols]
    return eps_cols + push_cols + rest


def _label_learning_series_column(col: str) -> str:
    if col == "epsilon_is":
        return r"$\varepsilon$-greedy + IS"
    if col.startswith("push_p"):
        pct = col.removeprefix("push_p")
        try:
            return f"push $p={int(pct)}\\%$"
        except ValueError:
            return col
    return col


def plot_learning_series_csv(
    series_csv_path: str,
    out_path: str,
    *,
    roll_window: int = 100,
) -> None:
    """Średnia krocząca kary — wszystkie kolumny poza ``episode`` (np. sweep wag push)."""
    episodes: list[int] = []
    series: dict[str, list[float]] = {}
    with open(series_csv_path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        cols = _ordered_series_columns(list(r.fieldnames or []))
        for row in r:
            episodes.append(int(row["episode"]))
            for name in cols:
                if name not in series:
                    series[name] = []
                series[name].append(float(row[name]))

    if not episodes or not series:
        raise ValueError(f"Brak serii w {series_csv_path}")

    fig, ax = plt.subplots(figsize=(10.0, 5.2))
    fig.patch.set_facecolor("white")

    w = max(1, min(roll_window, len(episodes)))

    def rolling_mean(vals: list[float]) -> list[float]:
        out: list[float] = []
        for i in range(len(vals)):
            lo = max(0, i - w + 1)
            chunk = vals[lo : i + 1]
            out.append(float(np.mean(chunk)))
        return out

    cmap = plt.cm.tab10.colors
    for i, key in enumerate(cols):
        if key not in series:
            continue
        ys = rolling_mean(series[key])
        ax.plot(
            episodes,
            ys,
            color=cmap[i % len(cmap)],
            linewidth=1.85,
            label=_label_learning_series_column(key),
        )

    ax.set_xlabel("epizod", fontsize=11)
    ax.set_ylabel(f"średnia kara (okno {w})", fontsize=11)
    ax.legend(loc="best", fontsize=10, framealpha=0.92)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="y", alpha=0.28, linestyle="-", linewidth=0.8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
