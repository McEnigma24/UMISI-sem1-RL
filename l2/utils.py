from __future__ import annotations

import csv
import os
import re
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


def order_learning_series_columns(fieldnames: list[str]) -> list[str]:
    """Kolejność serii: ``epsilon_is``, potem pary ``push_p{n}_is`` / ``push_p{n}_nois``, na końcu pozostałe."""
    cols = [c for c in fieldnames if c and c != "episode"]
    out: list[str] = []
    seen: set[str] = set()
    if "epsilon_is" in cols:
        out.append("epsilon_is")
        seen.add("epsilon_is")
    pcts: set[int] = set()
    for c in cols:
        m = re.fullmatch(r"push_p(\d+)_(is|nois)", c)
        if m:
            pcts.add(int(m.group(1)))
    for pct in sorted(pcts):
        for suffix in ("is", "nois"):
            name = f"push_p{pct}_{suffix}"
            if name in cols:
                out.append(name)
                seen.add(name)
    for c in cols:
        if c not in seen:
            out.append(c)
            seen.add(c)
    return out


def iter_push_bias_pct_pairs(fieldnames: list[str]) -> list[int]:
    """Procenty ``p``, dla których są obie kolumny IS i bez IS."""
    cols = set(fieldnames)
    pcts: set[int] = set()
    for c in fieldnames:
        m = re.fullmatch(r"push_p(\d+)_(is|nois)", c)
        if m:
            pcts.add(int(m.group(1)))
    return [pct for pct in sorted(pcts) if f"push_p{pct}_is" in cols and f"push_p{pct}_nois" in cols]


def iter_push_bias_pcts(fieldnames: list[str]) -> list[int]:
    """Wszystkie ``p`` występujące w kolumnach sweep."""
    pcts: set[int] = set()
    for c in fieldnames:
        m = re.fullmatch(r"push_p(\d+)_(is|nois)", c)
        if m:
            pcts.add(int(m.group(1)))
    return sorted(pcts)


# Baseline oraz pary (IS = jaśniejszy, bez IS = ciemniejszy) w tej samej rodzinie barw.
PUSH_SWEEP_BASELINE_COLOR = "#2ca02c"
_PUSH_P_COLORS: dict[int, tuple[str, str]] = {
    10: ("#6baed6", "#08519c"),
    20: ("#fdae6b", "#b35806"),
    30: ("#bfbbd9", "#54278f"),
}


def push_pair_colors_for_pct(pct: int) -> tuple[str, str]:
    """Zwraca (kolor IS jaśniejszy, kolor bez IS ciemniejszy)."""
    if pct in _PUSH_P_COLORS:
        return _PUSH_P_COLORS[pct]
    return ("#bdbdbd", "#404040")


def push_sweep_series_color(col: str) -> str:
    if col == "epsilon_is":
        return PUSH_SWEEP_BASELINE_COLOR
    m = re.fullmatch(r"push_p(\d+)_(is|nois)", col)
    if m:
        pct = int(m.group(1))
        light, dark = push_pair_colors_for_pct(pct)
        return light if m.group(2) == "is" else dark
    return "#636363"


def order_push_sweep_columns_grouped_is_then_nois(fieldnames: list[str]) -> list[str]:
    """Kolejność na wykresie „wszystko”: baseline, potem wszystkie IS, potem wszystkie bez IS."""
    cols = [c for c in fieldnames if c and c != "episode"]
    fn = set(cols)
    out: list[str] = []
    if "epsilon_is" in fn:
        out.append("epsilon_is")
    for pct in iter_push_bias_pcts(fieldnames):
        key = f"push_p{pct}_is"
        if key in fn:
            out.append(key)
    for pct in iter_push_bias_pcts(fieldnames):
        key = f"push_p{pct}_nois"
        if key in fn:
            out.append(key)
    for c in cols:
        if c not in out:
            out.append(c)
    return out


def _label_learning_series_column(col: str) -> str:
    if col == "epsilon_is":
        return r"$\varepsilon$-greedy + IS"
    m = re.fullmatch(r"push_p(\d+)_(is|nois)", col)
    if m:
        pct = int(m.group(1))
        if m.group(2) == "is":
            return f"push $p={pct}\\%$, IS"
        return f"push $p={pct}\\%$, bez IS"
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
    columns: list[str] | None = None,
    title: str | None = None,
    figsize: tuple[float, float] = (10.0, 5.2),
) -> None:
    """Średnia krocząca kar epizodowych; ``columns`` opcjonalnie ogranicza serie (kolejność na legendzie)."""
    episodes: list[int] = []
    series: dict[str, list[float]] = {}
    with open(series_csv_path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        fnames = list(r.fieldnames or [])
        if columns is None:
            cols = order_learning_series_columns(fnames)
        else:
            cols = [c for c in columns if c in fnames and c != "episode"]
        for row in r:
            episodes.append(int(row["episode"]))
            for name in cols:
                if name not in series:
                    series[name] = []
                series[name].append(float(row[name]))

    if not episodes or not series:
        raise ValueError(f"Brak serii w {series_csv_path}")

    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("white")

    w = max(1, min(roll_window, len(episodes)))

    def rolling_mean(vals: list[float]) -> list[float]:
        out: list[float] = []
        for i in range(len(vals)):
            lo = max(0, i - w + 1)
            chunk = vals[lo : i + 1]
            out.append(float(np.mean(chunk)))
        return out

    for key in cols:
        if key not in series:
            continue
        ys = rolling_mean(series[key])
        ax.plot(
            episodes,
            ys,
            color=push_sweep_series_color(key),
            linewidth=1.9,
            label=_label_learning_series_column(key),
        )

    ax.set_xlabel("epizod", fontsize=11)
    ax.set_ylabel(f"średnia kara (okno {w})", fontsize=11)
    if title:
        ax.set_title(title, fontsize=12)
    ax.legend(loc="best", fontsize=10, framealpha=0.92)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="y", alpha=0.28, linestyle="-", linewidth=0.8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_push_bias_is_pair(
    series_csv_path: str,
    out_path: str,
    *,
    pct: int,
    roll_window: int = 100,
    include_baseline: bool = True,
) -> None:
    """IS vs bez IS dla jednego ``p``; opcjonalnie zielony baseline ε-greedy + IS."""
    col_is = f"push_p{pct}_is"
    col_no = f"push_p{pct}_nois"
    episodes: list[int] = []
    s_is: list[float] = []
    s_no: list[float] = []
    s_eps: list[float] | None = None
    with open(series_csv_path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        fn = list(r.fieldnames or [])
        if col_is not in fn or col_no not in fn:
            raise ValueError(f"Brak kolumn {col_is} / {col_no} w {series_csv_path}")
        if include_baseline and "epsilon_is" in fn:
            s_eps = []
        for row in r:
            episodes.append(int(row["episode"]))
            s_is.append(float(row[col_is]))
            s_no.append(float(row[col_no]))
            if s_eps is not None:
                s_eps.append(float(row["epsilon_is"]))

    if not episodes:
        raise ValueError(f"Brak danych w {series_csv_path}")

    w = max(1, min(roll_window, len(episodes)))

    def rolling_mean(vals: list[float]) -> list[float]:
        out: list[float] = []
        for i in range(len(vals)):
            lo = max(0, i - w + 1)
            chunk = vals[lo : i + 1]
            out.append(float(np.mean(chunk)))
        return out

    y_is = rolling_mean(s_is)
    y_no = rolling_mean(s_no)
    c_light, c_dark = push_pair_colors_for_pct(pct)

    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    fig.patch.set_facecolor("white")
    if s_eps is not None:
        y_eps = rolling_mean(s_eps)
        ax.plot(
            episodes,
            y_eps,
            color=PUSH_SWEEP_BASELINE_COLOR,
            linewidth=2.0,
            linestyle="--",
            label=_label_learning_series_column("epsilon_is"),
            alpha=0.95,
        )
    ax.plot(episodes, y_is, color=c_light, linewidth=2.0, label=_label_learning_series_column(col_is))
    ax.plot(episodes, y_no, color=c_dark, linewidth=2.0, label=_label_learning_series_column(col_no))
    ax.set_xlabel("epizod", fontsize=11)
    ax.set_ylabel(f"średnia kara (okno {w})", fontsize=11)
    ax.set_title(f"Push $p={pct}\\%$: IS vs bez IS", fontsize=12)
    ax.legend(loc="best", fontsize=10, framealpha=0.92)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="y", alpha=0.28, linestyle="-", linewidth=0.8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def render_push_bias_sweep_plots(
    series_csv_path: str,
    *,
    plots_dir: str,
    roll_window: int = 100,
) -> list[str]:
    """Wykresy sweep: najpierw panel sam IS (+baseline), potem sam bez IS, potem pełny (IS, potem bez IS), na końcu pary p."""
    out_paths: list[str] = []

    with open(series_csv_path, newline="", encoding="utf-8") as f:
        fieldnames = list(csv.DictReader(f).fieldnames or [])

    fn = set(fieldnames)
    pcts = iter_push_bias_pcts(fieldnames)

    cols_all_is: list[str] = []
    if "epsilon_is" in fn:
        cols_all_is.append("epsilon_is")
    for pct in pcts:
        k = f"push_p{pct}_is"
        if k in fn:
            cols_all_is.append(k)

    cols_all_nois: list[str] = []
    if "epsilon_is" in fn:
        cols_all_nois.append("epsilon_is")
    for pct in pcts:
        k = f"push_p{pct}_nois"
        if k in fn:
            cols_all_nois.append(k)

    grouped = order_push_sweep_columns_grouped_is_then_nois(fieldnames)

    path_is = os.path.join(plots_dir, "push_bias_sweep_all_is.png")
    plot_learning_series_csv(
        series_csv_path,
        path_is,
        roll_window=roll_window,
        columns=cols_all_is,
        title="Push: warianty z importance sampling (+ baseline)",
        figsize=(10.0, 5.2),
    )
    out_paths.append(path_is)

    path_nois = os.path.join(plots_dir, "push_bias_sweep_all_nois.png")
    plot_learning_series_csv(
        series_csv_path,
        path_nois,
        roll_window=roll_window,
        columns=cols_all_nois,
        title="Push: warianty bez importance sampling (+ baseline)",
        figsize=(10.0, 5.2),
    )
    out_paths.append(path_nois)

    combined = os.path.join(plots_dir, "push_bias_sweep.png")
    plot_learning_series_csv(
        series_csv_path,
        combined,
        roll_window=roll_window,
        columns=grouped,
        title="Pełne porównanie (IS, potem bez IS)",
        figsize=(11.0, 5.5),
    )
    out_paths.append(combined)

    for pct in iter_push_bias_pct_pairs(fieldnames):
        pair_png = os.path.join(plots_dir, f"push_bias_sweep_p{pct}_is_vs_nois.png")
        plot_push_bias_is_pair(series_csv_path, pair_png, pct=pct, roll_window=roll_window)
        out_paths.append(pair_png)

    return out_paths
