"""Wkleja PNG do sprawozdanie_l2_zakrety.ipynb jako załączniki markdown (attachment:...).

Uruchom z katalogu l2:  python embed_sprawozdanie_images.py
"""
from __future__ import annotations

import base64
import json
from pathlib import Path


def _mime_png(path: Path) -> dict[str, str]:
    b = path.read_bytes()
    return {"image/png": base64.standard_b64encode(b).decode("ascii")}


def _first_existing(root: Path, candidates: list[Path]) -> Path | None:
    for c in candidates:
        p = root / c if not c.is_absolute() else c
        if p.is_file():
            return p
    return None


def main() -> None:
    root = Path(__file__).resolve().parent
    nb_path = root / "sprawozdanie_l2_zakrety.ipynb"
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    cells = nb["cells"]

    def set_md_attachments(
        idx: int, attachments: dict[str, dict[str, str]], source_lines: list[str]
    ) -> None:
        c = cells[idx]
        if c.get("cell_type") != "markdown":
            raise SystemExit(f"Cell {idx} is not markdown")
        c["attachments"] = attachments
        c["source"] = source_lines

    def _gr_or_plots(rel: str) -> Path | None:
        return _first_existing(
            root,
            [Path(f"greedy_rollout_corner_b/{rel}"), Path(f"plots/{rel}")],
        )

    # --- cell 4: corner_b (krótki wstęp + jeden przykładowy tor z treningu) ---
    hero = _first_existing(
        root,
        [
            Path("corrner_b___track_800.png"),
            Path("greedy_rollout_corner_b/track_29950.png"),
            Path("plots/track_29950.png"),
        ],
    )
    if hero is not None:
        set_md_attachments(
            4,
            {"eb_corner_b.png": _mime_png(hero)},
            [
                "## corner_b\n",
                "\n",
                "`python solution.py train_b` — trening na corner_b (w `main`: "
                "$\\alpha=0.3$, $n=5$, $\\varepsilon=0.05$, 30 000 epizodów), potem rollout zachłanny "
                "(`run_greedy_rollouts`, `GreedyPolicyDriver`, numery `track_30001+`).\n",
                "\n",
                "Przykładowy tor z fazy uczenia (tor + strzałki; wciąż polityka zachowania $b$):\n",
                "\n",
                "![corner_b — trening](attachment:eb_corner_b.png)\n",
            ],
        )
    else:
        print("Pominięto cell 4 — brak obrazu corner_b")

    # --- cell 5: te same pliki co wcześniej, tylko inna kolejność — najpierw trening (penalties), potem greedy (track_30001+) ---
    pens_rel = [
        "penalties_50.png",
        "penalties_450.png",
        "penalties_1350.png",
        "penalties_5550.png",
        "penalties_26100.png",
    ]
    p_names = ["gr_p50.png", "gr_p450.png", "gr_p1350.png", "gr_p5550.png", "gr_p26100.png"]
    att: dict[str, dict[str, str]] = {}
    lines: list[str] = [
        "## Trasy przy zachłannej polityce (po uczeniu)\n",
        "\n",
        "Poniżej **ta sama sesja** `train_b`, ale w kolejności czytelnej dla czytelnika: najpierw "
        "**wykresy kary z treningu** (`penalties_*`, numery epizodów $\\leq 30000$), potem **trasy "
        "rolloutu zachłannego** (`track_30001+`, osobny `Experiment` z `GreedyPolicyDriver` na tym samym $Q$).\n",
        "\n",
        "### Kara w czasie treningu (`penalties_*`)\n",
        "\n",
        "Średnia krocząca w oknie `problem.AVERAGING_WINDOW_SIZE` (25), zapis co "
        "`problem.DRAWING_FREQUENCY` (50) epizodów.\n",
        "\n",
    ]
    for rel, aname in zip(pens_rel, p_names, strict=True):
        p = _gr_or_plots(rel)
        if p is None:
            print("Brak penalties:", rel)
            continue
        att[aname] = _mime_png(p)
        lines.append(f"![{p.stem}](attachment:{aname})\n")

    lines += [
        "\n",
        "### Zachłanne przejazdy — rollout po uczeniu (`track_30001+`)\n",
        "\n",
        "```python\n",
        "greedy_driver = GreedyPolicyDriver(q)\n",
        "eval_exp = Experiment(\n",
        "    environment=environment,\n",
        "    driver=greedy_driver,\n",
        "    number_of_episodes=n_episodes,\n",
        "    current_episode_no=episode_label_offset,\n",
        "    draw_every_episode=True,\n",
        ")\n",
        "eval_exp.run()\n",
        "```\n",
        "\n",
    ]
    t_names = ["gr_t30001.png", "gr_t30002.png", "gr_t30003.png", "gr_t30004.png"]
    for i in range(4):
        ep = 30001 + i
        p = _gr_or_plots(f"track_{ep}.png")
        if p is None:
            print("Brak rollout track:", ep)
            continue
        aname = t_names[i]
        att[aname] = _mime_png(p)
        lines.append(f"![track ep. {ep}](attachment:{aname})\n")

    if att:
        set_md_attachments(5, att, lines)
    else:
        print("Pominięto cell 5 — brak plików w greedy_rollout_corner_b / plots")

    # --- cell 6: corner_c ---
    ps = _first_existing(
        root,
        [
            Path("plots/param_study_corner_c.png"),
            Path("case_study_SUCCESS/param_study_corner_c.png"),
            Path("case_study_2/param_study_corner_c.png"),
        ],
    )
    c1, c2, c3 = (
        root / "corrner_c___track_550.png",
        root / "corrner_c___track_550-2.png",
        root / "corrner_c___track_550-3.png",
    )
    att6: dict[str, dict[str, str]] = {}
    src6: list[str] = [
        "## corner_c — studium $\\alpha$ i $n$\n",
        "\n",
        "`python solution.py param_study` — siatka $(n,\\alpha)$ zdefiniowana w `cmd_param_study` "
        "w `solution.py` (corner, epizody, listy `n_list` / `alphas`). Wyniki dopisywane są do "
        "`plots/param_study_<corner>.csv`; po serii uruchomień skrypt może od razu wygenerować "
        "`plots/param_study_<corner>.png` (`utils.plot_param_study_n_alpha`). "
        "Sam wykres bez ponownego treningu: "
        "`python solution.py param_study_plot --param-study-corner corner_c`.\n",
        "\n",
    ]
    if ps is not None:
        att6["ps_corner_c.png"] = _mime_png(ps)
        src6.append("![param study corner_c](attachment:ps_corner_c.png)\n\n")
    else:
        src6.append("*(Brak pliku wykresu param_study — uruchom param_study / param_study_plot.)*\n\n")
    src6.append("Trasy corner_c (~550 kroków w epizodzie):\n\n")
    for path, nick in [(c1, "cct1.png"), (c2, "cct2.png"), (c3, "cct3.png")]:
        if path.is_file():
            att6[nick] = _mime_png(path)
            src6.append(f"![corner_c track](attachment:{nick})\n")
        else:
            print("Brak:", path)
    set_md_attachments(6, att6, src6)

    # --- cell 9: corner_d ---
    d550 = root / "corner_d" / "track_550.png"
    if d550.is_file():
        set_md_attachments(
            9,
            {"cd_t550.png": _mime_png(d550)},
            [
                "## corner_d\n",
                "\n",
                "Tor trudniejszy — parametry zwykle przenoszone ze studium na corner_c "
                "(np. $n=4$, $\\alpha=0.5$ w `train_off_policy`). Rysunki z treningu zapisują się "
                "jak wyżej (np. podkatalog z runu lub `plots/`).\n",
                "\n",
                "![corner_d track 550](attachment:cd_t550.png)\n",
            ],
        )
    else:
        print("Pominięto cell 9 — brak:", d550)

    nb_path.write_text(json.dumps(nb, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Zapisano:", nb_path)


if __name__ == "__main__":
    main()
