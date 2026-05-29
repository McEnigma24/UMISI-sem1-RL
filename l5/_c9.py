from tensorboard.backend.event_processing import event_accumulator as ea


def pick_latest_tensorboard_run(runs_dir: Path) -> Path:
    if not runs_dir.is_dir():
        raise FileNotFoundError(f"Brak katalogu TensorBoard: {runs_dir}")
    subs = [p for p in runs_dir.iterdir() if p.is_dir()]
    if not subs:
        raise FileNotFoundError(f"Brak podkatalogów w {runs_dir}")

    def latest_mtime(d: Path) -> float:
        files = list(d.glob("*"))
        if not files:
            return d.stat().st_mtime
        return max(f.stat().st_mtime for f in files)

    return max(subs, key=latest_mtime)


# Ustaw na None aby wybrać najnowszy; albo np. RUNS_DIR / "May24_12-00-00"
TB_RUN: Path | None = None
tb_run = TB_RUN or pick_latest_tensorboard_run(RUNS_DIR)
print("TensorBoard:", tb_run)

acc = ea.EventAccumulator(str(tb_run), size_guidance={"scalars": 0})
acc.Reload()
scalar_tags = acc.Tags().get("scalars", [])
print("Skalary:", sorted(scalar_tags))


def plot_scalar(tag: str, title: str | None = None) -> None:
    if tag not in scalar_tags:
        print(f"(brak tagu {tag!r})")
        return
    events = acc.Scalars(tag)
    steps = [e.step for e in events]
    vals = [e.value for e in events]
    plt.figure(figsize=(9, 3.2))
    plt.plot(steps, vals, lw=1)
    plt.xlabel("krok środowiska")
    plt.ylabel(tag)
    plt.title(title or tag)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


plt.rcParams["figure.dpi"] = 110
plot_scalar("test_ep_return", "Test: średni zwrot (deterministyczna polityka)")
plot_scalar("test_ep_length", "Test: średnia długość epizodu")
plot_scalar("alpha", "Temperatura entropii α (uczona)")
plot_scalar("loss_q")
plot_scalar("loss_pi")
plot_scalar("loss_alpha", "Strata dla log α")
plot_scalar("ep_return", "Trening: zwrot ostatniego epizodu (log)")
plot_scalar("ep_length", "Trening: długość ostatniego epizodu")