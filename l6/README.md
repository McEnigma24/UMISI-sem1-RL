# L6 — Decision Transformer (FetchPickAndPlace-v4)

## Python — wyłącznie **3.12.x**

Cały katalog L6 jest ustawiony pod **Python 3.12** (plik [`.python-version`](.python-version), nagłówek [`requirements.txt`](requirements.txt); **nanodt** z PyPI wymaga `>=3.12`).

- [`.python-version`](.python-version): `3.12` — **`uv venv`** wybierze 3.12 po `uv python install 3.12`.

```powershell
cd UMISI-sem1-RL\l6
uv python install 3.12
uv venv
```

```powershell
.\.venv\Scripts\python.exe -V
```

Musi być **Python 3.12.x**.

---

## Instalacja zależności (jeden plik, PyTorch z CUDA)

Wszystko jest w **[`requirements.txt`](requirements.txt)**:

- na początku **`--extra-index-url https://download.pytorch.org/whl/cu124`** — **`torch`** instaluje się jako **build z obsługą CUDA 12.4** (nie CPU-only z samego PyPI);
- reszta pakietów (gymnasium, SB3, **minari[hdf5,create]** — zapis datasetów wymaga Jax, nanodt itd.) i tak pobierana jest głównie z **PyPI**.

```powershell
.\.venv\Scripts\Activate.ps1
uv pip install -r requirements.txt
uv cache clean

uv pip install torch --index-url https://download.pytorch.org/whl/cu126
```

Jeśli **`uv pip`** zgłasza **`invalid peer certificate: UnknownIssuer`** (PyPI / proxy firmowy / antywirus SSL), spróbuj użyć **systemowego magazynu certyfikatów**:

```powershell
uv pip install --system-certs -r requirements.txt
```

To samo możesz dodać przy instalacji pojedynczych pakietów (np. `torch` z indeksu PyTorch).

**Sterownik NVIDIA** musi być zgodny z wheelami **CUDA 12.4** (w razie wątpliwości: [pytorch.org/get-started](https://pytorch.org/get-started/locally/)). Inny suffix (`cu121` itd.) → w `requirements.txt` podmień URL w linii `--extra-index-url`.

Sprawdzenie:

```powershell
python -c "import torch; print('cuda?', torch.cuda.is_available(), '|', torch.__version__)"
```

- **`cuda? True`** — zainstalowany torch z CUDA i driver widzi GPU.
- **`cuda? False`** — nadal możesz mieć **build +CUDA**, ale bez karty / bez sterownika obliczenia idą na CPU; to zachowanie PyTorch, nie „zły” plik requirements.

---

## Trening eksperta (PPO, FetchPickAndPlace-v4)

Skrypt **[`train_expert_ppo.py`](train_expert_ppo.py)** — Stable-Baselines3 **PPO** + **`MultiInputPolicy`** (obserwacja typu Dict), zapis do `weights/<timestamp>/` (`ppo_model.zip`, `manifest.json`, opcjonalnie `best/` z EvalCallback).

```powershell
python train_expert_ppo.py --check-device
python train_expert_ppo.py --timesteps 1_000_000 --tensorboard
python train_expert_ppo.py --eval-only weights\...\ppo_model.zip
```

Fetch ma nagrodę rzadką — przy słabym sukcesie zwiększ `--timesteps` lub rozważ `FetchPickAndPlaceDense-v4` (inny `env_id`, nagroda gęsta). W Gymnasium **v3** jest oznaczone jako deprecated — domyślnie używamy **v4**.

---

## Trening eksperta (SAC + HER — zalecane na Fetch)

Skrypt **[`train_expert_sac_her_fetch.py`](train_expert_sac_her_fetch.py)** — **SAC** z **`HerReplayBuffer`**, **`MultiInputPolicy`**.

### Gdzie zapisujemy wagi (domyślnie: `weights/besties/`)

- **Bez** `--fresh-run-dir`: jeden katalog akumulacji (domyślnie `weights/besties` względem `l6/`), podkatalog **per env**, np. `besties/FetchReach-v4/`. Kolejne joby na LSC dopisują checkpointy i nadpisują `sac_her_model.zip` po treningu.
- **`--fresh-run-dir`**: poprzednie zachowanie — nowy timestamp pod `weights/` (suite) albo `allocate_run_dir` dla pojedynczego env.
- **`--suite-dir PATH`**: własny root (np. projekt na SCRATCH); layout `use_subdir` jak wcześniej dla wielu envów.

W każdym katalogu env: `tensorboard/`, `best/` (EvalCallback), `checkpoints/` (okresowe `.zip`), `latest/sac_her_resume.zip` (kopia do wznowienia), `manifest.json`, opcjonalnie **`quality_metadata.json`**.

### Wczesny stop (streak) i certyfikacja

- Po `min_steps_before_early_stop` co `early_stop_check_freq` kroków: `rollout_eval`; warunek passu: `success_rate_final >= próg`.
- **`--early-stop-streak K`**: wymagane **K kolejnych** passów z rzędu; jeden słabszy rollout zeruje licznik. Dopiero wtedy trening się kończy (przed pełnym `--timesteps`).
- Progi per env w stałej `DEFAULT_EARLY_STOP_SUCCESS_THRESHOLD_BY_ENV` w skrypcie (m.in. Reach **0.95**, Push **0.85** …); **`--success-threshold`** nadpisuje jednym progiem; **`--early-stop-threshold-fallback`** dla nieznanego `env_id`. **`--no-early-stop`**: pełny budżet, brak logiki streak / certyfikacji skip.

### `quality_metadata.json` (reconcile)

- Certyfikat zapisuje się **tuż po** zapisaniu `sac_her_model.zip` (przed długim `rollout_eval` 30 ep.) — wcześniejsza kolejność mogła gubić plik przy błędzie ewaluacji lub killu Slurm. Bloki: `criteria` (próg z tolerancją float, streak, częstotliwości, `early_stop_enabled`, …) i `certification` (streak, metryki ostatniego rolloutu early-stop, `artifact_primary`).
- Na starcie: w logu widać `quality_metadata.json: … (istnieje: …)`. **`[skip]`** gdy (poza streakiem) `criteria` się zgadzają, certyfikat kompletny, plik wag istnieje oraz **H ≥ max(K_file, K_now)** — np. w pliku K_file=10 i H=10, a w kodzie K_now=5 → nadal OK (wcześniej trudniejszy wymóg). Inaczej: `[reconcile]` i dalszy trening.
- Suite: przy każdym `lsc_run` skrypt przechodzi **wszystkie cztery** envy z `FETCH_SPARSE_V4` (log `[suite] (i/4) …`); dla każdego osobno skip / resume / start.

### Checkpointy i resume

- Domyślnie **`CheckpointCallback`**: co `--checkpoint-save-freq` kroków środowiska (wewnętrznie `/ n_envs` dla SB3). **`--no-checkpoints`**: wyłącza. **`--checkpoint-save-replay-buffer`**: duże pliki (replay w `.zip`).
- **`--resume PATH`**: wymusza plik wag (tylko przy **jednym** env; przy suite ignorowane — każde env ma własne `latest/`).
- **`--additional-timesteps N`**: po wczytaniu checkpointu uczy do `num_timesteps + N`.

Domyślny budżet kroków na env: stała **`MY_TIMESTEPS_LIMIT`** w skrypcie (obecnie 1_500_000), nadpisywana przez **`--timesteps`**.

```powershell
python train_expert_sac_her_fetch.py
python train_expert_sac_her_fetch.py --env-id FetchPickAndPlace-v4
python train_expert_sac_her_fetch.py --fresh-run-dir
python train_expert_sac_her_fetch.py --accumulate-dir weights/moj_zbior
python train_expert_sac_her_fetch.py --early-stop-streak 10 --min-steps-before-early-stop 6000
python train_expert_sac_her_fetch.py --resume weights/besties/FetchReach-v4/latest/sac_her_resume.zip
python train_expert_sac_her_fetch.py --no-early-stop
python train_expert_sac_her_fetch.py --check-device
```

Zbiorczy **`manifest_suite.json`** w root (`besties/` lub katalogu timestamp) — w polu `runs` są wpisy z treningu lub `skipped: true` po skipie.

---

## Nagrywanie śladów eksperta do Minari (pod DT / offline RL)

Skrypt **[`record_expert_minari.py`](record_expert_minari.py)** ładuje checkpoint **SAC** lub **PPO** (`MultiInputPolicy`), odpala rollout w `gym.make(env_id)` owiniętym w **`minari.DataCollector`**, a na końcu woła **`create_dataset`**. Pliki HDF5 Minari zapisują się domyślnie w **`l6/minari_datasets/`** (root `MINARI_DATASETS_PATH` — w repo, nie w `~/.minari/datasets`). Dodatkowo powstaje **manifest** runu: `minari_recordings/<timestamp>/manifest.json` (`minari_dataset_id`, `minari_datasets_root`, ścieżka `.../data`, `post_eval`).

- **`--minari-datasets-root`**: opcjonalnie inny katalog główny Minari (nadpisuje domyślne `l6/minari_datasets`). Żeby `python -m minari list local` widział te same datasety co skrypt, ustaw w PowerShell: `$env:MINARI_DATASETS_PATH="C:\...\UMISI-sem1-RL\l6\minari_datasets"` przed wywołaniem CLI Minari.
- **Migracja**: jeśli wcześniej nagrywałeś do `~/.minari/datasets`, ten skrypt domyślnie **nie** widzi tych plików — przenieś katalog `l6/...` z home do `l6/minari_datasets/` albo nagraj ponownie (z `--overwrite` przy tej samej nazwie datasetu w *nowym* root).

**SAC + `HerReplayBuffer`:** `SAC.load` wymaga przekazanego środowiska — skrypt tworzy `gym.make(--env-id)` już przy wczytywaniu modelu (i zamyka je przed nagrywaniem). **`--env-id` musi zgadzać się z treningiem** (np. `FetchReach-v4` do zipa z tego env).

**Minari + `info`:** bufor epizodu łączy `info` przez `jax.tree_map` — wszystkie kroki muszą mieć te same klucze. Fetch zwraca puste `info` przy `reset()`, a `is_success` dopiero w `step()`; skrypt owija środowisko w **`_MinariGoalInfoPadWrapper`**, który uzupełnia brakujące `is_success` (tylko gdy włączone nagrywanie `info`, domyślnie tak).

- **Wymagane**: po aktualizacji `requirements.txt` jest **`minari[hdf5,create]`** (JAX — bez tego zapis z `DataCollector` kończy się błędem).
- **`--dataset-id`**: opcjonalnie (domyślnie `l6/<env-slug>/expert-<sac|ppo>-v0`). Przy kolizji z istniejącym datasetem: **`--overwrite`**.
- **`--algo`**: `auto` (najpierw SAC, potem PPO), albo wymusz `sac` / `ppo`.
- **`--model`**: pełna ścieżka do pliku `.zip` (w PowerShell **`...` to dosłownie trzy kropki**, nie skrót — użyj prawdziwego katalogu np. `weights\2026-05-31_11-20_sac_her_fetch_suite\FetchReach-v4\sac_her_model.zip`).
- Przy błędzie SB3 typu **`use_sde` / multiple values** przy `SAC.load`, skrypt próbuje **patch `SAC._setup_model`** (naprawa `policy_class` / `policy_kwargs`), a jeśli to nie pomoże — **tymczasowy zip** po poprawce pól w `data`.

```powershell
python record_expert_minari.py --model weights\2026-05-31_11-20_sac_her_fetch_suite\FetchReach-v4\sac_her_model.zip --env-id FetchReach-v4 --n-episodes 800
# --overwrite
```

## Trening Decision Transformera (nanoDT)

Skrypt **[`train_dt_minari_fetch.py`](train_dt_minari_fetch.py)** ładuje lokalny dataset Minari (domyślnie `l6/fetchreach-v4/expert-sac-v0`), spłaszcza obserwacje **Dict** do wektora (ta sama kolejność kluczy co w manifeście: alfabetycznie), trenuje **nanoDT** (`NanoDTAgent`) i zapisuje wagę + manifest w **`dt_weights/<timestamp UTC>/`**.

- **Patch pętli treningowej**: plik [`nanodt_train_loop.py`](nanodt_train_loop.py) — `itertools.cycle(iter(DataLoader))` oraz `num_workers=0` na Windows (upstream nanoDT 0.1.0 potrafi rzucić `StopIteration` przy krótkich runach / eval).
- **Źródło API**: [nanoDT na PyPI](https://pypi.org/project/nanodt/), [repo](https://github.com/lubiluk/nanoDT), przykład [`examples/train_dt.py`](https://github.com/lubiluk/nanoDT/blob/main/examples/train_dt.py).

```powershell
python train_dt_minari_fetch.py --dataset-id l6/fetchreach-v4/expert-sac-v0 --max-iters 50000
python train_dt_minari_fetch.py --dataset-id l6/fetchreach-v4/expert-sac-v0 --max-iters 2000
```

### Early stopping i metryki online (opcjonalnie)

- **`--max-iters`**: twardy limit iteracji (jak wcześniej).
- **Loss offline** (jak w upstream `estimate_loss`): **`--early-stop-patience`** + **`--early-stop-loss-max`** — zatrzymanie po tylu kolejnych evalach (co `--eval-interval`), w których loss (wg `--early-stop-on val|train|both`) jest **≥** progu.
- **Plateau val loss**: **`--early-stop-plateau-epsilon`** i **`--early-stop-plateau-window`** (≥ 2) — jeśli w ostatnich `window` punktach eval `max(val) - min(val) < epsilon`, trening się kończy.
- **Krótki rollout w env** (wolniejszy trening): **`--online-eval-every-iters`** + **`--online-eval-episodes`** — najlepiej gdy okres jest współdzielny z `--eval-interval` (inaczej online eval uruchomi się tylko przy iteracjach spełniających oba moduły). RTG: **`--online-eval-target-return`** (domyślnie heurystyka ze średniego zwrotu w próbce epizodów z Minari).
- **Stop po sukcesie online**: **`--early-stop-success-min`** (np. `0.95`) — gdy `success_rate_final` z rolloutu ≥ próg.
- **Stop przy słabym sukcesie**: **`--early-stop-success-max`** + **`--early-stop-success-patience`** — gdy przez tyle kolejnych **online** evalów `success_rate_final` ≤ max.
- **Checkpointy pośrednie**: **`--save-every-iters`** — pliki `ckpt_iter_########.pth` w tym samym katalogu co `dt_model.pth` (format jak `NanoDTAgent.save`, plus pola `minari_dataset_id`, `env_id`, `flat_observation_keys`, `checkpoint_iter`, `checkpoint_kind`).

W **`manifest.json`** i w dopisanych polach **`dt_model.pth`** znajdziesz m.in. `early_stop`, `env_id`, `flat_observation_keys`, `minari_dataset_id`.

### Ewaluacja online DT + baseline i wykresy

1. **[`eval_dt_minari_fetch.py`](eval_dt_minari_fetch.py)** — rollout DT z poprawnym `act(..., rew=...)`; **baseline SB3 domyślnie szukany automatycznie**: zip z `minari_recordings/*/manifest.json` z tym samym `minari_dataset_id` co w manifeście treningu (najnowszy run), albo zmienna środowiskowa **`L6_EVAL_BASELINE_MODEL`**, albo **`--baseline-model`**. Wyłączenie: **`--no-baseline`**. Wynik: **`eval_metrics.json`** (np. w `dt_eval_runs/<timestamp>/`).
2. **[`plot_dt_vs_baseline.py`](plot_dt_vs_baseline.py)** — siatka małych wykresów (jedna metryka = jedna skala osi Y), żeby np. success 0–1 nie ginął przy mean_return ~ −50; domyślnie `figures/dt_vs_baseline.png` obok JSON.

```powershell
python eval_dt_minari_fetch.py --model dt_weights\...\dt_model.pth --manifest dt_weights\...\manifest.json `
  --baseline-model weights\...\FetchReach-v4\sac_her_model.zip --n-episodes 50
python plot_dt_vs_baseline.py --input dt_eval_runs\...\eval_metrics.json
```

**TensorBoard** (cały suite naraz):

```powershell
tensorboard --logdir weights/besties
```
