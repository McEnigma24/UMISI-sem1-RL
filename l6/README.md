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
- reszta pakietów (gymnasium, SB3, minari, nanodt itd.) i tak pobierana jest głównie z **PyPI**.

```powershell
.\.venv\Scripts\Activate.ps1
uv pip install -r requirements.txt
uv cache clean

uv pip install torch --index-url https://download.pytorch.org/whl/cu126
```

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

Podgląd TensorBoard (cały zestaw pod `besties/`):

```powershell
tensorboard --logdir weights/besties
```
