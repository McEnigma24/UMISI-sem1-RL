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

Skrypt **[`train_expert_sac_her_fetch.py`](train_expert_sac_her_fetch.py)** — **SAC** z **`HerReplayBuffer`** (Hindsight Experience Replay), **`MultiInputPolicy`**. To setup zbliżony do benchmarków z literatury dla środowisk goal-conditioned (lepszy niż samo PPO na sparse).

- **Domyślnie** trenuje **po kolei** na czterech środowiskach **sparse v4**: `FetchReach-v4`, `FetchPush-v4`, `FetchSlide-v4`, `FetchPickAndPlace-v4` (osobny folder / checkpoint na env).
- **TensorBoard**: dla każdego env logi w `.../<env>/tensorboard/` — wbudowane metryki SAC (straty, `ent_coef`, Q itd.) + prefix **`fetch/`** (zwrot i długość epizodu, `is_success`, średnie kroczące).
- Zapis: `weights/<timestamp>_sac_her_fetch_suite/<EnvId>/` (`sac_her_model.zip`, `manifest.json`, `best/`) oraz zbiorczy **`manifest_suite.json`**.
- **Wczesny stop**: po `min_steps_before_early_stop` (domyślnie 150k) co `early_stop_check_freq` kroków — `rollout_eval`; warunek `success_rate_final >= próg`. **Domyślnie próg jest inny dla każdego Fetch** (stała `DEFAULT_EARLY_STOP_SUCCESS_THRESHOLD_BY_ENV` w skrypcie): Reach **0.93**, Push **0.78**, Slide **0.66**, PickAndPlace **0.62** (amatorski poziom vs. często ~0.9+ w mocnych runach). Dla nieznanego `--env-id` używany jest **`--early-stop-threshold-fallback`** (domyślnie **0.75**). **`--success-threshold P`** ustawia **jeden** próg dla wszystkich envów (nadpisuje tabelę). Pełny budżet bez progu: `--no-early-stop`. W `manifest.json` → `training.early_stop_success_threshold_resolved` i `..._source`.

```powershell
python train_expert_sac_her_fetch.py
python train_expert_sac_her_fetch.py --env-id FetchPickAndPlace-v4
python train_expert_sac_her_fetch.py --timesteps 5_000_000
python train_expert_sac_her_fetch.py --success-threshold 0.85
python train_expert_sac_her_fetch.py --early-stop-threshold-fallback 0.7
python train_expert_sac_her_fetch.py --no-early-stop
python train_expert_sac_her_fetch.py --check-device
```

Domyślnie **3_000_000** kroków **na każde** środowisko w suite (łącznie ~12M kroków na cztery Fetch — długo, ale sensownie pod „eksperta”), o ile wczesny stop nie skróci danego env. Zawsze zweryfikuj `--eval-only` / `success_*` w `manifest.json`.

Podgląd logów (cały suite naraz):

```powershell
tensorboard --logdir weights\<data>_sac_her_fetch_suite
```
