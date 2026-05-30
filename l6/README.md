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
```

Opcjonalnie ten sam efekt przez skrypt (sprawdza 3.12 przed instalacją):

```powershell
python install_deps.py
# lub: .\install_deps.ps1
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
