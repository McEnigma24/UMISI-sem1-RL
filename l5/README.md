# SAC + HER Implementation Exercise

## Overview

This exercise focuses on implementing **Hindsight Experience Replay (HER)** and **Soft Actor-Critic (SAC)** with automatic alpha adjustment. These are two powerful techniques in reinforcement learning that, when combined, can significantly enhance the performance of agents in complex environments.

## Soft Actor-Critic (SAC)

SAC is an off-policy reinforcement learning algorithm that optimizes a stochastic policy in an entropy-regularized framework. The key idea is to balance exploration and exploitation by maximizing a trade-off between expected reward and policy entropy. This results in a more robust and stable learning process.

### Key Features of SAC:
- **Entropy Regularization**: Encourages exploration by adding an entropy term to the objective.
- **Automatic Alpha Adjustment**: Dynamically tunes the entropy coefficient to balance exploration and exploitation.
- **Stability**: Uses a soft Q-function and target networks for stable learning.

For more details, refer to the original SAC paper: ["Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor"](https://arxiv.org/abs/1801.01290).

Automatic alpha adjustement comes from this paper: ["Soft Actor-Critic Algorithms and Applications"](https://arxiv.org/abs/1812.05905)

## Hindsight Experience Replay (HER)

HER is a technique designed to improve sample efficiency in sparse-reward environments. The main idea is to treat failed attempts as successes by redefining the goal during replay. This allows the agent to learn from trajectories that would otherwise be discarded.

### Key Features of HER:
- **Goal Relabeling**: Modifies the goal in past experiences to make the trajectory appear successful.
- **Improved Learning**: Helps the agent learn even in environments with sparse or delayed rewards.

For more details, refer to the original HER paper: ["Hindsight Experience Replay"](https://arxiv.org/abs/1707.01495).

## Your Task

1. **Implement Hindsight Experience Replay (HER).** (8 pts):
    - Modify the replay buffer to support goal relabeling.
    - Ensure that the agent can learn from both original and relabeled goals.

2. **Implement Automatic Alpha Adjustment in SAC.** (4 pts):
    - Dynamically tune the entropy coefficient during training.
    - Ensure the agent balances exploration and exploitation effectively.

## Getting Started

1. Set up the environment — **wybierz jedną** z opcji:

    ### A) Docker (bez venva na hoście)

    Zależności są w obrazie; katalog `l5/` (kod, `runs/`, `weights/`) montujesz do kontenera.

    ```bash
    cd l5
    chmod +x docker/enter_dev_container.sh start.sh   # raz, jeśli trzeba
    ./docker/enter_dev_container.sh
    ```

    W środku (w `/workspace`, czyli Twoim `l5/`):

    ```bash
    ./start.sh
    # np. wczytanie runu: ./start.sh --load weights/2026-05-24_14-30
    ```

    `start.sh` owija trening w `xvfb-run`, bo na końcu `train.py` używa `render_mode="human"`. Bez wirtualnego framebuffera: `SKIP_XVFB=1 ./start.sh`.

    TensorBoard w tym samym kontenerze: `tensorboard --logdir runs --bind_all`, w przeglądarce `http://localhost:6006` (port jest wystawiony w `enter_dev_container.sh`).

    Wersja Pythona jest tylko w ``l5/config`` (``PYTAG``); ``Dockerfile`` jej nie pinuje — ``docker/enter_dev_container.sh`` przekazuje ``--build-arg PYTAG=…``. Ręczny ``docker build`` bez tego argumentu zakończy się komunikatem błędu z Dockerfile.

    Inna nazwa obrazu: ustaw w ``l5/config`` ``DOCKER_IMAGE_BASE`` i ``PYTAG`` (domyślnie ``rl-l5`` + ``-3.12`` → ``rl-l5-3.12``). Pełna nadpisanie: ``DOCKER_IMAGE=moje:tag ./docker/enter_dev_container.sh``.

    Plik ``l5/config`` sam ustawia ``DOCKER_GPUS`` i ``DOCKER_RUN_FLAGS``: jeśli działa ``nvidia-smi`` **oraz** krótki test ``docker run --gpus all busybox true``, włączane jest ``--gpus all`` (pierwszy raz może pobrać obraz ``busybox``). Inaczej kontener startuje bez GPU. Pełne GPU wymaga [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) na hoście, z którego wołasz Dockera.

    ### B) Lokalnie z pip

    ```bash
    pip install -r requirements.txt
    ```

    **Windows + Python 3.13:** older pins (`mujoco` 3.1.x with `gymnasium-robotics` 1.3.x) install MuJoCo from source and fail unless `MUJOCO_PATH` is set. The versions in `requirements.txt` use a MuJoCo build that ships Windows wheels for 3.13.

    **TensorBoard on Python 3.13:** the standard library module `imghdr` was removed; `requirements.txt` pulls in `standard-imghdr` so `tensorboard --logdir runs` keeps working.

    Prefer a **virtual environment** for this lab (`python -m venv .venv` then activate it) so upgrading `gymnasium` does not conflict with other tools in your base conda env (for example `stable-baselines3` pins an older `gymnasium`).

    **`report.ipynb` (Cursor / VS Code):** the notebook needs the **`l5/.venv`** interpreter. Install deps in that venv, then run **`l5/register_jupyter_kernel.ps1`** once and pick kernel **Python (UMISI l5 .venv)**, or use **Enter interpreter path** → `l5\.venv\Scripts\python.exe`. If you open the repo as workspace **`WIN_sem_1`**, `.vscode/settings.json` already suggests that interpreter.

2. Train and evaluate with `train.py`:
    - **Training** (default): `python train.py` — writes a run under `weights/<YYYY-mm-dd_HH-MM>/` with `metadata.json` (full hyperparameter signature) and `policy.pt`.
    - **Load + render test only**: `python train.py --load weights/<that-folder>` — aborts unless `metadata.json` matches the current hyperparameters in `train.py`.

3. **Sweep entropy coefficient α** (optional, for comparing many fixed values in TensorBoard and in `report.ipynb`):

    From `l5/` (with the same venv / deps as for `train.py`):

    ```bash
    python train_alpha_sweep.py --dry-run    # print planned TB paths only
    python train_alpha_sweep.py              # seven fixed α + final run with alpha="auto"
    python train_alpha_sweep.py --skip-auto  # only the seven fixed α (no auto run)
    ```

    Each run logs to `runs/<sweep_id>/alpha_<...>/` and saves weights under `weights/<sweep_id>/<timestamp>/`. `metadata.json` includes `tensorboard_log_dir` (path relative to `l5/`) so the notebook can pair logs without guessing. A growing `weights/<sweep_id>/manifest.json` lists each finished run.

    **TensorBoard:** `tensorboard --logdir runs/<sweep_id>` (or `--logdir runs` to see all sweeps).

    **Cost:** each run uses `TRAIN_N_STEPS` from `train.py` (default 100_000 env steps). The default sweep is **eight** trainings (seven fixed α + one `auto`) ≈ eight times the wall-clock of one training.

    **Loading a specific sweep checkpoint** (`train.py --load` or `load_sac_from_run_dir` in the notebook): `train.py` must still match the run’s full signature — set `SAC_KWARGS["alpha"]` (and any other differing fields) in `train.py` to the values stored in that run’s `metadata.json` before loading.

4. Review the provided codebase and identify where HER and automatic alpha adjustment need to be implemented.

5. Test your implementation on the provided environments and analyze the results.

## Submission

Submit your completed implementation along with a brief report discussing:
- The challenges you faced.
- The results you obtained.
- Any insights or observations.

Good luck, and happy coding!