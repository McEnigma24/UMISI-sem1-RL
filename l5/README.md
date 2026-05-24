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

    W katalogu `l5` zależności są w obrazie; kod i artefakty (`runs/`, `weights/`) zostają na hoście dzięki montowaniu katalogu (skrypty w `docker/`, analogicznie do innych projektów z repo).

    ```bash
    cd l5
    chmod +x docker/*.sh   # raz, jeśli pliki nie mają bitu wykonywania
    ./docker/build.sh
    ./docker/run.sh
    ```

    Domyślnie `./docker/run.sh` uruchamia `python train.py`. Pod spodem jest `xvfb-run` (entrypoint obrazu), bo na końcu treningu `train.py` odpala `render_mode="human"`.

    Wczytanie runu i test z renderem:

    ```bash
    ./docker/run.sh python train.py --load weights/2026-05-24_14-30
    ```

    TensorBoard (logi w `./runs` na hoście, przeglądarka: `http://localhost:6006`):

    ```bash
    ./docker/tensorboard.sh
    ```

    Powłoka w kontenerze (bez entrypointu z xvfb):

    ```bash
    ./docker/shell.sh
    ```

    Inna nazwa obrazu (np. własny tag):

    ```bash
    DOCKER_IMAGE=moj-l5:dev ./docker/build.sh
    DOCKER_IMAGE=moj-l5:dev ./docker/run.sh python train.py
    ```

    Wyłączenie `xvfb-run` (np. własny `DISPLAY`):

    ```bash
    SKIP_XVFB=1 ./docker/run.sh python -c "import sys; print(sys.version)"
    ```

    ### B) Lokalnie z pip

    ```bash
    pip install -r requirements.txt
    ```

    **Windows + Python 3.13:** older pins (`mujoco` 3.1.x with `gymnasium-robotics` 1.3.x) install MuJoCo from source and fail unless `MUJOCO_PATH` is set. The versions in `requirements.txt` use a MuJoCo build that ships Windows wheels for 3.13.

    **TensorBoard on Python 3.13:** the standard library module `imghdr` was removed; `requirements.txt` pulls in `standard-imghdr` so `tensorboard --logdir runs` keeps working.

    Prefer a **virtual environment** for this lab (`python -m venv .venv` then activate it) so upgrading `gymnasium` does not conflict with other tools in your base conda env (for example `stable-baselines3` pins an older `gymnasium`).

2. Train and evaluate with `train.py`:
    - **Training** (default): `python train.py` — writes a run under `weights/<YYYY-mm-dd_HH-MM>/` with `metadata.json` (full hyperparameter signature) and `policy.pt`.
    - **Load + render test only**: `python train.py --load weights/<that-folder>` — aborts unless `metadata.json` matches the current hyperparameters in `train.py`.

3. Review the provided codebase and identify where HER and automatic alpha adjustment need to be implemented.

4. Test your implementation on the provided environments and analyze the results.

## Submission

Submit your completed implementation along with a brief report discussing:
- The challenges you faced.
- The results you obtained.
- Any insights or observations.

Good luck, and happy coding!