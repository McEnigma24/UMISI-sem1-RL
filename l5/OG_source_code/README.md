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

1. Set up the environment:
    ```bash
    pip install -r requirements.txt
    ```
    Or any other way of doing the equivalent.

2. Review the provided codebase and identify where HER and automatic alpha adjustment need to be implemented.

3. Test your implementation on the provided environments and analyze the results.

## Submission

Submit your completed implementation along with a brief report discussing:
- The challenges you faced.
- The results you obtained.
- Any insights or observations.

Good luck, and happy coding!