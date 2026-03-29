from __future__ import annotations

import argparse
import collections
import csv
import random
from typing import Literal

import numpy as np
import sklearn.preprocessing as skl_preprocessing

import problem as problem_module
from problem import (
    Action,
    available_actions,
    Corner,
    Driver,
    Experiment,
    Environment,
    State,
)

ALMOST_INFINITE_STEP = 100000

BehaviorMode = Literal["epsilon_greedy", "push_forward"]


class RandomDriver(Driver):
    def __init__(self) -> None:
        self.current_step: int = 0

    def start_attempt(self, state: State) -> Action:
        self.current_step = 0
        return random.choice(available_actions(state))

    def control(self, state: State, last_reward: int) -> Action:
        self.current_step += 1
        return random.choice(available_actions(state))

    def finished_learning(self) -> bool:
        return self.current_step > problem_module.MAX_LEARNING_STEPS


class GreedyPolicyDriver(Driver):
    """Rollout z polityką zachłanną względem Q (bez uczenia)."""

    def __init__(
        self,
        q: dict[tuple[State, Action], float],
        max_steps: int | None = None,
    ) -> None:
        self.q = q
        self.max_steps = max_steps if max_steps is not None else problem_module.MAX_LEARNING_STEPS
        self.current_step: int = 0
        self.done: bool = False

    def start_attempt(self, state: State) -> Action:
        self.current_step = 0
        self.done = False
        return self._greedy_action(state)

    def control(self, state: State, last_reward: int) -> Action:
        self.current_step += 1
        if last_reward == 0 or self.current_step >= self.max_steps:
            self.done = True
            return Action(0, 0)
        return self._greedy_action(state)

    def _greedy_action(self, state: State) -> Action:
        acts = available_actions(state)
        values = [self.q[state, a] for a in acts]
        m = max(values)
        best = [a for a, v in zip(acts, values) if v == m]
        return random.choice(best)

    def finished_learning(self) -> bool:
        return self.done


class OffPolicyNStepSarsaDriver(Driver):
    def __init__(
        self,
        step_size: float,
        step_no: int,
        experiment_rate: float,
        discount_factor: float,
        *,
        use_importance_sampling: bool = True,
        behavior_mode: BehaviorMode = "epsilon_greedy",
        push_forward_bias: float = 0.5,
    ) -> None:
        self.step_size: float = step_size
        self.step_no: int = step_no
        self.experiment_rate: float = experiment_rate
        self.discount_factor: float = discount_factor
        self.use_importance_sampling: bool = use_importance_sampling
        self.behavior_mode: BehaviorMode = behavior_mode
        self.push_forward_bias: float = push_forward_bias

        self.q: dict[tuple[State, Action], float] = collections.defaultdict(float)
        self.current_step: int = 0
        self.final_step: int = ALMOST_INFINITE_STEP
        self.finished: bool = False
        self.states: dict[int, State] = {}
        self.actions: dict[int, Action] = {}
        self.rewards: dict[int, int] = {}

    def start_attempt(self, state: State) -> Action:
        self.current_step = 0
        self.states[self._access_index(self.current_step)] = state
        action = self._select_action(self._behavior_policy(state, available_actions(state)))
        self.actions[self._access_index(self.current_step)] = action
        self.final_step = ALMOST_INFINITE_STEP
        self.finished = False
        return action

    def control(self, state: State, last_reward: int) -> Action:
        if self.current_step < self.final_step:
            self.rewards[self._access_index(self.current_step + 1)] = last_reward
            self.states[self._access_index(self.current_step + 1)] = state
            if self.final_step == ALMOST_INFINITE_STEP and (
                last_reward == 0 or self.current_step == problem_module.MAX_LEARNING_STEPS
            ):
                self.final_step = self.current_step
            action = self._select_action(self._behavior_policy(state, available_actions(state)))
            self.actions[self._access_index(self.current_step + 1)] = action
        else:
            action = Action(0, 0)

        update_step = self.current_step - self.step_no + 1
        if update_step >= 0:
            return_value_weight = self._return_value_weight(update_step)
            return_value = self._return_value(update_step)
            state_t = self.states[self._access_index(update_step)]
            action_t = self.actions[self._access_index(update_step)]
            self.q[state_t, action_t] += (
                self.step_size * return_value_weight * (return_value - self.q[state_t, action_t])
            )

        if update_step == self.final_step - 1:
            self.finished = True

        self.current_step += 1
        return action

    def _return_value(self, update_step: int) -> float:
        tau = update_step
        gamma = self.discount_factor
        n = self.step_no
        g = 0.0
        for i in range(1, n + 1):
            r = float(self.rewards[self._access_index(tau + i)])
            g += (gamma ** (i - 1)) * r
            if r == 0.0:
                return g
        s_h = self.states[self._access_index(tau + n)]
        a_h = self.actions[self._access_index(tau + n)]
        g += (gamma**n) * self.q[s_h, a_h]
        return g

    def _return_value_weight(self, update_step: int) -> float:
        if not self.use_importance_sampling:
            return 1.0
        tau = update_step
        rho = 1.0
        for k in range(tau + 1, tau + self.step_no):
            sk = self.states[self._access_index(k)]
            ak = self.actions[self._access_index(k)]
            acts = available_actions(sk)
            pi = self.greedy_policy(sk, acts)
            b = self._behavior_policy(sk, acts)
            rho *= pi[ak] / b[ak]
        return rho

    def finished_learning(self) -> bool:
        return self.finished

    def _access_index(self, index: int) -> int:
        return index % (self.step_no + 1)

    @staticmethod
    def _select_action(actions_distribution: dict[Action, float]) -> Action:
        actions = list(actions_distribution.keys())
        probabilities = list(actions_distribution.values())
        i = np.random.choice(list(range(len(actions))), p=probabilities)
        return actions[i]

    def _behavior_policy(self, state: State, actions: list[Action]) -> dict[Action, float]:
        if self.behavior_mode == "epsilon_greedy":
            return self.epsilon_greedy_policy(state, actions)
        return self.push_forward_policy(state, actions)

    def epsilon_greedy_policy(self, state: State, actions: list[Action]) -> dict[Action, float]:
        n_a = len(actions)
        eps = self.experiment_rate
        greedy = self._greedy_probabilities(state, actions)
        probs = eps / n_a + (1.0 - eps) * greedy
        return {a: float(p) for a, p in zip(actions, probs)}

    def push_forward_policy(self, state: State, actions: list[Action]) -> dict[Action, float]:
        """
        Polityka behawioralna: mieszanka „pędzenia” (preferuj max a_x + a_y)
        oraz ε-zachłannej — wszystkie akcje mają dodatnie p (ważne dla IS).
        """
        p = self.push_forward_bias
        accel_scores = np.array([float(a.a_x + a.a_y) for a in actions])
        max_s = float(np.max(accel_scores))
        push_mask = (accel_scores == max_s).astype(float)
        push = self._normalise(push_mask)
        greedy_mix = np.array(list(self.epsilon_greedy_policy(state, actions).values()))
        probs = p * push + (1.0 - p) * greedy_mix
        s = probs.sum()
        if s > 0:
            probs = probs / s
        else:
            probs = self._random_probabilities(actions)
        return {a: float(probs[i]) for i, a in enumerate(actions)}

    def greedy_policy(self, state: State, actions: list[Action]) -> dict[Action, float]:
        probabilities = self._greedy_probabilities(state, actions)
        return {action: float(probability) for action, probability in zip(actions, probabilities)}

    def _greedy_probabilities(self, state: State, actions: list[Action]) -> np.ndarray:
        values = [self.q[state, action] for action in actions]
        maximal_spots = (values == np.max(values)).astype(float)
        return self._normalise(maximal_spots)

    @staticmethod
    def _random_probabilities(actions: list[Action]) -> np.ndarray:
        maximal_spots = np.array([1.0 for _ in actions])
        return OffPolicyNStepSarsaDriver._normalise(maximal_spots)

    @staticmethod
    def _normalise(probabilities: np.ndarray) -> np.ndarray:
        return skl_preprocessing.normalize(probabilities.reshape(1, -1), norm="l1")[0]


def run_greedy_rollouts(
    environment: Environment,
    q: dict[tuple[State, Action], float],
    n_episodes: int,
    episode_label_offset: int,
) -> None:
    """Rysuje trasy dla zachłannej polityki względem nauczonego Q."""
    greedy_driver = GreedyPolicyDriver(q)
    eval_exp = Experiment(
        environment=environment,
        driver=greedy_driver,
        number_of_episodes=n_episodes,
        current_episode_no=episode_label_offset,
        draw_every_episode=True,
    )
    eval_exp.run()


def train_off_policy(
    corner_name: str,
    episodes: int,
    step_size: float,
    step_no: int,
    experiment_rate: float,
    discount: float,
    *,
    use_importance_sampling: bool = True,
    behavior_mode: BehaviorMode = "epsilon_greedy",
    push_forward_bias: float = 0.5,
    steering_fail_chance: float = 0.01,
) -> tuple[OffPolicyNStepSarsaDriver, list[int]]:
    driver = OffPolicyNStepSarsaDriver(
        step_size=step_size,
        step_no=step_no,
        experiment_rate=experiment_rate,
        discount_factor=discount,
        use_importance_sampling=use_importance_sampling,
        behavior_mode=behavior_mode,
        push_forward_bias=push_forward_bias,
    )
    experiment = Experiment(
        environment=Environment(
            corner=Corner(name=corner_name),
            steering_fail_chance=steering_fail_chance,
        ),
        driver=driver,
        number_of_episodes=episodes,
    )
    experiment.run()
    penalties = list(experiment.penalties or [])
    return driver, penalties


def mean_last_window(penalties: list[int], window: int) -> float:
    if len(penalties) < window:
        return float(np.mean(penalties)) if penalties else 0.0
    return float(np.mean(penalties[-window:]))


def cmd_param_study() -> None:
    corner = "corner_c"
    episodes = 600
    window = 100
    alphas = [0.1, 0.2, 0.3, 0.5]
    max_steps_list = [300, 500, 800]

    out_path = "plots/param_study_corner_c.csv"
    rows: list[dict[str, float | int | str]] = []
    old_max = problem_module.MAX_LEARNING_STEPS
    try:
        for alpha in alphas:
            for max_steps in max_steps_list:
                problem_module.MAX_LEARNING_STEPS = max_steps
                _, penalties = train_off_policy(
                    corner,
                    episodes,
                    step_size=alpha,
                    step_no=5,
                    experiment_rate=0.05,
                    discount=1.0,
                )
                rows.append(
                    {
                        "alpha": alpha,
                        "max_learning_steps": max_steps,
                        "mean_penalty_last_window": mean_last_window(penalties, window),
                    }
                )
    finally:
        problem_module.MAX_LEARNING_STEPS = old_max

    import os

    os.makedirs("plots", exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["alpha", "max_learning_steps", "mean_penalty_last_window"])
        w.writeheader()
        w.writerows(rows)
    print(f"Zapisano {out_path}")


def cmd_compare_push_and_is() -> None:
    """corner_c: ε-greedy vs pędzący agent; z IS i bez IS (pędzący)."""
    corner = "corner_c"
    episodes = 800
    window = 100
    problem_module.MAX_LEARNING_STEPS = 500
    configs: list[tuple[str, BehaviorMode, bool, float]] = [
        ("epsilon_is", "epsilon_greedy", True, 0.0),
        ("push_is", "push_forward", True, 0.5),
        ("push_no_is", "push_forward", False, 0.5),
    ]
    out_path = "plots/compare_behavior_is.csv"
    rows: list[dict[str, object]] = []
    for name, mode, use_is, pfb in configs:
        _, penalties = train_off_policy(
            corner,
            episodes,
            step_size=0.3,
            step_no=5,
            experiment_rate=0.05,
            discount=1.0,
            use_importance_sampling=use_is,
            behavior_mode=mode,
            push_forward_bias=pfb,
        )
        rows.append(
            {
                "name": name,
                "mean_penalty_last_window": mean_last_window(penalties, window),
            }
        )
    import os

    os.makedirs("plots", exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["name", "mean_penalty_last_window"])
        w.writeheader()
        w.writerows(rows)
    print(f"Zapisano {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="RL L3 — off-policy n-step SARSA")
    parser.add_argument(
        "mode",
        nargs="?",
        default="random",
        choices=[
            "random",
            "train_b",
            "train_c",
            "train_d",
            "greedy_viz",
            "param_study",
            "compare_push_is",
        ],
    )
    args = parser.parse_args()

    if args.mode == "random":
        Experiment(
            environment=Environment(
                corner=Corner(name="corner_b"),
                steering_fail_chance=0.01,
            ),
            driver=RandomDriver(),
            number_of_episodes=100,
        ).run()
        return

    if args.mode == "train_b":
        driver, _ = train_off_policy(
            "corner_b",
            30000,
            step_size=0.3,
            step_no=5,
            experiment_rate=0.05,
            discount=1.0,
        )
        env = Environment(corner=Corner(name="corner_b"), steering_fail_chance=0.01)
        run_greedy_rollouts(env, driver.q, n_episodes=5, episode_label_offset=30001)
        return

    if args.mode == "train_c":
        train_off_policy(
            "corner_c",
            8000,
            step_size=0.3,
            step_no=5,
            experiment_rate=0.05,
            discount=1.0,
        )
        return

    if args.mode == "train_d":
        train_off_policy(
            "corner_d",
            40000,
            step_size=0.3,
            step_no=5,
            experiment_rate=0.05,
            discount=1.0,
        )
        return

    if args.mode == "greedy_viz":
        driver, _ = train_off_policy(
            "corner_b",
            5000,
            step_size=0.3,
            step_no=5,
            experiment_rate=0.05,
            discount=1.0,
        )
        env = Environment(corner=Corner(name="corner_b"), steering_fail_chance=0.01)
        run_greedy_rollouts(env, driver.q, n_episodes=5, episode_label_offset=5001)
        return

    if args.mode == "param_study":
        cmd_param_study()
        return

    if args.mode == "compare_push_is":
        cmd_compare_push_and_is()
        return


if __name__ == "__main__":
    main()
