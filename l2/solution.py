from __future__ import annotations

import argparse
import collections
import csv
import json
import os
import random
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Literal

import numpy as np
from tqdm import tqdm

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

PLOTS_DIR = "plots"

# #region agent log
_AGENT_DEBUG_LOG = Path(__file__).resolve().parents[2] / "debug-cabcfa.log"
_BAD_PROB_LOG_BUDGET = 8


def _agent_debug_log(
    location: str,
    message: str,
    *,
    data: dict,
    hypothesis_id: str,
    run_id: str = "post-fix",
) -> None:
    try:
        line = json.dumps(
            {
                "sessionId": "cabcfa",
                "timestamp": int(time.time() * 1000),
                "location": location,
                "message": message,
                "data": data,
                "hypothesisId": hypothesis_id,
                "runId": run_id,
            },
            default=str,
        )
        with _AGENT_DEBUG_LOG.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
    except OSError:
        pass


# #endregion

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
        # T z Sutton–Barto: pierwszy indeks *po* końcu epizodu (terminal albo limit kroków).
        # Ustawiane na current_step + 1 w momencie obserwacji końca; ostatnia aktualizacja przy tau = T - 1.
        self.episode_T: int = ALMOST_INFINITE_STEP
        self.finished: bool = False
        self.states: dict[int, State] = {}
        self.actions: dict[int, Action] = {}
        self.rewards: dict[int, int] = {}

    def start_attempt(self, state: State) -> Action:
        self.current_step = 0
        self.states[self._access_index(self.current_step)] = state
        action = self._select_action(self._behavior_policy(state, available_actions(state)))
        self.actions[self._access_index(self.current_step)] = action
        self.episode_T = ALMOST_INFINITE_STEP
        self.finished = False
        return action

    def control(self, state: State, last_reward: int) -> Action:
        if self.episode_T == ALMOST_INFINITE_STEP or self.current_step < self.episode_T:
            self.rewards[self._access_index(self.current_step + 1)] = last_reward
            self.states[self._access_index(self.current_step + 1)] = state
            if self.episode_T == ALMOST_INFINITE_STEP and (
                last_reward == 0 or self.current_step == problem_module.MAX_LEARNING_STEPS
            ):
                self.episode_T = self.current_step + 1
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

        if self.episode_T != ALMOST_INFINITE_STEP and update_step == self.episode_T - 1:
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

    def _select_action(self, actions_distribution: dict[Action, float]) -> Action:
        actions = list(actions_distribution.keys())
        probabilities = list(actions_distribution.values())
        # #region agent log
        global _BAD_PROB_LOG_BUDGET
        p_arr = np.asarray(probabilities, dtype=np.float64)
        p_sum = float(np.nansum(p_arr))
        bad = (not np.all(np.isfinite(p_arr))) or (not np.isclose(p_sum, 1.0, rtol=0.0, atol=1e-8))
        if bad and _BAD_PROB_LOG_BUDGET > 0:
            _BAD_PROB_LOG_BUDGET -= 1
            _agent_debug_log(
                "solution.py:_select_action",
                "invalid p before np.random.choice",
                data={
                    "behavior_mode": self.behavior_mode,
                    "experiment_rate": float(self.experiment_rate),
                    "n_actions": len(actions),
                    "p_sum": p_sum,
                    "p_min": float(np.nanmin(p_arr)) if p_arr.size else None,
                    "p_max": float(np.nanmax(p_arr)) if p_arr.size else None,
                    "any_nan": bool(np.any(np.isnan(p_arr))),
                    "any_inf": bool(np.any(np.isinf(p_arr))),
                    "prob_sample": [float(x) for x in probabilities[:6]],
                },
                hypothesis_id="H1_H2_H3_H4",
            )
        # #endregion
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
        """L1-normalizacja nieujemnego wektora; przy sumie 0 lub NaN — rozkład jednostajny."""
        v = np.asarray(probabilities, dtype=np.float64).reshape(-1)
        n = int(v.size)
        if n == 0:
            return v
        s = float(np.nansum(v))
        if not np.isfinite(s) or s <= 0.0:
            return np.full(n, 1.0 / n)
        return v / s


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
    enable_drawing: bool = True,
    episode_step_progress: bool = False,
    show_episode_progress: bool = True,
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
        enable_drawing=enable_drawing,
        episode_step_progress=episode_step_progress,
        show_episode_progress=show_episode_progress,
    )
    experiment.run()
    penalties = list(experiment.penalties or [])
    return driver, penalties


def mean_last_window(penalties: list[int], window: int) -> float:
    if len(penalties) < window:
        return float(np.mean(penalties)) if penalties else 0.0
    return float(np.mean(penalties[-window:]))


def normalized_episode_penalty_cost(mean_penalty: float, max_learning_steps: int) -> float:
    """|średnia kara na epizod| / (|kara za krok| * max_kroków) = |średnia| / max_learning_steps.

    Kara za krok to -1, maksymalna możliwa suma (co do modułu) przy limicie kroków to max_learning_steps.
    Im mniejsza wartość, tym lepiej; na wykresie „im wyżej tym gorzej” użyj tej samej wielkości
    (dobre polityki leżą blisko 0, bardzo złe zbliżają się do 1).
    """
    return abs(float(mean_penalty)) / float(max_learning_steps)


PARAM_STUDY_CSV_FIELDS = [
    "alpha",
    "n",
    "max_learning_steps",
    "mean_penalty_last_window",
    "normalized_cost",
]


def _resolve_param_study_jobs(requested: int, num_tasks: int) -> int:
    if num_tasks <= 0:
        return 1
    if requested <= 0:
        cpu = os.cpu_count() or 1
        return max(1, min(cpu, num_tasks))
    return max(1, min(requested, num_tasks))


def _param_study_worker(payload: tuple) -> dict[str, float | int]:
    """Jedna para (n, α) — top-level pod multiprocessing (import w procesach potomnych)."""
    (
        corner,
        episodes,
        alpha,
        n,
        experiment_rate,
        discount,
        window,
        max_episode_steps,
    ) = payload
    _, penalties = train_off_policy(
        corner,
        episodes,
        step_size=alpha,
        step_no=n,
        experiment_rate=experiment_rate,
        discount=discount,
        enable_drawing=False,
        episode_step_progress=False,
        show_episode_progress=False,
    )
    mean_p = mean_last_window(penalties, window)
    return {
        "alpha": alpha,
        "n": n,
        "max_learning_steps": max_episode_steps,
        "mean_penalty_last_window": round(mean_p, 4),
        "normalized_cost": round(
            normalized_episode_penalty_cost(mean_p, max_episode_steps), 6
        ),
    }


def cmd_param_study(*, episode_step_progress: bool = False, jobs: int = 1) -> None:
    """Studium jak w instrukcji (corner_c): wpływ α i n (n-krokowy SARSA).

    Limit długości epizodu: `problem.MAX_LEARNING_STEPS` (bez nadpisywania globala).
    Zapisuje znormalizowany koszt: |średnia kara w ostatnim oknie| / MAX_LEARNING_STEPS.

    Bez rysowania PNG; każdy wiersz CSV dopisywany od razu (flush + fsync), żeby przy przerwaniu
    nie stracić wcześniejszych wyników. Ponowne uruchomienie dopisuje do istniejącego pliku —
    usuń CSV, jeśli chcesz świeży zestaw od zera.

    Opcjonalny pasek „Kroki w epizodzie”: `python solution.py param_study --episode-step-progress`
    (tylko przy `-j 1`).

    Równoległość: `-j N` lub `-j 0` (min(rdzenie, liczba zadań)). Jeden pasek w głównym procesie:
    „param_study (zadania)” = ile par (n, α) już domknięto. W workerach **brak** tqdm po epizodach
    / krokach (żeby nie mieszać wielu procesów w jednym terminalu).
    """



    corner = "corner_c"
    window = 100


    # n_list = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]

    # episodes = 1000
    # episodes = 450
    # episodes = 300


    # episodes = 600
    # n_list = [1, 2, 4, 8, 16]

    # episodes = 600
    # n_list = [32, 64, 128]

    # alphas = [round(i * 0.1, 1) for i in range(11)]
    # alphas = [0.1]


    corner = "corner_d"
    episodes = 600
    alphas = [0.5]
    n_list = [4]



    max_episode_steps = problem_module.MAX_LEARNING_STEPS
    experiment_rate = 0.05
    discount = 1.0

    tasks: list[tuple[int, float]] = [(n, a) for n in n_list for a in alphas]
    n_workers = _resolve_param_study_jobs(jobs, len(tasks))

    if n_workers > 1 and episode_step_progress:
        print(
            "Uwaga: --episode-step-progress jest ignorowane przy równoległym param_study (-j > 1)."
        )

    os.makedirs(PLOTS_DIR, exist_ok=True)
    out_path = os.path.join(PLOTS_DIR, "param_study_corner_c.csv")
    need_header = not os.path.exists(out_path) or os.path.getsize(out_path) == 0
    rows_written = 0
    with open(out_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=PARAM_STUDY_CSV_FIELDS)
        if need_header:
            w.writeheader()
            f.flush()
            os.fsync(f.fileno())
        if n_workers == 1:
            for n in tqdm(n_list, desc="n (kroków SARSA)"):
                for alpha in alphas:
                    _, penalties = train_off_policy(
                        corner,
                        episodes,
                        step_size=alpha,
                        step_no=n,
                        experiment_rate=experiment_rate,
                        discount=discount,
                        enable_drawing=False,
                        episode_step_progress=episode_step_progress,
                        show_episode_progress=True,
                    )
                    mean_p = mean_last_window(penalties, window)
                    row = {
                        "alpha": alpha,
                        "n": n,
                        "max_learning_steps": max_episode_steps,
                        "mean_penalty_last_window": round(mean_p, 4),
                        "normalized_cost": round(
                            normalized_episode_penalty_cost(mean_p, max_episode_steps), 6
                        ),
                    }
                    w.writerow(row)
                    f.flush()
                    os.fsync(f.fileno())
                    rows_written += 1
        else:
            payloads = [
                (
                    corner,
                    episodes,
                    alpha,
                    n,
                    experiment_rate,
                    discount,
                    window,
                    max_episode_steps,
                )
                for n, alpha in tasks
            ]
            with ProcessPoolExecutor(max_workers=n_workers) as pool:
                futures = [pool.submit(_param_study_worker, p) for p in payloads]
                for fut in tqdm(
                    as_completed(futures),
                    total=len(futures),
                    desc="param_study",
                    unit="zadanie",
                    dynamic_ncols=True,
                    mininterval=0.2,
                ):
                    row = fut.result()
                    w.writerow(row)
                    f.flush()
                    os.fsync(f.fileno())
                    rows_written += 1
    print(f"Dopisano {rows_written} wierszy do {out_path}")
    if rows_written > 0:
        import utils as utils_module

        png_path = os.path.join(PLOTS_DIR, "param_study_corner_c.png")
        utils_module.plot_param_study_n_alpha(out_path, png_path)
        print(f"Wykres: {png_path}")


def cmd_compare_push_and_is() -> None:
    """corner_c: ε-greedy vs pędzący agent; z IS i bez IS (pędzący)."""
    corner = "corner_c"
    episodes = 800
    window = 100
    configs: list[tuple[str, BehaviorMode, bool, float]] = [
        ("epsilon_is", "epsilon_greedy", True, 0.0),
        ("push_is", "push_forward", True, 0.5),
        ("push_no_is", "push_forward", False, 0.5),
    ]
    out_path = os.path.join(PLOTS_DIR, "compare_behavior_is.csv")
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
    os.makedirs(PLOTS_DIR, exist_ok=True)
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
            "param_study_plot",
            "compare_push_is",
        ],
    )
    parser.add_argument(
        "--episode-step-progress",
        action="store_true",
        help=(
            "Tylko dla param_study: włącz wewnętrzny pasek tqdm "
            "(„Kroki w epizodzie …”). Domyślnie wyłączony."
        ),
    )
    parser.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=1,
        metavar="N",
        help=(
            "param_study: liczba równoległych procesów (1 = sekwencyjnie). "
            "0 = min(rdzenie CPU, liczba par n×α). "
            "Pasek postępu: ukończone zadania (n,α); bez tqdm epizodów w workerach."
        ),
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
        cmd_param_study(
            episode_step_progress=args.episode_step_progress,
            jobs=args.jobs,
        )
        return

    if args.mode == "param_study_plot":
        import utils as utils_module

        csv_path = os.path.join(PLOTS_DIR, "param_study_corner_c.csv")
        png_path = os.path.join(PLOTS_DIR, "param_study_corner_c.png")
        utils_module.plot_param_study_n_alpha(csv_path, png_path)
        print(f"Zapisano {png_path}")
        return

    if args.mode == "compare_push_is":
        cmd_compare_push_and_is()
        return


if __name__ == "__main__":
    main()
