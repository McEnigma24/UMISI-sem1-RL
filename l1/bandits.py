import matplotlib.pyplot as plt
import numpy as np

from abc import abstractmethod
from itertools import accumulate
import random
from typing import Protocol


class KArmedBandit(Protocol):
    @abstractmethod
    def arms(self) -> list[str]:
        raise NotImplementedError

    @abstractmethod
    def reward(self, arm: str) -> float:
        raise NotImplementedError


class BanditLearner(Protocol):
    name: str
    color: str

    @abstractmethod
    def reset(self, arms: list[str], time_steps: int):
        raise NotImplementedError

    @abstractmethod
    def pick_arm(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def acknowledge_reward(self, arm: str, reward: float) -> None:
        pass


class BanditProblem:
    def __init__(self, time_steps: int, bandit: KArmedBandit, learner: BanditLearner):
        self.time_steps: int = time_steps
        self.bandit: KArmedBandit = bandit
        self.learner: BanditLearner = learner
        self.learner.reset(self.bandit.arms(), self.time_steps)

    def run(self) -> list[float]:
        rewards = []
        for _ in range(self.time_steps):
            arm = self.learner.pick_arm()
            reward = self.bandit.reward(arm)
            self.learner.acknowledge_reward(arm, reward)
            rewards.append(reward)
        return rewards


POTENTIAL_HITS = {
    "In Praise of Dreams": 0.8,
    "We Built This City": 0.9,
    "April Showers": 0.5,
    "Twenty Four Hours": 0.3,
    "Dirge for November": 0.1,
}



class TopHitBandit(KArmedBandit):
    def __init__(self, potential_hits: dict[str, float]):
        self.potential_hits: dict[str, float] = potential_hits

    def arms(self) -> list[str]:
        return list(self.potential_hits)

    def reward(self, arm: str) -> float:
        thumb_up_probability = self.potential_hits[arm]
        return 1.0 if random.random() <= thumb_up_probability else 0.0

    def max_expected_reward(self) -> float:
        """Wartość oczekiwana nagrody przy zawsze optymalnej decyzji (do obliczania regretu)."""
        return max(self.potential_hits.values()) if self.potential_hits else 0.0


def make_random_bandit(k: int, rng: np.random.Generator | None = None) -> TopHitBandit:
    """Tworzy bandytę z k ramionami i losowymi p (z Unif(0,1)) – do eksperymentów z losowymi problemami."""
    rng = rng or np.random.default_rng()
    probs = {str(i): float(rng.uniform(0, 1)) for i in range(k)}
    return TopHitBandit(probs)


class NonStationaryTopHitBandit(KArmedBandit):
    """
    Bandyta niestacjonarny: dryf preferencji – z każdym krokiem prawdopodobieństwa
    zmieniają się o niewielką wartość (błądzenie losowe, np. N(0, drift_std)),
    obcięte do [0, 1].
    """

    def __init__(self, initial_probs: dict[str, float], drift_std: float = 0.01, rng: np.random.Generator | None = None):
        self.potential_hits = dict(initial_probs)
        self.drift_std = drift_std
        self._rng = rng or np.random.default_rng()

    def arms(self) -> list[str]:
        return list(self.potential_hits)

    def _drift(self) -> None:
        """Jedno kroki błądzenia: każda p zmienia się o N(0, drift_std), clip [0,1]."""
        for arm in self.potential_hits:
            delta = float(self._rng.normal(0, self.drift_std))
            self.potential_hits[arm] = np.clip(self.potential_hits[arm] + delta, 0.0, 1.0)

    def reward(self, arm: str) -> float:
        self._drift()
        p = self.potential_hits[arm]
        return 1.0 if self._rng.random() <= p else 0.0



class RandomLearner(BanditLearner):
    def __init__(self):
        self.name = "Random"
        self.color = "black"
        self.arms: list[str] = []


    def reset(self, arms: list[str], time_steps: int):
        self.arms = arms

    def pick_arm(self) -> str:
        return random.choice(self.arms)

    def acknowledge_reward(self, arm: str, reward: float) -> None:
        pass

class ExploreThenCommitLearner(BanditLearner):
    def __init__(self, m: int = 100, color: str = "blue"):
        self.name = f"ExploreThenCommit {m}"
        self.color = color
        self.arms: list[str] = []
        self.sum_rewards: dict[str, float] = {}

        self.t = 0
        self.commit_arm = None

        self.m = m      # explore steps
        self.k = 0      # number of arms

    def reset(self, arms: list[str], time_steps: int):
        self.arms = arms
        self.sum_rewards = {arm: 0.0 for arm in arms}

        self.t = 0
        self.commit_arm = None

        self.m = self.m
        self.k = len(arms)

    def pick_arm(self) -> str:
        self.t += 1

        if self.t < (self.m * self.k):
            return self.arms[(self.t - 1) % self.k]
        elif self.t == (self.m * self.k):
            # robimy średnią dla każdego arm
            average_rewards = {arm: self.sum_rewards[arm] / self.m for arm in self.arms}
            self.commit_arm = max(average_rewards, key=average_rewards.get)
            return self.commit_arm
        else:
            return self.commit_arm

    def acknowledge_reward(self, arm: str, reward: float) -> None:
        if self.t < (self.m * self.k):
            self.sum_rewards[arm] += reward

class GreedyLearner(BanditLearner):
    def __init__(self, epsilon: float = 0.1, color: str = "yellow"):
        self.name = f"Greedy {epsilon}"
        self.color = color
        self.arms: list[str] = []
        self.epsilon = epsilon
        self.action_counters: dict[str, int] = None
        self.action_moving_avg: dict[str, float] = None

    def reset(self, arms: list[str], time_steps: int):
        self.arms = arms
        self.epsilon = self.epsilon
        self.action_counters = {arm: 0 for arm in arms}
        self.action_moving_avg = {arm: 0 for arm in arms}

    def pick_arm(self) -> str:
        if random.random() < self.epsilon:
            return random.choice(self.arms)
        else:
            return max(self.action_moving_avg, key=self.action_moving_avg.get)

    def acknowledge_reward(self, arm: str, reward: float) -> None:
        self.action_counters[arm] += 1
        self.action_moving_avg[arm] = self.action_moving_avg[arm] + (1 / self.action_counters[arm]) * (reward - self.action_moving_avg[arm])

class GreedyConstantStepLearner(BanditLearner):
    """
    Zachłanny z learning rate (średnia wykładniczo ważona aktualnością).
    Q(a) += alpha * (R - Q(a)) tylko dla wybranej akcji – przydatne przy bandytach niestacjonarnych.
    """

    def __init__(self, epsilon: float = 0.1, alpha: float = 0.1, color: str = "orange"):
        self.name = f"Greedy const α={alpha}"
        self.color = color
        self.arms: list[str] = []
        self.epsilon = epsilon
        self.alpha = alpha
        self.action_moving_avg: dict[str, float] = {}

    def reset(self, arms: list[str], time_steps: int):
        self.arms = arms
        self.action_moving_avg = {arm: 0.0 for arm in arms}

    def pick_arm(self) -> str:
        if random.random() < self.epsilon:
            return random.choice(self.arms)
        return max(self.action_moving_avg, key=self.action_moving_avg.get)

    def acknowledge_reward(self, arm: str, reward: float) -> None:
        self.action_moving_avg[arm] += self.alpha * (reward - self.action_moving_avg[arm])

class UCB1(BanditLearner):
    """Upper Confidence Bound (UCB1): wybiera ramię z maksymalną wartością Q(a) + c*sqrt(ln(t)/N(a))."""

    def __init__(self, c: float = 2.0, color: str = "green"):
        self.name = f"UCB1 c={c}"
        self.color = color
        self.c = c  # współczynnik eksploracji (typowo sqrt(2) lub 2)
        self.arms: list[str] = []
        self.sum_rewards: dict[str, float] = {}
        self.action_counters: dict[str, int] = {}
        self.t: int = 0  # numer kroku (1-indexed)
        self.k = 0

    def reset(self, arms: list[str], time_steps: int):
        self.arms = arms
        self.sum_rewards = {arm: 0.0 for arm in arms}
        self.action_counters = {arm: 0 for arm in arms}
        self.t = 0
        self.k = len(arms)

    def pick_arm(self) -> str:
        self.t += 1
        # Pierwsze k kroków: zagraj każdym ramieniem po raz pierwszy (inicjalizacja)
        if self.t <= self.k:
            return self.arms[self.t - 1]
        # UCB1: wybierz ramię z maksymalnym Q(a) + c * sqrt(ln(t) / N(a))
        ucb_values = {}
        for arm in self.arms:
            n = self.action_counters[arm]
            if n == 0:
                ucb_values[arm] = float("inf")  # nigdy nie wybrane = priorytet
            else:
                q = self.sum_rewards[arm] / n
                bonus = self.c * np.sqrt(np.log(self.t) / n)
                ucb_values[arm] = q + bonus
        return max(ucb_values, key=ucb_values.get)

    def acknowledge_reward(self, arm: str, reward: float) -> None:
        self.action_counters[arm] += 1
        self.sum_rewards[arm] += reward

class GradientBanditLearner(BanditLearner):
    """
    Wariant gradientowy (Sutton & Barto):
    - Preferencje H(a) zamiast estymaty wartości; prawdopodobieństwa przez soft-max.
    - Aktualizacja: H(a) += alpha * (R - R̄) * (𝟙[a=A] - π(a)).
    - Baseline R̄ = średnia nagroda (running average). Pełna sieć neuronowa nie jest potrzebna.
    """

    def __init__(self, alpha: float = 0.1, color: str = "magenta"):
        self.name = f"Gradient α={alpha}"
        self.color = color
        self.alpha = alpha
        self.arms: list[str] = []
        self.H: dict[str, float] = {}  # preferencje
        self.avg_reward: float = 0.0   # baseline R̄ (średnia ze wszystkich nagród)
        self.t: int = 0

    def reset(self, arms: list[str], time_steps: int):
        self.arms = arms
        self.H = {arm: 0.0 for arm in arms}
        self.avg_reward = 0.0
        self.t = 0

    def _softmax_probs(self) -> dict[str, float]:
        """π(a) = exp(H(a)) / Σ_b exp(H(b)); stabilne numerycznie (odejmowanie max)."""
        vals = np.array([self.H[arm] for arm in self.arms])
        vals = vals - np.max(vals) # transforms values to range -> ( -inf, 0 >     e^(-inf) ==> 0, e^0 = 1      y dostajemy prawdopodobienstwa w zakresie (0, 1) == stalibność numeryczne -> nie ma wystrzału przez overflowu (do inf)
        exp_vals = np.exp(vals)
        total = exp_vals.sum()
        # e^x / sum(e^x)
        return {arm: exp_vals[i] / total for i, arm in enumerate(self.arms)}

    def pick_arm(self) -> str:
        probs = self._softmax_probs()
        return np.random.choice(self.arms, p=[probs[a] for a in self.arms]) # still random (model temperature could be used to control how much it favors highest value)

    def acknowledge_reward(self, arm: str, reward: float) -> None:
        self.t += 1
        # π_t(a) – prawdopodobieństwa w momencie wyboru (przed aktualizacją H)
        probs = self._softmax_probs()
        # baseline R̄ = średnia nagród z kroków 1..t-1 (przed włączeniem tej nagrody)
        baseline = self.avg_reward
        self.avg_reward += (reward - self.avg_reward) / self.t  # działa ponieważ wcześniejsza średnia razy wsześniejsza ilość elementów daj nam sumę wszystkich wcześniejszych elementów
                                                                # nowa_srednia = (   (stara_srednia * (t-1))    + nowa_nagroda) / t
                                                                #                       ^^suma wcz. el.^^
                                                                #                (   st_średnia * t  - st_średnia + nowa_nagroda) / t
                                                                #                    st_średnia + (nowa_nagroda - st_średnia) / t
                                                                #
                                                                # nowa_średnia += (nowa_nagroda - st_średnia) / t                niezłe
        # H(a) += alpha * (R_t - R̄) * (𝟙[a=A_t] - π_t(a))
        for a in self.arms:
            one_if_chosen = 1.0 if a == arm else 0.0
            self.H[a] += self.alpha * (reward - baseline) * (one_if_chosen - probs[a])

            #                          moc korekty, ugruntowana w średniej z wcześniejszych -> inaczej mielibyśmy silne przeskoki nawet kiedy zbliżalibyśmy się do poprawnego rozwiązania
            #                                                - dla niepoprawnej
            #                                                + dla poprawnej

class ThompsonSamplingLearner(BanditLearner):
    """
    Próbkowanie Thompsona dla bandytów Bernoulliego (nagroda 0/1).
    Dla każdego ramienia utrzymujemy rozkład Beta(α, β); po obserwacji
    nagrody r: α += r, β += (1-r). Wybór: próbkujemy θ_a ~ Beta(α_a, β_a)
    i wybieramy ramię z max θ_a.
    """

    def __init__(self, prior_alpha: float = 1.0, prior_beta: float = 1.0, color: str = "darkviolet"):
        self.name = "Thompson"
        self.color = color
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta
        self.arms: list[str] = []
        self.alpha: dict[str, float] = {}
        self.beta: dict[str, float] = {}

    def reset(self, arms: list[str], time_steps: int):
        self.arms = arms
        self.alpha = {arm: self.prior_alpha for arm in arms}
        self.beta = {arm: self.prior_beta for arm in arms}

    def pick_arm(self) -> str:
        samples = {
            arm: np.random.beta(self.alpha[arm], self.beta[arm])
            for arm in self.arms
        }
        return max(samples, key=samples.get)

    def acknowledge_reward(self, arm: str, reward: float) -> None:
        # Bernoulli: sukces -> α+1, porażka -> β+1
        self.alpha[arm] += reward
        self.beta[arm] += 1.0 - reward





TIME_STEPS = 1000
TRIALS_PER_LEARNER = 50
K_ARMS_RANDOM = 5  # liczba ramion w eksperymentach z losowymi problemami
# losowe problemy+regret oraz studium parametryczne
RUN_RANDOM_AND_PARAMETER_STUDY = False
# bandyci niestacjonarni (dryf + zachłanny z learning rate)
RUN_NONSTATIONARY = True


def evaluate_learner(learner: BanditLearner) -> None:
    runs_results = []
    for _ in range(TRIALS_PER_LEARNER):
        bandit = TopHitBandit(POTENTIAL_HITS)
        problem = BanditProblem(time_steps=TIME_STEPS, bandit=bandit, learner=learner)
        rewards = problem.run()
        accumulated_rewards = list(accumulate(rewards))
        runs_results.append(accumulated_rewards)

    runs_results = np.array(runs_results)
    mean_accumulated_rewards = np.mean(runs_results, axis=0)
    std_accumulated_rewards = np.std(runs_results, axis=0)
    plt.plot(mean_accumulated_rewards, label=learner.name, color=learner.color)
    plt.fill_between(
        range(len(mean_accumulated_rewards)),
        mean_accumulated_rewards - std_accumulated_rewards,
        mean_accumulated_rewards + std_accumulated_rewards,
        color=learner.color,
        alpha=0.2,
    )


def evaluate_learner_random_problems(
    learner: BanditLearner,
    n_trials: int = TRIALS_PER_LEARNER,
    time_steps: int = TIME_STEPS,
    k_arms: int = K_ARMS_RANDOM,
    rng: np.random.Generator | None = None,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Losowe problemy (nowy bandyta z losowymi p przy każdym trialu) + regret.
    Regret w kroku t: t * max_a p_a - suma nagród do t (ile tracimy względem optymalnej decyzji).
    Jeśli podasz seed, w każdym trialu bandyta jest z rng(seed + trial) – te same instancje dla wszystkich learnerów.
    Zwraca: (mean_cumulative_rewards, std_cumulative_rewards, mean_cumulative_regret, std_cumulative_regret).
    """
    cum_rewards_list: list[list[float]] = []
    cum_regret_list: list[list[float]] = []

    for trial in range(n_trials):
        if seed is not None:
            trial_rng = np.random.default_rng(seed + trial)
            bandit = make_random_bandit(k_arms, rng=trial_rng)
        else:
            rng = rng or np.random.default_rng()
            bandit = make_random_bandit(k_arms, rng=rng)
        problem = BanditProblem(time_steps=time_steps, bandit=bandit, learner=learner)
        rewards = problem.run()
        max_prob = bandit.max_expected_reward()
        cum_rewards = list(accumulate(rewards))
        # regret w chwili t (0-indexed): (t+1)*max_prob - cum_rewards[t]
        cum_regret = [(t + 1) * max_prob - cum_rewards[t] for t in range(time_steps)]
        cum_rewards_list.append(cum_rewards)
        cum_regret_list.append(cum_regret)

    arr_rewards = np.array(cum_rewards_list)
    arr_regret = np.array(cum_regret_list)
    return (
        np.mean(arr_rewards, axis=0),
        np.std(arr_rewards, axis=0),
        np.mean(arr_regret, axis=0),
        np.std(arr_regret, axis=0),
    )


def plot_random_problems_and_regret(
    learners: list[BanditLearner],
    n_trials: int = TRIALS_PER_LEARNER,
    time_steps: int = TIME_STEPS,
    k_arms: int = K_ARMS_RANDOM,
    seed: int | None = None,
) -> None:
    """Wykres: suma nagród i regret w czasie (uśrednione po wielu losowych problemach)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    for learner in learners:
        mean_cr, std_cr, mean_regret, std_regret = evaluate_learner_random_problems(
            learner, n_trials=n_trials, time_steps=time_steps, k_arms=k_arms, seed=seed
        )
        t = np.arange(1, time_steps + 1)
        ax1.plot(mean_cr, label=learner.name, color=learner.color)
        ax1.fill_between(t - 1, mean_cr - std_cr, mean_cr + std_cr, color=learner.color, alpha=0.2)
        ax2.plot(mean_regret, label=learner.name, color=learner.color)
        ax2.fill_between(t - 1, mean_regret - std_regret, mean_regret + std_regret, color=learner.color, alpha=0.2)

    ax1.set_xlabel("Krok")
    ax1.set_ylabel("Skumulowana nagroda")
    ax1.set_title("Losowe problemy – skumulowana nagroda")
    ax1.legend()
    ax1.set_xlim(0, time_steps)

    ax2.set_xlabel("Krok")
    ax2.set_ylabel("Regret (skumulowany)")
    ax2.set_title("Losowe problemy – regret")
    ax2.legend()
    ax2.set_xlim(0, time_steps)

    plt.tight_layout()
    plt.show()


def run_parameter_study(
    time_steps: int = TIME_STEPS,
    n_trials: int = TRIALS_PER_LEARNER,
    k_arms: int = K_ARMS_RANDOM,
    seed: int | None = None,
) -> None:
    """
    Studium parametryczne: pełny przekrój meta-parametrów dla każdego algorytmu.
    Dla każdej wartości parametru: losowe problemy, uśredniona skumulowana nagroda / regret.
    """
    configs: list[tuple[str, list[BanditLearner], str]] = [
        ("Explore-Then-Commit (m)", [ExploreThenCommitLearner(m=m, color=f"C{i}") for i, m in enumerate([5, 10, 25, 50, 100, 200])], "m"),
        ("ε-zachłanny (ε)", [GreedyLearner(epsilon=e, color=f"C{i}") for i, e in enumerate([0.01, 0.05, 0.1, 0.2, 0.3, 0.5])], "ε"),
        ("UCB1 (c)", [UCB1(c=c, color=f"C{i}") for i, c in enumerate([0.5, 1.0, np.sqrt(2), 2.0, 3.0])], "c"),
        ("Gradient (α)", [GradientBanditLearner(alpha=a, color=f"C{i}") for i, a in enumerate([0.05, 0.1, 0.2, 0.4, 0.8])], "α"),
        ("Thompson/BernTS", [ThompsonSamplingLearner(color="C0")], "—"),
    ]

    for title, learners, param_label in configs:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        for learner in learners:
            mean_cr, std_cr, mean_regret, std_regret = evaluate_learner_random_problems(
                learner, n_trials=n_trials, time_steps=time_steps, k_arms=k_arms, seed=seed
            )
            t = np.arange(time_steps)
            ax1.plot(mean_cr, label=learner.name, color=learner.color)
            ax1.fill_between(t, mean_cr - std_cr, mean_cr + std_cr, color=learner.color, alpha=0.2)
            ax2.plot(mean_regret, label=learner.name, color=learner.color)
            ax2.fill_between(t, mean_regret - std_regret, mean_regret + std_regret, color=learner.color, alpha=0.2)

        ax1.set_xlabel("Krok")
        ax1.set_ylabel("Skumulowana nagroda")
        ax1.set_title(f"Studium parametryczne: {title}")
        ax1.legend()
        ax1.set_xlim(0, time_steps)

        ax2.set_xlabel("Krok")
        ax2.set_ylabel("Regret (skumulowany)")
        ax2.set_title(f"Regret: {title}")
        ax2.legend()
        ax2.set_xlim(0, time_steps)

        plt.tight_layout()
        plt.show()


def run_nonstationary_experiment(
    learners: list[BanditLearner],
    initial_probs: dict[str, float] | None = None,
    k_arms: int = 5,
    time_steps: int = TIME_STEPS,
    n_trials: int = TRIALS_PER_LEARNER,
    drift_std: float = 0.01,
    seed: int | None = None,
) -> None:
    """
    Bandyci niestacjonarni: dryf preferencji (błądzenie losowe N(0, drift_std) co krok).
    Dla każdego trialu wszyscy learnerzy widzą ten sam ciąg dryfu (ten sam seed+trial).
    """
    rng = np.random.default_rng(seed)
    if initial_probs is None:
        initial_probs = {str(i): float(rng.uniform(0.2, 0.8)) for i in range(k_arms)}

    base = seed if seed is not None else 0
    fig, ax = plt.subplots(figsize=(10, 6))
    for learner in learners:
        runs = []
        for trial in range(n_trials):
            bandit = NonStationaryTopHitBandit(
                dict(initial_probs), drift_std=drift_std, rng=np.random.default_rng(base + trial)
            )
            problem = BanditProblem(time_steps=time_steps, bandit=bandit, learner=learner)
            rewards = problem.run()
            runs.append(list(accumulate(rewards)))
        arr = np.array(runs)
        mean_cr = np.mean(arr, axis=0)
        std_cr = np.std(arr, axis=0)
        ax.plot(mean_cr, label=learner.name, color=learner.color)
        ax.fill_between(np.arange(time_steps), mean_cr - std_cr, mean_cr + std_cr, color=learner.color, alpha=0.2)

    ax.set_xlabel("Krok")
    ax.set_ylabel("Skumulowana nagroda")
    ax.set_title(f"Bandyci niestacjonarni (dryf σ={drift_std})")
    ax.legend()
    ax.set_xlim(0, time_steps)
    plt.tight_layout()
    plt.show()


def main():
    learners = [
        RandomLearner(),

        # ExploreThenCommitLearner(10, color="blue"),
        # ExploreThenCommitLearner(100, color="green"),
        # ExploreThenCommitLearner(250, color="red"),

        # GreedyLearner(0.1, color="yellow"),
        # GreedyLearner(0.2, color="orange"),
        # GreedyLearner(0.3, color="purple"),

        # UCB1(c=2.0, color="teal"),
        # UCB1(c=np.sqrt(2), color="cyan"),

        GradientBanditLearner(alpha=0.1, color="magenta"),
        GradientBanditLearner(alpha=0.4, color="brown"),

        ThompsonSamplingLearner(color="darkviolet"),
    ]
    for learner in learners:
        evaluate_learner(learner)

    plt.xlabel('Czas')
    plt.ylabel('Suma uzyskanych nagród')
    plt.xlim(0, TIME_STEPS)
    plt.ylim(0, TIME_STEPS)
    plt.legend()
    plt.show()

    if RUN_RANDOM_AND_PARAMETER_STUDY:
        # Losowe problemy + regret (agregacja po wielu instancjach)
        learners_all = [
            RandomLearner(),
            ExploreThenCommitLearner(50, color="blue"),
            GreedyLearner(0.1, color="yellow"),
            UCB1(c=2.0, color="teal"),
            GradientBanditLearner(alpha=0.1, color="magenta"),
            ThompsonSamplingLearner(color="darkviolet"),
        ]
        plot_random_problems_and_regret(learners_all, seed=42)
        # Studium parametryczne (pełny przekrój parametrów)
        run_parameter_study(seed=42)

    if RUN_NONSTATIONARY:
        # Bandyci niestacjonarni: dryf preferencji; zachłanny ze średnią wykładniczo ważoną (alpha) vs zwykły
        run_nonstationary_experiment(
            learners=[
                RandomLearner(),
                GreedyLearner(0.1, color="gold"),
                GreedyConstantStepLearner(epsilon=0.1, alpha=0.1, color="orange"),
                GreedyConstantStepLearner(epsilon=0.1, alpha=0.2, color="coral"),
                UCB1(c=2.0, color="teal"),
                GradientBanditLearner(alpha=0.1, color="magenta"),
                ThompsonSamplingLearner(color="darkviolet"),
            ],
            initial_probs=dict(POTENTIAL_HITS),
            time_steps=TIME_STEPS,
            n_trials=TRIALS_PER_LEARNER,
            drift_std=0.01,
            seed=42,
        )


if __name__ == '__main__':
    main()
