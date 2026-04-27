"""
Actor-Critic (A2C / one-step TD) — CartPole-v1 & LunarLander-v3
================================================================
Cwiczenie: zaimplementuj brakujace fragmenty oznaczone komentarzami TODO.

Uruchomienie
------------
  # Trening CartPole (domyslnie):
  KERAS_BACKEND=torch python solution.py --env cartpole

  # Trening LunarLander:
  KERAS_BACKEND=torch python solution.py --env lunar

  # Podglad nauczonego modelu (bez treningu):
  KERAS_BACKEND=torch python solution.py --env lunar --render sciezka/do/modelu.keras

  # Wznowienie treningu z checkpointu:
  KERAS_BACKEND=torch python solution.py --env lunar --resume sciezka/do/modelu.keras


                                      ()
         ,
       _=|_
     _[_## ]_
_  +[_[_+_]P/    _    |_       ____      _=--|-~
 ~--/\\I_I_[\\--~ ~~--[o]--==-|##==]-=-~~  o]H
-~ /[_[_|_]_]\\  -_  [[=]]    |====]  __  !j]H
  /    "|"    \\     ^U-U^  - |    - ~ .~  U/~
 ~~--__~~~--__~~-__   H_H_    |_     --   _H_
-. _  ~~~#######~~~     ~~~-    ~~--  ._ - ~~-=
           ~~~=~~  -~~--  _     . -      _ _ -
"""

import os
os.environ.setdefault("KERAS_BACKEND", "torch")

import click
import gymnasium as gym
import keras
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

matplotlib.use("Agg")

if (backend := keras.backend.backend()) != "torch":
    raise NotImplementedError(
        f"Backend {backend!r} nie jest wspierany!"
    )
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[backend] PyTorch  |  urzadzenie: {DEVICE.upper()}")


def format_state(state: np.ndarray) -> np.ndarray:
    return np.reshape(state, (1, -1)).astype(np.float32)


def to_numpy(tensor) -> np.ndarray:
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    return np.array(tensor)


ENV_CONFIGS: dict[str, dict] = {
    "cartpole": {
        "gym_id":           "CartPole-v1",
        "hidden":           (128, 128),
        "learning_rate":    1e-3,
        "discount_factor":  0.99,
        "entropy_coeff":    0.0,
        "separate_trunks":  False,
        "n_episodes":       2000,
        "solved_threshold": 475.0,
        "window":           100,
        "plot_every":       25,
        "save_every":       100,
        "plot_dir":         "plots_cartpole",
    },
    "lunar": {
        "gym_id":           "LunarLander-v3",
        "hidden":           (256, 256),
        "learning_rate":    3e-4,
        "discount_factor":  0.99,
        "entropy_coeff":    0.01,
        "separate_trunks":  False,
        "n_episodes":       5000,
        "solved_threshold": 200.0,
        "window":           100,
        "plot_every":       50,
        "save_every":       50,
        "plot_dir":         "plots_lunar",
    },
}


def build_actor_critic(
    obs_dim: int,
    n_actions: int,
    hidden: tuple[int, int] = (256, 256),
) -> keras.Model:
    """
    Actor-Critic ze wspolnym trzonem sieci.

    Wejscie : obserwacja  (batch, obs_dim)
    Wyjscia : logits      (batch, n_actions)  — surowe logity akcji
              value       (batch, 1)          — V(s)
    """
    obs = keras.Input(shape=(obs_dim,), name="obs")

    x = keras.layers.Dense(hidden[0], activation="tanh", name="fc1")(obs)
    x = keras.layers.Dense(hidden[1], activation="tanh", name="fc2")(x)

    logits = keras.layers.Dense(n_actions, name="logits")(x)
    value  = keras.layers.Dense(1,         name="value")(x)

    return keras.Model(inputs=obs, outputs=[logits, value], name="ActorCritic")


def build_actor_critic_separate(
    obs_dim: int,
    n_actions: int,
    hidden: tuple[int, int] = (256, 256),
) -> keras.Model:
    """
    Actor-Critic z NIEZALEZNYMI trzonkami — brak wspoldzielonych wag.
    Kazda glowica ma wlasne dwie warstwy ukryte (dwa razy wiecej parametrow).
    """
    raise NotImplementedError("TODO: zaimplementuj rozlaczne trzonki aktora i krytyka!")


class ActorCriticController:
    """
    Agent A2C (one-step) z opcjonalna regularyzacja entropii.
    """

    def __init__(
        self,
        environment: gym.Env,
        learning_rate: float = 3e-4,
        discount_factor: float = 0.99,
        entropy_coeff: float = 0.01,
        hidden: tuple[int, int] = (256, 256),
        separate: bool = False,
    ) -> None:
        self.gamma         = discount_factor
        self.entropy_coeff = entropy_coeff

        obs_dim   = int(np.prod(environment.observation_space.shape))
        n_actions = int(environment.action_space.n)

        build_fn = build_actor_critic_separate if separate else build_actor_critic
        self.model     = build_fn(obs_dim, n_actions, hidden=hidden)
        self.optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

        self.last_action: int | None = None
        self.last_error_squared: float = 0.0

        if DEVICE == "cuda":
            self.model = self.model.to(DEVICE)

    def choose_action(self, state: np.ndarray) -> int:
        x = format_state(state)

        # TODO: wywolaj model (tryb inferencji) i pobierz logity
        # Podpowiedz: self.model(x, training=False) zwraca..?
        logits = None

        # TODO: oblicz prawdopodobienstwa akcji za pomoca softmax na logitach
        # Podpowiedz: może pomogą keras.ops..?
        probs = None

        # TODO: przekonwertuj prawdopodobienstwa do numpy (przyda sie też metoda ravel i pomocnicze to_numpy), 
        # znormalizuj dla bezpieczenstwa (podziel przez sume) i wylosuj akcje uzywajac np.random.choice
        # Podpowiedz: astype(np.float64) często ratuje sytuację...
        action = 0

        self.last_action = action
        return action

    def learn(
        self,
        state: np.ndarray,
        reward: float,
        new_state: np.ndarray,
        terminal: bool,
    ) -> None:
        assert self.last_action is not None
        x_s = format_state(state)
        x_s_next = format_state(new_state)
        grads = self.compute_grads(self.last_action, x_s, x_s_next, reward, terminal)
        self.optimizer.apply(grads, self.model.trainable_variables)

    def compute_loss(self, action, x_s, x_s_next, reward, terminal):
        """
        Oblicza laczna strate: L = L_krytyk + L_aktor + L_entropia
        """
        logits_s, value_s = self.model(x_s, training=True)
        v_s = value_s[0, 0]

        # TODO: oblicz cel TD (target)
        # Podpowiedz: nie zapomnij o keras.ops.stop_gradient...
        r = keras.ops.cast(reward, "float32")
        target = r  # TODO: zastap poprawna formula

        # TODO: oblicz blad TD (delta = target - V(s)) i zapisz jego kwadrat
        # do self.last_error_squared (przydatne do wykresow)
        # Podpowiedz: a tu się przyda keras.ops.square...
        error = keras.ops.cast(0.0, "float32")  # TODO: zastap poprawna formula
        self.last_error_squared = 0.0

        # TODO: oblicz strate krytyka jako kwadrat bledu TD
        critic_loss = keras.ops.cast(0.0, "float32")

        # TODO: oblicz log-prawdopodobienstwa akcji (log-softmax)
        # Podpowiedz: log_probs = logits_s - keras.ops.logsumexp...
        # Nastepnie wybierz log-prawdopodobienstwo konkretnej wybranej akcji
        log_prob_a = keras.ops.cast(0.0, "float32")

        # TODO: oblicz strate aktora (policy gradient)
        # Podpowiedz: tu tez sie przyda keras.ops.stop_gradient...
        actor_loss = keras.ops.cast(0.0, "float32")

        # TODO: oblicz bonus z entropii
        # i mnoz przez -entropy_coeff (maksymalizacja entropii = minimalizacja -entropii)
        # Podpowiedz: log_probs zostalo juz obliczone powyzej...
        entropy_loss = keras.ops.cast(0.0, "float32")

        return critic_loss + actor_loss + entropy_loss

    def compute_grads(self, action, x_s, x_s_next, reward, terminal):
        raw = [v.value for v in self.model.trainable_variables]
        for t in raw:
            if t.grad is not None:
                t.grad.zero_()
        loss = self.compute_loss(action, x_s, x_s_next, reward, terminal)
        loss.backward()
        return [t.grad if t.grad is not None else torch.zeros_like(t) for t in raw]


def render_model(model_path: str, env_name: str = "cartpole", n_episodes: int = 5) -> None:
    """Wczytuje checkpoint .keras i renderuje n_episodes z widokiem dla czlowieka."""
    cfg = ENV_CONFIGS[env_name]
    print(f"Wczytywanie modelu z: {model_path}  (srodowisko: {cfg['gym_id']})")
    env   = gym.make(cfg["gym_id"], render_mode="human")
    model = keras.saving.load_model(model_path)
    model.summary()

    for ep in range(n_episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0.0

        while not done:
            logits, _ = model(format_state(state), training=False)
            probs = to_numpy(keras.ops.softmax(logits, axis=-1)).ravel().astype(np.float64)
            probs /= probs.sum()

            action = int(np.argmax(probs)) # zachlanne — bez eksploracji - moga byc lepsze wyniki!
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += float(reward)

        print(f"  Epizod {ep+1}: laczna nagroda = {total_reward:.1f}")
    env.close()


def train(env_name: str = "cartpole", resume_path: str | None = None) -> None:
    cfg = ENV_CONFIGS[env_name]
    os.makedirs(cfg["plot_dir"], exist_ok=True)

    env = gym.make(cfg["gym_id"], render_mode=None)
    agent = ActorCriticController(
        env,
        learning_rate   = cfg["learning_rate"],
        discount_factor = cfg["discount_factor"],
        entropy_coeff   = cfg["entropy_coeff"],
        hidden          = cfg["hidden"],
        separate        = cfg["separate_trunks"],
    )

    if resume_path:
        print(f"Wznawianie z checkpointu: {resume_path}")
        agent.model = keras.saving.load_model(resume_path)
        if DEVICE == "cuda":
            agent.model = agent.model.to(DEVICE)

    agent.model.summary()

    past_rewards: list[float] = []
    past_errors:  list[float] = []

    N_EPISODES = cfg["n_episodes"]
    WINDOW     = cfg["window"]
    THRESHOLD  = cfg["solved_threshold"]
    PLOT_EVERY = cfg["plot_every"]
    SAVE_EVERY = cfg["save_every"]
    PLOT_DIR   = cfg["plot_dir"]

    for ep in tqdm(range(N_EPISODES), desc=f"{cfg['gym_id']} trening"):
        state, _ = env.reset()
        done = False
        reward_sum = 0.0
        ep_errors: list[float] = []

        while not done:
            action = agent.choose_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            reward = float(reward)
            agent.learn(state, reward, next_state, done)

            state = next_state
            reward_sum += reward
            ep_errors.append(agent.last_error_squared)

        past_rewards.append(reward_sum)
        past_errors.append(float(np.mean(ep_errors)))

        if len(past_rewards) >= WINDOW:
            mean_w = np.mean(past_rewards[-WINDOW:])
            if mean_w >= THRESHOLD:
                print(f"\nUkonczone w epizodzie {ep}!\n Srednia nagroda z ostatnich {WINDOW} epizodow: {mean_w:.1f}")
                agent.model.save(f"{env_name}_solved_ep{ep}.keras")
                break

        if ep % PLOT_EVERY == 0 and len(past_rewards) >= WINDOW:
            rolling_err = [np.mean(past_errors[i:i+WINDOW]) for i in range(len(past_errors)  - WINDOW)]
            rolling_rew = [np.mean(past_rewards[i:i+WINDOW]) for i in range(len(past_rewards) - WINDOW)]

            fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(10, 7))
            ax0.plot(rolling_err, color="tab:red")
            ax0.axhline(0, color="gray", linewidth=0.5)
            ax0.set_title(f"Sredni kwadratowy blad TD (okno {WINDOW})")
            ax0.set_ylabel("MSE")
            ax1.plot(rolling_rew, color="tab:green")
            ax1.axhline(THRESHOLD, color="gold", linewidth=1, linestyle="--", label="prog ukonczenia")
            ax1.legend(fontsize=8)
            ax1.set_title(f"Suma nagrod w epizodzie (okno {WINDOW})")
            ax1.set_ylabel("Zwrot")
            ax1.set_xlabel("Epizod")
            plt.tight_layout()
            plt.savefig(f"{PLOT_DIR}/learning_{ep:05d}.png", dpi=100)
            plt.close(fig)

        if ep % SAVE_EVERY == 0 and ep > 0:
            ckpt = f"{env_name}_checkpoint_ep{ep}.keras"
            agent.model.save(ckpt)
            print(f"\n  [checkpoint] zapisano -> {ckpt}")

    env.close()
    agent.model.save(f"{env_name}_final.keras")
    print(f"\nGotowe. Model zapisano -> {env_name}_final.keras")


@click.command()
@click.option("--env", type=click.Choice(["cartpole", "lunar"]), default="cartpole", show_default=True, help="Srodowisko do treningu.")
@click.option("--render", metavar="MODEL.keras", default=None, help="Wczytaj zapisany model i renderuj epizody (bez treningu).")
@click.option("--resume", metavar="MODEL.keras", default=None, help="Wznow trening z zapisanego checkpointu.")
def main(env: str, render: str | None, resume: str | None) -> None:
    if render:
        render_model(render, env_name=env)
    else:
        train(env_name=env, resume_path=resume)


if __name__ == "__main__":
    main()
