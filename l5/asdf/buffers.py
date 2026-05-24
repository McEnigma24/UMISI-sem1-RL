from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any, Optional, Union

import gymnasium as gym
import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from .utils import combined_shape


class BaseBuffer(ABC):
    @abstractmethod
    def __init__(
        self, env: gym.Env, size: int = 100000, device: Optional[torch.device] = None
    ) -> None:
        self.device = device

        self.actions = torch.zeros(
            combined_shape(size, env.action_space.shape),
            dtype=torch.float32,
            device=device,
        )
        self.rewards = torch.zeros(size, dtype=torch.float32, device=device)
        self.terminations = torch.zeros(size, dtype=torch.float32, device=device)
        self.truncations = torch.zeros(size, dtype=torch.float32, device=device)
        self.infos = np.empty((size,), dtype=object)
        self._ptr, self.size, self.max_size = 0, 0, size

    def store(
        self,
        observation: Union[NDArray, dict[str, NDArray]],
        action: NDArray,
        reward: float,
        next_observation: Union[NDArray, dict[str, NDArray]],
        terminated: bool,
        truncated: bool,
        info: dict[str, Any],
    ) -> None:
        self._store_observations(observation, next_observation)
        self.actions[self._ptr] = torch.as_tensor(action, dtype=torch.float32)
        self.rewards[self._ptr] = torch.as_tensor(reward, dtype=torch.float32)
        self.terminations[self._ptr] = torch.as_tensor(terminated, dtype=torch.float32)
        self.truncations[self._ptr] = torch.as_tensor(truncated, dtype=torch.float32)
        self.infos[self._ptr] = info
        self._ptr = (self._ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    @abstractmethod
    def _store_observations(
        self,
        observation: Union[NDArray, dict[str, NDArray]],
        next_observation: Union[NDArray, dict[str, NDArray]],
    ) -> None: ...

    def sample_batch(
        self, batch_size: int = 32
    ) -> dict[str, Union[Tensor, dict[str, Tensor]]]:
        idxs = torch.randint(0, self.size, size=(batch_size,))
        # idxs = np.random.randint(0, self.size, size=batch_size)
        return self.batch(idxs)

    def batch(self, idxs: Tensor) -> dict[str, Union[Tensor, dict[str, Tensor]]]:
        data = dict(
            action=self.actions[idxs],
            reward=self.rewards[idxs],
            terminated=self.terminations[idxs],
            truncated=self.truncations[idxs],
            info=self.infos[idxs],
        )
        observations = self._observations_batch(idxs)
        data.update(observations)

        return data

    @abstractmethod
    def _observations_batch(
        self, idxs: Tensor
    ) -> dict[str, Union[Tensor, dict[str, Tensor]]]: ...

    def start_episode(self):
        pass

    def end_episode(self):
        pass

    def clear(self):
        self.actions.zero_()
        self.rewards.zero_()
        self.terminations.zero_()
        self.truncations.zero_()
        self.infos.fill(None)
        self._ptr, self.size = 0, 0


class DictReplayBuffer(BaseBuffer):
    """
    A dictionary experience replay buffer for off-policy agents.
    """

    def __init__(
        self, env: gym.Env, size: int = 100000, device: Optional[torch.device] = None
    ):
        assert isinstance(env.observation_space, gym.spaces.Dict)
        super().__init__(env=env, size=size, device=device)

        obs_space = {
            k: combined_shape(size, v.shape) for k, v in env.observation_space.items()
        }

        self.observations: dict[str, Tensor] = {
            k: torch.zeros(obs_space[k], dtype=torch.float32, device=device)
            for k, v in env.observation_space.items()
        }
        self.next_observations: dict[str, Tensor] = {
            k: torch.zeros(obs_space[k], dtype=torch.float32, device=device)
            for k, v in env.observation_space.items()
        }

    def _store_observations(
        self,
        observation: dict[str, NDArray],
        next_observation: dict[str, NDArray],
    ) -> None:
        for k in observation.keys():
            self.observations[k][self._ptr] = torch.as_tensor(
                observation[k], dtype=torch.float32
            )
        for k in next_observation.keys():
            self.next_observations[k][self._ptr] = torch.as_tensor(
                next_observation[k], dtype=torch.float32
            )

    def _observations_batch(self, idxs: Tensor) -> dict[str, dict[str, Tensor]]:
        return dict(
            observation={k: v[idxs] for k, v in self.observations.items()},
            next_observation={k: v[idxs] for k, v in self.next_observations.items()},
        )




class HerReplayBuffer(DictReplayBuffer):
    """
    Hindsight Experience Replay (HER) on top of a dict observation replay buffer.

    Transitions are accumulated per episode; when the episode ends, each step is
    stored once with the original goal and ``n_sampled_goal`` additional times with
    relabeled ``desired_goal`` and reward from ``env.unwrapped.compute_reward``.
    """

    def __init__(
        self,
        env: gym.Env,
        size: int = 100000,
        device: Optional[torch.device] = None,
        n_sampled_goal: int = 1,
        goal_selection_strategy: str = "final",
    ):
        super().__init__(env=env, size=size, device=device)
        self.env = env
        self.n_sampled_goal = n_sampled_goal
        self.selection_strategy = goal_selection_strategy
        self._episode_transitions: list[dict[str, Any]] = []
        self._rng = np.random.default_rng()

    @staticmethod
    def _copy_obs(obs: dict[str, NDArray]) -> dict[str, NDArray]:
        return {k: np.asarray(v, dtype=np.float32).copy() for k, v in obs.items()}

    def start_episode(self) -> None:
        self._episode_transitions.clear()

    def end_episode(self) -> None:
        ep = self._episode_transitions
        if not ep:
            return
        env_compute = self.env.unwrapped
        if not hasattr(env_compute, "compute_reward"):
            raise AttributeError(
                "HER requires env.unwrapped.compute_reward (goal-conditioned Gymnasium env)."
            )

        for t_idx, tr in enumerate(ep):
            self._store_one_transition(
                tr["observation"],
                tr["action"],
                tr["reward"],
                tr["next_observation"],
                tr["terminated"],
                tr["truncated"],
                tr["info"],
            )
            for _ in range(self.n_sampled_goal):
                new_goal = self._sample_goal(ep, t_idx)
                o = self._copy_obs(tr["observation"])
                o2 = self._copy_obs(tr["next_observation"])
                o["desired_goal"] = np.asarray(new_goal, dtype=np.float32).copy()
                o2["desired_goal"] = np.asarray(new_goal, dtype=np.float32).copy()
                info = deepcopy(tr["info"])
                r = float(
                    env_compute.compute_reward(
                        o2["achieved_goal"], o2["desired_goal"], info
                    )
                )
                if "is_success" in info:
                    info["is_success"] = bool(np.isclose(r, 0.0))
                self._store_one_transition(
                    o, tr["action"], r, o2, tr["terminated"], tr["truncated"], info
                )

        self._episode_transitions.clear()

    def _store_one_transition(
        self,
        observation: dict[str, NDArray],
        action: NDArray,
        reward: float,
        next_observation: dict[str, NDArray],
        terminated: bool,
        truncated: bool,
        info: dict[str, Any],
    ) -> None:
        super().store(
            observation=observation,
            action=action,
            reward=reward,
            next_observation=next_observation,
            terminated=terminated,
            truncated=truncated,
            info=info,
        )

    def _sample_goal(self, episode: list[dict[str, Any]], transition_idx: int) -> NDArray:
        """Pick a substitute desired goal for HER (vector in achieved_goal space)."""
        T = len(episode)
        strat = self.selection_strategy

        if strat == "final":
            return np.asarray(
                episode[-1]["next_observation"]["achieved_goal"], dtype=np.float32
            ).copy()

        if strat == "future":
            candidates: list[NDArray] = []
            for j in range(transition_idx + 1, T):
                candidates.append(
                    np.asarray(
                        episode[j]["next_observation"]["achieved_goal"],
                        dtype=np.float32,
                    ).copy()
                )
            if not candidates:
                return self._sample_goal_episode(episode)
            pick = int(self._rng.integers(0, len(candidates)))
            return np.asarray(candidates[pick], dtype=np.float32).copy()

        if strat == "episode":
            return self._sample_goal_episode(episode)

        raise ValueError(
            f"Unknown goal_selection_strategy={strat!r}; "
            "expected 'final', 'future', or 'episode'."
        )

    def _sample_goal_episode(self, episode: list[dict[str, Any]]) -> NDArray:
        j = int(self._rng.integers(0, len(episode)))
        return np.asarray(
            episode[j]["next_observation"]["achieved_goal"], dtype=np.float32
        ).copy()

    def store(
        self,
        observation: dict[str, NDArray],
        action: NDArray,
        reward: float,
        next_observation: dict[str, NDArray],
        terminated: bool,
        truncated: bool,
        info: dict[str, Any],
    ) -> None:
        self._episode_transitions.append(
            dict(
                observation=self._copy_obs(observation),
                action=np.asarray(action, dtype=np.float32).copy(),
                reward=reward,
                next_observation=self._copy_obs(next_observation),
                terminated=terminated,
                truncated=truncated,
                info=deepcopy(info),
            )
        )

    def clear(self) -> None:
        super().clear()
        self._episode_transitions.clear()
