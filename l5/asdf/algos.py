import itertools
import time
from copy import deepcopy
from typing import Optional

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from tqdm import trange

from .buffers import BaseBuffer
from .loggers import BaseLogger
from .loggers import SilentLogger
from .policies import MlpPolicy
from .utils import count_vars



class SAC:
    """Soft Actor-Critic (SAC)"""

    def __init__(
        self,
        env,
        policy: MlpPolicy,
        buffer: BaseBuffer,
        seed=0,
        gamma=0.99,
        polyak=0.995,
        lr=1e-3,
        alpha="auto",
        target_entropy="auto",
        batch_size=100,
        start_steps=10000,
        update_after=1000,
        update_every=50,
        n_test_episodes=25,
        max_episode_len=1000,
        n_updates=None,
        logger: BaseLogger = SilentLogger(),
    ):
        """
        Soft Actor-Critic (SAC)
        Args:
            env_fn : A function which creates a copy of the environment.
                The environment must satisfy the OpenAI Gym API.
            policy: The constructor method for a PyTorch Module with an ``act``
                method, a ``pi`` module, a ``q1`` module, and a ``q2`` module.
                The ``act`` method and ``pi`` module should accept batches of
                observations as inputs, and ``q1`` and ``q2`` should accept a batch
                of observations and a batch of actions as inputs. When called,
                ``act``, ``q1``, and ``q2`` should return:
                ===========  ================  ======================================
                Call         Output Shape      Description
                ===========  ================  ======================================
                ``act``      (batch, act_dim)  | Numpy array of actions for each
                                            | observation.
                ``q1``       (batch,)          | Tensor containing one current estimate
                                            | of Q* for the provided observations
                                            | and actions. (Critical: make sure to
                                            | flatten this!)
                ``q2``       (batch,)          | Tensor containing the other current
                                            | estimate of Q* for the provided observations
                                            | and actions. (Critical: make sure to
                                            | flatten this!)
                ===========  ================  ======================================
                Calling ``pi`` should return:
                ===========  ================  ======================================
                Symbol       Shape             Description
                ===========  ================  ======================================
                ``a``        (batch, act_dim)  | Tensor containing actions from policy
                                            | given observations.
                ``logp_pi``  (batch,)          | Tensor containing log probabilities of
                                            | actions in ``a``. Importantly: gradients
                                            | should be able to flow back into ``a``.
                ===========  ================  ======================================
            policy_kwargs (dict): Any kwargs appropriate for the Policy object
                you provided to SAC.
            buffer (BaseBuffer): Maximum length of replay buffer.
            buffer_kwargs (dict):  Any kwargs appropriate for the Buffer object
            seed (int): Seed for random number generators.
            gamma (float): Discount factor. (Always between 0 and 1.)
            polyak (float): Interpolation factor in polyak averaging for target
                networks. Target networks are updated towards main networks
                according to:
                .. math:: \\theta_{\\text{targ}} \\leftarrow
                    \\rho \\theta_{\\text{targ}} + (1-\\rho) \\theta
                where :math:`\\rho` is polyak. (Always between 0 and 1, usually
                close to 1.)
            lr (float): Learning rate (used for both policy and value learning).
            alpha (float): Entropy regularization coefficient. (Equivalent to
                inverse of reward scale in the original SAC paper.)
            batch_size (int): Minibatch size for SGD.
            start_steps (int): Number of steps for uniform-random action selection,
                before running real policy. Helps exploration.
            update_after (int): Number of env interactions to collect before
                starting to do gradient descent updates. Ensures replay buffer
                is full enough for useful updates.
            update_every (int): Number of env interactions that should elapse
                between gradient descent updates. Note: Regardless of how long
                you wait between updates, the ratio of env steps to gradient steps
                is locked to 1.
            n_test_episodes (int): Number of episodes to test the deterministic
                policy at the end of each epoch.
            max_episode_len (int): Maximum length of trajectory / episode / rollout.
            n_updates (int): The number of updates per `update_every`,
                by default it is the same as save_freq
            gpu_buffer (bool): Whether or not to store replay buffer in GPU memory
            gpu_computation (bool): Whether or not to store computation graph on GPU memory
            logger (BaseLogger): Logger to use
        """
        self.logger = logger

        torch.manual_seed(seed)
        np.random.seed(seed)

        self.env = env
        self.gamma = gamma
        self.polyak = polyak
        self.n_test_episodes = n_test_episodes
        self.max_episode_len = max_episode_len
        self.start_steps = start_steps
        self.update_after = update_after
        self.update_every = update_every
        self.n_updates = n_updates
        self.batch_size = batch_size
        self.target_entropy = target_entropy
        # Entropy coefficient / Entropy temperature
        # Inverse of the reward scale
        self.alpha = alpha
        # TODO: fill this in
        # Additional properties for automatic alpha adjustment...

        # Create actor-critic module and target networks
        self.policy = policy
        self.policy_targ = deepcopy(self.policy)

        # Freeze target networks with respect to optimizers (only update via polyak averaging)
        for p in self.policy_targ.parameters():
            p.requires_grad = False

        # List of parameters for both Q-networks (save this for convenience)
        self.q_params = itertools.chain(
            self.policy.q1.parameters(), self.policy.q2.parameters()
        )

        # Experience buffer
        self.buffer = buffer

        # Count variables (protip: try to get a feel for how different size networks behave!)
        var_counts = tuple(
            count_vars(module)
            for module in [self.policy.pi, self.policy.q1, self.policy.q2]
        )
        self.logger.log_msg(
            "\nNumber of parameters: \t pi: %d, \t q1: %d, \t q2: %d\n" % var_counts
        )

        # Set up optimizers for policy and q-function
        self.pi_optimizer = Adam(self.policy.pi.parameters(), lr=lr)
        self.q_optimizer = Adam(self.q_params, lr=lr)

        # Target entropy is used when learning the entropy coefficient
        if self.target_entropy == "auto":
            # automatically set target entropy if needed
            self.target_entropy = -np.prod(self.env.action_space.shape).astype(
                np.float32
            )
        else:
            # Force conversion
            # this will also throw an error for unexpected string
            self.target_entropy = float(self.target_entropy)

        # Use the same device for alpha as for the policy
        alpha_device = next(self.policy.parameters()).device

        # The entropy coefficient or entropy can be learned automatically
        # see Automating Entropy Adjustment for Maximum Entropy RL section
        # of https://arxiv.org/abs/1812.05905
        if isinstance(self.alpha, str) and self.alpha.startswith("auto"):
            # Default initial value of alpha when learned
            init_value = 1.0
            if "_" in self.alpha:
                init_value = float(self.alpha.split("_")[1])
                assert (
                    init_value > 0.0
                ), "The initial value of alpha must be greater than 0"

            # Consider: optimizing the log of the entropy coeff which is slightly different from the paper
            # as discussed in https://github.com/rail-berkeley/softlearning/issues/37

            # TODO: fill this in
            # self.alpha = ...
            # self.alpha_optimizer = ...
        else:
            # Force conversion to float
            # this will throw an error if a malformed string (different from 'auto') is passed
            self.alpha = torch.tensor(
                float(self.alpha), dtype=torch.float32, device=alpha_device
            )

    # Set up function for computing SAC Q-losses
    def compute_loss_q(self, data):
        o, a, r, o2, ter, tru = (
            data["observation"],
            data["action"],
            data["reward"],
            data["next_observation"],
            data["terminated"],
            data["truncated"],
        )

        q1 = self.policy.q1(o, a)
        q2 = self.policy.q2(o, a)

        # Bellman backup for Q functions
        with torch.no_grad():
            # Target actions come from *current* policy
            a2, logp_a2 = self.policy.pi(o2)

            # Target Q-values
            q1_pi_targ = self.policy_targ.q1(o2, a2)
            q2_pi_targ = self.policy_targ.q2(o2, a2)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            backup = r + self.gamma * (1 - ter) * (q_pi_targ - self.alpha * logp_a2)

        # MSE loss against Bellman backup
        loss_q1 = F.mse_loss(q1, backup)
        loss_q2 = F.mse_loss(q2, backup)
        loss_q = 0.5 * (loss_q1 + loss_q2)

        # Useful info for logging
        q_info = dict(
            q1_vals=q1.detach().cpu().numpy(), q2_vals=q2.detach().cpu().numpy()
        )

        return loss_q, q_info

    # Set up function for computing SAC pi loss
    def compute_loss_pi(self, data):
        o = data["observation"]
        pi, logp_pi = self.policy.pi(o)
        q1_pi = self.policy.q1(o, pi)
        q2_pi = self.policy.q2(o, pi)
        q_pi = torch.min(q1_pi, q2_pi)

        # Entropy-regularized policy loss
        loss_pi = (self.alpha * logp_pi - q_pi).mean()

        # Useful info for logging
        pi_info = dict(log_pi=logp_pi.detach().cpu().numpy())

        return loss_pi, logp_pi, pi_info

    def compute_loss_alpha(self, logp_pi):
        # Important: detach the variable from the graph
        # so we don't change it with other losses
        # see https://github.com/rail-berkeley/softlearning/issues/60
        # TODO: fill this in
        alpha_loss = 0
        alpha = 0
        return alpha_loss, alpha
        # Remember to use target_entropy

    def update(self, data) -> dict[str, float]:
        self.policy.train()
        # First run one gradient descent step for Q1 and Q2
        self.q_optimizer.zero_grad()
        loss_q, q_info = self.compute_loss_q(data)
        loss_q.backward()
        self.q_optimizer.step()

        # Freeze Q-networks so you don"t waste computational effort
        # computing gradients for them during the policy learning step.
        for p in self.q_params:
            p.requires_grad = False

        # Next run one gradient descent step for pi.
        self.pi_optimizer.zero_grad()
        loss_pi, logp_pi, pi_info = self.compute_loss_pi(data)
        loss_pi.backward()
        self.pi_optimizer.step()

        # Unfreeze Q-networks so you can optimize it at next DDPG step.
        for p in self.q_params:
            p.requires_grad = True

        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(
                self.policy.parameters(), self.policy_targ.parameters()
            ):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)

        # Optimize entropy coefficient, also called
        # entropy temperature or alpha in the paper
        if self.alpha_optimizer is not None:
            alpha_loss, alpha = self.compute_loss_alpha(logp_pi)
            # TODO: fill this in
            # Update alpha...
        else:
            alpha_loss = np.array(0)

        return {
            "q": loss_q.item(),
            "pi": loss_pi.item(),
            "alpha": alpha_loss.item(),
        }

    def test(
        self,
        env: gym.Env,
        n_episodes: int,
        sleep: float = 0,
        store_experience: bool = False,
        render: bool = False,
        up_to_buffer_size=False,
    ) -> dict:
        ep_returns = []
        ep_lengths = []
        ep_successes = []
        self.policy.eval()

        if render:
            frames = []

        for _ in range(n_episodes):
            (o, i), d, ep_ret, ep_len = env.reset(), False, 0, 0

            if store_experience:
                self.buffer.start_episode()

            if render:
                frames.append([])

            while not (d or (ep_len == self.max_episode_len)):
                # Take deterministic actions at test time
                a = self.policy.act(o, True)

                o2, r, ter, tru, i = env.step(a)

                if store_experience:
                    # Store experience to replay buffer
                    self.buffer.store(o, a, r, o2, ter, tru, i)

                o = o2

                if render:
                    frame = env.render()
                    frames[-1].append(frame)

                if sleep > 0:
                    time.sleep(sleep)

                d = ter or tru
                ep_ret += r
                ep_len += 1

            if store_experience:
                self.buffer.end_episode()

            ep_returns.append(ep_ret)
            ep_lengths.append(ep_len)

            if "is_success" in i:
                ep_successes.append(i["is_success"])

            if up_to_buffer_size and self.buffer.size == self.buffer.max_size:
                break

        results = {
            "mean_ep_ret": np.array(ep_returns).mean(),
            "mean_ep_len": np.array(ep_lengths).mean(),
        }

        if len(ep_successes) > 0:
            results["success_rate"] = np.array(ep_successes).mean()

        if render:
            results["ep_frames"] = frames

        return results


    def train(self, n_steps, log_interval=1000, callbacks=[]):
        # Prepare for interaction with environment
        (o, i), ep_ret, ep_len = self.env.reset(), 0, 0
        self.buffer.start_episode()
        test_ep_return = None

        # Main loop: collect experience in env and update/log each epoch
        with trange(n_steps) as prgs:
            for t in prgs:
                # Until start_steps have elapsed, randomly sample actions
                # from a uniform distribution for better exploration. Afterwards,
                # use the learned policy.
                if t > self.start_steps:
                    a = self.policy.act(o)
                else:
                    a = self.env.action_space.sample()

                # Step the env
                o2, r, ter, tru, i = self.env.step(a)
                ep_ret += r
                ep_len += 1

                # Store experience to replay buffer
                self.buffer.store(o, a, r, o2, ter, tru, i)

                # Super critical, easy to overlook step: make sure to update
                # most recent observation!
                o = o2

                # End of trajectory handling
                if (ter or tru) or (ep_len == self.max_episode_len):
                    self.logger.log_scalar("ep_return", ep_ret, t)
                    self.logger.log_scalar("ep_length", ep_len, t)
                    self.buffer.end_episode()
                    (o, i), ep_ret, ep_len = self.env.reset(), 0, 0
                    self.buffer.start_episode()

                # Update handling
                if t >= self.update_after and t % self.update_every == 0:
                    for j in range(self.n_updates or self.update_every):
                        batch = self.buffer.sample_batch(self.batch_size)
                        losses = self.update(data=batch)
                        self.logger.log_scalar("loss_q", losses["q"], t)
                        self.logger.log_scalar("loss_pi", losses["pi"], t)
                        self.logger.log_scalar("loss_alpha", losses["alpha"], t)
                        self.logger.log_scalar("alpha", self.alpha, t)

                # End of epoch handling
                if t % log_interval == 0:
                    # Test the performance of the deterministic version of the agent.
                    results = self.test(self.env, self.n_test_episodes)
                    prgs.set_description(f"test_ep_return {results['mean_ep_ret']:.3g}")
                    self.logger.log_scalar("test_ep_return", results["mean_ep_ret"], t)
                    self.logger.log_scalar("test_ep_length", results["mean_ep_len"], t)

                    stop = False

                    for c in callbacks:
                        stop = c(results)
                        if stop:
                            break

                    if stop:
                        break

        return test_ep_return


    def save(self, path: str):
        torch.save(
            {
                "policy_state_dict": self.policy.state_dict(),
                "alpha": self.alpha,
                "pi_optimizer_state_dict": self.pi_optimizer.state_dict(),
                "q_optimizer_state_dict": self.q_optimizer.state_dict(),
                # TODO: fill this in
                # "alpha_optimizer_state_dict": ...,
            },
            path,
        )

    def load(self, path: str):
        checkpoint = torch.load(path)
        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        self.pi_optimizer.load_state_dict(checkpoint["pi_optimizer_state_dict"])
        self.q_optimizer.load_state_dict(checkpoint["q_optimizer_state_dict"])
        self.alpha = checkpoint["alpha"]
        # TODO: fill this in
        # self.alpha_optimizer = ...
