# A script to train a Soft Actor-Critic (SAC) agent with Hindsight Experience Replay (HER) on a specified gym environment.
# By default it uses the PandaReach-v3 environment from the panda_gym package.
# But it can be easily modified to use any other gym environment, including those from the gymnasium_robotics package.
# Just uncomment the import statement for gymnasium_robotics and register the environments and use for example the FetchReach-v3 environment.

# Also give it a try on push and pick and place tasks: PandaPush-v3, PandaPickAndPlace-v3, FetchPush-v3, FetchPickAndPlace-v3

import argparse

import gymnasium as gym
# Uncomment the following line to use gymnasium_robotics environments
# import gymnasium_robotics
import panda_gym
import torch

# Uncomment the following lines to register gymnasium_robotics environments
# gym.register_envs(gymnasium_robotics)

from asdf.algos import SAC
from asdf.buffers import HerReplayBuffer
from asdf.extractors import DictExtractor
from asdf.loggers import TensorboardLogger
from asdf.policies import MlpPolicy


# There are two challenges in this exercise:
# 1. Implement the Hindsight Experience Replay (HER) algorithm.
#    This is done in the HerReplayBuffer class.
# 2. Improve the SAC algorithm with an automatically adjusted temperature (alpha) parameter.
#    This is done in the SAC class.
def main(env_id: str) -> None:
    if torch.cuda.is_available():
        device = "cuda"
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = "mps"
        print("Using Apple Silicon GPU")
    else:
        print("Using CPU")
        device = "cpu"

    env = gym.make(env_id)

    policy = MlpPolicy(
        env.observation_space,
        env.action_space,
        hidden_sizes=[64, 64],
        extractor_type=DictExtractor,
    )
    policy.to(device)

    buffer = HerReplayBuffer(
        env=env,
        size=1_000_000,
        n_sampled_goal=3,
        goal_selection_strategy="future",
        device=device,
    )
    logger = TensorboardLogger()
    logger.open()

    algo = SAC(
        env,
        policy=policy,
        buffer=buffer,
        update_every=1,
        update_after=1000,
        batch_size=64,
        # alpha="auto", # use automatic alpha adjustment (uncoment when implemented)
        alpha=0.05, # use fixed alpha (comment out when implementing automatic alpha adjustment)
        gamma=0.9,
        # polyak=0.95,
        lr=1e-4,
        logger=logger,
        max_episode_len=100,
        start_steps=1_000,
    )
    algo.train(n_steps=100_000, log_interval=1000)
    env.close()
    logger.close()

    policy.cpu()
    env = gym.make(env_id, render_mode="human")
    test_rew, test_ep_len = algo.test(env, n_episodes=50, sleep=1 / 30)
    env.close()
    print(f"Test reward {test_rew}, Test episode length: {test_ep_len}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env", type=str, default="PandaReach-v3", help="Gym environment ID"
    )

    args = parser.parse_args()

    main(args.env)
