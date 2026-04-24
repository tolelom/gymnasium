import os
from dataclasses import asdict

import gymnasium as gym
import gymnasium_env  # noqa: F401  (환경 등록용)

from stable_baselines3 import A2C
from stable_baselines3.common.env_checker import check_env

from training.configs.grid_world_config import (
    A2CConfig,
    GridWorldConfig,
    TrainingConfig,
)


def train():
    env_cfg = GridWorldConfig()
    train_cfg = TrainingConfig()
    algo_cfg = A2CConfig()

    env = gym.make(
        "gymnasium_env/GridWorld-v0",
        render_mode=env_cfg.render_mode,
        size=env_cfg.size,
    )
    check_env(env)

    model = A2C(
        env=env,
        verbose=train_cfg.verbose,
        tensorboard_log=train_cfg.tensorboard_log,
        **asdict(algo_cfg),
    )
    model.learn(total_timesteps=train_cfg.total_timesteps)

    os.makedirs(train_cfg.save_path, exist_ok=True)
    model.save(os.path.join(train_cfg.save_path, "a2c_gridworld"))
    env.close()


if __name__ == "__main__":
    train()
