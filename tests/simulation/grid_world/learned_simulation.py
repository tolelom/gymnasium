"""학습된 모델로 GridWorld를 시뮬레이션하는 코드"""

import os

import gymnasium as gym
import gymnasium_env  # noqa: F401
from stable_baselines3 import A2C, PPO, DQN

from tests.simulation.common.simulation_utils import (
    calculate_episode_stats,
    print_episode_stats,
    run_evaluation_episodes,
)


MODEL_LOADERS = {"A2C": A2C.load, "PPO": PPO.load, "DQN": DQN.load}


def load_model(model_path: str, model_type: str):
    if model_type not in MODEL_LOADERS:
        raise ValueError(f"지원하지 않는 모델 타입: {model_type}")
    return MODEL_LOADERS[model_type](model_path)


def load_and_simulate(model_path, model_type="A2C", episodes=5, render_mode="human"):
    """학습된 모델을 로드해 시뮬레이션하고 통계를 출력."""
    env = gym.make("gymnasium_env/GridWorld-v0", render_mode=render_mode, size=5)
    model = load_model(model_path, model_type)

    rewards, steps = run_evaluation_episodes(
        env,
        model,
        n_episodes=episodes,
        deterministic=True,
        render=(render_mode == "human"),
        delay=0.8 if render_mode == "human" else 0.0,
    )
    env.close()

    stats = calculate_episode_stats(rewards, steps)
    print_episode_stats(stats, title=f"{model_type} @ {os.path.basename(model_path)}")
    return stats


def compare_models(model_configs, episodes=5):
    """여러 모델을 렌더링 없이 비교."""
    results = {}
    for model_path, model_type, name in model_configs:
        if not os.path.exists(model_path):
            print(f"모델 파일 없음: {model_path}")
            continue

        env = gym.make("gymnasium_env/GridWorld-v0", size=5)
        model = load_model(model_path, model_type)
        rewards, steps = run_evaluation_episodes(
            env, model, n_episodes=episodes, deterministic=True, render=False
        )
        env.close()

        results[name] = calculate_episode_stats(rewards, steps)
        print_episode_stats(results[name], title=name)

    return results


if __name__ == "__main__":
    configs = [
        ("models/a2c_gridworld.zip", "A2C", "A2C 기본"),
        ("models/ppo_gridworld.zip", "PPO", "PPO 기본"),
    ]
    compare_models(configs, episodes=5)
