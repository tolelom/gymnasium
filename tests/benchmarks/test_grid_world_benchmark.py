"""GridWorld 환경 성능 벤치마크 (pytest-benchmark)"""

import gymnasium as gym
import gymnasium_env  # noqa: F401
import pytest


SIZES = [5, 10, 20]


@pytest.mark.parametrize("size", SIZES)
def test_env_creation(benchmark, size):
    def create_and_close():
        env = gym.make("gymnasium_env/GridWorld-v0", size=size)
        env.close()

    benchmark(create_and_close)


@pytest.mark.parametrize("size", SIZES)
def test_step_performance(benchmark, size):
    env = gym.make("gymnasium_env/GridWorld-v0", size=size)
    env.reset()

    def run_step():
        _, _, terminated, truncated, _ = env.step(env.action_space.sample())
        if terminated or truncated:
            env.reset()

    try:
        benchmark(run_step)
    finally:
        env.close()


@pytest.mark.parametrize("size", SIZES)
def test_rgb_render(benchmark, size):
    env = gym.make("gymnasium_env/GridWorld-v0", size=size, render_mode="rgb_array")
    env.reset()
    try:
        benchmark(env.render)
    finally:
        env.close()


@pytest.mark.parametrize("algo_name", ["A2C", "PPO"])
def test_short_training(benchmark, algo_name):
    sb3 = pytest.importorskip("stable_baselines3")
    algo = getattr(sb3, algo_name)

    def train():
        env = gym.make("gymnasium_env/GridWorld-v0", size=5)
        model = algo("MultiInputPolicy", env, verbose=0)
        model.learn(total_timesteps=1000)
        env.close()

    benchmark.pedantic(train, rounds=1, iterations=1)
