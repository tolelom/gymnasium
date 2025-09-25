import gymnasium as gym
import gymnasium_env

from stable_baselines3 import A2C
from stable_baselines3.common.env_checker import check_env


env = gym.make("gymnasium_env/GridWorld-v0", render_mode="rgb_array")
check_env(env)

model = A2C("MultiInputPolicy", env, verbose=1).learn(total_timesteps=10000)