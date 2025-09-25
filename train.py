import gymnasium as gym
import gymnasium_env

env = gym.make('gymnasium_env/GridWorld-v0')

obs, info = env.reset()

for _ in range(100):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        obs, info = env.reset()

env.close()

print("Environment tested successfully!")