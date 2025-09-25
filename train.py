import gymnasium as gym
import gymnasium_env
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

env = gym.make('gymnasium_env/GridWorld-v0')

check_env(env)

model = PPO("MultiInputPolicy", env, verbose=1)
model.learn(total_timesteps=10_000)

# 4. 모델 저장
model.save("ppo_gridworld")

# 5. 학습된 모델 테스트
obs, info = env.reset()
for i in range(100):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()