"""Stable Baselines3 호환성 테스트"""

import pytest
import numpy as np
import gymnasium as gym
import gymnasium_env

try:
    from stable_baselines3 import A2C, PPO, DQN
    from stable_baselines3.common.env_checker import check_env
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.evaluation import evaluate_policy

    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False


@pytest.mark.skipif(not SB3_AVAILABLE, reason="Stable Baselines3 not installed")
class TestSB3Compatibility:
    """Stable Baselines3 호환성 테스트 클래스"""

    def test_sb3_env_checker(self, wrapped_grid_world_env):
        """SB3 환경 검증 테스트"""
        check_env(wrapped_grid_world_env)

    def test_a2c_compatibility(self, wrapped_grid_world_env):
        """A2C 알고리즘 호환성 테스트"""
        model = A2C("MultiInputPolicy", wrapped_grid_world_env, verbose=0)
        model.learn(total_timesteps=100)

        obs, _ = wrapped_grid_world_env.reset()
        action, _ = model.predict(obs, deterministic=True)
        assert wrapped_grid_world_env.action_space.contains(action)

    def test_ppo_compatibility(self, wrapped_grid_world_env):
        """PPO 알고리즘 호환성 테스트"""
        model = PPO("MultiInputPolicy", wrapped_grid_world_env, verbose=0)
        model.learn(total_timesteps=100)

        obs, _ = wrapped_grid_world_env.reset()
        action, _ = model.predict(obs, deterministic=True)
        assert wrapped_grid_world_env.action_space.contains(action)

    def test_dqn_compatibility(self):
        """DQN은 Dict observation을 직접 지원하지 않음"""
        pytest.skip("DQN requires flattened observations for Dict spaces")

    def test_vectorized_env(self):
        """벡터화된 환경 테스트"""
        def make_env():
            return gym.make('gymnasium_env/GridWorld-v0', size=5)

        vec_env = DummyVecEnv([make_env for _ in range(2)])

        observations = vec_env.reset()
        assert len(observations) == 2

        actions = [vec_env.action_space.sample() for _ in range(2)]
        observations, rewards, dones, infos = vec_env.step(actions)

        assert len(observations) == 2
        assert len(rewards) == 2
        assert len(dones) == 2
        assert len(infos) == 2

        vec_env.close()

    def test_model_save_load(self, wrapped_grid_world_env, tmp_path):
        """모델 저장/로드 테스트"""
        model = A2C("MultiInputPolicy", wrapped_grid_world_env, verbose=0)
        model.learn(total_timesteps=50)

        model_path = tmp_path / "model.zip"
        model.save(str(model_path))

        loaded_model = A2C.load(str(model_path))

        obs, _ = wrapped_grid_world_env.reset()
        original_action, _ = model.predict(obs, deterministic=True)
        loaded_action, _ = loaded_model.predict(obs, deterministic=True)

        assert original_action == loaded_action

    def test_evaluation(self, wrapped_grid_world_env):
        """모델 평가 테스트"""
        model = A2C("MultiInputPolicy", wrapped_grid_world_env, verbose=0)
        model.learn(total_timesteps=100)

        mean_reward, std_reward = evaluate_policy(
            model, wrapped_grid_world_env, n_eval_episodes=3, deterministic=True
        )

        assert isinstance(mean_reward, (int, float))
        assert isinstance(std_reward, (int, float))
        assert not np.isnan(mean_reward)
        assert not np.isnan(std_reward)

    def test_observation_preprocessing(self, wrapped_grid_world_env):
        """Dict observation 처리 확인"""
        obs, _ = wrapped_grid_world_env.reset()

        model = A2C("MultiInputPolicy", wrapped_grid_world_env, verbose=0)
        action, _ = model.predict(obs)
        assert wrapped_grid_world_env.action_space.contains(action)


if __name__ == "__main__":
    pytest.main([__file__])