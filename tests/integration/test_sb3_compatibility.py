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

    @pytest.fixture
    def env(self):
        """테스트용 환경 픽스처"""
        env = gym.make('gymnasium_env/GridWorld-v0', size=5)
        yield env
        env.close()

    def test_sb3_env_checker(self, env):
        """SB3 환경 검증 테스트"""
        try:
            check_env(env)
        except Exception as e:
            pytest.fail(f"Environment failed SB3 check_env: {e}")

    def test_a2c_compatibility(self, env):
        """A2C 알고리즘 호환성 테스트"""
        try:
            # A2C 모델 생성 (매우 적은 스텝으로 빠른 테스트)
            model = A2C("MultiInputPolicy", env, verbose=0)

            # 짧은 학습 실행
            model.learn(total_timesteps=100)

            # 예측 테스트
            obs, _ = env.reset()
            action, _states = model.predict(obs, deterministic=True)
            assert env.action_space.contains(action)

        except Exception as e:
            pytest.fail(f"A2C compatibility test failed: {e}")

    def test_ppo_compatibility(self, env):
        """PPO 알고리즘 호환성 테스트"""
        try:
            # PPO 모델 생성
            model = PPO("MultiInputPolicy", env, verbose=0)

            # 짧은 학습 실행
            model.learn(total_timesteps=100)

            # 예측 테스트
            obs, _ = env.reset()
            action, _states = model.predict(obs, deterministic=True)
            assert env.action_space.contains(action)

        except Exception as e:
            pytest.fail(f"PPO compatibility test failed: {e}")

    def test_dqn_compatibility(self, env):
        """DQN 알고리즘 호환성 테스트"""
        try:
            # DQN은 Dict observation을 직접 지원하지 않으므로
            # 이 테스트는 예상된 실패일 수 있음
            pytest.skip("DQN requires flattened observations for Dict spaces")

        except Exception as e:
            # DQN이 Dict 관찰공간을 지원하지 않는다면 예상된 동작
            pass

    def test_vectorized_env(self):
        """벡터화된 환경 테스트"""
        try:
            # 다중 환경 생성
            def make_env():
                return gym.make('gymnasium_env/GridWorld-v0', size=5)

            vec_env = DummyVecEnv([make_env for _ in range(2)])

            # 환경 초기화 테스트
            observations = vec_env.reset()
            assert len(observations) == 2

            # 액션 실행 테스트
            actions = [vec_env.action_space.sample() for _ in range(2)]
            observations, rewards, dones, infos = vec_env.step(actions)

            assert len(observations) == 2
            assert len(rewards) == 2
            assert len(dones) == 2
            assert len(infos) == 2

            vec_env.close()

        except Exception as e:
            pytest.fail(f"Vectorized environment test failed: {e}")

    def test_model_save_load(self, env):
        """모델 저장/로드 테스트"""
        import tempfile
        import os

        try:
            # A2C 모델 생성 및 짧은 학습
            model = A2C("MultiInputPolicy", env, verbose=0)
            model.learn(total_timesteps=50)

            # 임시 파일에 모델 저장
            with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as tmp_file:
                model_path = tmp_file.name

            model.save(model_path)

            # 모델 로드 및 테스트
            loaded_model = A2C.load(model_path)

            obs, _ = env.reset()
            original_action, _ = model.predict(obs, deterministic=True)
            loaded_action, _ = loaded_model.predict(obs, deterministic=True)

            # 같은 상태에서 같은 액션을 예측해야 함
            assert original_action == loaded_action

            # 임시 파일 정리
            os.unlink(model_path)

        except Exception as e:
            pytest.fail(f"Model save/load test failed: {e}")

    def test_evaluation(self, env):
        """모델 평가 테스트"""
        try:
            # 간단한 모델 생성
            model = A2C("MultiInputPolicy", env, verbose=0)
            model.learn(total_timesteps=100)

            # 모델 평가
            mean_reward, std_reward = evaluate_policy(
                model, env, n_eval_episodes=3, deterministic=True
            )

            # 평가 결과가 유효한 값인지 확인
            assert isinstance(mean_reward, (int, float))
            assert isinstance(std_reward, (int, float))
            assert not np.isnan(mean_reward)
            assert not np.isnan(std_reward)

        except Exception as e:
            pytest.fail(f"Model evaluation test failed: {e}")

    def test_observation_preprocessing(self, env):
        """관찰 전처리 테스트"""
        try:
            obs, _ = env.reset()

            # SB3가 Dict 관찰을 올바르게 처리하는지 확인
            model = A2C("MultiInputPolicy", env, verbose=0)

            # 모델이 관찰을 받아들일 수 있는지 테스트
            action, _ = model.predict(obs)
            assert env.action_space.contains(action)

        except Exception as e:
            pytest.fail(f"Observation preprocessing test failed: {e}")


if __name__ == "__main__":
    pytest.main([__file__])
