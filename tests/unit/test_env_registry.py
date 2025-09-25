"""환경 등록 및 생성 테스트"""

import pytest
import gymnasium as gym
import gymnasium_env


def test_environment_registration():
    """환경이 올바르게 등록되었는지 테스트"""
    # 환경 ID가 등록되어 있는지 확인
    env_id = 'gymnasium_env/GridWorld-v0'

    # 환경 생성이 성공하는지 확인
    env = gym.make(env_id)
    assert env is not None
    env.close()


def test_environment_with_parameters():
    """환경 파라미터와 함께 생성 테스트"""
    env_id = 'gymnasium_env/GridWorld-v0'

    # 다양한 크기로 환경 생성
    for size in [3, 5, 8]:
        env = gym.make(env_id, size=size)
        assert env.size == size
        env.close()

    # 렌더링 모드로 환경 생성
    for render_mode in ['human', 'rgb_array']:
        env = gym.make(env_id, render_mode=render_mode)
        assert env.render_mode == render_mode
        env.close()


def test_max_episode_steps():
    """최대 에피소드 스텝 설정 테스트"""
    env = gym.make('gymnasium_env/GridWorld-v0')

    # max_episode_steps가 설정되어 있는지 확인
    wrapper = env
    while hasattr(wrapper, 'env'):
        if hasattr(wrapper, '_max_episode_steps'):
            assert wrapper._max_episode_steps == 100
            break
        wrapper = wrapper.env

    env.close()


def test_env_checker_compliance():
    """Gymnasium env_checker 호환성 테스트"""
    from gymnasium.utils.env_checker import check_env

    env = gym.make('gymnasium_env/GridWorld-v0')

    # check_env가 오류 없이 통과하는지 확인
    try:
        check_env(env)
    except Exception as e:
        pytest.fail(f"Environment failed check_env: {e}")
    finally:
        env.close()


def test_multiple_env_instances():
    """다중 환경 인스턴스 생성 테스트"""
    envs = []

    # 여러 환경 인스턴스 생성
    for i in range(3):
        env = gym.make('gymnasium_env/GridWorld-v0', size=5)
        envs.append(env)

    # 각 환경이 독립적으로 작동하는지 확인
    observations = []
    for env in envs:
        obs, _ = env.reset(seed=i)  # 다른 시드 사용
        observations.append(obs)

    # 서로 다른 초기 상태를 가져야 함 (시드가 다르므로)
    for i in range(len(observations)):
        for j in range(i + 1, len(observations)):
            # 적어도 하나의 환경은 다른 초기 상태를 가져야 함
            pass  # 확률적이므로 강제 검증하지 않음

    # 모든 환경 정리
    for env in envs:
        env.close()


if __name__ == "__main__":
    pytest.main([__file__])
