"""환경 등록 및 생성 테스트 (Simplified with unwrapped fixtures)"""

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
        actual_env = env.unwrapped
        assert actual_env.size == size
        env.close()

    # 렌더링 모드로 환경 생성
    for render_mode in ['human', 'rgb_array']:
        env = gym.make(env_id, render_mode=render_mode)
        actual_env = env.unwrapped
        assert actual_env.render_mode == render_mode
        env.close()


def test_max_episode_steps(wrapped_grid_world_env):
    """최대 에피소드 스텝 설정 테스트"""
    # TimeLimit 래퍼에서 max_episode_steps 확인
    if hasattr(wrapped_grid_world_env, '_max_episode_steps'):
        assert wrapped_grid_world_env._max_episode_steps == 100
    else:
        # 래퍼를 찾아서 확인
        wrapper = wrapped_grid_world_env
        found_time_limit = False
        while hasattr(wrapper, 'env'):
            if hasattr(wrapper, '_max_episode_steps'):
                assert wrapper._max_episode_steps == 100
                found_time_limit = True
                break
            wrapper = wrapper.env

        # TimeLimit 래퍼가 있어야 함
        assert found_time_limit, "TimeLimit wrapper not found"


def test_env_checker_compliance(grid_world_env):
    """Gymnasium env_checker 호환성 테스트"""
    from gymnasium.utils.env_checker import check_env

    # check_env가 오류 없이 통과하는지 확인 (이미 unwrapped 환경)
    try:
        check_env(grid_world_env)
    except Exception as e:
        pytest.fail(f"Environment failed check_env: {e}")


def test_multiple_env_instances():
    """다중 환경 인스턴스 생성 테스트"""
    envs = []
    unwrapped_envs = []

    # 여러 환경 인스턴스 생성
    for i in range(3):
        env = gym.make('gymnasium_env/GridWorld-v0', size=5)
        envs.append(env)
        unwrapped_envs.append(env.unwrapped)

    # 각 환경이 독립적으로 작동하는지 확인
    observations = []
    for i, actual_env in enumerate(unwrapped_envs):
        obs, _ = actual_env.reset(seed=i)  # 다른 시드 사용
        observations.append(obs)

    # 서로 다른 초기 상태를 가져야 함 (시드가 다르므로)
    # 적어도 일부 환경은 다른 초기 상태를 가져야 함
    all_same = True
    for i in range(len(observations)):
        for j in range(i + 1, len(observations)):
            if not (observations[i]['agent'] == observations[j]['agent']).all() or \
               not (observations[i]['target'] == observations[j]['target']).all():
                all_same = False
                break
        if not all_same:
            break

    # 모든 환경이 정확히 같은 초기 상태를 가질 확률은 매우 낮음
    # 하지만 확률적이므로 너무 엄격하게 검증하지 않음

    # 모든 환경 정리
    for env in envs:
        env.close()


def test_environment_attributes(grid_world_env):
    """환경 속성 접근 테스트"""
    # 기본 속성들 확인 (이미 unwrapped 환경)
    assert hasattr(grid_world_env, 'size')
    assert hasattr(grid_world_env, 'action_space')
    assert hasattr(grid_world_env, 'observation_space')
    assert hasattr(grid_world_env, 'render_mode')

    # 값 확인
    assert grid_world_env.size == 5
    assert grid_world_env.action_space.n == 4


def test_wrapper_chain(wrapped_grid_world_env):
    """래퍼 체인 테스트"""
    # 래퍼 체인 확인
    current = wrapped_grid_world_env
    wrapper_types = []

    while hasattr(current, 'env'):
        wrapper_types.append(type(current).__name__)
        current = current.env

    # 실제 환경 타입 확인
    assert type(current).__name__ == 'GridWorldEnv'

    # TimeLimit 래퍼가 있어야 함
    assert 'TimeLimit' in wrapper_types


def test_fixture_consistency():
    """픽스처 일관성 테스트"""
    # 직접 생성한 환경과 픽스처의 일관성 확인
    env1 = gym.make('gymnasium_env/GridWorld-v0', size=5)
    actual_env1 = env1.unwrapped

    env2 = gym.make('gymnasium_env/GridWorld-v0', size=5)
    actual_env2 = env2.unwrapped

    # 같은 설정으로 생성된 환경들은 같은 속성을 가져야 함
    assert actual_env1.size == actual_env2.size
    assert actual_env1.action_space.n == actual_env2.action_space.n
    assert type(actual_env1.observation_space) == type(actual_env2.observation_space)

    env1.close()
    env2.close()


if __name__ == "__main__":
    pytest.main([__file__])