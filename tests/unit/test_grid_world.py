"""GridWorld 환경 단위 테스트"""

import pytest
import numpy as np
import gymnasium as gym
import gymnasium_env


class TestGridWorldEnv:
    """GridWorld 환경 테스트 클래스"""

    @pytest.fixture
    def env(self):
        """테스트용 GridWorld 환경 픽스처"""
        env = gym.make('gymnasium_env/GridWorld-v0', size=5)
        yield env
        env.close()

    def test_environment_creation(self):
        """환경 생성 테스트"""
        env = gym.make('gymnasium_env/GridWorld-v0', size=5)
        assert env is not None
        assert env.size == 5
        env.close()

    def test_observation_space(self, env):
        """관찰 공간 테스트"""
        assert env.observation_space is not None
        assert 'agent' in env.observation_space.spaces
        assert 'target' in env.observation_space.spaces

        # 관찰 공간 범위 확인
        agent_space = env.observation_space.spaces['agent']
        target_space = env.observation_space.spaces['target']

        assert agent_space.low[0] == 0 and agent_space.high[0] == 4
        assert target_space.low[0] == 0 and target_space.high[0] == 4

    def test_action_space(self, env):
        """행동 공간 테스트"""
        assert env.action_space.n == 4  # 4방향 이동

        # 모든 행동이 유효한지 확인
        for action in range(4):
            assert env.action_space.contains(action)

    def test_reset(self, env):
        """환경 초기화 테스트"""
        observation, info = env.reset()

        # 관찰값 구조 확인
        assert 'agent' in observation
        assert 'target' in observation
        assert 'distance' in info

        # 위치값 유효성 확인
        agent_pos = observation['agent']
        target_pos = observation['target']

        assert 0 <= agent_pos[0] < 5 and 0 <= agent_pos[1] < 5
        assert 0 <= target_pos[0] < 5 and 0 <= target_pos[1] < 5

        # 에이전트와 목표가 다른 위치에 있는지 확인
        assert not np.array_equal(agent_pos, target_pos)

        # 맨해튼 거리 계산 확인
        expected_distance = np.abs(agent_pos[0] - target_pos[0]) + np.abs(agent_pos[1] - target_pos[1])
        assert abs(info['distance'] - expected_distance) < 1e-6

    def test_step_mechanics(self, env):
        """스텝 실행 메커니즘 테스트"""
        observation, info = env.reset()
        initial_pos = observation['agent'].copy()

        # 오른쪽 이동 (action=0)
        observation, reward, terminated, truncated, info = env.step(0)

        # 반환값 타입 확인
        assert isinstance(reward, (int, float))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

        # 위치 변화 확인 (경계에 있지 않은 경우)
        if initial_pos[0] < 4:  # 오른쪽으로 이동 가능한 경우
            assert observation['agent'][0] == initial_pos[0] + 1
            assert observation['agent'][1] == initial_pos[1]

    def test_boundary_conditions(self, env):
        """경계 조건 테스트"""
        # 환경을 특정 위치로 강제 설정
        env.reset()

        # 왼쪽 경계에서 왼쪽으로 이동
        env._agent_location = np.array([0, 2])
        observation, _, _, _, _ = env.step(2)  # LEFT
        assert observation['agent'][0] == 0  # 경계에서 멈춤

        # 오른쪽 경계에서 오른쪽으로 이동
        env._agent_location = np.array([4, 2])
        observation, _, _, _, _ = env.step(0)  # RIGHT
        assert observation['agent'][0] == 4  # 경계에서 멈춤

    def test_goal_reaching(self, env):
        """목표 도달 테스트"""
        env.reset()

        # 에이전트를 목표 근처로 강제 이동
        target_pos = env._target_location.copy()
        adjacent_pos = target_pos.copy()
        adjacent_pos[0] = max(0, min(4, target_pos[0] - 1))  # 왼쪽으로 한 칸

        env._agent_location = adjacent_pos

        # 목표로 이동
        if target_pos[0] > adjacent_pos[0]:
            action = 0  # RIGHT
        else:
            action = 2  # LEFT

        observation, reward, terminated, truncated, info = env.step(action)

        if np.array_equal(observation['agent'], target_pos):
            assert terminated == True
            assert reward == 1.0
            assert info['distance'] == 0.0

    def test_reward_system(self, env):
        """보상 시스템 테스트"""
        observation, info = env.reset()

        # 목표에 도달하지 않은 경우의 보상
        initial_distance = info['distance']
        if initial_distance > 0:  # 이미 목표에 도달하지 않은 경우
            observation, reward, terminated, truncated, info = env.step(env.action_space.sample())

            if not terminated:
                assert reward == -0.01  # 스텝 패널티
            else:
                assert reward == 1.0  # 목표 도달 보상

    def test_different_sizes(self):
        """다양한 그리드 크기 테스트"""
        for size in [3, 5, 8, 10]:
            env = gym.make('gymnasium_env/GridWorld-v0', size=size)
            observation, info = env.reset()

            # 관찰 공간이 크기에 맞게 설정되었는지 확인
            agent_pos = observation['agent']
            target_pos = observation['target']

            assert 0 <= agent_pos[0] < size and 0 <= agent_pos[1] < size
            assert 0 <= target_pos[0] < size and 0 <= target_pos[1] < size

            env.close()

    def test_deterministic_reset(self, env):
        """결정적 초기화 테스트"""
        # 같은 시드로 두 번 초기화
        observation1, info1 = env.reset(seed=42)
        observation2, info2 = env.reset(seed=42)

        # 같은 시드로 초기화한 경우 같은 상태여야 함
        assert np.array_equal(observation1['agent'], observation2['agent'])
        assert np.array_equal(observation1['target'], observation2['target'])
        assert info1['distance'] == info2['distance']

    def test_rendering_modes(self):
        """렌더링 모드 테스트"""
        # human 모드
        env_human = gym.make('gymnasium_env/GridWorld-v0', render_mode='human')
        env_human.reset()
        env_human.render()  # 오류 없이 실행되어야 함
        env_human.close()

        # rgb_array 모드
        env_rgb = gym.make('gymnasium_env/GridWorld-v0', render_mode='rgb_array')
        env_rgb.reset()
        rgb_array = env_rgb.render()
        if rgb_array is not None:
            assert isinstance(rgb_array, np.ndarray)
            assert len(rgb_array.shape) == 3  # (height, width, channels)
        env_rgb.close()


if __name__ == "__main__":
    pytest.main([__file__])
