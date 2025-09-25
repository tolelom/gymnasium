"""GridWorld 환경 단위 테스트 (Simplified with unwrapped fixtures)"""

import pytest
import numpy as np
import gymnasium as gym
import gymnasium_env


class TestGridWorldEnv:
    """GridWorld 환경 테스트 클래스"""

    def test_environment_creation(self):
        """환경 생성 테스트"""
        env = gym.make('gymnasium_env/GridWorld-v0', size=5)
        assert env is not None
        actual_env = env.unwrapped
        assert actual_env.size == 5
        env.close()

    def test_observation_space(self, grid_world_env):
        """관찰 공간 테스트"""
        assert grid_world_env.observation_space is not None
        assert 'agent' in grid_world_env.observation_space.spaces
        assert 'target' in grid_world_env.observation_space.spaces

        # 관찰 공간 범위 확인
        agent_space = grid_world_env.observation_space.spaces['agent']
        target_space = grid_world_env.observation_space.spaces['target']

        assert agent_space.low[0] == 0 and agent_space.high[0] == 4
        assert target_space.low[0] == 0 and target_space.high[0] == 4

    def test_action_space(self, grid_world_env):
        """행동 공간 테스트"""
        assert grid_world_env.action_space.n == 4  # 4방향 이동

        # 모든 행동이 유효한지 확인
        for action in range(4):
            assert grid_world_env.action_space.contains(action)

    def test_reset(self, grid_world_env):
        """환경 초기화 테스트"""
        observation, info = grid_world_env.reset()

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

    def test_step_mechanics(self, grid_world_env):
        """스텝 실행 메커니즘 테스트"""
        observation, info = grid_world_env.reset()
        initial_pos = observation['agent'].copy()

        # 오른쪽 이동 (action=0)
        observation, reward, terminated, truncated, info = grid_world_env.step(0)

        # 반환값 타입 확인
        assert isinstance(reward, (int, float))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

        # 위치 변화 확인 (경계에 있지 않은 경우)
        if initial_pos[0] < 4:  # 오른쪽으로 이동 가능한 경우
            assert observation['agent'][0] == initial_pos[0] + 1
            assert observation['agent'][1] == initial_pos[1]

    def test_boundary_conditions(self, grid_world_env):
        """경계 조건 테스트"""
        # 환경을 특정 위치로 강제 설정
        grid_world_env.reset()

        # 왼쪽 경계에서 왼쪽으로 이동
        grid_world_env._agent_location = np.array([0, 2])
        observation, _, _, _, _ = grid_world_env.step(2)  # LEFT
        assert observation['agent'][0] == 0  # 경계에서 멈춤

        # 오른쪽 경계에서 오른쪽으로 이동
        grid_world_env._agent_location = np.array([4, 2])
        observation, _, _, _, _ = grid_world_env.step(0)  # RIGHT
        assert observation['agent'][0] == 4  # 경계에서 멈춤

        # 위쪽 경계에서 위로 이동
        grid_world_env._agent_location = np.array([2, 4])
        observation, _, _, _, _ = grid_world_env.step(1)  # UP
        assert observation['agent'][1] == 4  # 경계에서 멈춤

        # 아래쪽 경계에서 아래로 이동
        grid_world_env._agent_location = np.array([2, 0])
        observation, _, _, _, _ = grid_world_env.step(3)  # DOWN
        assert observation['agent'][1] == 0  # 경계에서 멈춤

    def test_goal_reaching(self, grid_world_env):
        """목표 도달 테스트"""
        grid_world_env.reset()

        # 에이전트를 목표 근처로 강제 이동
        target_pos = grid_world_env._target_location.copy()
        adjacent_pos = target_pos.copy()
        adjacent_pos[0] = max(0, min(4, target_pos[0] - 1))  # 왼쪽으로 한 칸

        grid_world_env._agent_location = adjacent_pos

        # 목표로 이동
        if target_pos[0] > adjacent_pos[0]:
            action = 0  # RIGHT
        else:
            action = 2  # LEFT

        observation, reward, terminated, truncated, info = grid_world_env.step(action)

        if np.array_equal(observation['agent'], target_pos):
            assert terminated == True
            assert reward == 1.0
            assert info['distance'] == 0.0

    def test_reward_system(self, grid_world_env):
        """보상 시스템 테스트"""
        observation, info = grid_world_env.reset()

        # 목표에 도달하지 않은 경우의 보상
        initial_distance = info['distance']
        if initial_distance > 0:  # 이미 목표에 도달하지 않은 경우
            observation, reward, terminated, truncated, info = grid_world_env.step(grid_world_env.action_space.sample())

            if not terminated:
                assert reward == -0.01  # 스텝 패널티
            else:
                assert reward == 1.0  # 목표 도달 보상

    def test_different_sizes(self):
        """다양한 그리드 크기 테스트"""
        for size in [3, 5, 8, 10]:
            env = gym.make('gymnasium_env/GridWorld-v0', size=size)
            actual_env = env.unwrapped
            observation, info = actual_env.reset()

            # 관찰 공간이 크기에 맞게 설정되었는지 확인
            agent_pos = observation['agent']
            target_pos = observation['target']

            assert 0 <= agent_pos[0] < size and 0 <= agent_pos[1] < size
            assert 0 <= target_pos[0] < size and 0 <= target_pos[1] < size

            # 실제 환경 크기 확인
            assert actual_env.size == size

            env.close()

    def test_deterministic_reset(self, grid_world_env):
        """결정적 초기화 테스트"""
        # 같은 시드로 두 번 초기화
        observation1, info1 = grid_world_env.reset(seed=42)
        observation2, info2 = grid_world_env.reset(seed=42)

        # 같은 시드로 초기화한 경우 같은 상태여야 함
        assert np.array_equal(observation1['agent'], observation2['agent'])
        assert np.array_equal(observation1['target'], observation2['target'])
        assert info1['distance'] == info2['distance']

    def test_rendering_modes(self):
        """렌더링 모드 테스트"""
        # human 모드
        env_human = gym.make('gymnasium_env/GridWorld-v0', render_mode='human')
        actual_env_human = env_human.unwrapped
        actual_env_human.reset()
        try:
            actual_env_human.render()  # 오류 없이 실행되어야 함
        except Exception as e:
            # 헤드리스 환경에서는 렌더링이 실패할 수 있음
            pytest.skip(f"Rendering failed in headless environment: {e}")
        finally:
            env_human.close()

        # rgb_array 모드
        env_rgb = gym.make('gymnasium_env/GridWorld-v0', render_mode='rgb_array')
        actual_env_rgb = env_rgb.unwrapped
        actual_env_rgb.reset()
        rgb_array = actual_env_rgb.render()
        if rgb_array is not None:
            assert isinstance(rgb_array, np.ndarray)
            assert len(rgb_array.shape) == 3  # (height, width, channels)
        env_rgb.close()

    def test_action_effects(self, grid_world_env):
        """각 액션의 효과 테스트"""
        grid_world_env.reset()

        # 중앙 위치에서 테스트
        grid_world_env._agent_location = np.array([2, 2])
        initial_pos = grid_world_env._agent_location.copy()

        # 각 방향으로 이동 테스트
        actions_and_expected = [
            (0, np.array([1, 0])),   # RIGHT
            (1, np.array([0, 1])),   # UP
            (2, np.array([-1, 0])),  # LEFT
            (3, np.array([0, -1]))   # DOWN
        ]

        for action, expected_delta in actions_and_expected:
            grid_world_env._agent_location = initial_pos.copy()
            observation, _, _, _, _ = grid_world_env.step(action)
            expected_pos = initial_pos + expected_delta
            assert np.array_equal(observation['agent'], expected_pos)

    def test_distance_calculation(self, grid_world_env):
        """거리 계산 정확성 테스트"""
        grid_world_env.reset()

        # 특정 위치 설정
        grid_world_env._agent_location = np.array([1, 1])
        grid_world_env._target_location = np.array([3, 4])

        observation = grid_world_env._get_obs()
        info = grid_world_env._get_info()

        # 맨해튼 거리 계산
        expected_distance = abs(3 - 1) + abs(4 - 1)  # |3-1| + |4-1| = 5
        assert info['distance'] == expected_distance

        # 거리 0인 경우 테스트
        grid_world_env._agent_location = np.array([2, 2])
        grid_world_env._target_location = np.array([2, 2])

        info = grid_world_env._get_info()
        assert info['distance'] == 0.0


if __name__ == "__main__":
    pytest.main([__file__])