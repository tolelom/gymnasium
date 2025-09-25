"""pytest 설정 및 공용 픽스처"""

import pytest
import gymnasium as gym
import gymnasium_env
import numpy as np


@pytest.fixture
def grid_world_env():
    """기본 GridWorld 환경 픽스처"""
    env = gym.make('gymnasium_env/GridWorld-v0', size=5)
    yield env
    env.close()


@pytest.fixture
def large_grid_world_env():
    """큰 GridWorld 환경 픽스처"""
    env = gym.make('gymnasium_env/GridWorld-v0', size=10)
    yield env
    env.close()


@pytest.fixture
def grid_world_env_with_render():
    """렌더링 가능한 GridWorld 환경 픽스처"""
    env = gym.make('gymnasium_env/GridWorld-v0', size=5, render_mode='rgb_array')
    yield env
    env.close()


@pytest.fixture
def deterministic_env():
    """결정적 GridWorld 환경 픽스처 (고정 시드)"""
    env = gym.make('gymnasium_env/GridWorld-v0', size=5)
    env.reset(seed=42)
    yield env
    env.close()


@pytest.fixture
def sample_episode_data():
    """샘플 에피소드 데이터 픽스처"""
    return {
        'rewards': [0.5, 0.8, 1.0, 0.2, 0.9, 0.1, 0.7],
        'steps': [25, 30, 15, 50, 20, 45, 22],
        'success_count': 4
    }


# pytest 설정
def pytest_configure(config):
    """pytest 실행 시 초기 설정"""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "benchmark: marks tests as performance benchmarks"
    )


def pytest_collection_modifyitems(config, items):
    """테스트 항목 수정"""
    if config.getoption("--runslow"):
        # --runslow가 주어진 경우 slow 마커를 건너뛰지 않음
        return
    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


def pytest_addoption(parser):
    """명령행 옵션 추가"""
    parser.addoption(
        "--runslow", action="store_true", default=False, help="run slow tests"
    )
