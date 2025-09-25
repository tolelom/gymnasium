"""pytest 설정 및 공용 픽스처"""

import pytest
import gymnasium as gym
import gymnasium_env
import numpy as np

@pytest.fixture
def grid_world_env():
    """기본 GridWorld 환경 픽스처 (unwrapped)"""
    env = gym.make('gymnasium_env/GridWorld-v0', size=5)
    actual_env = env.unwrapped  # ✨ 여기서 unwrap!
    yield actual_env
    env.close()

@pytest.fixture
def wrapped_grid_world_env():
    """래퍼가 적용된 환경 (필요시 사용)"""
    env = gym.make('gymnasium_env/GridWorld-v0', size=5)
    yield env  # wrapped 상태 그대로
    env.close()
