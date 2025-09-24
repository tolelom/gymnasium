from typing import Optional
from enum import Enum

import numpy as np
import pygame

import gymnasium as gym
from gymnasium import spaces

class Actions(Enum):
    RIGHT = 0
    UP = 1
    LEFT = 2
    DOWN = 3

class GridWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, size: int = 5):
        self.size = size
        self.window_size = 512

        self.observation_space = gym.spaces.Dict(
            {
                "agent": gym.spaces.Box(0, size - 1, shape=(2,), dtype=np.int32),
                "target": gym.spaces.Box(0, size - 1, shape=(2,), dtype=np.int32),
            }
        )
        self._agent_location = np.array([-1, -1], dtype=np.int32)
        self._target_location = np.array([-1, -1], dtype=np.int32)

        self.action_space = gym.spaces.Discrete(4)

        self._action_to_direction = {
            Actions.RIGHT.value: np.array([1, 0]),
            Actions.UP.value: np.array([0, 1]),
            Actions.LEFT.value: np.array([-1, 0]),
            Actions.DOWN.value: np.array([0, -1]),
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.window = None
        self.clock = None

    def _get_obs(self):
        return {
            "agent": self._agent_location,
            "target": self._target_location,
        }

    def _get_info(self):
        return {
            "distance": np.linalg.norm(
                self._agent_location - self._target_location, ord=1
            )
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self._agent_location = self.np_random.integers(0, self.size, size=2, dtype=int)
        self._target_location = self._agent_location
        while np.array_equal(self._target_location, self._agent_location):
            self._target_location = self.np_random.integers(0, self.size, size=2, dtype=int)

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        direction = self._action_to_direction[action]

        self._agent_location = np.clip(
            self._agent_location + direction, 0, self.size - 1
        )

        terminated = np.array_equal(self._agent_location, self._target_location)

        truncated = False

        reward = 1 if terminated else -0.01

        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, truncated, info

    def render(self):
        """Render the environment for human viewing."""
        if self.render_mode == "human":
            # Print a simple ASCII representation
            for y in range(self.size - 1, -1, -1):  # Top to bottom
                row = ""
                for x in range(self.size):
                    if np.array_equal([x, y], self._agent_location):
                        row += "A "  # Agent
                    elif np.array_equal([x, y], self._target_location):
                        row += "T "  # Target
                    else:
                        row += ". "  # Empty
                print(row)
            print()

