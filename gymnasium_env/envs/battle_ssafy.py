from typing import Optional, Tuple
from enum import Enum

import numpy as np
import pygame

import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import env_checker


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
            Actions.RIGHT.value: np.array([1, 0], dtype=np.int32),
            Actions.UP.value: np.array([0, 1], dtype=np.int32),
            Actions.LEFT.value: np.array([-1, 0], dtype=np.int32),
            Actions.DOWN.value: np.array([0, -1], dtype=np.int32),
        }
        self.last_agent_action_to_derection = 0

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

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[dict, dict]:
        super().reset(seed=seed)

        self._agent_location = self.np_random.integers(0, self.size, size=2, dtype=np.int32)
        self._target_location = self._agent_location
        while np.array_equal(self._target_location, self._agent_location):
            self._target_location = self.np_random.integers(0, self.size, size=2, dtype=np.int32)

        observation = self._get_obs()
        info = self._get_info()

        self.last_agent_action_to_derection = 0

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        if isinstance(action, np.ndarray):
            action = action.item()

        direction = self._action_to_direction[action]

        self.last_agent_action_to_derection = action
        self._agent_location = np.clip(
            self._agent_location + direction, 0, self.size - 1
        )

        terminated = np.array_equal(self._agent_location, self._target_location)

        truncated = False

        reward = 1 if terminated else -0.01

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

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
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.window_size, self.window_size)
            )
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = (
                self.window_size / self.size
        )

        agent_image = pygame.image.load("C:/Users/SSAFY/PycharmProjects/gymnasium/gymnasium_env/envs/agent.png")
        target_image = pygame.image.load("C:/Users/SSAFY/PycharmProjects/gymnasium/gymnasium_env/envs/target.png")
        grass_image = pygame.image.load("C:/Users/SSAFY/PycharmProjects/gymnasium/gymnasium_env/envs/grass.png")
        rock_image = pygame.image.load("C:/Users/SSAFY/PycharmProjects/gymnasium/gymnasium_env/envs/rock.png")
        tree_image = pygame.image.load("C:/Users/SSAFY/PycharmProjects/gymnasium/gymnasium_env/envs/tree.png")

        agent_image = pygame.transform.scale(agent_image, (pix_square_size, pix_square_size));
        target_image = pygame.transform.scale(target_image, (pix_square_size, pix_square_size));
        tree_image = pygame.transform.scale(tree_image, (pix_square_size, pix_square_size));
        rock_image = pygame.transform.scale(rock_image, (pix_square_size, pix_square_size));
        grass_image = pygame.transform.scale(grass_image, (pix_square_size, pix_square_size));

        for x in range(self.size):
            for y in range(self.size):
                canvas.blit(grass_image, (pix_square_size * x, pix_square_size * y))

        if self.last_agent_action_to_derection == 1:
            agent_image = pygame.transform.rotate(agent_image, 90)
        elif self.last_agent_action_to_derection == 0:
            agent_image = pygame.transform.rotate(agent_image, 180)
        elif self.last_agent_action_to_derection == 3:
            agent_image = pygame.transform.rotate(agent_image, 270)


        print("last_agent_action_to_derection", self.last_agent_action_to_derection)
        canvas.blit(target_image, (pix_square_size * self._target_location))
        canvas.blit(agent_image, (pix_square_size * self._agent_location))


        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())

            pygame.event.pump()
            pygame.display.update()

            self.clock.tick(self.metadata["render_fps"])
        else:
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
