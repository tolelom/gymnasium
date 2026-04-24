"""GridWorld 학습 설정"""

from dataclasses import dataclass


@dataclass
class GridWorldConfig:
    size: int = 5
    render_mode: str = "rgb_array"


@dataclass
class TrainingConfig:
    total_timesteps: int = 10000
    verbose: int = 1
    tensorboard_log: str = "./logs/"
    save_path: str = "./models/"


@dataclass
class A2CConfig:
    learning_rate: float = 0.0007
    n_steps: int = 5
    gamma: float = 0.99
    gae_lambda: float = 1.0
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    policy: str = "MultiInputPolicy"


@dataclass
class PPOConfig:
    learning_rate: float = 0.0003
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    policy: str = "MultiInputPolicy"
