"""GridWorld 학습 설정 파일"""

from typing import Dict, Any
from dataclasses import dataclass


@dataclass
class GridWorldConfig:
    """GridWorld 환경 설정"""
    size: int = 5
    render_mode: str = "rgb_array"
    max_episode_steps: int = 100


@dataclass
class TrainingConfig:
    """학습 설정"""
    total_timesteps: int = 10000
    verbose: int = 1
    tensorboard_log: str = "./logs/"
    save_path: str = "./models/"
    eval_episodes: int = 10
    eval_freq: int = 1000


@dataclass
class A2CConfig:
    """A2C 알고리즘 설정"""
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
    """PPO 알고리즘 설정"""
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


@dataclass
class DQNConfig:
    """DQN 알고리즘 설정 (참고용 - Dict observation 지원 제한)"""
    learning_rate: float = 0.0001
    buffer_size: int = 10000
    learning_starts: int = 1000
    batch_size: int = 32
    tau: float = 1.0
    gamma: float = 0.99
    train_freq: int = 4
    gradient_steps: int = 1
    target_update_interval: int = 1000
    exploration_fraction: float = 0.1
    exploration_initial_eps: float = 1.0
    exploration_final_eps: float = 0.05
    policy: str = "MlpPolicy"  # Dict observation 미지원으로 사용 제한


# 기본 설정들
DEFAULT_CONFIGS = {
    'grid_world': {
        'env': GridWorldConfig(),
        'training': TrainingConfig(),
        'algorithms': {
            'A2C': A2CConfig(),
            'PPO': PPOConfig(),
            'DQN': DQNConfig()  # 사용 시 주의 필요
        }
    }
}


def get_config(env_name: str = 'grid_world',
               algorithm: str = 'A2C') -> Dict[str, Any]:
    """
    설정 가져오기

    Args:
        env_name: 환경 이름
        algorithm: 알고리즘 이름

    Returns:
        설정 딕셔너리
    """
    if env_name not in DEFAULT_CONFIGS:
        raise ValueError(f"Unknown environment: {env_name}")

    env_config = DEFAULT_CONFIGS[env_name]

    if algorithm not in env_config['algorithms']:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    return {
        'env_config': env_config['env'],
        'training_config': env_config['training'],
        'algorithm_config': env_config['algorithms'][algorithm]
    }


def create_hyperparameter_sweep() -> Dict[str, Dict[str, Any]]:
    """
    하이퍼파라미터 스윕을 위한 설정 생성

    Returns:
        다양한 하이퍼파라미터 조합
    """
    sweep_configs = {}

    # A2C 하이퍼파라미터 스윕
    learning_rates = [0.0003, 0.0007, 0.001]
    n_steps_list = [5, 10, 16]

    for i, lr in enumerate(learning_rates):
        for j, n_steps in enumerate(n_steps_list):
            config_name = f"A2C_lr{lr}_steps{n_steps}"

            base_config = get_config('grid_world', 'A2C')
            base_config['algorithm_config'].learning_rate = lr
            base_config['algorithm_config'].n_steps = n_steps

            sweep_configs[config_name] = base_config

    return sweep_configs


def save_config(config: Dict[str, Any], filepath: str):
    """설정을 JSON 파일로 저장"""
    import json
    from dataclasses import asdict

    # 데이터클래스를 딕셔너리로 변환
    serializable_config = {}
    for key, value in config.items():
        if hasattr(value, '__dict__'):
            serializable_config[key] = asdict(value)
        else:
            serializable_config[key] = value

    with open(filepath, 'w') as f:
        json.dump(serializable_config, f, indent=2)


def load_config(filepath: str) -> Dict[str, Any]:
    """JSON 파일에서 설정 로드"""
    import json

    with open(filepath, 'r') as f:
        data = json.load(f)

    # 다시 데이터클래스로 변환 (필요한 경우)
    config = {}
    for key, value in data.items():
        if key == 'env_config':
            config[key] = GridWorldConfig(**value)
        elif key == 'training_config':
            config[key] = TrainingConfig(**value)
        elif key == 'algorithm_config':
            # 알고리즘 타입에 따라 적절한 클래스 사용
            if value.get('policy') == 'MultiInputPolicy':
                if 'clip_range' in value:
                    config[key] = PPOConfig(**value)
                else:
                    config[key] = A2CConfig(**value)
            else:
                config[key] = DQNConfig(**value)
        else:
            config[key] = value

    return config


if __name__ == "__main__":
    # 설정 테스트
    print("GridWorld Configuration Test")

    # 기본 설정 출력
    config = get_config('grid_world', 'A2C')
    print("\nA2C Configuration:")
    for key, value in config.items():
        print(f"{key}: {value}")

    # 하이퍼파라미터 스윕 테스트
    sweep = create_hyperparameter_sweep()
    print(f"\nHyperparameter sweep generated {len(sweep)} configurations")

    # 설정 저장/로드 테스트
    test_file = "test_config.json"
    save_config(config, test_file)
    loaded_config = load_config(test_file)
    print(f"\nConfiguration saved and loaded successfully")

    import os

    os.remove(test_file)  # 테스트 파일 정리
