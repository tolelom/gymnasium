"""공통 시뮬레이션 유틸리티 함수들"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Tuple
import time
import gymnasium as gym


def calculate_episode_stats(episode_rewards: List[float],
                            episode_steps: List[int]) -> Dict[str, float]:
    """
    에피소드 통계 계산

    Args:
        episode_rewards: 에피소드별 보상 리스트
        episode_steps: 에피소드별 스텝 수 리스트

    Returns:
        통계 딕셔너리
    """
    if not episode_rewards or not episode_steps:
        return {}

    return {
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'min_reward': np.min(episode_rewards),
        'max_reward': np.max(episode_rewards),
        'mean_steps': np.mean(episode_steps),
        'std_steps': np.std(episode_steps),
        'min_steps': np.min(episode_steps),
        'max_steps': np.max(episode_steps),
        'success_rate': np.mean([r > 0 for r in episode_rewards]) * 100,
        'total_episodes': len(episode_rewards)
    }


def print_episode_stats(stats: Dict[str, float], title: str = "Episode Statistics"):
    """
    에피소드 통계 출력

    Args:
        stats: calculate_episode_stats에서 반환된 통계
        title: 출력 제목
    """
    print(f"\n{'=' * 50}")
    print(f"{title:^50}")
    print(f"{'=' * 50}")
    print(f"총 에피소드: {stats.get('total_episodes', 0)}")
    print(f"성공률: {stats.get('success_rate', 0):.1f}%")
    print(f"평균 보상: {stats.get('mean_reward', 0):.3f} ± {stats.get('std_reward', 0):.3f}")
    print(f"보상 범위: [{stats.get('min_reward', 0):.3f}, {stats.get('max_reward', 0):.3f}]")
    print(f"평균 스텝: {stats.get('mean_steps', 0):.1f} ± {stats.get('std_steps', 0):.1f}")
    print(f"스텝 범위: [{stats.get('min_steps', 0)}, {stats.get('max_steps', 0)}]")
    print("=" * 50)


def plot_training_curves(episode_rewards: List[float],
                         episode_steps: List[int],
                         window_size: int = 10,
                         save_path: str = None):
    """
    훈련 곡선 플롯

    Args:
        episode_rewards: 에피소드별 보상
        episode_steps: 에피소드별 스텝 수
        window_size: 이동 평균 윈도우 크기
        save_path: 저장할 파일 경로 (None이면 표시만)
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    episodes = range(1, len(episode_rewards) + 1)

    # 보상 곡선
    ax1.plot(episodes, episode_rewards, alpha=0.6, label='Episode Reward')
    if len(episode_rewards) >= window_size:
        moving_avg = np.convolve(episode_rewards,
                                 np.ones(window_size) / window_size, mode='valid')
        ax1.plot(range(window_size, len(episode_rewards) + 1), moving_avg,
                 'r-', linewidth=2, label=f'Moving Average ({window_size})')

    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.set_title('Training Rewards')
    ax1.legend()
    ax1.grid(True)

    # 스텝 곡선
    ax2.plot(episodes, episode_steps, alpha=0.6, label='Episode Steps')
    if len(episode_steps) >= window_size:
        moving_avg_steps = np.convolve(episode_steps,
                                       np.ones(window_size) / window_size, mode='valid')
        ax2.plot(range(window_size, len(episode_steps) + 1), moving_avg_steps,
                 'r-', linewidth=2, label=f'Moving Average ({window_size})')

    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Steps')
    ax2.set_title('Training Steps')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()


def run_evaluation_episodes(env: gym.Env,
                            model,
                            n_episodes: int = 10,
                            deterministic: bool = True,
                            render: bool = False,
                            delay: float = 0.0) -> Tuple[List[float], List[int]]:
    """
    모델 평가 에피소드 실행

    Args:
        env: 테스트할 환경
        model: 평가할 모델 (predict 메서드 필요)
        n_episodes: 실행할 에피소드 수
        deterministic: 결정적 정책 사용 여부
        render: 렌더링 여부
        delay: 스텝 간 딜레이 (초)

    Returns:
        (episode_rewards, episode_steps) 튜플
    """
    episode_rewards = []
    episode_steps = []

    for episode in range(n_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        step_count = 0

        if render:
            env.render()
            time.sleep(delay)

        while True:
            if hasattr(model, 'predict'):
                action, _ = model.predict(obs, deterministic=deterministic)
            else:
                # 랜덤 정책 fallback
                action = env.action_space.sample()

            obs, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            step_count += 1

            if render:
                env.render()
                time.sleep(delay)

            if terminated or truncated:
                break

        episode_rewards.append(episode_reward)
        episode_steps.append(step_count)

        if render:
            print(f"Episode {episode + 1}: Reward={episode_reward:.3f}, Steps={step_count}")

    return episode_rewards, episode_steps


def create_environment_summary(env: gym.Env) -> Dict[str, Any]:
    """
    환경 요약 정보 생성

    Args:
        env: 분석할 환경

    Returns:
        환경 정보 딕셔너리
    """
    summary = {
        'env_id': env.spec.id if env.spec else "Unknown",
        'action_space': {
            'type': type(env.action_space).__name__,
            'shape': getattr(env.action_space, 'shape', None),
            'n': getattr(env.action_space, 'n', None)
        },
        'observation_space': {
            'type': type(env.observation_space).__name__,
            'shape': getattr(env.observation_space, 'shape', None)
        },
        'max_episode_steps': getattr(env, '_max_episode_steps', None),
        'render_modes': getattr(env.metadata, 'render_modes', []) if hasattr(env, 'metadata') else []
    }

    # Dict 관찰공간인 경우 상세 정보 추가
    if hasattr(env.observation_space, 'spaces'):
        summary['observation_space']['spaces'] = {
            key: {
                'type': type(space).__name__,
                'shape': getattr(space, 'shape', None),
                'low': getattr(space, 'low', None),
                'high': getattr(space, 'high', None)
            }
            for key, space in env.observation_space.spaces.items()
        }

    return summary


def print_environment_summary(env: gym.Env):
    """환경 요약 정보 출력"""
    summary = create_environment_summary(env)

    print("\n" + "=" * 60)
    print("ENVIRONMENT SUMMARY".center(60))
    print("=" * 60)
    print(f"Environment ID: {summary['env_id']}")
    print(f"Max Episode Steps: {summary['max_episode_steps']}")
    print(f"Render Modes: {summary['render_modes']}")

    print(f"\nAction Space:")
    print(f"  Type: {summary['action_space']['type']}")
    if summary['action_space']['n']:
        print(f"  Size: {summary['action_space']['n']}")
    if summary['action_space']['shape']:
        print(f"  Shape: {summary['action_space']['shape']}")

    print(f"\nObservation Space:")
    print(f"  Type: {summary['observation_space']['type']}")
    if 'spaces' in summary['observation_space']:
        print(f"  Spaces:")
        for key, space_info in summary['observation_space']['spaces'].items():
            print(f"    {key}: {space_info['type']} {space_info['shape']}")
            if space_info['low'] is not None:
                print(f"      Range: [{space_info['low']}, {space_info['high']}]")
    elif summary['observation_space']['shape']:
        print(f"  Shape: {summary['observation_space']['shape']}")

    print("=" * 60)


def compare_algorithms(results: Dict[str, Dict[str, float]],
                       metric: str = 'success_rate') -> List[Tuple[str, float]]:
    """
    알고리즘 성능 비교

    Args:
        results: {algorithm_name: stats_dict} 형태의 결과
        metric: 비교할 메트릭 (success_rate, mean_reward, mean_steps 등)

    Returns:
        (algorithm_name, metric_value) 튜플의 정렬된 리스트
    """
    comparison = []
    for algo_name, stats in results.items():
        if metric in stats:
            comparison.append((algo_name, stats[metric]))

    # 성공률과 보상은 높은 순으로, 스텝은 낮은 순으로 정렬
    reverse = metric in ['success_rate', 'mean_reward', 'max_reward']
    return sorted(comparison, key=lambda x: x[1], reverse=reverse)


def save_experiment_results(results: Dict[str, Any],
                            filename: str = "experiment_results.json"):
    """
    실험 결과를 JSON 파일로 저장

    Args:
        results: 저장할 결과 딕셔너리
        filename: 저장할 파일명
    """
    import json
    from datetime import datetime

    # 시간 스탬프 추가
    results['timestamp'] = datetime.now().isoformat()

    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        print(f"Results saved to {filename}")
    except Exception as e:
        print(f"Failed to save results: {e}")


if __name__ == "__main__":
    # 유틸리티 함수 테스트
    print("Simulation utilities test")

    # 더미 데이터로 통계 계산 테스트
    dummy_rewards = [0.5, 0.8, 1.0, 0.2, 0.9]
    dummy_steps = [25, 30, 15, 50, 20]

    stats = calculate_episode_stats(dummy_rewards, dummy_steps)
    print_episode_stats(stats, "Test Statistics")
