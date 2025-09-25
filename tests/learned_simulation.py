"""
학습된 모델로 GridWorld를 시뮬레이션하는 코드
훈련된 강화학습 에이전트의 성능을 시각적으로 확인합니다.
"""
import gymnasium as gym
import gymnasium_env
from stable_baselines3 import A2C, PPO, DQN
import numpy as np
import time
import os


def load_and_simulate(model_path, model_type='A2C', episodes=5, render_mode='human'):
    """
    학습된 모델을 로드하여 GridWorld 시뮬레이션을 실행합니다.

    Args:
        model_path: 학습된 모델 파일 경로
        model_type: 모델 타입 ('A2C', 'PPO', 'DQN')
        episodes: 실행할 에피소드 수
        render_mode: 렌더링 모드 ('human', 'rgb_array')
    """
    print(f"=== 학습된 {model_type} 모델 시뮬레이션 ===")

    # 환경 생성
    env = gym.make('gymnasium_env/GridWorld-v0', render_mode=render_mode, size=5)

    # 모델 로드
    try:
        if model_type == 'A2C':
            model = A2C.load(model_path)
        elif model_type == 'PPO':
            model = PPO.load(model_path)
        elif model_type == 'DQN':
            model = DQN.load(model_path)
        else:
            raise ValueError(f"지원하지 않는 모델 타입: {model_type}")

        print(f"✅ 모델 로드 완료: {model_path}")
    except Exception as e:
        print(f"❌ 모델 로드 실패: {e}")
        return

    # 시뮬레이션 통계
    episode_rewards = []
    episode_steps = []
    success_count = 0

    for episode in range(episodes):
        print(f"\n--- 에피소드 {episode + 1}/{episodes} ---")

        obs, info = env.reset()
        episode_reward = 0
        step_count = 0

        print(f"시작 상태:")
        print(f"  에이전트: {obs['agent']}")
        print(f"  목표: {obs['target']}")
        print(f"  초기 거리: {info['distance']:.1f}")

        # 환경 렌더링
        if render_mode == 'human':
            env.render()
            time.sleep(1)  # 초기 상태 확인 시간

        while True:
            # 학습된 정책으로 행동 선택
            action, _ = model.predict(obs, deterministic=True)
            action_names = ['오른쪽', '위', '왼쪽', '아래']

            print(f"스텝 {step_count + 1}: 정책 선택 -> {action_names[action]}")

            # 행동 실행
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            step_count += 1

            print(f"  결과: 위치 {obs['agent']}, 보상 {reward:.3f}, 거리 {info['distance']:.1f}")

            # 렌더링 및 딜레이
            if render_mode == 'human':
                env.render()
                time.sleep(0.8)  # 사람이 보기 좋은 속도

            # 종료 조건 확인
            if terminated:
                print(f"🎯 목표 달성! {step_count}스텝으로 성공")
                success_count += 1
                break
            elif truncated:
                print(f"⏱️ 시간 초과 ({step_count}스텝)")
                break

        episode_rewards.append(episode_reward)
        episode_steps.append(step_count)

        print(f"에피소드 결과: 보상 {episode_reward:.3f}, 스텝 {step_count}")

        # 다음 에피소드 전 대기
        if render_mode == 'human' and episode < episodes - 1:
            time.sleep(2)

    env.close()

    # 최종 통계 출력
    print(f"\n=== {model_type} 모델 성능 분석 ===")
    print(f"총 에피소드: {episodes}")
    print(f"성공률: {success_count / episodes * 100:.1f}% ({success_count}/{episodes})")
    print(f"평균 보상: {np.mean(episode_rewards):.3f} ± {np.std(episode_rewards):.3f}")
    print(f"평균 스텝: {np.mean(episode_steps):.1f} ± {np.std(episode_steps):.1f}")
    if success_count > 0:
        success_steps = [steps for i, steps in enumerate(episode_steps)
                         if episode_rewards[i] > 0]
        print(f"성공 시 평균 스텝: {np.mean(success_steps):.1f}")
    print(f"최고 성능: {np.max(episode_rewards):.3f} 보상, {np.min(episode_steps)} 스텝")


def compare_models(model_configs, episodes=3):
    """
    여러 학습된 모델을 비교하여 시뮬레이션합니다.

    Args:
        model_configs: [(model_path, model_type, name), ...] 형태의 리스트
        episodes: 각 모델당 실행할 에피소드 수
    """
    print("=== 다중 모델 성능 비교 ===")

    results = {}

    for model_path, model_type, name in model_configs:
        if not os.path.exists(model_path):
            print(f"⚠️ 모델 파일이 존재하지 않음: {model_path}")
            continue

        print(f"\n🤖 {name} 모델 테스트 중...")

        # 환경 생성 (렌더링 없이 빠르게)
        env = gym.make('gymnasium_env/GridWorld-v0', size=5)

        try:
            # 모델 로드
            if model_type == 'A2C':
                model = A2C.load(model_path)
            elif model_type == 'PPO':
                model = PPO.load(model_path)
            elif model_type == 'DQN':
                model = DQN.load(model_path)

            episode_rewards = []
            episode_steps = []
            success_count = 0

            # 빠른 시뮬레이션 (렌더링 없이)
            for episode in range(episodes):
                obs, info = env.reset()
                episode_reward = 0
                step_count = 0

                while True:
                    action, _ = model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, info = env.step(action)
                    episode_reward += reward
                    step_count += 1

                    if terminated:
                        success_count += 1
                        break
                    elif truncated:
                        break

                episode_rewards.append(episode_reward)
                episode_steps.append(step_count)

            env.close()

            # 결과 저장
            results[name] = {
                'success_rate': success_count / episodes * 100,
                'avg_reward': np.mean(episode_rewards),
                'avg_steps': np.mean(episode_steps),
                'std_reward': np.std(episode_rewards),
                'std_steps': np.std(episode_steps)
            }

            print(f"✅ {name}: 성공률 {results[name]['success_rate']:.1f}%, "
                  f"평균 보상 {results[name]['avg_reward']:.3f}")

        except Exception as e:
            print(f"❌ {name} 테스트 실패: {e}")

    # 비교 결과 출력
    if results:
        print(f"\n{'=' * 50}")
        print("🏆 모델 성능 비교 결과")
        print(f"{'=' * 50}")
        print(f"{'모델명':<15} {'성공률':<10} {'평균보상':<12} {'평균스텝':<10}")
        print("-" * 50)

        for name, stats in sorted(results.items(),
                                  key=lambda x: x[1]['success_rate'],
                                  reverse=True):
            print(f"{name:<15} {stats['success_rate']:>7.1f}%  "
                  f"{stats['avg_reward']:>9.3f}    {stats['avg_steps']:>7.1f}")


def interactive_model_demo():
    """대화형 모델 데모 - 사용자가 모델을 선택하여 시뮬레이션"""
    print("=== 학습된 모델 데모 시스템 ===")

    # 사용 가능한 모델 검색
    model_dir = "models"  # 모델 저장 디렉토리
    if not os.path.exists(model_dir):
        print(f"⚠️ 모델 디렉토리가 없습니다: {model_dir}")
        print("먼저 모델을 학습시켜주세요.")
        return

    available_models = []
    for filename in os.listdir(model_dir):
        if filename.endswith('.zip'):
            available_models.append(filename)

    if not available_models:
        print("❌ 사용 가능한 모델이 없습니다.")
        return

    print("사용 가능한 모델:")
    for i, model in enumerate(available_models):
        print(f"{i + 1}. {model}")

    try:
        choice = int(input(f"모델 선택 (1-{len(available_models)}): ")) - 1
        if 0 <= choice < len(available_models):
            selected_model = available_models[choice]
            model_path = os.path.join(model_dir, selected_model)

            # 모델 타입 추정
            if 'a2c' in selected_model.lower():
                model_type = 'A2C'
            elif 'ppo' in selected_model.lower():
                model_type = 'PPO'
            elif 'dqn' in selected_model.lower():
                model_type = 'DQN'
            else:
                model_type = input("모델 타입을 입력하세요 (A2C/PPO/DQN): ").upper()

            episodes = int(input("실행할 에피소드 수 (기본 3): ") or 3)

            load_and_simulate(model_path, model_type, episodes, 'human')
        else:
            print("올바른 번호를 선택하세요.")
    except (ValueError, KeyboardInterrupt):
        print("입력이 취소되었습니다.")


if __name__ == "__main__":
    print("GridWorld 학습된 모델 시뮬레이션")
    print("1. 특정 모델 시뮬레이션")
    print("2. 다중 모델 성능 비교")
    print("3. 대화형 모델 데모")

    try:
        choice = input("선택 (1-3): ")

        if choice == "1":
            model_path = input("모델 파일 경로: ")
            model_type = input("모델 타입 (A2C/PPO/DQN): ").upper()
            episodes = int(input("에피소드 수 (기본 5): ") or 5)
            load_and_simulate(model_path, model_type, episodes)

        elif choice == "2":
            # 예시 모델 설정 (실제 경로로 수정 필요)
            configs = [
                ("models/a2c_gridworld.zip", "A2C", "A2C 기본"),
                ("models/ppo_gridworld.zip", "PPO", "PPO 기본"),
                ("models/dqn_gridworld.zip", "DQN", "DQN 기본"),
            ]
            compare_models(configs, episodes=5)

        elif choice == "3":
            interactive_model_demo()

        else:
            print("올바른 선택지를 입력하세요.")

    except KeyboardInterrupt:
        print("\n프로그램이 중단되었습니다.")
