"""
무작위 입력으로 GridWorld 시뮬레이션하는 코드
에이전트가 랜덤하게 행동하며 환경을 탐색합니다.
"""
import gymnasium as gym
import gymnasium_env
import time
import numpy as np


def random_simulation():
    """무작위 행동으로 GridWorld 환경을 시뮬레이션합니다."""
    print("=== GridWorld 무작위 시뮬레이션 시작 ===")

    # 환경 생성 (human 모드로 시각화)
    env = gym.make('gymnasium_env/GridWorld-v0', render_mode='human', size=5)

    # 통계 변수들
    total_episodes = 10
    episode_rewards = []
    episode_steps = []

    for episode in range(total_episodes):
        print(f"\n--- 에피소드 {episode + 1}/{total_episodes} ---")

        # 환경 초기화
        observation, info = env.reset()
        episode_reward = 0
        step_count = 0

        print(f"시작 위치:")
        print(f"  에이전트: {observation['agent']}")
        print(f"  목표: {observation['target']}")
        print(f"  맨해튼 거리: {info['distance']:.1f}")

        # ASCII 렌더링으로 초기 상태 표시
        env.render()

        while True:
            # 무작위 행동 선택
            action = env.action_space.sample()
            action_names = ['오른쪽', '위', '왼쪽', '아래']

            print(f"스텝 {step_count + 1}: {action_names[action]} 이동")

            # 행동 실행
            observation, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            step_count += 1

            print(f"  현재 위치: {observation['agent']}")
            print(f"  보상: {reward:.3f}, 거리: {info['distance']:.1f}")

            # 에피소드 종료 확인
            if terminated:
                print(f"🎉 목표 도달! 총 {step_count}스텝으로 완료")
                break
            elif truncated:
                print(f"⏰ 최대 스텝 수 도달 ({step_count}스텝)")
                break

            # 시각화를 위한 딜레이
            time.sleep(0.5)

        episode_rewards.append(episode_reward)
        episode_steps.append(step_count)

        print(f"에피소드 보상: {episode_reward:.3f}")
        print(f"에피소드 스텝: {step_count}")


random_simulation()