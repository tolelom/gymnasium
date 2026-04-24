# Gymnasium 커스텀 환경

강화학습을 위한 커스텀 Gymnasium 환경 모음. Stable Baselines3 호환.

## 환경

- **GridWorld** — 5x5 그리드에서 타겟을 찾아가는 기본 환경
- **BattleSsafy** — 배틀시티 기반 환경 (개발 중, 현재 GridWorld 복제 상태)

## Tech Stack

- Gymnasium + Stable Baselines3 (A2C, PPO)
- pygame (렌더링)
- pytest + pytest-benchmark (테스트/벤치마크)
- TensorBoard (학습 시각화)

## 구조

```
gymnasium_env/envs/     # 커스텀 환경 구현
training/               # 학습 코드 + 하이퍼파라미터 설정
tests/                  # 단위/통합/벤치마크 테스트
tests/simulation/       # 학습된/랜덤 정책 시뮬레이션 스크립트
```

## 설치

```bash
pip install -e ".[dev]"
```

## 학습

```bash
python training/grid_world/train.py
# 모델: ./models/a2c_gridworld.zip, 로그: ./logs/
tensorboard --logdir ./logs/
```

## 테스트

```bash
# 단위 + 통합
pytest tests/unit tests/integration

# 벤치마크
pytest tests/benchmarks --benchmark-only
```

## 시뮬레이션

```bash
# 랜덤 정책
python tests/simulation/grid_world/random_simulation.py

# 학습된 모델 (models/ 아래에 저장된 zip 필요)
python tests/simulation/grid_world/learned_simulation.py
```

## 관련 프로젝트

- [battle_city_deep_learning](https://github.com/tolelom/battle_city_deep_learning) — 배틀시티 강화학습
