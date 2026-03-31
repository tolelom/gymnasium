# Gymnasium 커스텀 환경

강화학습을 위한 커스텀 Gymnasium 환경 모음. Stable Baselines3 호환.

## 환경

- **GridWorld** — 5x5 그리드에서 타겟을 찾아가는 기본 환경
- **BattleSsafy** — 배틀시티 기반 환경

## Tech Stack

- Gymnasium + Stable Baselines3 (A2C, PPO)
- pygame (렌더링)
- pytest + benchmark (테스트)
- TensorBoard (학습 시각화)

## 구조

```
gymnasium_env/envs/     # 커스텀 환경 구현
training/               # 학습 코드 + 하이퍼파라미터 설정
tests/                  # 단위/통합/벤치마크 테스트
```

## 실행

```bash
pip install -e ".[dev]"

# 학습
python training/grid_world/train.py

# 테스트
pytest tests/
```

## 관련 프로젝트

- [battle_city_deep_learning](https://github.com/tolelom/battle_city_deep_learning) — 배틀시티 강화학습
